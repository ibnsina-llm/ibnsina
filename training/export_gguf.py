#!/usr/bin/env python3
"""Export a nanochat-trained Llama-arch (nanochat/llama.py) or Qwen3-arch (nanochat/qwen3.py) checkpoint to standard GGUF:
architecture "llama" or "qwen3" (read from the checkpoint meta), GPT-2-style byte-level BPE vocab + merges derived from our
tiktoken ranks, pre-tokenizer "llama-bpe" (requires a tokenizer trained with the Llama-3 regex, i.e. train_tokenizer.py
--pattern llama3 — a property of the tokenizer, not the arch), nanochat chat template. Q/K permute for llama only:
LLM_ARCH_LLAMA ropes in "normal" (interleaved) mode, LLM_ARCH_QWEN3 in NEOX mode == the HF half-split layout as trained,
so qwen3 tensors go out unpermuted, plus per-layer blk.N.attn_q_norm.weight / attn_k_norm.weight (per-head RMSNorm gains).

  python export_gguf.py --base-dir /data/nc --source base|sft --model-tag toy [--step N] --out /data/export/toy-f16.gguf [--dtype f16|f32]
  llama-quantize /data/export/toy-f16.gguf /data/export/toy-Q4_K_M.gguf Q4_K_M
"""
import argparse, json, os, pickle, sys
import numpy as np
import torch
import gguf
from gguf import GGUFWriter, TokenType

SPECIALS = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
            "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]
# nanochat's render_conversation: bos, then <|user_start|>..<|user_end|> / <|assistant_start|>..<|assistant_end|>; a system message is
# folded into the first user turn as "system\n\nuser". BOS is added by the runtime (add_bos_token=true), not by the template.
CHAT_TEMPLATE = ("{% set ns = namespace(sys='') %}{% for m in messages %}"
                 "{% if m['role'] == 'system' %}{% set ns.sys = m['content'] ~ '\\n\\n' %}"
                 "{% elif m['role'] == 'user' %}<|user_start|>{{ ns.sys }}{{ m['content'] }}<|user_end|>{% set ns.sys = '' %}"
                 "{% elif m['role'] == 'assistant' %}<|assistant_start|>{{ m['content'] }}<|assistant_end|>{% endif %}{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant_start|>{% endif %}")


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def bpe_split(ranks, token, max_rank):
    """Re-run BPE on `token` using only merges of rank < max_rank; the final two parts are the merge that produced it."""
    parts = [bytes([b]) for b in token]
    while True:
        best_i, best_r = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best_r is None or r < best_r):
                best_i, best_r = i, r
        if best_r is None or best_r >= max_rank:
            break
        parts = parts[:best_i] + [parts[best_i] + parts[best_i + 1]] + parts[best_i + 2:]
    return parts


def build_vocab(enc):
    ranks = enc._mergeable_ranks; specials = enc._special_tokens
    b2u = bytes_to_unicode(); to_str = lambda b: "".join(b2u[x] for x in b)
    n = len(ranks) + len(specials); tokens = [None] * n; types = [TokenType.NORMAL] * n
    for tb, r in ranks.items():
        tokens[r] = to_str(tb)
    for s, i in specials.items():
        tokens[i] = s; types[i] = TokenType.CONTROL
    assert all(t is not None for t in tokens), "gap in token ids"
    merges = []
    for tb, r in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(tb) == 1:
            continue
        parts = bpe_split(ranks, tb, r)
        assert len(parts) == 2, f"token {tb!r} (rank {r}) did not split into two parts: {parts}"
        merges.append(f"{to_str(parts[0])} {to_str(parts[1])}")
    return tokens, types, merges, specials


def permute(w, n_head):
    """HF half-split RoPE layout -> llama.cpp interleaved ('normal' RoPE mode of LLM_ARCH_LLAMA)."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=os.environ.get("NANOCHAT_BASE_DIR", "/data/nc")); ap.add_argument("--source", default="sft", choices=["base", "sft"])
    ap.add_argument("--model-tag", required=True); ap.add_argument("--step", type=int, default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32"]); ap.add_argument("--name", default=None)
    a = ap.parse_args()
    ck = os.path.join(a.base_dir, {"base": "base_checkpoints", "sft": "chatsft_checkpoints"}[a.source], a.model_tag)
    step = a.step if a.step is not None else max(int(f.split("_")[1].split(".")[0]) for f in os.listdir(ck) if f.startswith("model_"))
    meta = json.load(open(os.path.join(ck, f"meta_{step:06d}.json"))); cfg = meta["model_config"]
    arch = cfg.get("arch")
    assert arch in ("llama", "qwen3"), f"checkpoint arch is {cfg.get('arch', 'gpt')} — only nanochat/llama.py and nanochat/qwen3.py checkpoints are GGUF-exportable"
    sd = torch.load(os.path.join(ck, f"model_{step:06d}.pt"), map_location="cpu")
    enc = pickle.load(open(os.path.join(a.base_dir, "tokenizer", "tokenizer.pkl"), "rb"))
    tokens, types, merges, specials = build_vocab(enc)
    V, D, L, H, HKV = cfg["vocab_size"], cfg["n_embd"], cfg["n_layer"], cfg["n_head"], cfg["n_kv_head"]
    assert len(tokens) == V, f"tokenizer has {len(tokens)} tokens but model vocab is {V}"
    ffn = sd["transformer.h.0.mlp.c_gate.weight"].shape[0]; head_dim = D // H
    print(f"{a.source}/{a.model_tag} step {step}: L={L} D={D} H={H} HKV={HKV} ffn={ffn} V={V}; {len(merges)} merges; dtype {a.dtype}")

    w = GGUFWriter(a.out, arch)  # every {arch}.* KV key below is templated off this string by gguf-py
    w.add_name(a.name or f"persian-{a.model_tag}-{a.source}")
    w.add_context_length(cfg["sequence_len"]); w.add_embedding_length(D); w.add_block_count(L); w.add_feed_forward_length(ffn)
    w.add_head_count(H); w.add_head_count_kv(HKV); w.add_rope_dimension_count(head_dim); w.add_rope_freq_base(float(cfg.get("rope_base", 500000.0)))
    if arch == "qwen3":
        # qwen3.attention.{key,value}_length: llama.cpp defaults these to n_embd/n_head (== our head_dim), but convert_hf_to_gguf
        # writes them for Qwen3 (whose head_dim can differ), so write them too. VERIFIED 2026-09-01 via toy parity: no other qwen3-only KV required —
        # Qwen3Model.set_gguf_parameters adds nothing else beyond the common keys already written above.
        w.add_key_length(head_dim); w.add_value_length(head_dim)
    w.add_layer_norm_rms_eps(float(cfg.get("norm_eps", 1e-5))); w.add_vocab_size(V)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if a.dtype == "f16" else gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("llama-bpe")  # pre-tokenizer follows tokenizer v2 (llama3 regex), not the model arch
    w.add_token_list(tokens); w.add_token_types(types); w.add_token_merges(merges)
    bos, a_end = specials["<|bos|>"], specials["<|assistant_end|>"]
    w.add_bos_token_id(bos); w.add_eos_token_id(a_end); w.add_eot_token_id(a_end); w.add_pad_token_id(bos)
    w.add_add_bos_token(True); w.add_add_eos_token(False); w.add_chat_template(CHAT_TEMPLATE)

    mat = (lambda t: t.float().numpy().astype(np.float16)) if a.dtype == "f16" else (lambda t: t.float().numpy())
    vec = lambda t: t.float().numpy()
    # llama.cpp RoPE mode: llama = NORM (interleaved -> permute from HF half-split), qwen3 = NEOX (half-split = our layout, no permute).
    # VERIFIED 2026-09-01: toy d12 qwen3 greedy-decode parity — torch == llama.cpp f16 == Q4_K_M (BOS-consistent llama-cli path).
    maybe_permute = (lambda t, nh: t) if arch == "qwen3" else permute
    w.add_tensor("token_embd.weight", mat(sd["transformer.wte.weight"][:V]))
    w.add_tensor("output_norm.weight", vec(sd["norm_f.weight"]))
    w.add_tensor("output.weight", mat(sd["lm_head.weight"][:V]))
    for i in range(L):
        p = f"transformer.h.{i}."
        w.add_tensor(f"blk.{i}.attn_norm.weight", vec(sd[p + "attn_norm.weight"]))
        w.add_tensor(f"blk.{i}.attn_q.weight", maybe_permute(mat(sd[p + "attn.c_q.weight"]), H))
        w.add_tensor(f"blk.{i}.attn_k.weight", maybe_permute(mat(sd[p + "attn.c_k.weight"]), HKV))
        if arch == "qwen3":
            w.add_tensor(f"blk.{i}.attn_q_norm.weight", vec(sd[p + "attn.q_norm.weight"]))  # per-head RMSNorm gain, size head_dim
            w.add_tensor(f"blk.{i}.attn_k_norm.weight", vec(sd[p + "attn.k_norm.weight"]))
        w.add_tensor(f"blk.{i}.attn_v.weight", mat(sd[p + "attn.c_v.weight"]))
        w.add_tensor(f"blk.{i}.attn_output.weight", mat(sd[p + "attn.c_proj.weight"]))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", vec(sd[p + "mlp_norm.weight"]))
        w.add_tensor(f"blk.{i}.ffn_gate.weight", mat(sd[p + "mlp.c_gate.weight"]))
        w.add_tensor(f"blk.{i}.ffn_up.weight", mat(sd[p + "mlp.c_up.weight"]))
        w.add_tensor(f"blk.{i}.ffn_down.weight", mat(sd[p + "mlp.c_down.weight"]))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
