#!/usr/bin/env python3
"""T0 — train nanochat's BPE tokenizer (rustbpe, GPT-4-style split, 9 nanochat special tokens) on a stratified sample of train_v1_open.

Run inside the nanochat uv env on the pipeline VM:
  cd /data/nanochat && uv run python /data/pipeline/training/train_tokenizer.py --max-chars 10e9 --vocab-size 32768
Stratification (chars): Persian-dominant; each slice capped at its quota so code/math/English are represented.
Docs are cropped to --doc-cap chars (nanochat default 10k) so a few huge docs don't dominate.
Saves to $NANOCHAT_BASE_DIR/tokenizer (tokenizer.pkl + token_bytes.pt, exactly what scripts/tok_train.py writes).
"""
import argparse, os, random, sys, time, json
from pathlib import Path
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "/data/nanochat")
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
import nanochat.tokenizer as _nt
# Llama-3's exact pre-tokenizer regex (llama.cpp LLAMA_VOCAB_PRE_TYPE_LLAMA3 / tokenizer.ggml.pre="llama-bpe"); nanochat's own differs in digit grouping ({1,2}) and newline runs.
LLAMA3_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
from nanochat.common import get_base_dir

TRAIN = Path("/data/mix/train_v1_open/train")
QUOTA = {  # share of the training sample, by characters
    "fa_web": 0.56, "fa_lit": 0.04, "wiki_fa": 0.04, "en_edu": 0.14, "wiki_en": 0.02, "code_pyts": 0.06, "code_other": 0.05, "math": 0.06, "parallel": 0.03,
}


def bucket(source, slice_):
    if slice_ == "wiki":
        return "wiki_fa" if source == "fawiki" else "wiki_en"
    return slice_


def text_iterator(max_chars, doc_cap, seed):
    rnd = random.Random(seed)
    shards = sorted(TRAIN.glob("shard_*.parquet")); rnd.shuffle(shards)
    want = {k: v * max_chars for k, v in QUOTA.items()}; got = {k: 0 for k in QUOTA}; ndocs = {k: 0 for k in QUOTA}
    t0 = time.time(); total = 0
    for p in shards:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            b = pf.read_row_group(rg, columns=["text", "source", "slice"])
            texts, srcs, sls = b.column("text").to_pylist(), b.column("source").to_pylist(), b.column("slice").to_pylist()
            for t, s, sl in zip(texts, srcs, sls):
                k = bucket(s, sl)
                if k not in want or got[k] >= want[k]:
                    continue
                if len(t) > doc_cap:
                    t = t[:doc_cap]
                got[k] += len(t); ndocs[k] += 1; total += len(t)
                yield t
        done = all(got[k] >= want[k] * 0.98 for k in want)
        print(f"[sample] {p.name}: {total/1e9:.2f}B chars  " + " ".join(f"{k}={got[k]/max(1,want[k]):.0%}" for k in want) + f"  ({time.time()-t0:.0f}s)", flush=True)
        if done or total >= max_chars:
            break
    Path("/data/tok").mkdir(exist_ok=True)
    Path("/data/tok/sample_stats.json").write_text(json.dumps({"chars": got, "docs": ndocs, "total_chars": total, "seed": seed, "doc_cap": doc_cap}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=float, default=10e9); ap.add_argument("--doc-cap", type=int, default=10_000)
    ap.add_argument("--vocab-size", type=int, default=32768); ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--pattern", choices=["nanochat", "llama3"], default="nanochat", help="pre-tokenizer regex; llama3 = exact Llama-3 regex so GGUF can use tokenizer.ggml.pre=llama-bpe")
    a = ap.parse_args()
    if a.pattern == "llama3":
        _nt.SPLIT_PATTERN = LLAMA3_PATTERN
        print("pre-tokenizer: Llama-3 regex (llama-bpe)", flush=True)
    print(f"vocab {a.vocab_size} (incl. {len(SPECIAL_TOKENS)} specials: {SPECIAL_TOKENS}), max_chars {a.max_chars:.2e}, doc_cap {a.doc_cap}", flush=True)
    t0 = time.time()
    tok = RustBPETokenizer.train_from_iterator(text_iterator(int(a.max_chars), a.doc_cap, a.seed), a.vocab_size)
    print(f"trained in {(time.time()-t0)/60:.1f} min; vocab {tok.get_vocab_size()}", flush=True)
    out = os.path.join(get_base_dir(), "tokenizer"); tok.save(out)
    test = "سلام دنیا! این یک آزمایش است. ۱۲۳ Hello world, def f(x): return x**2  ∫ x dx"
    assert tok.decode(tok.encode(test)) == test
    special_ids = set(tok.encode_special(s) for s in tok.get_special_tokens())
    tb = [0 if i in special_ids else len(tok.decode_single_token_bytes(i)) for i in range(tok.get_vocab_size())]
    torch.save(torch.tensor(tb, dtype=torch.int32), os.path.join(out, "token_bytes.pt"))
    print(f"saved to {out}: {os.listdir(out)}", flush=True)
