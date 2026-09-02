# 3 B run — config (Qwen3 arch; FROZEN proposal — launch only after the owner's go and the toy qwen3 GGUF loop passes)

Ruling 2026-08-29 + owner confirmation: the 3B trains on the **Qwen3 architecture** (llama.cpp arch `qwen3`) with our patched
nanochat stack. Qwen3 = the Llama shape (RMSNorm, RoPE, GQA, SwiGLU, untied head) **plus per-head QK-norm** — learnable RMSNorm
(size head_dim) on Q and K after the head reshape, before RoPE. That is the fix for the 1.5B's attention-logit growth
(max |q·k/√d| hit 102 by step 27k on the Llama arch, which has no QK-norm). Stack = nanochat training loop (Muon+AdamW, resume/sync,
guard patches) + `nanochat/qwen3.py` (NANOCHAT_ARCH=qwen3 via `nanochat/arch.py`) + `export_gguf.py` (arch `qwen3` → ollama/llama.cpp/LM Studio).

## Shape
| | layers | dim | heads / kv | head_dim | SwiGLU | params (dense) | of which embeddings |
|---|---|---|---|---|---|---|---|
| **3B (chosen)** | 40 | 2560 | 20 / **4 (GQA)** | 128 | 7168 | **≈3.00 B** | 0.17 B |
| alt: wider-shallower | 28 | 3072 | 24 / 4 | 128 | 8192 | 2.93 B | 0.20 B |
| 1.5B (shipped) | 28 | 2048 | 16 / 4 | 128 | 6144 | 1.48 B | 0.13 B |

Chosen shape is exactly nanochat's depth×aspect convention: **depth 40 × aspect-ratio 64 = 2560** (already a multiple of head-dim 128 → 20 heads).
Param arithmetic (vocab 32,768, untied head, no padding since 32768 % 64 = 0):
- embeddings: 2 × 32768 × 2560 = **167.77 M**
- attention / layer: q,o 2×2560² + k,v 2×(4×128)×2560 = 13.11 M + 2.62 M = **15.73 M**
- SwiGLU / layer: 3 × 2560 × 7168 = **55.05 M** (auto 8/3 rule would give 6912 → 2.90 B total; pinned up via `--ffn-hidden=7168` = 2.8×dim to land on 3.0 B)
- matrices: 40 × 70.78 M = **2 831.16 M**; norms: 40×(2×2560 + 2×128) + 2560 = **0.22 M** (QK-norm gains: 40×2×128 = 10,240 params)
- **total = 2 999.14 M ≈ 3.00 B**

GQA 4 kv-heads: KV cache 40×2×4×128×2 B = 80 KiB/token in f16 → 2048-token context = **160 MiB** (phone-viable after Q4). RoPE base 500k,
RMSNorm eps 1e-5, untied head, no bias anywhere (Qwen3 dropped Qwen2's QKV bias), vocab 32,768 (**tokenizer v2**, Llama-3 regex → `tokenizer.ggml.pre = llama-bpe`).

## Data / tokens
**Owner ruling 2026-09-01 (overtraining thesis): ~100 B tokens.** Chosen **99.6 B** = `--num-iterations=190000` × 524,288 ≈ **33.2 tokens/param**
(vs 24.8 t/p achieved on the 1.5B). Unique data ≈ 48.4 B (`train_v1_1_open` 46.35 B + synth_v1 2 B per the SY-B ② ruling) → **≈2.06 epochs** —
well inside the ≤4-epoch safe zone for data reuse; synth shards fold into the mix as waves land, replacing repeats, never shrinking the real-data share.

## Launch (8×H100 spot; `train_run.sh big3b …` with the full env suite)
```
NANOCHAT_ARCH=qwen3 NANOCHAT_SPIKE_GUARD=1.0 NANOCHAT_GRAD_GUARD=4.0 NANOCHAT_GUARD_VALVE=100 NANOCHAT_SYNC_COLLECTIVES=1
--depth=40 --aspect-ratio=64 --head-dim=128 --n-kv-head=4 --ffn-hidden=7168 --window-pattern=L
--max-seq-len=2048 --device-batch-size=4 --total-batch-size=524288 --num-iterations=190000 --warmup-steps=500 --warmdown-ratio=0.4
--save-every=1000 --eval-every=2000 --eval-tokens=41943040 --sample-every=-1 --core-metric-every=-1
```
- **Mandatory guard env suite** (non-negotiable, all resumes too): loss guard armed (`NANOCHAT_SPIKE_GUARD=1.0`, the default — set it explicitly anyway),
  `NANOCHAT_GRAD_GUARD=4.0`, `NANOCHAT_GUARD_VALVE=100`, `NANOCHAT_SYNC_COLLECTIVES=1`; escape-ladder hooks `NANOCHAT_LR_SCALE` and
  `NANOCHAT_WARMDOWN_START` stay available on resume. `train_run.sh` records only NANOCHAT_ARCH in `/data/runs/big3b.env` — **append the four guard
  vars to that file right after the first launch** so preemption resumes keep them.
- LRs per nanochat convention (defaults; no flags): Muon matrix_lr 0.02 (shape-aware, not dim-scaled), AdamW groups auto-scaled ∝1/√(2560/768)=0.548
  → effective unembedding 0.00219, embedding 0.1095; RMSNorm gains (incl. q_norm/k_norm) 0.01. Warmdown: linear over the last 40 % (step 114,000 → end).
- Batch math: 524,288 / (8 GPU × 4 × 2048) = 8 grad-accum micro-steps. (device-batch 8 OOMs on 80 GB for the 3B — 74.7 GB allocated at launch, 2026-09-01; total batch and thus the optimization recipe unchanged. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the run env for allocator headroom.)
- Throughput: 6×2.92 B matmul FLOP/token ≈ 2.2× the 1.5B → **≈0.18–0.21 M tok/s → 132–154 h (5.5–6.4 days)** for 99.6 B tokens, ≈ **$3,700–4,400** spot including restarts.
  Report at $3,700 per the rules, continue.
- `--fp8`: **do not use** (diverged on the 1.5B; same ruling here).
- Checkpoints every 1000 steps + 30-min GCS sync; prune to the last 3 on disk (each ≈ 12 GB model + optimizer shards).

## QK-norm note (why this arch, what to expect from the probe)
With per-head RMSNorm, q and k rows are unit-RMS per head, so |q·k/√d| ≤ √d·g_q·g_k ≈ 11.3 × (gain product) at head_dim 128:
**expect `attn_probe.py` max |logit| to sit O(10–30) and stay flat**, vs. the 1.5B's monotone climb to 102 by step 27k. Probe every checkpoint anyway
(`attn_probe.py` now applies q_norm/k_norm for qwen3 so it sees the real logits). **The early-warmdown (`NANOCHAT_WARMDOWN_START`) and
`NANOCHAT_LR_SCALE` rungs stay armed regardless** — QK-norm removes the known failure mode, not the unknown ones; the spike/grad guards and the
probe schedule are unchanged from the 1.5B playbook.

## After pretraining
1. SFT v2 (`chat_sft_fa.py`) on the same stack per the sft_v2 mandate; same chat template as the 1.5B.
2. Evals: ParsiNLU MC/entailment/QQP, PersianMedQA, Persian val bpb, English CORE sanity; Khayyam Challenge if granted.
3. Export: `export_gguf.py` (arch `qwen3`, no Q/K permute — NEOX RoPE, + attn_q_norm/attn_k_norm tensors) → F16 → `llama-quantize` Q8_0 / Q4_K_M
   → llama.cpp, ollama Modelfile, LM Studio; HF safetensors mirror (Qwen3ForCausalLM) for transformers.

## Pre-launch gate
- [ ] owner go on this frozen config · [ ] **toy qwen3 loop PASS** (`toy_qwen3_loop.sh`: tokenizer parity, greedy parity f16+Q4_K_M, llama-cli generation coherent)
- [ ] `train_v1_1_open` shards + tokenizer v2 in place · [ ] guard env suite in `/data/runs/big3b.env` · [ ] checkpoint pruning on
- [ ] helper VMs stopped (CPU quota) · [ ] budget note at $3,700 · [ ] GCS-evacuation deadline (credits 2026-09-25) checked against the 132–154 h runtime
