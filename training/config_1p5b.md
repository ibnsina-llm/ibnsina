# 1.5 B run — config (Llama-compatible stack; launch only after STOP T-C and the toy GGUF loop passes)

Ruling 2026-08-29: every release model trains in a Llama-compatible architecture so it ships as standard GGUF (llama.cpp, ollama,
LM Studio, phone runtimes). Stack = nanochat's training loop (Muon+AdamW, dataloader, resume/sync tooling) + `nanochat/llama.py`
(exact llama.cpp `llama` semantics) + `export_gguf.py`. nanollama (GPLv3, own SentencePiece vocab) was rejected for licence + tokenizer reasons.

## Shape
| | layers | dim | heads / kv | SwiGLU | params (dense) | of which embeddings |
|---|---|---|---|---|---|---|
| **1.5B (chosen)** | 28 | 2048 | 16 / **4 (GQA)** | 6144 | **≈1.48 B** | 0.13 B |
| alt: SmolLM2-1.7B shape | 24 | 2048 | 32 / 32 | 8192 | 1.71 B | 0.13 B |
| pilot (nanochat arch) | 18 | 1152 | 9 / 9 | relu² 4x | 0.36 B | 0.08 B |

GQA 4 kv-heads: KV cache 4× smaller than MHA (2048-token context ≈ 112 MB in f16) — matters on phones. RoPE base 500k, RMSNorm eps 1e-5, untied head, no bias, vocab 32,768 (**tokenizer v2**, Llama-3 regex → `tokenizer.ggml.pre = llama-bpe`).

## Data
`train_v1_1_open` (v1 46.35 B tokens + chap textbooks) — one epoch ≈ 31 tokens/param. `--num-iterations` = tokens / 524 288 ≈ **88 000** at a 524k batch.

## Launch (8×H100 spot; `train_run.sh big …` with `NANOCHAT_ARCH=llama`)
```
NANOCHAT_ARCH=llama  --depth=28 --aspect-ratio=73 --head-dim=128 --n-kv-head=4 --ffn-hidden=6144 --window-pattern=L
--max-seq-len=2048 --device-batch-size=16 --total-batch-size=524288 --num-iterations=88000 --warmup-steps=500 --warmdown-ratio=0.4
--save-every=1000 --eval-every=2000 --eval-tokens=41943040 --sample-every=-1 --core-metric-every=-1
```
(`--aspect-ratio=73` → 28×73 = 2044 → rounded to the head-dim multiple 2048, 16 heads.)
- Throughput from the pilot (d18: 1.6 M tok/s, 44 % MFU): a 1.48 B Llama block ≈ 6×1.35 B FLOP/token → **≈0.40–0.45 M tok/s → 29–32 h** for 46 B tokens, ≈ **$800–900** spot including restarts. Report at $800 per the rules, continue.
- Expected efficiency cost vs nanochat's own architecture: the tweaks we give up (value embeddings, x0/residual lambdas, QK-norm + logit softcap, smear/backout) are worth roughly 5–10 % tokens-to-loss in the nanochat/modded-nanogpt lineage; Muon (the big win, ~30 %+ over AdamW) stays. Sliding windows (SSSL) were only a speed feature and are already off on SDPA. The toy loop measures this at d12 (`ab-gpt` vs `ab-llama`, same tokens/data/tokenizer) and the number goes into the T-C report.
- Checkpoints every 1000 steps (~20 min) + 30-min sync; preemption/24 h-STOP resume is automatic (`gpu_startup.sh` → `pilot_pipeline.sh`-style orchestration with `RUN=big`). Prune to the last 3 checkpoints on disk (each ≈ 6 GB model + 8 optimizer shards).
- `--fp8`: **do not use** — on the 1.5B it gave +20 % throughput but diverged after warm-up (loss 3.18 → 4.66 by step 4,200); the run was rolled back to bf16 from checkpoint 2,000. A meaningful trial needs ≥1,500 full-LR steps.

## After pretraining
1. SFT v2 on the same stack (`chat_sft_fa.py`): v1 mixture + 5–10 k multi-turn translated conversations + permissive Persian math if found; keep MMLU aux-train as format teacher.
2. Evals: ParsiNLU MC/entailment/QQP (pilot baseline), Khayyam Challenge if access is granted, Persian val bpb, English CORE sanity.
3. Export: `export_gguf.py` → F16 → `llama-quantize` Q8_0 / Q4_K_M → llama.cpp, ollama Modelfile, LM Studio; HF safetensors mirror for transformers.

## Pre-launch gate
- [ ] T-C approved · [ ] toy loop PASS (tokenizer parity, greedy parity, Q4_K_M runs) · [ ] `train_v1_1_open` shards + tokenizer v2 in place · [ ] fp8 A/B · [ ] checkpoint pruning · [ ] helper VMs stopped (CPU quota) · [ ] budget note at $800
