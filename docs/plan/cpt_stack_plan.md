# CPT stack decision memo — Qwen3-30B-A3B verification lane

Status: STAGED PLAN — nothing in this memo is running. Date: 2026-08-31.
Scope: FSDP training-stack proof (owner gate (b)) for continued pretraining (CPT) of
Qwen/Qwen3-30B-A3B-Base on ONE `a2-ultragpu-8g` (8×A100-80GB, 640 GB HBM total), spot
with preemption. Gate (a), the llama.cpp export test, is staged separately
(`cpt_export_test.sh`; currently blocked on evalbox — see "Evalbox findings" at the end).

Fallback base if gate (a) fails and cannot be remedied: Qwen3-14B dense (same stacks
apply; everything below gets simpler, not harder).

---

## 1. Candidate stacks

### A. torchtitan (PyTorch-native pretraining)
- **MoE support maturity:** MoE is first-class (grouped-GEMM experts, expert
  parallelism), but Qwen3 lives under `torchtitan/experiments/` — experimental tier,
  not the supported core. Requires HF-safetensors → DCP checkpoint import, and a
  DCP → HF export at the end so the llama.cpp/GGUF release path (Llama-arch ruling)
  still applies. Two custom conversion steps on a verification lane = two new ways
  to fail silently.
- **bf16 + activation checkpointing:** native FSDP2 mixed-precision policy, selective
  or full AC per layer. Best-in-class.
- **Throughput:** best of the three. Grouped experts + EP: expect 30–40% MFU.
- **Checkpoint/resume:** native async DCP; solid.
- **Guard hooks:** trivial — the training loop is plain, readable PyTorch.

### B. HF Transformers + accelerate (FSDP2), custom loop  ← recommended
- **MoE support maturity:** `Qwen3MoeForCausalLM` has been in transformers since
  v4.51 (spring 2025); it is the reference implementation the released checkpoint
  ships against. Weights, config, tokenizer load directly — zero conversion in, and
  the trained model saves straight back to HF safetensors, which is exactly what our
  llama.cpp convert path consumes. The export gate stays honest end-to-end.
- **bf16 + activation checkpointing:** accelerate `fsdp_version: 2` gives DTensor
  FSDP2 sharding, bf16 mixed precision, and full/selective AC on decoder layers.
- **Throughput:** the HF Qwen3-MoE forward loops over experts rather than grouped
  GEMM — expect 15–25% MFU. Acceptable for the verification lane; the honest cost is
  wall-clock (numbers below).
- **Checkpoint/resume:** sharded state via torch.distributed.checkpoint (DCP) or
  `accelerate.save_state`; dataloader/RNG state saved manually in our loop.
- **Guard hooks:** we write the loop, so every mandated guard sits exactly where it
  must (map in §3). Trainer/TRL callbacks canNOT do a pre-update step-skip cleanly;
  a custom loop can.

### C. LLaMA-Factory / axolotl (config-driven frameworks)
- **MoE support maturity:** both drive the same transformers modeling code as B, so
  model support is fine; CPT is LLaMA-Factory's `pt` stage / axolotl pretrain mode.
- **bf16 + AC:** yes, via flags.
- **Throughput:** same as B minus framework overhead.
- **Checkpoint/resume:** HF Trainer checkpoints — heavyweight full-state saves, slow
  with a 30B MoE, and dataloader-resume is approximate.
- **Guard hooks:** the disqualifier. Guards must live *inside* the optimizer step
  (pre-update grad-norm skip, all-rank loss gather, quarantine-on-anomaly). In these
  frameworks that means monkey-patching Trainer internals — at which point the
  framework is a liability, not a convenience.

## 2. Recommendation

**Stack B: HF Transformers + accelerate FSDP2 with a hand-rolled training loop.**
Rationale: (1) the released checkpoint's native implementation, no format conversion
in or out, so the GGUF/ollama release ruling is exercised on the real artifact;
(2) FSDP2 + bf16 + AC comfortably fits 30B total / 3.3B active on 8×A100-80GB with
room for a sane batch; (3) the mandatory guard suite requires owning the step loop,
and this is the stack where owning the loop is idiomatic rather than a fork.
Trade-off accepted: 1.5–2× slower than torchtitan. Decision point recorded at Gate 3:
if measured throughput < ~18k tok/s, either accept longer wall-clock for the full CPT
or invest 1–2 days porting to torchtitan *after* the verification lane passes —
torchtitan is the performance upgrade path, not the verification vehicle.

### Throughput estimate (stack B, honest ranges)
- Active params ≈ 3.3B → ≈ 6 × 3.3e9 ≈ 2.0e10 training FLOPs/token.
- 8×A100 bf16 peak = 2,496 TFLOPS. At 15–25% MFU → **~19k–31k tokens/s**
  (plan on ~25k; torchtitan at 30–40% MFU would be ~37k–50k).
- **15B tokens:** ~7–9 days compute; **25B tokens:** ~12–15 days. Add ~20% for spot
  preemption/restart churn. Suggested shape: seq 4096, packed; micro-batch × grad-accum
  tuned to ~0.5M tokens/global-batch.
- Cost order-of-magnitude: a2-ultragpu-8g spot ≈ $12–16/hr (verify at launch;
  on-demand ≈ $40/hr). Verification lane itself (§4) is ≈ 4–6 GPU-box-hours ≈ <$100.

## 3. Guard suite — non-negotiable standard equipment, hook map

All five guards from the owner mandate, placed in the custom loop:

1. **Pre-update grad-norm guard** — after `backward()`, before `optimizer.step()`:
   compute the global grad norm across all FSDP2 DTensor shards
   (`clip_grad_norm_` returns it). Non-finite or > threshold → **skip the step**
   (zero grads, do not step, do not advance LR), log step id + batch shard ids,
   increment trip counter. K consecutive trips → quarantine (guard 5).
2. **All-rank loss guard** — every step, `all_gather` the scalar loss from all 8
   ranks (gather result must stay alive until `wait()` — the nanochat torn-gather
   lesson applies verbatim to any async collective here). Any rank NaN/Inf, or any
   rank deviating > x% from the median → halt + quarantine. Catches single-GPU/ECC
   flakiness that a mean-reduced loss hides.
3. **Periodic attention-logit / router-health probes** — every M steps (e.g. 100) run
   one probe micro-batch with hooks on q/k projections: report per-layer max
   attention logit (drift → logit explosion early warning). MoE-specific and
   mandatory for this base: collect router stats — per-layer expert-load histogram,
   load entropy, max-expert share, router aux/z-loss value. Entropy collapse or one
   expert absorbing the batch → alert; sustained → quarantine. (Qwen3-30B-A3B:
   128 experts, 8 active, no shared expert — router health IS model health.)
4. **Checkpoint-every-N + GCS mirror** — DCP sharded save (model + optimizer +
   dataloader offset + per-rank RNG) every N steps (~30 min wall) to local SSD,
   `gcloud storage rsync` mirrored to
   `$CORPUS_BUCKET/cpt_ckpt/` by a background thread;
   a step-marker file written LAST makes each checkpoint atomically discoverable.
   Keep last 3 local, every checkpoint mirrored. Resume = highest complete marker in
   GCS. Spot preemption gives ~30 s ACPI notice — a shutdown script does a
   best-effort final marker, but the design assumes zero-notice kills.
5. **Quarantine-on-anomaly** — any guard trip beyond thresholds: freeze, snapshot
   full state + offending batch ids to a `quarantine/` GCS prefix, exit non-zero so
   the supervisor does NOT auto-restart into the same failure. Human (owner) reviews
   before the run continues.

## 4. Build plan — 2–3 days, gated; budget flows only after Gate 3

**Day 1 — bring-up + Gate 1 (smoke test).**
Provision a2-ultragpu-8g spot from image/startup-script; pin torch/transformers/
accelerate versions; write the custom loop with all five guards ARMED from the first
step. Load Qwen3-30B-A3B-Base under FSDP2 bf16 + AC. Smoke: tiny batch, seq 2048,
50–100 steps on a small re-tokenized corpus sample.
*Gate 1 pass:* initial loss ≈ base-model perplexity on held-out Persian (sanity, not
random-init-level), loss decreasing, no guard trips, memory headroom ≥ 10% per GPU.

**Day 2 — Gate 2 (preemption-resume proof).**
Run ~30 min, `kill -9` the whole job mid-step (and once via a simulated preemption:
`gcloud compute instances stop` on the spot VM), restore from the GCS mirror on a
fresh boot. With deterministic mode (fixed seeds, `use_deterministic_algorithms`,
step-indexed data order) compare against an uninterrupted reference run.
*Gate 2 pass:* bitwise-identical loss sequence post-resume, OR (if a kernel blocks
determinism) loss-continuity: |Δloss| within tolerance (≤0.02) over the 50 steps
after resume vs the reference, plus identical dataloader offset. No sample skipped
or double-consumed.

**Day 2–3 — Gate 3 (1–2 h stable CPT run, guards armed).**
Real packed corpus sample, production batch shape, checkpoints + GCS mirror live,
probe dashboards (grad norm, per-rank loss, router entropy, max attn logit) written
to GCS as JSONL. *Gate 3 pass:* 1–2 h with zero unexplained guard trips, throughput
measured and recorded (decision point vs 18k tok/s, §2), checkpoint cadence and
mirror lag acceptable. **THEN and only then budget flows to the 15–25B-token run.**

## 5. Data note — tokenizer and packing (staged, do NOT run)

CPT trains the **base model's own tokenizer** (Qwen3), NOT our tokenizer. Corpus
packing job: re-tokenize `train_v1_1_open` + `synth_v1` shards with the Qwen3
tokenizer, pack to seq 4096 (document-boundary-aware), write shards to GCS.
Estimated ~1 day CPU on `corpus-pipeline2`. This job is a prerequisite for Gate 3's
"real corpus sample" and for the full run; it is **not started** — schedule it only
after Gates 1–2 look healthy so a gate failure doesn't strand a day of CPU.

## 6. Evalbox findings for gate (a) — export test (checked 2026-08-31, read-only)

- **Disk:** /data shows 94 GB free (193 GB, 52% used) → below the 100 GB script gate
  and well below the ~145 GB real peak (61 GB safetensors + 61 GB bf16 GGUF + 19 GB
  Q4_K_M). Remedy: clear old eval artifacts under /data/models or resize the disk.
- **llama.cpp too old (BLOCKER):** `convert_hf_to_gguf.py` has no `Qwen3Moe` and no
  `Qwen3ForCausalLM` at all — the build predates Qwen3 (no git metadata to date it).
  Remedy: pull/re-clone latest llama.cpp and rebuild converter **and** CUDA binaries
  (arch table lives in the C++ too). The staged script's preflight stage enforces
  both checks and prints the exact rebuild commands.
- **transformers in /data/venv:** 5.16.1 — Qwen3-MoE-capable, fine.
- **GPU:** 37.0/40.0 GB in use, 100% util (eval job) — script auto-falls back to CPU
  generation if <~24 GB VRAM free at run time.

Export-test script (staged, bash -n clean, not run):
`/private/tmp/claude-501/-Users-sinameraji/970c7194-c84a-4d7a-affb-8281695d3b38/scratchpad/cpt_export_test.sh`
— move it into the repo (e.g. `scripts/cpt/`) once the owner signs off on the lane.
