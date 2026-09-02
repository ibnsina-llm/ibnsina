# CPT lane — Qwen3-30B-A3B continued pretraining (runbook)

Status: SCAFFOLD, nothing launched. Ratified stack (docs/plan/cpt_stack_plan.md, stack B): HF Transformers +
accelerate **FSDP2** with a **hand-rolled loop**, bf16 + activation checkpointing, target **15B tokens** on one
`a2-ultragpu-8g` (8×A100-80GB) **spot** box. The 1.5B guard suite is standard equipment — armed from step 1 of
G1, inherited by every resume. **Budget flows only after Gate 3 passes** (plan §4).

Scrub rule: **no literal project id or bucket name anywhere in `cpt/`** — always `$GCP_PROJECT`
(placeholder default `YOUR-GCP-PROJECT`) and `$CORPUS_BUCKET` (placeholder default `gs://YOUR-BUCKET`).

| file | what |
|---|---|
| `pack_data.py` | parquet(`text`) → packed uint32 `.npy` shards + `manifest.json` (Qwen3 tokenizer, CPU, multiprocess) |
| `train_cpt.py` | the loop: FSDP2 bf16+AC, AdamW cosine-to-10%, full guard suite, checkpoints+mirror, quarantine, `--resume` |
| `launch_cpt.sh` | nohup launcher + `RUN.env` (full guard env recorded — resumes inherit) + GCS sync sidecar / post-save hook |
| `gates.sh` | G1 smoke / G2 kill-and-resume proof / G3 1–2 h guarded run, as runnable assertions |

## 1. Box

Preferred: the existing launcher (zone rotation, spot ladder, 24 h max-run, corpus-worker SA):

```bash
GCP_PROJECT=... CORPUS_BUCKET=gs://... bash training/gpu_launch.sh cpt-box a100spot8 training/gpu_startup.sh 1000
```

Raw sketch (what that expands to; env vars, never literals):

```bash
gcloud compute instances create cpt-box --project=$GCP_PROJECT --zone=us-central1-a \
  --machine-type=a2-ultragpu-8g --provisioning-model=SPOT --instance-termination-action=STOP \
  --max-run-duration=24h --discard-local-ssds-at-termination-timestamp=true --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size=1000GB --boot-disk-type=pd-balanced --network-interface=nic-type=GVNIC \
  --service-account=corpus-worker@$GCP_PROJECT.iam.gserviceaccount.com --scopes=cloud-platform \
  --metadata=install-nvidia-driver=True --labels=purpose=persian-corpus,role=train,auto-kill=true
```

TODO-VERIFY at launch: a2-ultragpu-8g spot price (~$12–16/h assumed), zones with capacity (a/c as of 2026-08-29),
disk sizing (model 61 GB + 3 local ckpts ≈ 3×~180 GB sharded model+optim → 1 TB boot disk minimum).

## 2. Setup on the box

```bash
sudo mkdir -p /data/cpt && cd /data
gcloud storage rsync -r $CORPUS_BUCKET/code/pipeline /data/pipeline        # repo mirror convention
pip install 'transformers==5.16.*' accelerate                              # torch comes pinned by the DLVM image
huggingface-cli download Qwen/Qwen3-30B-A3B-Base                           # ~61 GB, do ONCE, before gates
```

TODO-VERIFY pins: evalbox had transformers 5.16.1 (Qwen3-MoE-capable); pin the accelerate version that carries
`fsdp_version: 2` and record both in the Gate-1 note. TODO-VERIFY flash-attn wheel for the image's torch/CUDA if
switching `CPT_ATTN=flash_attention_2` (default is `sdpa`).

## 3. Data (prerequisite for G3 + full run — schedule AFTER G1/G2 look healthy, plan §5)

On `corpus-pipeline2` (CPU, ~1 day): re-tokenize the `train_v1_1_open` + `synth_v1` text mix with the **base
model's tokenizer** (NOT our v2 tokenizer):

```bash
python cpt/pack_data.py --in-glob '/data/cpt/mix/*.parquet' --out-dir /data/cpt/data/train_v1_1_qwen3 \
  --tokenizer Qwen/Qwen3-30B-A3B-Base --seq-len 4096
gcloud storage rsync -r /data/cpt/data/train_v1_1_qwen3 $CORPUS_BUCKET/cpt_data/train_v1_1_qwen3
```

**Shard format** (the contract between `pack_data.py` and `train_cpt.py`):
- `shard_FFFF_PP.npy` — uint32 array, shape `(n_rows, seq_len+1)`; docs joined by one EOS, no BOS, rows cross
  doc boundaries; per-input-file sub-row tail dropped (counted in manifest).
- `manifest.json` — `tokenizer`, `eos_id`, `seq_len`, `row_len` (= seq_len+1), `dtype`, `shards[{file,n_seqs,n_tokens}]`,
  totals, `dropped_tail_tokens`. The trainer asserts `row_len == seq_len+1`.
- A row is fed as `input_ids` AND `labels` (HF shifts internally → seq_len targets/row).

## 4. Gates (exact commands; budget flows only after G3)

```bash
# G1 — smoke. Off-box first (proxy model, CPU/1 GPU, synthetic shards), then on-box with the real model:
bash cpt/gates.sh g1                                   # proxy: loop+guards+ckpt+resume logic, no downloads > ~1 GB
G1_FSDP=1 G1_SEQ=2048 bash cpt/gates.sh g1 Qwen/Qwen3-30B-A3B-Base    # on-box, 8 GPUs, FSDP2
#   plan §4 extras checked by hand on-box: initial loss ≈ base-model ppl on held-out Persian (not random-init),
#   loss decreasing, no guard trips, ≥10% memory headroom per GPU (nvidia-smi during the run).

# G2 — kill-and-resume proof (real model, small step budget, deterministic mode):
CPT_DATA_DIR=/data/cpt/data/train_v1_1_qwen3 bash cpt/gates.sh g2 g2run
#   pass: 'resume: step K' + first post-resume step K+1 + |EMA(K) − EMA(K+10)| ≤ G2_TOL (0.05) + guards re-armed.
#   Also run the plan's second form once: gcloud compute instances stop (simulated preemption), fresh boot,
#   launch_cpt.sh g2run — the empty local dir must repopulate from the GCS mirror.

# G3 — 1–2 h guarded run at production batch shape:
G3_ARGS='--save-every 100' CPT_DATA_DIR=/data/cpt/data/train_v1_1_qwen3 bash cpt/gates.sh g3 cpt30b 120
#   pass: every line PASS — zero trips/alerts/quarantine, probes flowing, ≥2 complete ckpts, GCS mirror live.
#   Record mean tok/s: < 18k ⇒ decision point (accept wall-clock vs port to torchtitan AFTER the lane passes).
```

Full run afterwards (owner go required): `bash cpt/launch_cpt.sh cpt30b-full --save-every 100` — defaults give
8×2×8×4096 = 524,288 tok/step ≈ 0.5 M, 15 B tokens ≈ 28.6 k steps, ~7–9 days at 19–31 k tok/s (plan §2).

## 5. Guard env table (all recorded into `RUN.env` on first launch — resumes inherit, the 1.5B lesson)

| env (default) | semantics | 1.5B analog |
|---|---|---|
| `CPT_SPIKE_GUARD` (1.0) | skip update when all-reduced step-mean loss > debiased EMA + margin nats; 0 disables | `NANOCHAT_SPIKE_GUARD` (same semantics) |
| `CPT_GRAD_GUARD` (4.0) | skip when global grad norm > k× its EMA; 0 disables | `NANOCHAT_GRAD_GUARD` (4.0 on the big run) |
| `CPT_GUARD_VALVE` (100) | accept anyway after N consecutive skips (anti-starvation) | `NANOCHAT_GUARD_VALVE` (100 on the big run) |
| `CPT_GUARD_WARMUP` (5) | EMA updates before guards judge | 5-step warm-in (same) |
| `CPT_QUARANTINE_AFTER` (8) | consecutive trips ⇒ quarantine + exit 3; 0 disables | manual (`finish_compressed.sh`) — now automated |
| `CPT_RANK_DEV` (2.0) | any rank > x nats off the median per-rank loss ⇒ trip; NaN/Inf on any rank ⇒ immediate quarantine | per-rank print on trips only — now a tripwire |
| `CPT_PROBE_EVERY` (100) | router-health probe cadence (steps); also fires at step 1 (baseline) | `attn_probe.py` per-checkpoint (attn-logit form) |
| `CPT_ROUTER_ENT_MIN` (0.5) / `CPT_ROUTER_MAXFRAC` (0.25) / `CPT_ROUTER_PATIENCE` (3) | normalized load-entropy floor / max expert share / consecutive alerts before quarantine | — (MoE-specific, new) |
| `CPT_SAVE_EVERY` (100) / `CPT_KEEP_LAST` (3) | ckpt cadence (~30 min) / local retention (GCS keeps all) | `--save-every` + `ckpt_sync.sh` prune |
| `CPT_RESUME` via `--resume` | auto-detect latest `COMPLETE` marker; fresh start when none | `train_run.sh` pull + `--resume-from-step` |

Guard-parity deltas vs the nanochat patch, on purpose: (1) grad reference is the TRUE global FSDP norm from
`clip_grad_norm_`, not the RMS-over-ranks of per-rank norms; (2) a valve-accept does **not** update the EMA
references here (owner mandate: refs untouched on skipped *and* valve steps — nanochat fed the EMA on valve
accepts); (3) skipped steps do not advance the LR schedule (LR is driven by the accepted-update count).
Collectives are all synchronous — the torn-gather failure mode is excluded by construction.

## 6. Checkpoints, mirror, quarantine

- Local: `$CPT_HOME/ckpt/RUN/step_XXXXXXXX/` = `accelerate save_state` shards + `meta.json` (loader cursor,
  guard state, token count) + `COMPLETE` marker **written last**. Keep last 3; every one mirrored.
- GCS: `$CORPUS_BUCKET/cpt_ckpt/RUN/` via the watch sidecar + post-save hook (`launch_cpt.sh sync RUN`).
  Mirror is two-pass (markers uploaded last) so a `COMPLETE` in GCS implies a full dir. Resume on a fresh boot
  pulls the mirror automatically when the local dir is empty.
- Quarantine: guard escalation copies the **last good** checkpoint to `$CPT_HOME/quarantine/RUN/…_stepN_REASON/`
  (+ `anomaly.json`, batch-head dumps in `logs/trip_batches/`), writes `ckpt/RUN/QUARANTINED`, exits 3.
  `launch_cpt.sh` refuses a quarantined run until the owner reviews (`rm .../QUARANTINED` or `FORCE=1`).
- Preemption: SIGTERM → checkpoint at the next step boundary; zero-notice kill → resume from the last marker
  (G2 proves both). `RUN.env` + `RUN.args` make every relaunch identical — never pass fresh args to resume.

## 7. Open TODO-VERIFY (resolve during G1 bring-up, record in the gate notes)

1. `FullyShardedDataParallelPlugin` kwargs (`fsdp_version=2`, `transformer_cls_names_to_wrap`,
   `activation_checkpointing`, `state_dict_type`) against the pinned accelerate; wrap class name
   `Qwen3MoeDecoderLayer` against the pinned transformers.
2. `save_state`/`load_state` sharded format is world-size-bound (fine on the fixed 8-GPU box) — confirm.
3. Memory: micro-bsz 2 at seq 4096 with AC on 80 GB (fallback `CPT_MICRO_BSZ=1 CPT_GRAD_ACCUM=16`).
4. CPT peak LR 2e-5 (cosine to 10%) for 30B-A3B — sanity-check against Gate-1 loss behavior.
5. Router probe: `output_router_logits` kwarg + `router_logits` output + `num_experts_per_tok`/`num_experts`
   config names; calibrate `CPT_ROUTER_ENT_MIN`/`MAXFRAC` from the step-1 baseline probe of the BASE model.
6. `router_aux_loss_coef`/`output_router_logits` config fields if enabling aux loss (`CPT_ROUTER_AUX>0`) —
   default off; decide after the baseline probe.
7. `sdpa` vs `flash_attention_2` on the image's torch/CUDA; grad-accum `no_sync` (FSDP2
   `set_requires_gradient_sync`) if G3 throughput lands under 18k tok/s.
8. Qwen3 base tokenizer `eos_token_id` (`<|endoftext|>`) — pack_data asserts non-None.
9. Spot price + zone capacity + disk sizing at launch time (§1).
