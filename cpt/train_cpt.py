#!/usr/bin/env python3
"""Hand-rolled CPT loop: Qwen3-30B-A3B(-Base) on HF Transformers + accelerate FSDP2 (bf16, activation ckpt).
Stack ruling: docs/plan/cpt_stack_plan.md (stack B). Guard suite = the 1.5B mandate, ported from
training/nanochat_patches/spike_guard_patch.py semantics (parity table in cpt/README.md):
  (a) all-rank loss guard    skip the update when the all-reduced step-mean loss > debiased EMA + CPT_SPIKE_GUARD
                             nats; per-rank gather too: any rank NaN/Inf or > CPT_RANK_DEV nats off the median trips.
  (b) grad-norm guard        skip when the GLOBAL grad norm (clip_grad_norm_ across FSDP2 shards) exceeds
                             CPT_GRAD_GUARD x its running EMA. EMA references are NEVER updated on skipped or
                             valve steps, and n counts accepted updates only (a skip streak cannot decay the ref).
  (c) router-health probe    every CPT_PROBE_EVERY steps: per-layer expert-load entropy (normalized) + max expert
                             share; below/above thresholds -> CPT_ALERT; CPT_ROUTER_PATIENCE consecutive alerts
                             -> quarantine. Router health IS model health for 128-expert/top-8 A3B.
  (d) checkpointing          every CPT_SAVE_EVERY steps: accelerate save_state (model+optim+RNG) + meta.json
                             (dataloader cursor + guard state) + COMPLETE marker written LAST; post-save hook
                             `bash $CPT_SYNC_SCRIPT sync $CPT_RUN` mirrors to GCS (non-blocking); keep last N local.
  (e) quarantine-on-anomaly  CPT_QUARANTINE_AFTER consecutive trips, any non-finite, or a sustained router alert:
                             COPY the last good checkpoint into quarantine/ (never overwrite it), write
                             anomaly.json + a QUARANTINED flag (launch_cpt.sh refuses the run), exit 3 so no
                             supervisor restarts into the same failure.
Data: uint32 .npy shards of shape (n_rows, seq_len+1) + manifest.json from cpt/pack_data.py. A row is fed as
input_ids AND labels (HF shifts internally -> seq_len targets). Config: CPT_* env vars, overridable by flags;
--resume auto-detects the latest COMPLETE checkpoint (fresh start when none — launch_cpt.sh always passes it).
Collectives: every gather/all_reduce here is SYNCHRONOUS — the nanochat torn-gather bug (async all_gather input
freed before wait()) cannot recur by construction. Keep it that way.
"""
import argparse
import bisect
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def E(k, d, c=str):
    v = os.environ.get(k)
    return c(v) if v not in (None, "") else d


def get_args():
    p = argparse.ArgumentParser(description="CPT loop (env CPT_* defaults, flags override)")
    a = p.add_argument
    a("--run", default=E("CPT_RUN", "cpt0"))
    a("--model-id", default=E("CPT_MODEL_ID", "Qwen/Qwen3-30B-A3B-Base"))
    a("--data-dir", default=E("CPT_DATA_DIR", "/data/cpt/data/train_v1_1_qwen3"))
    a("--ckpt-dir", default=E("CPT_CKPT_DIR", ""))            # default $CPT_HOME/ckpt/<run>
    a("--quarantine-dir", default=E("CPT_QUARANTINE_DIR", ""))
    a("--log-dir", default=E("CPT_LOG_DIR", ""))
    a("--seq-len", type=int, default=E("CPT_SEQ_LEN", 4096, int))
    a("--micro-bsz", type=int, default=E("CPT_MICRO_BSZ", 2, int))    # TODO-VERIFY fits 80GB w/ AC at seq 4096
    a("--grad-accum", type=int, default=E("CPT_GRAD_ACCUM", 8, int))  # 8r x 2 x 8 x 4096 = 524,288 tok/step
    a("--lr", type=float, default=E("CPT_LR", 2e-5, float))           # TODO-VERIFY CPT peak LR for 30B-A3B
    a("--min-lr-frac", type=float, default=E("CPT_MIN_LR_FRAC", 0.10, float))  # cosine-to-10% (mandate)
    a("--warmup", type=int, default=E("CPT_WARMUP", 500, int))
    a("--total-tokens", type=float, default=E("CPT_TOTAL_TOKENS", 15e9, float))
    a("--max-steps", type=int, default=E("CPT_MAX_STEPS", 0, int))    # >0 overrides --total-tokens
    a("--weight-decay", type=float, default=E("CPT_WEIGHT_DECAY", 0.1, float))
    a("--grad-clip", type=float, default=E("CPT_GRAD_CLIP", 1.0, float))
    a("--save-every", type=int, default=E("CPT_SAVE_EVERY", 100, int))  # ~30 min at ~21 s/step (plan §3.4)
    a("--keep-last", type=int, default=E("CPT_KEEP_LAST", 3, int))
    a("--sync-script", default=E("CPT_SYNC_SCRIPT", ""))              # post-save hook: bash SCRIPT sync RUN
    a("--spike-guard", type=float, default=E("CPT_SPIKE_GUARD", 1.0, float))  # nats over ref EMA; 0 disables
    a("--grad-guard", type=float, default=E("CPT_GRAD_GUARD", 4.0, float))    # x ref EMA; 0 disables
    a("--valve", type=int, default=E("CPT_GUARD_VALVE", 100, int))    # consecutive skips before accepting anyway
    a("--guard-warmup", type=int, default=E("CPT_GUARD_WARMUP", 5, int))      # EMA updates before guards judge
    a("--quarantine-after", type=int, default=E("CPT_QUARANTINE_AFTER", 8, int))  # consecutive trips; 0 disables
    a("--rank-dev", type=float, default=E("CPT_RANK_DEV", 2.0, float))        # nats off the median; 0 disables
    a("--probe-every", type=int, default=E("CPT_PROBE_EVERY", 100, int))
    a("--router-ent-min", type=float, default=E("CPT_ROUTER_ENT_MIN", 0.5, float))   # TODO-VERIFY vs base-model baseline
    a("--router-maxfrac", type=float, default=E("CPT_ROUTER_MAXFRAC", 0.25, float))  # uniform top-8/128 -> 0.008
    a("--router-patience", type=int, default=E("CPT_ROUTER_PATIENCE", 3, int))
    a("--router-aux", type=float, default=E("CPT_ROUTER_AUX", 0.0, float))    # >0 enables HF aux loss  TODO-VERIFY
    a("--attn", default=E("CPT_ATTN", "sdpa"))  # switch to flash_attention_2 once verified on-box  TODO-VERIFY
    a("--seed", type=int, default=E("CPT_SEED", 1337, int))
    a("--log-every", type=int, default=E("CPT_LOG_EVERY", 1, int))
    a("--no-fsdp", action="store_true", default=bool(E("CPT_NO_FSDP", 0, int)))  # smoke: 1 proc / proxy model
    a("--deterministic", action="store_true", default=bool(E("CPT_DETERMINISTIC", 0, int)))
    a("--resume", action="store_true")
    args = p.parse_args()
    home = os.environ.get("CPT_HOME", "/data/cpt")
    args.ckpt_dir = args.ckpt_dir or f"{home}/ckpt/{args.run}"
    args.quarantine_dir = args.quarantine_dir or f"{home}/quarantine/{args.run}"
    args.log_dir = args.log_dir or f"{home}/logs"
    return args


class PackedLoader:
    """Streams (micro_bsz, row_len) int64 batches from the packed .npy shards. A single global cursor
    advances world*micro_bsz rows per micro-step; rank r takes the r-th micro slice — so the resumable
    state is ONE int, identical on every rank, with zero communication. Shards are mmapped lazily."""

    def __init__(self, data_dir, micro_bsz, seq_len, rank, world):
        man = json.load(open(Path(data_dir) / "manifest.json"))
        assert man["row_len"] == seq_len + 1, f"manifest row_len {man['row_len']} != seq_len+1 {seq_len + 1}"
        self.dir = Path(data_dir)
        self.mb, self.rank, self.world = micro_bsz, rank, world
        self.files = [s["file"] for s in man["shards"]]
        self.cum = np.cumsum([0] + [s["n_seqs"] for s in man["shards"]])
        self.total = int(self.cum[-1])
        assert self.total >= world * micro_bsz, "corpus smaller than one micro-batch sweep"
        self.cursor, self.epoch = 0, 0
        self._mm = {}

    def _row(self, i):
        s = bisect.bisect_right(self.cum, i) - 1
        if s not in self._mm:
            self._mm[s] = np.load(self.dir / self.files[s], mmap_mode="r")
        return self._mm[s][i - self.cum[s]]

    def next_batch(self):
        base = self.cursor + self.rank * self.mb
        rows = np.stack([np.asarray(self._row((base + j) % self.total)) for j in range(self.mb)])
        self.cursor += self.world * self.mb
        if self.cursor >= self.total:
            self.cursor -= self.total
            self.epoch += 1
        return torch.from_numpy(rows.astype(np.int64))

    def state_dict(self):
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, d):
        self.cursor, self.epoch = int(d["cursor"]), int(d["epoch"])


class Guards:
    """Spike-guard state, nanochat semantics: EMAs debiased by n; n counts ACCEPTED updates only,
    so a skip streak can never decay the reference. References frozen on skipped AND valve steps."""

    def __init__(self):
        self.loss_ema = 0.0
        self.grad_ema = 0.0
        self.n = 0
        self.consec = 0
        self.skips = 0
        self.router_streak = 0

    def loss_ref(self):
        return self.loss_ema / (1 - 0.9 ** self.n) if self.n else None

    def grad_ref(self):
        return self.grad_ema / (1 - 0.9 ** self.n) if self.n else None

    def accept(self, loss, gn):
        self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        self.grad_ema = 0.9 * self.grad_ema + 0.1 * gn
        self.n += 1
        self.consec = 0

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, d):
        self.__dict__.update(d)


def _f(x):
    if hasattr(x, "full_tensor"):  # FSDP2 clip_grad_norm_ can hand back a DTensor
        try:
            x = x.full_tensor()
        except Exception:
            pass
    return float(x)


def latest_complete(ckpt_dir):
    done = sorted(d for d in Path(ckpt_dir).glob("step_*") if (d / "COMPLETE").exists())
    return done[-1] if done else None


def main():
    args = get_args()
    from accelerate import Accelerator
    if args.no_fsdp:
        acc = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")
    else:
        from accelerate.utils import FullyShardedDataParallelPlugin
        fsdp = FullyShardedDataParallelPlugin(  # TODO-VERIFY kwarg names against the pinned accelerate version
            fsdp_version=2,
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["Qwen3MoeDecoderLayer"],  # TODO-VERIFY class name in pinned transformers
            activation_checkpointing=True,
            state_dict_type="SHARDED_STATE_DICT",  # TODO-VERIFY: naming under fsdp_version=2
            reshard_after_forward=True,
            cpu_ram_efficient_loading=True,
        )
        acc = Accelerator(mixed_precision="bf16", fsdp_plugin=fsdp)
    rank, world = acc.process_index, acc.num_processes

    def log0(*m):
        if acc.is_main_process:
            print(*m, flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2 ** 32)
    if args.deterministic:  # G2 mode — bitwise or near-bitwise resume comparison (plan §4 Gate 2)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(args.model_id)
    if args.router_aux > 0 and hasattr(cfg, "router_aux_loss_coef"):  # TODO-VERIFY field names on Qwen3MoeConfig
        cfg.router_aux_loss_coef = args.router_aux
        cfg.output_router_logits = True
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_id, config=cfg, torch_dtype=dtype,
                                                 attn_implementation=args.attn)
    model.train()
    log0(f"model {args.model_id} loaded in {time.time() - t0:.0f}s | "
         f"params {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    model = acc.prepare(model)  # FSDP2: shard FIRST, then build the optimizer on the DTensor params
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": nodecay, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    optimizer = acc.prepare(optimizer)

    loader = PackedLoader(args.data_dir, args.micro_bsz, args.seq_len, rank, world)
    tokens_per_step = world * args.micro_bsz * args.grad_accum * args.seq_len
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.total_tokens / tokens_per_step)
    warmup = min(args.warmup, max(1, total_steps // 10))
    min_lr = args.lr * args.min_lr_frac

    def lr_at(t):  # t = accepted-update count; skipped steps do NOT advance LR (mandate)
        if t < warmup:
            return args.lr * (t + 1) / warmup
        prog = min(1.0, (t - warmup) / max(1, total_steps - warmup))
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * prog))

    guards = Guards()
    start_step, opt_steps, tokens = 0, 0, 0
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    metrics_path = Path(args.log_dir) / f"metrics-{args.run}.jsonl"
    probe_path = Path(args.log_dir) / f"router-{args.run}.jsonl"

    if args.resume:
        last = latest_complete(args.ckpt_dir)
        if last is None:
            log0("resume: no checkpoint found -> fresh start")
        else:
            acc.load_state(str(last))  # TODO-VERIFY: sharded state requires the SAME world size (fixed 8-GPU box)
            meta = json.load(open(last / "meta.json"))
            start_step, opt_steps, tokens = meta["step"], meta["opt_steps"], meta["tokens"]
            loader.load_state_dict(meta["loader"])
            guards.load_state_dict(meta["guards"])  # 1.5B lesson: guard refs resume too, never re-warm blind
            log0(f"resume: step {start_step} tokens {tokens} (ckpt {last})")

    log0(f"CPT guards armed | spike {args.spike_guard} | grad {args.grad_guard}x | valve {args.valve} | "
         f"warmup_refs {args.guard_warmup} | quarantine_after {args.quarantine_after} | rank_dev {args.rank_dev} | "
         f"probe_every {args.probe_every} (ent_min {args.router_ent_min} maxfrac {args.router_maxfrac} "
         f"patience {args.router_patience})")
    log0(f"plan: {total_steps} steps x {tokens_per_step} tok = {total_steps * tokens_per_step / 1e9:.2f}B tokens | "
         f"world {world} | micro {args.micro_bsz} x accum {args.grad_accum} x seq {args.seq_len} | lr {args.lr:.2e}")

    preempt = {"flag": False}  # spot preemption gives ~30 s ACPI notice -> SIGTERM; save at the next boundary
    signal.signal(signal.SIGTERM, lambda s, f: preempt.__setitem__("flag", True))

    def save_ckpt(step):
        t = time.time()
        d = Path(args.ckpt_dir) / f"step_{step:08d}"
        acc.save_state(str(d))  # sharded model+optim+per-rank RNG
        if acc.is_main_process:
            json.dump({"step": step, "opt_steps": opt_steps, "tokens": tokens,
                       "loader": loader.state_dict(), "guards": guards.state_dict(),
                       "run": args.run, "time": time.strftime("%FT%TZ", time.gmtime())},
                      open(d / "meta.json", "w"))
        acc.wait_for_everyone()
        if acc.is_main_process:
            (d / "COMPLETE").touch()  # marker LAST -> checkpoint atomically discoverable
            if args.sync_script:      # post-save hook: GCS mirror, non-blocking
                subprocess.Popen(["bash", args.sync_script, "sync", args.run],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            done = sorted(p_ for p_ in Path(args.ckpt_dir).glob("step_*") if (p_ / "COMPLETE").exists())
            for old in done[:-args.keep_last]:  # GCS keeps everything; local keeps last N
                shutil.rmtree(old, ignore_errors=True)
        log0(f"ckpt step {step} -> {d} ({time.time() - t:.0f}s)")

    def quarantine(reason, detail, step):
        if acc.is_main_process:
            os.makedirs(args.quarantine_dir, exist_ok=True)
            last = latest_complete(args.ckpt_dir)
            dst = Path(args.quarantine_dir) / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_step{step}_{reason}"
            if last is not None:
                shutil.copytree(last, dst)  # COPY the last good checkpoint — never overwrite it
            else:
                os.makedirs(dst, exist_ok=True)
            json.dump(detail, open(dst / "anomaly.json", "w"), indent=1)
            Path(args.ckpt_dir, "QUARANTINED").write_text(f"{reason} step {step}\n")  # launch_cpt.sh refuses this run
            print(f"CPT_QUARANTINE step {step} reason {reason} -> {dst}", flush=True)
            if args.sync_script:  # blocking final mirror so the evidence reaches GCS before we die
                subprocess.run(["bash", args.sync_script, "sync", args.run], timeout=900, check=False)
        acc.wait_for_everyone()
        sys.exit(3)  # non-zero: the supervisor must NOT auto-restart into the same failure (owner reviews)

    probed_na = {"done": False}

    def router_probe(step, batch):
        try:
            with torch.no_grad():
                out = model(input_ids=batch, output_router_logits=True)  # TODO-VERIFY kwarg on Qwen3MoeForCausalLM
        except TypeError:
            out = None  # dense proxy model (G1 smoke) has no router kwarg
        rl = getattr(out, "router_logits", None) if out is not None else None
        if not rl:
            if not probed_na["done"]:
                log0(f"router probe n/a step {step} (dense model or no router_logits)")
                probed_na["done"] = True
            return
        mcfg = acc.unwrap_model(model).config
        k = int(getattr(mcfg, "num_experts_per_tok", 1))  # TODO-VERIFY attr names (Qwen3MoeConfig: 8 of 128)
        stats = []
        for logits in rl:
            l2 = logits.float().reshape(-1, logits.shape[-1])
            cnt = torch.bincount(l2.topk(k, dim=-1).indices.reshape(-1), minlength=l2.shape[-1]).float()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(cnt)  # global expert load; synchronous — same stats (and verdict) on every rank
            load = cnt / cnt.sum().clamp(min=1)
            ent = float(-(load * (load + 1e-9).log()).sum() / math.log(l2.shape[-1]))  # normalized: uniform = 1.0
            stats.append((ent, float(load.max().item())))
        min_ent = min(s[0] for s in stats)
        max_frac = max(s[1] for s in stats)
        worst = min(range(len(stats)), key=lambda i: stats[i][0])
        log0(f"CPT_PROBE step {step} min_ent {min_ent:.3f} max_frac {max_frac:.3f} "
             f"worst_layer {worst} layers {len(stats)}")
        if acc.is_main_process:
            with open(probe_path, "a") as f:
                f.write(json.dumps({"step": step,
                                    "per_layer": [[round(e, 4), round(m, 4)] for e, m in stats]}) + "\n")
        if min_ent < args.router_ent_min or max_frac > args.router_maxfrac:
            guards.router_streak += 1
            log0(f"CPT_ALERT router step {step}: min_ent {min_ent:.3f} (min {args.router_ent_min}) "
                 f"max_frac {max_frac:.3f} (max {args.router_maxfrac}) — "
                 f"streak {guards.router_streak}/{args.router_patience}")
            if guards.router_streak >= args.router_patience:
                quarantine("ROUTER", {"step": step, "min_ent": min_ent, "max_frac": max_frac,
                                      "per_layer": [[e, m] for e, m in stats]}, step)
        else:
            guards.router_streak = 0

    for step in range(start_step + 1, total_steps + 1):
        if preempt["flag"]:
            log0(f"SIGTERM -> checkpoint at step boundary {step - 1} and exit")
            save_ckpt(step - 1)
            sys.exit(0)
        ts = time.time()
        cur_lr = lr_at(opt_steps)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        micro_losses, forens = [], []
        for _m in range(args.grad_accum):
            batch = loader.next_batch().to(acc.device)
            out = model(input_ids=batch, labels=batch)  # HF shifts labels -> seq_len targets per row
            # per-micro grad sync accepted for the verification lane; TODO-VERIFY no_sync/set_requires_gradient_sync
            # for FSDP2 before the full 15B-token run if throughput demands it
            acc.backward(out.loss / args.grad_accum)
            micro_losses.append(out.loss.detach())
            forens.append(batch[:, :256].cpu())  # batch heads kept for trip forensics

        step_loss = torch.stack(micro_losses).mean()
        per_rank = acc.gather(step_loss.reshape(1)).float()  # SYNCHRONOUS gather (torn-gather lesson)
        mean_l = per_rank.mean().item()
        med_l = per_rank.median().item()
        rank_dev = (per_rank - med_l).abs().max().item()
        gn = _f(acc.clip_grad_norm_(model.parameters(), args.grad_clip))  # global pre-clip norm across shards

        ref, gref = guards.loss_ref(), guards.grad_ref()
        trip = None
        if not (math.isfinite(mean_l) and math.isfinite(gn) and bool(torch.isfinite(per_rank).all())):
            trip = "NONFINITE"
        elif guards.n >= args.guard_warmup:
            if args.spike_guard > 0 and mean_l > ref + args.spike_guard:
                trip = "LOSS"
            elif args.grad_guard > 0 and gn > args.grad_guard * gref:
                trip = "GRAD"
            elif args.rank_dev > 0 and rank_dev > args.rank_dev:
                trip = "RANKDIV"

        # identical inputs on every rank (per_rank gathered, gn global) -> identical verdict, no extra comm
        if trip == "NONFINITE" or (trip and 0 < args.quarantine_after <= guards.consec + 1):
            quarantine(trip, {"step": step, "per_rank": [round(v, 4) for v in per_rank.tolist()],
                              "grad_norm": gn, "loss_ref": ref, "grad_ref": gref,
                              "consec": guards.consec + 1, "loader": loader.state_dict()}, step)

        skipped = False
        if trip and guards.consec < args.valve:
            skipped = True  # skip: no optimizer step, no LR advance, refs untouched; batch stays consumed
            guards.consec += 1
            guards.skips += 1
            log0(f"CPT_GUARD trip {trip} step {step}: loss {mean_l:.4f} (ref {ref}) gn {gn:.3f} (gref {gref}) "
                 f"rank_dev {rank_dev:.3f} per-rank {[round(v, 3) for v in per_rank.tolist()]} "
                 f"consec {guards.consec} cursor {loader.state_dict()}")
            if guards.consec <= 4:  # forensics for the first skips of a streak (nanochat convention)
                fd = Path(args.log_dir) / "trip_batches"
                os.makedirs(fd, exist_ok=True)
                torch.save({"step": step, "rank": rank, "trip": trip,
                            "micro_losses": [_f(l) for l in micro_losses],
                            "x_heads": forens, "loader": loader.state_dict()},
                           fd / f"step{step}_rank{rank}.pt")
        elif trip:
            log0(f"CPT_GUARD valve step {step}: {trip} still tripping after {args.valve} consecutive skips "
                 f"-> accepting the update (refs untouched)")
            guards.consec = 0
        else:
            guards.accept(mean_l, gn)

        if not skipped:
            optimizer.step()
            opt_steps += 1
        optimizer.zero_grad(set_to_none=True)
        tokens += tokens_per_step

        dt = time.time() - ts
        tps = tokens_per_step / dt
        if step % args.log_every == 0 or skipped:
            ema = guards.loss_ref() or mean_l
            log0(f"step {step}/{total_steps} | loss {mean_l:.4f} | ema {ema:.4f} | gn {gn:.3f} | "
                 f"lr {cur_lr:.2e} | tps {tps:.0f} | toks {tokens} | skips {guards.skips} | "
                 f"epoch {loader.epoch}{' | SKIPPED' if skipped else ''}")
            if acc.is_main_process:
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"step": step, "loss": round(mean_l, 5), "ema": round(ema, 5),
                                        "gn": round(gn, 4), "lr": cur_lr, "tps": round(tps, 1),
                                        "tokens": tokens, "skips": guards.skips,
                                        "skipped": skipped}) + "\n")

        if args.probe_every > 0 and (step % args.probe_every == 0 or step == start_step + 1):
            router_probe(step, batch)  # re-forwards the last micro-batch, no_grad (baseline probe on step 1)

        if args.save_every > 0 and step % args.save_every == 0:
            save_ckpt(step)

    if total_steps > start_step and (args.save_every <= 0 or total_steps % args.save_every != 0):
        save_ckpt(total_steps)
    log0(f"CPT_DONE step {total_steps} tokens {tokens} skips {guards.skips}")


if __name__ == "__main__":
    main()
