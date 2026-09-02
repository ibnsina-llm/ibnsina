"""Post-run weight sanity check for a nanochat base run.

For each requested checkpoint step: val bpb computed exactly like base_train's periodic eval (same loader, same
evaluate_bpb, same token_bytes) plus greedy generations for a fixed set of Persian/English prompts. Meant to be run on
the training box after the base stage, or on any GPU box that has the checkpoints synced from GCS.

  cd /data/nanochat && NANOCHAT_ARCH=llama NANOCHAT_BASE_DIR=/data/nc \\
    uv run --no-sync python /data/pipeline/training/sanity_check.py --tag big --steps 27000,88000 \\
      --eval-tokens 20971520 --log /data/logs/train-big.log --out /data/logs/sanity_big.json

Reads only; writes the JSON report. Also parses the training log's "Validation bpb" lines so the report shows whether the
val trajectory after the resume point continued the pre-resume trend (the "no residue" check).
"""
import argparse, json, os, re, sys, time
import torch
sys.path.insert(0, os.getcwd())
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes
from nanochat.engine import Engine

PROMPTS = [
    "پایتخت ایران",
    "حافظ شیرازی، شاعر بزرگ قرن هشتم هجری،",
    "برای درست کردن چای ابتدا",
    "در سال ۱۳۵۷ خورشیدی",
    "دیابت نوع دو بیماری است که",
    "The capital of Iran is",
    "def fibonacci(n):",
]

def val_bpb(model, tokenizer, device, B, T, eval_tokens):
    token_bytes = get_token_bytes(device=device)
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, B, T, split="val", device=device)
    steps = max(1, eval_tokens // (B * T))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return float(evaluate_bpb(model, loader, steps, token_bytes))

def generations(model, tokenizer, max_tokens):
    eng = Engine(model, tokenizer); out = {}
    for p in PROMPTS:
        toks = tokenizer.encode(p, prepend="<|bos|>")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            res = eng.generate(toks, num_samples=1, max_tokens=max_tokens, temperature=1.0, top_k=1)   # top_k=1 -> greedy
        rows = list(res) if not isinstance(res, (list, tuple)) else res
        # Engine.generate yields per-step tokens or returns rows; normalise to a flat token list
        flat = []
        for r in rows:
            if isinstance(r, (list, tuple)):
                for x in r: flat += (x if isinstance(x, list) else [x])
            elif isinstance(r, torch.Tensor): flat += r.flatten().tolist()
            else: flat.append(r)
        out[p] = tokenizer.decode([t for t in flat if isinstance(t, int)])[:400]
    return out

def parse_val_log(path):
    vals = []
    if not path or not os.path.exists(path): return vals
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"Step (\d+) \| Validation bpb: ([0-9.]+)", line)
        if m: vals.append((int(m.group(1)), float(m.group(2))))
    # keep the LAST value logged for each step (later passes override earlier, rolled-back ones)
    last = {}
    for s, v in vals: last[s] = v
    return sorted(last.items())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="big"); ap.add_argument("--steps", required=True, help="comma-separated checkpoint steps")
    ap.add_argument("--eval-tokens", type=int, default=20_971_520); ap.add_argument("--device-batch-size", type=int, default=8); ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=48); ap.add_argument("--log", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = torch.device("cuda")
    report = {"tag": a.tag, "eval_tokens": a.eval_tokens, "checkpoints": {}, "val_trajectory_from_log": parse_val_log(a.log), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for step in [int(s) for s in a.steps.split(",")]:
        t0 = time.time(); model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=a.tag, step=step); model.eval()
        bpb = val_bpb(model, tokenizer, device, a.device_batch_size, a.max_seq_len, a.eval_tokens)
        gens = generations(model, tokenizer, a.max_tokens)
        report["checkpoints"][str(step)] = {"val_bpb": bpb, "generations": gens, "seconds": round(time.time() - t0, 1)}
        print(f"step {step}: val bpb {bpb:.4f}", flush=True)
        for p, g in gens.items(): print(f"   [{p}] -> {g[:160]!r}", flush=True)
        del model; torch.cuda.empty_cache()
    traj = report["val_trajectory_from_log"]
    if len(traj) >= 4:
        pre = [(s, v) for s, v in traj if s <= 26000][-3:]; post = [(s, v) for s, v in traj if s >= 28000][:3]
        report["trend_note"] = f"pre-resume last vals {pre}; first post-resume vals {post}"
        print(report["trend_note"])
    json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1); print("wrote", a.out)

if __name__ == "__main__":
    main()
