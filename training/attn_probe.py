"""Attention-logit magnitude probe for nanochat-loop checkpoints (Llama, Qwen3 or GPT arch; Qwen3's q/k norms are applied).

For each checkpoint step: per-layer max and 99.9th-percentile |q·k/√d| after RoPE (causal positions only) on a few
validation batches, plus q/k RMS. A slow upward creep in the max logit precedes softmax saturation and gradient-norm
blow-ups by thousands of steps; probe every checkpoint. See the technical report, section 5.

  cd /path/to/nanochat && NANOCHAT_ARCH=llama NANOCHAT_BASE_DIR=/path/to/base_dir \\
    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python /path/to/training/attn_probe.py 20000,27000 --tag big --out attn_probe.json
"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.getcwd())
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
ARCH = os.environ.get("NANOCHAT_ARCH", "gpt")
apply_rotary_emb = __import__({"llama": "nanochat.llama", "qwen3": "nanochat.qwen3"}.get(ARCH, "nanochat.gpt"), fromlist=["apply_rotary_emb"]).apply_rotary_emb

ap = argparse.ArgumentParser()
ap.add_argument("steps", help="comma-separated checkpoint steps"); ap.add_argument("--tag", default="big")
ap.add_argument("--batches", type=int, default=3); ap.add_argument("--B", type=int, default=2); ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--out", default="attn_probe.json"); a = ap.parse_args()
dev = torch.device("cuda:0"); steps = [int(s) for s in a.steps.split(",")]
model, tokenizer, _ = load_model("base", dev, phase="eval", model_tag=a.tag, step=steps[0])
loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, a.B, a.T, split="val", device=dev)
batches = [next(loader)[0] for _ in range(a.batches)]
del model; torch.cuda.empty_cache()
out = {}
for step in steps:
    model, tokenizer, _ = load_model("base", dev, phase="eval", model_tag=a.tag, step=step); model.eval(); stats = {}
    def make_hook(li):
        def hook(mod, args, output):
            x, cos_sin = args[0], args[1]; Bq, Tq, _ = x.shape
            q = mod.c_q(x).view(Bq, Tq, mod.n_head, mod.head_dim); k = mod.c_k(x).view(Bq, Tq, mod.n_kv_head, mod.head_dim)
            if hasattr(mod, "q_norm"): q, k = mod.q_norm(q), mod.k_norm(k)  # qwen3 QK-norm (before RoPE), so the probe sees the real logits
            cos, sin = cos_sin; q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            k = k.repeat_interleave(mod.n_head // mod.n_kv_head, dim=2)
            q = q.float().transpose(1, 2); k = k.float().transpose(1, 2)
            logits = (q @ k.transpose(-1, -2)) / (mod.head_dim ** 0.5)
            mask = torch.tril(torch.ones(Tq, Tq, device=x.device, dtype=torch.bool))
            lg = logits.masked_select(mask).abs().float()
            s = stats.setdefault(li, {"max": 0.0, "p999": 0.0, "qrms": 0.0, "krms": 0.0, "n": 0})
            sample = lg[torch.randint(0, lg.numel(), (200000,), device=lg.device)]
            s["max"] = max(s["max"], lg.max().item()); s["p999"] = max(s["p999"], torch.quantile(sample, 0.999).item())
            s["qrms"] += q.pow(2).mean().sqrt().item(); s["krms"] += k.pow(2).mean().sqrt().item(); s["n"] += 1
        return hook
    hooks = [blk.attn.register_forward_hook(make_hook(i)) for i, blk in enumerate(model.transformer.h)]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for x in batches: model(x)
    for h in hooks: h.remove()
    worst = sorted(stats.items(), key=lambda kv: -kv[1]["max"])[:5]
    out[step] = {"max_logit_overall": max(v["max"] for v in stats.values()), "p999_overall": max(v["p999"] for v in stats.values()),
                 "per_layer_max": [round(stats[i]["max"], 1) for i in range(len(stats))],
                 "worst_layers": [(i, round(v["max"], 1), round(v["p999"], 1), round(v["qrms"] / v["n"], 2), round(v["krms"] / v["n"], 2)) for i, v in worst]}
    print(f"step {step}: max |logit| {out[step]['max_logit_overall']:.1f}  p99.9 {out[step]['p999_overall']:.1f}  per-layer max {out[step]['per_layer_max']}  worst (layer,max,p999,q_rms,k_rms) {out[step]['worst_layers']}", flush=True)
    del model; torch.cuda.empty_cache()
json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)
