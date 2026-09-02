"""NANOCHAT_WARMDOWN_START=<step>: begin the linear LR warmdown (and the Muon momentum warmdown) at <step> instead of
num_iterations*(1-warmdown_ratio), decaying to final_lr_frac at num_iterations. Idempotent patch for scripts/base_train.py.
Escape-ladder rung for the 1.5B run (attention-logit growth without QK-norm)."""
import sys
p = "scripts/base_train.py"; s = open(p).read()
if "NANOCHAT_WARMDOWN_START" in s:
    print("base_train: warmdown-start hook already present"); sys.exit(0)
old_lr = ("    if it < warmup_iters:\n        return (it + 1) / warmup_iters\n    elif it <= num_iterations - warmdown_iters:\n        return 1.0\n")
new_lr = ("    if it < warmup_iters:\n        return (it + 1) / warmup_iters\n"
          "    _wd_start = min(WARMDOWN_START, num_iterations - warmdown_iters) if WARMDOWN_START > 0 else num_iterations - warmdown_iters  # NANOCHAT_WARMDOWN_START\n"
          "    if WARMDOWN_START > 0 and it > _wd_start:\n"
          "        progress = (num_iterations - it) / max(1, num_iterations - _wd_start)\n"
          "        return progress * 1.0 + (1 - progress) * args.final_lr_frac\n"
          "    elif it <= num_iterations - warmdown_iters:\n        return 1.0\n")
assert s.count(old_lr) == 1, "lr schedule not found"; s = s.replace(old_lr, new_lr)
old_m = ("    elif it >= warmdown_start:\n        progress = (it - warmdown_start) / warmdown_iters\n        return 0.97 * (1 - progress) + 0.90 * progress\n")
new_m = ("    elif WARMDOWN_START > 0 and it >= min(WARMDOWN_START, warmdown_start):  # NANOCHAT_WARMDOWN_START\n"
         "        _ws = min(WARMDOWN_START, warmdown_start); progress = (it - _ws) / max(1, num_iterations - _ws)\n"
         "        return 0.97 * (1 - progress) + 0.90 * progress\n"
         "    elif it >= warmdown_start:\n        progress = (it - warmdown_start) / warmdown_iters\n        return 0.97 * (1 - progress) + 0.90 * progress\n")
assert s.count(old_m) == 1, "momentum schedule not found"; s = s.replace(old_m, new_m)
old_h = "# Learning rate schedule (linear warmup, constant, linear warmdown)\n"
assert s.count(old_h) == 1
s = s.replace(old_h, old_h + "WARMDOWN_START = int(os.environ.get('NANOCHAT_WARMDOWN_START', '0')); print0(f'WARMDOWN_START override: {WARMDOWN_START}')\n")
if "\nimport os\n" not in s and not s.startswith("import os\n"): s = "import os\n" + s
open(p, "w").write(s); print("base_train: warmdown-start hook added")
