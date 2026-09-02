"""NANOCHAT_LR_SCALE=<f> multiplies every optimizer group's learning rate for the rest of a (resumed) run. Idempotent patch
for scripts/base_train.py. Needed because optimizer.load_state_dict restores the saved initial_lr, so new --*-lr args are
ignored on resume. Used on the 1.5B run at step 27,000 (attention-logit growth without QK-norm -> LR x0.5)."""
import sys
p = "scripts/base_train.py"; s = open(p).read()
if "NANOCHAT_LR_SCALE" in s:
    print("base_train: LR scale hook already present"); sys.exit(0)
old = '        group["lr"] = group["initial_lr"] * lrm\n'
assert s.count(old) == 1, "lr line not found"
s = s.replace(old, '        group["lr"] = group["initial_lr"] * lrm * LR_SCALE  # NANOCHAT_LR_SCALE\n')
old2 = "# Training loop\n"
assert s.count(old2) == 1
s = s.replace(old2, "# Training loop\nLR_SCALE = float(os.environ.get('NANOCHAT_LR_SCALE', '1.0')); print0(f'LR_SCALE: {LR_SCALE}')\n")
if "\nimport os\n" not in s and not s.startswith("import os\n"): s = "import os\n" + s
open(p, "w").write(s); print("base_train: LR scale hook added")
