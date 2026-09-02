"""One-shot data skip on resume (loss-spike recovery). Idempotent patch for scripts/base_train.py.
NANOCHAT_RESUME_SKIP=<step>:<row_groups> advances the data-loader position only when resuming from exactly <step>;
later resumes are unaffected and checkpoint files are never modified."""
import sys
p = "scripts/base_train.py"; s = open(p).read()
if "NANOCHAT_RESUME_SKIP" in s:
    print("base_train: NANOCHAT_RESUME_SKIP hook already present"); sys.exit(0)
old = 'dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]\n'
hook = (
    '_skip = os.environ.get("NANOCHAT_RESUME_SKIP", "")\n'
    'if resuming and _skip and int(_skip.split(":")[0]) == args.resume_from_step:\n'
    '    dataloader_resume_state_dict["rg_idx"] += int(_skip.split(":")[1])\n'
    '    print0(f"NANOCHAT_RESUME_SKIP: advanced loader by {_skip.split(chr(58))[1]} row groups -> {dataloader_resume_state_dict}")\n'
)
assert s.count(old) == 1, "resume line not found"
s = s.replace(old, old + hook)
if "\nimport os\n" not in s and not s.startswith("import os\n"):
    s = "import os\n" + s
open(p, "w").write(s); print("base_train: NANOCHAT_RESUME_SKIP hook added")
