#!/bin/bash
# Apply the Persian + Llama/Qwen3-arch patches to a nanochat checkout (idempotent).  usage: apply_patches.sh [/data/nanochat]
set -eu; NC=${1:-/data/nanochat}; HERE=$(cd "$(dirname "$0")" && pwd)
cp "$HERE"/nanochat/llama.py "$HERE"/nanochat/qwen3.py "$HERE"/nanochat/arch.py "$NC"/nanochat/; cp "$HERE"/tasks/*.py "$NC"/tasks/; cp "$HERE"/scripts/*.py "$NC"/scripts/
python3 - "$NC" <<'PY'
import sys; nc = sys.argv[1]
def patch(path, pairs):
    s = open(path).read(); orig = s
    for old, new in pairs:
        if new in s: continue
        if old not in s: continue
        assert s.count(old) == 1, f"{path}: anchor not unique ({s.count(old)}): {old[:70]!r}"
        s = s.replace(old, new)
    open(path, "w").write(s); print(("patched " if s != orig else "already patched ") + path)
patch(f"{nc}/scripts/base_train.py", [
    ("n_kv_head=(args.n_kv_head or num_heads)", "n_kv_head=(args.n_kv_head if args.n_kv_head and num_heads % args.n_kv_head == 0 else num_heads)"),  # upgrade boxes patched with the first version
    ("from nanochat.gpt import GPT, GPTConfig, Linear", "from nanochat.arch import GPT, GPTConfig, Linear"),
    ('parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")',
     'parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")\nparser.add_argument("--n-kv-head", type=int, default=0, help="KV heads for GQA (0 = same as query heads)")\nparser.add_argument("--ffn-hidden", type=int, default=0, help="llama/qwen3 arches: SwiGLU hidden width (0 = auto)")'),
    ("        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,\n        window_pattern=args.window_pattern,\n    )",
     "        n_layer=depth, n_head=num_heads, n_kv_head=(args.n_kv_head if args.n_kv_head and num_heads % args.n_kv_head == 0 else num_heads), n_embd=model_dim,\n        window_pattern=args.window_pattern,\n        **({\"ffn_hidden\": args.ffn_hidden} if args.ffn_hidden else {}),\n    )"),
])
patch(f"{nc}/nanochat/checkpoint_manager.py", [
    ("from nanochat.gpt import GPT, GPTConfig", "from nanochat.arch import GPT, GPTConfig, model_classes_for"),
    ('    model_config = GPTConfig(**model_config_kwargs)\n    _patch_missing_keys(model_data, model_config)\n    with torch.device("meta"):\n        model = GPT(model_config)',
     '    ModelCls, ConfigCls = model_classes_for(model_config_kwargs)  # Llama/Qwen3 checkpoints carry arch="llama"/"qwen3"\n    model_config = ConfigCls(**model_config_kwargs)\n    if getattr(model_config, "arch", "gpt") not in ("llama", "qwen3"):\n        _patch_missing_keys(model_data, model_config)\n    with torch.device("meta"):\n        model = ModelCls(model_config)'),
    ('    if getattr(model_config, "arch", "gpt") != "llama":', '    if getattr(model_config, "arch", "gpt") not in ("llama", "qwen3"):'),  # upgrade boxes patched before qwen3
])
PY
echo "patches applied to $NC"

# one-shot data skip on resume (loss-spike recovery), see resume_skip_patch.py
(cd "$NC" && python3 "$HERE/resume_skip_patch.py")

# loss-spike guard (skip the update on a garbage batch), see spike_guard_patch.py
(cd "$NC" && python3 "$HERE/spike_guard_patch.py")

# keep async-collective inputs/Work objects alive until wait() in MuonAdamW (torn all_gather -> loss plateaus), see optim_patch.py
(cd "$NC" && python3 "$HERE/optim_patch.py")

# NANOCHAT_LR_SCALE: scale all group LRs for the rest of a resumed run, see lr_scale_patch.py
(cd "$NC" && python3 "$HERE/lr_scale_patch.py")

# NANOCHAT_WARMDOWN_START: early LR/momentum warmdown from a given step, see warmdown_patch.py
(cd "$NC" && python3 "$HERE/warmdown_patch.py")
