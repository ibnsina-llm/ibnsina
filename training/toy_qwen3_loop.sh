#!/bin/bash
# Distribution-loop validation for the Qwen3 arch on a small GPU box — the gate before the 3B (same gate the 1.5B passed):
#   tokenizer v2 (Llama-3 regex) -> Qwen3-arch d12 toy (QK-norm exercised) trained with nanochat's loop -> GGUF (F16, arch qwen3)
#   -> llama.cpp (CPU build) -> Q4_K_M -> llama-cli generation
#   + tokenizer parity (llama-tokenize vs tiktoken) + greedy-decode parity (llama.cpp vs torch; also proves the NO-permute NEOX rope path).
# Prereq: gpu_startup.sh ran (code in $PIPELINE_DIR, nanochat env, data shards present). Run as root: bash toy_qwen3_loop.sh
# Paths/env overridable: CORPUS_BUCKET, PIPELINE_DIR, NANOCHAT_DIR, NANOCHAT_BASE_DIR, SHARD_SRC, LLAMA_CPP_DIR.
set -u; export HOME=/root PATH=/root/.local/bin:$PATH
B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}; T=${PIPELINE_DIR:-/data/pipeline}/training; NC=${NANOCHAT_DIR:-/data/nanochat}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/data/nc2}; SHARD_SRC=${SHARD_SRC:-/data/nc/base_data_climbmix}
LCDIR=${LLAMA_CPP_DIR:-/data/llama.cpp}; LC=$LCDIR/build/bin; export LC
LOG=/data/logs/toy_qwen3_loop.log; mkdir -p /data/logs /data/export $NANOCHAT_BASE_DIR/tokenizer; exec > >(tee -a $LOG) 2>&1
echo "== $(date -u +%FT%TZ) toy qwen3 loop start"
# 1) tokenizer v2 + data (shared shards from the v1 base dir)
[ -f $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl ] || gcloud --no-user-output-enabled storage cp $B/tokenizer/v2_32k_llama/tokenizer.pkl $B/tokenizer/v2_32k_llama/token_bytes.pt $NANOCHAT_BASE_DIR/tokenizer/
[ -e $NANOCHAT_BASE_DIR/$(basename $SHARD_SRC) ] || ln -s $SHARD_SRC $NANOCHAT_BASE_DIR/$(basename $SHARD_SRC)
# 2) patches (Qwen3 arch + arch switch + guards + Persian tasks)
bash $T/nanochat_patches/apply_patches.sh $NC
cd $NC && uv pip install --quiet gguf sentencepiece 2>/dev/null || uv pip install gguf
COMMON="--run=dummy --core-metric-every=-1 --sample-every=-1 --window-pattern=L"
# 3) d12 toy on the Qwen3 arch (dim 768, 6 heads / 2 kv — GQA + QK-norm both exercised; the artifact goes through the whole loop)
NANOCHAT_ARCH=qwen3 uv run --no-sync torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- $COMMON --model-tag=toy-qwen3 --depth=12 --max-seq-len=1024 \
  --device-batch-size=16 --total-batch-size=131072 --num-iterations=300 --eval-every=100 --eval-tokens=1048576 --warmup-steps=20 > /data/logs/train-toy-qwen3.log 2>&1
grep -aE "^step 00299|Validation" /data/logs/train-toy-qwen3.log | tail -n 3 | cut -c1-120
# 4) export GGUF F16 (arch qwen3, q/k norms, no permute) -> llama.cpp CPU build -> Q4_K_M
uv run --no-sync python $T/export_gguf.py --base-dir $NANOCHAT_BASE_DIR --source base --model-tag toy-qwen3 --out /data/export/toy-qwen3-f16.gguf --name persian-toy-qwen3
if [ ! -x $LC/llama-quantize ]; then
  command -v cmake >/dev/null || { apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake >/dev/null 2>&1; }
  [ -d $LCDIR ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp $LCDIR
  (cd $LCDIR && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF && cmake --build build -j $(nproc) --target llama-quantize llama-cli llama-tokenize llama-simple) > /data/logs/llamacpp_build.log 2>&1 || { echo "!! llama.cpp build failed — see /data/logs/llamacpp_build.log"; tail -n 5 /data/logs/llamacpp_build.log; }
fi
ls $LC | tr "\n" " "; echo
$LC/llama-quantize /data/export/toy-qwen3-f16.gguf /data/export/toy-qwen3-Q4_K_M.gguf Q4_K_M 2>&1 | tail -n 2
ls -la /data/export/toy-qwen3-*.gguf
# 5) tokenizer parity: llama.cpp ids vs tiktoken ids on Persian / English / code / digits / newlines
uv run --no-sync python - <<'PY'
import os, pickle, subprocess, json
base = os.environ.get("NANOCHAT_BASE_DIR", "/data/nc2"); lc = os.environ["LC"]
enc = pickle.load(open(f"{base}/tokenizer/tokenizer.pkl", "rb"))
tests = ["زبان فارسی یکی از کهن‌ترین زبان‌های زنده جهان است.", "Persian is one of the oldest living languages in the world.", "def area(r):\n    return 3.14159 * r ** 2\n", "در سال 1403 حدود 12345678 نفر", "خط اول\n\n\nخط دوم  ", "Hello world! 😀 سلام"]
ok = True
for s in tests:
    ref = enc.encode_ordinary(s)
    out = subprocess.run([f"{lc}/llama-tokenize", "-m", "/data/export/toy-qwen3-f16.gguf", "-p", s, "--ids", "--no-bos", "--log-disable"], capture_output=True, text=True).stdout
    got = json.loads(out.strip().splitlines()[-1]) if out.strip() else None
    same = got == ref; ok &= same
    print(("OK  " if same else "DIFF") + f" {s[:40]!r}: tiktoken {ref[:12]} llama.cpp {got[:12] if got else out[-200:]}")
print("TOKENIZER PARITY:", "PASS" if ok else "FAIL")
PY
# 6) greedy-decode parity: torch (bf16, QK-norm in nanochat/qwen3.py) vs llama.cpp qwen3 graph, F16 and Q4_K_M
#    This is the check that the NEOX no-permute export and the q/k norm tensor placement are right — DIFF here means the export path is wrong.
uv run --no-sync python - <<'PY'
import os, subprocess, torch, sys
sys.argv = ["x"]
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
lc = os.environ["LC"]
ddp, rank, lrank, world, device = compute_init(autodetect_device_type())
model, tok, meta = load_model("base", device, phase="eval", model_tag="toy-qwen3")
prompt = "زبان فارسی"
ids = [tok.get_bos_token_id()] + tok.encode(prompt)
with torch.no_grad():
    for _ in range(16):
        logits = model(torch.tensor([ids], device=device))
        ids.append(int(logits[0, -1].argmax()))
torch_text = tok.decode(ids[1 + len(tok.encode(prompt)):])
print("torch greedy   :", repr(torch_text))
for q in ("f16", "Q4_K_M"):
    # llama-simple omits BOS (torch path prepends it) — a divergent greedy chain there is methodology, not export error.
    # Use llama-cli, which honours add_bos_token=true, for an apples-to-apples comparison.
    r = subprocess.run([f"{lc}/llama-cli", "-m", f"/data/export/toy-qwen3-{q}.gguf", "-st", "--simple-io", "--no-display-prompt", "-n", "16", "--temp", "0", "-p", prompt], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    out = (r.stdout or "").strip().splitlines(); gen = out[-1] if out else r.stderr[-300:]
    print(f"llama.cpp {q:6s}:", repr(gen[len(prompt):] if gen.startswith(prompt) else gen))
PY
# 7) llama-cli generation (the ollama-style path): 64 greedy tokens off the Q4_K_M — eyeball for coherent Persian
#    (a 300-step d12 toy won't be fluent; the gate is: loads as arch qwen3, generates Persian script, no garbage bytes / immediate EOS)
$LC/llama-cli -m /data/export/toy-qwen3-Q4_K_M.gguf -p "زبان فارسی" -n 64 --temp 0 -st --no-display-prompt 2>/data/logs/llama-cli-toy-qwen3.err | tail -n 5
OUT=$($LC/llama-cli -m /data/export/toy-qwen3-Q4_K_M.gguf -p "زبان فارسی" -n 64 --temp 0 -st --no-display-prompt 2>/dev/null)
[ -n "$OUT" ] && echo "LLAMA-CLI GENERATION: non-empty PASS (coherence: eyeball above)" || echo "LLAMA-CLI GENERATION: FAIL (empty output)"
# 8) upload artifacts
gcloud --no-user-output-enabled storage cp /data/export/toy-qwen3-f16.gguf /data/export/toy-qwen3-Q4_K_M.gguf $LOG $B/toy_gguf/ && echo "uploaded to $B/toy_gguf/"
echo "== $(date -u +%FT%TZ) toy qwen3 loop done"
