#!/bin/bash
# Distribution-loop validation on a small GPU box (L4 spot) — the gate before the 1.5B:
#   tokenizer v2 (Llama-3 regex) -> Llama-arch d4 toy trained with nanochat's loop -> GGUF (F16) -> llama.cpp (CPU build) -> Q4_K_M -> run
#   + tokenizer parity (llama-tokenize vs tiktoken) + greedy-decode parity (llama.cpp vs torch) + d12 GPT-vs-Llama A/B for the efficiency cost.
# Prereq: gpu_startup.sh ran (code in /data/pipeline, nanochat env, data-shards=4). Run as root: bash toy_gguf_loop.sh
set -u; export HOME=/root PATH=/root/.local/bin:$PATH; B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}; T=/data/pipeline/training; NC=/data/nanochat
export NANOCHAT_BASE_DIR=/data/nc2; LOG=/data/logs/toy_gguf_loop.log; mkdir -p /data/logs /data/export $NANOCHAT_BASE_DIR/tokenizer; exec > >(tee -a $LOG) 2>&1
echo "== $(date -u +%FT%TZ) toy loop start"
# 1) tokenizer v2 + data (shared shards from the v1 base dir)
gcloud --no-user-output-enabled storage cp $B/tokenizer/v2_32k_llama/tokenizer.pkl $B/tokenizer/v2_32k_llama/token_bytes.pt $NANOCHAT_BASE_DIR/tokenizer/
[ -e $NANOCHAT_BASE_DIR/base_data_climbmix ] || ln -s /data/nc/base_data_climbmix $NANOCHAT_BASE_DIR/base_data_climbmix
# 2) patches (Llama arch + arch switch + Persian tasks)
bash $T/nanochat_patches/apply_patches.sh $NC
cd $NC && uv pip install --quiet gguf sentencepiece 2>/dev/null || uv pip install gguf
COMMON="--run=dummy --core-metric-every=-1 --sample-every=-1 --window-pattern=L"
# 3) d4 toy on the Llama arch (the artifact that goes through the whole loop)
NANOCHAT_ARCH=llama uv run --no-sync torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- $COMMON --model-tag=toy-llama --depth=4 --max-seq-len=512 \
  --device-batch-size=16 --total-batch-size=32768 --num-iterations=300 --eval-every=100 --eval-tokens=262144 --warmup-steps=10 > /data/logs/train-toy-llama.log 2>&1
grep -aE "^step 00299|Validation" /data/logs/train-toy-llama.log | tail -n 3 | cut -c1-120
# 4) export GGUF F16 -> llama.cpp CPU build -> Q4_K_M
uv run --no-sync python $T/export_gguf.py --base-dir $NANOCHAT_BASE_DIR --source base --model-tag toy-llama --out /data/export/toy-llama-f16.gguf --name persian-toy-llama
if [ ! -x /data/llama.cpp/build/bin/llama-quantize ]; then
  command -v cmake >/dev/null || { apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake >/dev/null 2>&1; }
  [ -d /data/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /data/llama.cpp
  (cd /data/llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF && cmake --build build -j $(nproc) --target llama-quantize llama-cli llama-tokenize llama-simple) > /data/logs/llamacpp_build.log 2>&1 || { echo "!! llama.cpp build failed — see /data/logs/llamacpp_build.log"; tail -n 5 /data/logs/llamacpp_build.log; }
fi
LC=/data/llama.cpp/build/bin; ls $LC | tr "\n" " "; echo
$LC/llama-quantize /data/export/toy-llama-f16.gguf /data/export/toy-llama-Q4_K_M.gguf Q4_K_M 2>&1 | tail -n 2
ls -la /data/export/*.gguf
# 5) tokenizer parity: llama.cpp ids vs tiktoken ids on Persian / English / code / digits / newlines
uv run --no-sync python - <<'PY'
import pickle, subprocess, json, sys
enc = pickle.load(open("/data/nc2/tokenizer/tokenizer.pkl", "rb"))
tests = ["زبان فارسی یکی از کهن‌ترین زبان‌های زنده جهان است.", "Persian is one of the oldest living languages in the world.", "def area(r):\n    return 3.14159 * r ** 2\n", "در سال 1403 حدود 12345678 نفر", "خط اول\n\n\nخط دوم  ", "Hello world! 😀 سلام"]
ok = True
for s in tests:
    ref = enc.encode_ordinary(s)
    out = subprocess.run(["/data/llama.cpp/build/bin/llama-tokenize", "-m", "/data/export/toy-llama-f16.gguf", "-p", s, "--ids", "--no-bos", "--log-disable"], capture_output=True, text=True).stdout
    got = json.loads(out.strip().splitlines()[-1]) if out.strip() else None
    same = got == ref; ok &= same
    print(("OK  " if same else "DIFF") + f" {s[:40]!r}: tiktoken {ref[:12]} llama.cpp {got[:12] if got else out[-200:]}")
print("TOKENIZER PARITY:", "PASS" if ok else "FAIL")
PY
# 6) greedy-decode parity: torch (bf16) vs llama.cpp F16 and Q4_K_M
uv run --no-sync python - <<'PY'
import subprocess, torch, sys
sys.argv = ["x"]
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
ddp, rank, lrank, world, device = compute_init(autodetect_device_type())
model, tok, meta = load_model("base", device, phase="eval", model_tag="toy-llama")
prompt = "زبان فارسی"
ids = [tok.get_bos_token_id()] + tok.encode(prompt)
with torch.no_grad():
    for _ in range(16):
        logits = model(torch.tensor([ids], device=device))
        ids.append(int(logits[0, -1].argmax()))
torch_text = tok.decode(ids[1 + len(tok.encode(prompt)):])
print("torch greedy   :", repr(torch_text))
for q in ("f16", "Q4_K_M"):
    r = subprocess.run(["/data/llama.cpp/build/bin/llama-simple", "-m", f"/data/export/toy-llama-{q}.gguf", "-n", "16", prompt], capture_output=True, text=True)
    out = (r.stdout or "").strip().splitlines(); gen = out[-1] if out else r.stderr[-300:]
    print(f"llama.cpp {q:6s}:", repr(gen[len(prompt):] if gen.startswith(prompt) else gen))
PY
# 7) efficiency A/B: d12, 300 steps x 131k tokens, GPT arch vs Llama arch, same tokenizer v2 / data / batch (both full attention)
for arch in gpt llama; do
  NANOCHAT_ARCH=$arch uv run --no-sync torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- $COMMON --model-tag=ab-$arch --depth=12 --max-seq-len=1024 \
    --device-batch-size=16 --total-batch-size=131072 --num-iterations=300 --eval-every=100 --eval-tokens=1048576 --warmup-steps=20 --save-every=-1 > /data/logs/train-ab-$arch.log 2>&1
  echo "A/B $arch: $(grep -a 'Validation bpb' /data/logs/train-ab-$arch.log | tr '\n' ' ' | cut -c1-200)"
  echo "A/B $arch: $(grep -a '^step 00299' /data/logs/train-ab-$arch.log | cut -c1-120)"
  grep -aE "Number of parameters|num params|params:" /data/logs/train-ab-$arch.log | head -n 2 | cut -c1-120
done
# 8) upload artifacts
gcloud --no-user-output-enabled storage cp /data/export/toy-llama-f16.gguf /data/export/toy-llama-Q4_K_M.gguf $LOG $B/toy_gguf/ && echo "uploaded to $B/toy_gguf/"
echo "== $(date -u +%FT%TZ) toy loop done"
