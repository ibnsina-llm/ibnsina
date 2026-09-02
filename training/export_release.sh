#!/bin/bash
# Release export for a Llama-arch run: GGUF F16 -> Q8_0 + Q4_K_M, ollama Modelfile, model card with ibnsina-llm links -> gs://.../release/<NAME>/
# usage: export_release.sh RUN NAME [SOURCE=sft]      (on the GPU box; needs /data/llama.cpp built — builds it if missing)
set -u; RUN=$1; NAME=$2; SRC=${3:-sft}; export HOME=/root PATH=/root/.local/bin:$PATH NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/data/nc}
B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}; T=/data/pipeline/training; NC=/data/nanochat; E=/data/release/$NAME; mkdir -p $E; ORG=ibnsina-llm
if [ ! -x /data/llama.cpp/build/bin/llama-quantize ]; then
  command -v cmake >/dev/null || { apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake >/dev/null 2>&1; }
  [ -d /data/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /data/llama.cpp
  (cd /data/llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF && cmake --build build -j "$(nproc)" --target llama-quantize llama-cli llama-tokenize llama-simple) > /data/logs/llamacpp_build.log 2>&1
fi
LC=/data/llama.cpp/build/bin
cd $NC && uv run --no-sync python $T/export_gguf.py --base-dir $NANOCHAT_BASE_DIR --source $SRC --model-tag $RUN --out $E/$NAME-f16.gguf --name "$ORG/$NAME" || exit 1
$LC/llama-quantize $E/$NAME-f16.gguf $E/$NAME-Q8_0.gguf Q8_0 >/dev/null 2>&1 && $LC/llama-quantize $E/$NAME-f16.gguf $E/$NAME-Q4_K_M.gguf Q4_K_M >/dev/null 2>&1; ls -la $E/*.gguf
cat > $E/Modelfile <<MF
FROM ./$NAME-Q4_K_M.gguf
TEMPLATE """{{- range .Messages }}{{- if eq .Role "user" }}<|user_start|>{{ .Content }}<|user_end|>{{- else if eq .Role "assistant" }}<|assistant_start|>{{ .Content }}<|assistant_end|>{{- end }}{{- end }}<|assistant_start|>"""
PARAMETER stop "<|assistant_end|>"
PARAMETER stop "<|user_start|>"
PARAMETER stop "<|python_start|>"
PARAMETER temperature 0.6
MF
EVAL=$(ls /data/eval/results_$RUN.json 2>/dev/null); EVALTAB=""
[ -n "$EVAL" ] && EVALTAB=$(python3 -c "
import json; r=json.load(open('$EVAL')); print('| task | acc | random |'); print('|---|---:|---:|')
[print(f\"| {k} | {100*v['acc']:.1f} % | {100*v['random']:.0f} % |\") for k,v in r['tasks'].items()]")
gcloud --no-user-output-enabled storage rsync -r $B/code/pipeline/release /data/pipeline/release 2>/dev/null
CARD=/data/pipeline/release/MODEL_CARD_$NAME.md; [ -f "$CARD" ] || CARD=/data/pipeline/release/MODEL_CARD_ibnsina-1.5b.md
STEPS=$(ls $NANOCHAT_BASE_DIR/base_checkpoints/$RUN | grep -oE 'model_[0-9]{6}' | sort | tail -n 1 | grep -oE '[0-9]+' | sed 's/^0*//')
TOK=$(python3 -c "print(f'{${STEPS:-0}*524288/1e9:.1f} B')")
python3 - "$CARD" "$E/README.md" "${EVALTAB:-_(pending)_}" "$STEPS" "$TOK" <<'PY'
import sys; src, dst, ev, steps, tok = sys.argv[1:6]
open(dst, "w").write(open(src).read().replace("{{EVAL_TABLE}}", ev).replace("{{TRAIN_STEPS}}", f"{int(steps):,}" if steps else "?").replace("{{TRAIN_TOKENS}}", tok))
PY
cp $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl $NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt $E/ 2>/dev/null; [ -n "$EVAL" ] && cp $EVAL $E/
gcloud --no-user-output-enabled storage rsync -r $E $B/release/$NAME && echo "== release exported to $B/release/$NAME"
