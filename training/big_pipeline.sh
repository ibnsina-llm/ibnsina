#!/bin/bash
# IbnSina 1.5B orchestration ON the 8-GPU box (Llama arch). Idempotent + reboot-safe (gpu_startup.sh re-launches it while /data/big exists).
# Stages: patches -> fp8 A/B (200 steps each; decides --fp8) -> base (train_run.sh big, resumable) -> SFT (v2 if present, else v1) -> eval -> release export -> stop self.
set -u; export NANOCHAT_BASE_DIR=/data/nc NANOCHAT_ARCH=llama PATH=/root/.local/bin:$PATH HOME=/root
RUN=${RUN:-big}; NAME=${NAME:-ibnsina-1.5b}; B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}; NC=/data/nanochat; T=/data/pipeline/training
ITERS=${ITERS:-88000}; SHAPE="--depth=28 --aspect-ratio=73 --head-dim=128 --n-kv-head=4 --ffn-hidden=6144 --window-pattern=L --max-seq-len=2048"
BASE_ARGS="$SHAPE --device-batch-size=8 --total-batch-size=524288 --num-iterations=$ITERS --warmup-steps=500 --warmdown-ratio=0.4 --save-every=1000 --eval-every=2000 --eval-tokens=41943040 --sample-every=-1 --core-metric-every=-1"
exec >> /data/logs/big_pipeline.log 2>&1; echo "== $(date -u +%FT%TZ) big pipeline start run=$RUN"
[ -f /data/big_done ] && { echo "already done"; exit 0; }
mkdir -p /data/logs /data/eval /data/runs; bash $T/nanochat_patches/apply_patches.sh $NC >/dev/null
# tokenizer v2 for the Llama line
[ -f $NANOCHAT_BASE_DIR/tokenizer/.v2 ] || { gcloud --no-user-output-enabled storage cp $B/tokenizer/v2_32k_llama/tokenizer.pkl $B/tokenizer/v2_32k_llama/token_bytes.pt $NANOCHAT_BASE_DIR/tokenizer/ && touch $NANOCHAT_BASE_DIR/tokenizer/.v2; }
# ---- stage 0: fp8 A/B (200 steps each, same shape) — decided once, recorded in /data/runs/fp8.decision ----
# fp8 trial is OFF by default since the 1.5B diverged under fp8 after warm-up (2026-08-30); set FP8_TRIAL=1 to run the A/B (make it >=1500 steps past warm-up)
if [ "${FP8_TRIAL:-0}" = 1 ] && [ ! -f /data/runs/fp8.decision ] && [ ! -f /data/runs/$RUN.args ]; then
  for mode in off on; do
    extra=""; [ $mode = on ] && extra="--fp8"
    cd $NC && uv run --no-sync torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m scripts.base_train -- --run=dummy --model-tag=ab-fp8-$mode $SHAPE $extra \
      --device-batch-size=8 --total-batch-size=524288 --num-iterations=200 --warmup-steps=20 --eval-every=200 --eval-tokens=8388608 --sample-every=-1 --core-metric-every=-1 --save-every=-1 > /data/logs/train-ab-fp8-$mode.log 2>&1
    echo "fp8 $mode: $(grep -a '^step 00199' /data/logs/train-ab-fp8-$mode.log | cut -c1-120) | $(grep -a 'Validation bpb' /data/logs/train-ab-fp8-$mode.log | tail -n 1)"
  done
  python3 - <<'PY' > /data/runs/fp8.decision
import re
def parse(m):
    t = open(f"/data/logs/train-ab-fp8-{m}.log", errors="ignore").read()
    tps = [float(x.replace(",", "")) for x in re.findall(r"tok/sec: ([\d,]+)", t)[-50:]]
    bpb = re.findall(r"Validation bpb: ([\d.]+)", t); nan = "nan" in t.lower().split("loss:")[-1][:20] if "loss:" in t else False
    return (sum(tps) / len(tps) if tps else 0), (float(bpb[-1]) if bpb else 9), nan
off, on = parse("off"), parse("on")
ok = on[0] > off[0] * 1.08 and on[1] <= off[1] * 1.01 and not on[2]
print("--fp8" if ok else "")
print(f"# off tok/s {off[0]:.0f} bpb {off[1]:.4f} | on tok/s {on[0]:.0f} bpb {on[1]:.4f} nan={on[2]} -> {'USE fp8' if ok else 'no fp8'}")
PY
  cat /data/runs/fp8.decision; rm -rf $NANOCHAT_BASE_DIR/base_checkpoints/ab-fp8-*
fi
FP8=$(head -n 1 /data/runs/fp8.decision 2>/dev/null)
FINAL=$NANOCHAT_BASE_DIR/base_checkpoints/$RUN/model_$(printf %06d $ITERS).pt; tries=0
# ---- stage 1: base pretraining (resumes across preemptions / 24h STOP) ----
while [ ! -f "$FINAL" ]; do
  if ! tmux has-session -t "train-$RUN" 2>/dev/null; then
    tries=$((tries+1)); [ $tries -gt 6 ] && { echo "!! train-$RUN died $tries times — giving up, tell the curator"; exit 1; }
    echo "$(date -u +%FT%TZ) (re)launching base training (try $tries) fp8='$FP8'"
    if [ -f /data/runs/$RUN.args ]; then bash $T/train_run.sh $RUN; else NANOCHAT_ARCH=llama bash $T/train_run.sh $RUN $BASE_ARGS $FP8; fi; sleep 240
  fi
  sleep 120
done
echo "== $(date -u +%FT%TZ) base training done"; bash $T/ckpt_sync.sh sync $RUN /data/logs/train-$RUN.log
# ---- stage 2: SFT (v2 when the assembled set exists in GCS, else v1) ----
if ! ls $NANOCHAT_BASE_DIR/chatsft_checkpoints/$RUN/model_*.pt >/dev/null 2>&1; then
  SFT=v1; gcloud storage ls $B/sft/v2/train.jsonl >/dev/null 2>&1 && SFT=v2
  mkdir -p /data/sft && gcloud --no-user-output-enabled storage rsync -r $B/sft/$SFT /data/sft/$SFT && echo "SFT set: $SFT ($(wc -l < /data/sft/$SFT/train.jsonl) rows)"
  cd $NC && NANOCHAT_ARCH=llama uv run --no-sync torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m scripts.chat_sft_fa -- --run=dummy --model-tag=$RUN --chatcore-every=-1 --sft-dir=/data/sft/$SFT --device-batch-size=8 > /data/logs/sft-$RUN.log 2>&1
  echo "sft exit=$?"; ls $NANOCHAT_BASE_DIR/chatsft_checkpoints/$RUN/ || { echo "!! no SFT checkpoint"; exit 1; }
fi
gcloud --no-user-output-enabled storage rsync -r $NANOCHAT_BASE_DIR/chatsft_checkpoints/$RUN $B/checkpoints/${RUN}_sft; gcloud --no-user-output-enabled storage cp /data/logs/sft-$RUN.log $B/checkpoints/${RUN}_sft/
# ---- stage 3: evals ----
bash $T/fetch_eval.sh >/dev/null; cd $NC && NANOCHAT_ARCH=llama uv run --no-sync python -m scripts.eval_fa -i sft -g $RUN -o /data/eval/results_$RUN.json > /data/logs/eval-$RUN.log 2>&1; tail -n 8 /data/logs/eval-$RUN.log
gcloud --no-user-output-enabled storage cp /data/eval/results_$RUN.json /data/logs/eval-$RUN.log $B/release/$NAME/eval/
# ---- stage 4: release export (GGUF F16/Q8_0/Q4_K_M + Modelfile + model card) ----
bash $T/export_release.sh $RUN $NAME sft
touch /data/big_done; echo "== $(date -u +%FT%TZ) BIG PIPELINE DONE"
Z=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}'); N=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/name)
sleep 60; gcloud compute instances stop "$N" --zone "$Z" --discard-local-ssd=true --quiet && echo "stopping self ($N)"
