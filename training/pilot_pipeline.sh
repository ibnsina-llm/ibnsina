#!/bin/bash
# T2 pilot orchestration ON the 8-GPU box. Idempotent + reboot-safe: gpu_startup.sh re-launches it whenever /data/pilot exists.
# Stages: base (train_run.sh, resumable) -> sft (chat_sft_fa) -> eval (eval_fa) -> export (gs://.../pilot/<run>/) -> stop this instance.
set -u; export NANOCHAT_BASE_DIR=/data/nc PATH=/root/.local/bin:$PATH HOME=/root
RUN=${RUN:-pilot}; B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}; NC=/data/nanochat; T=/data/pipeline/training
ITERS=${ITERS:-15000}; DEPTH=${DEPTH:-18}
BASE_ARGS="--depth=$DEPTH --max-seq-len=2048 --device-batch-size=32 --total-batch-size=524288 --num-iterations=$ITERS --save-every=1000 --eval-every=1000 --eval-tokens=20971520 --sample-every=-1 --warmup-steps=100"
exec >> /data/logs/pilot_pipeline.log 2>&1
echo "== $(date -u +%FT%TZ) pipeline start run=$RUN"
[ -f /data/pilot_done ] && { echo "already done"; exit 0; }
mkdir -p /data/logs /data/eval /data/export; cp $T/nanochat_patches/tasks/*.py $NC/tasks/ && cp $T/nanochat_patches/scripts/*.py $NC/scripts/
FINAL=$NANOCHAT_BASE_DIR/base_checkpoints/$RUN/model_$(printf %06d $ITERS).pt; tries=0
# ---- stage 1: base pretraining (resumes across preemptions via train_run.sh; never restarts) ----
while [ ! -f "$FINAL" ]; do
  if ! tmux has-session -t "train-$RUN" 2>/dev/null; then
    tries=$((tries+1)); [ $tries -gt 4 ] && { echo "!! train-$RUN died $tries times — giving up, tell the curator"; exit 1; }
    echo "$(date -u +%FT%TZ) (re)launching base training (try $tries)"
    if [ -f /data/runs/$RUN.args ]; then bash $T/train_run.sh $RUN; else bash $T/train_run.sh $RUN $BASE_ARGS; fi; sleep 180
  fi
  sleep 60
done
echo "== $(date -u +%FT%TZ) base training done: $FINAL"; bash $T/ckpt_sync.sh sync $RUN /data/logs/train-$RUN.log
# ---- stage 2: SFT (FarsInstruct + translated SmolTalk + MMLU-format teacher); ~minutes, rerun from scratch if interrupted ----
if ! ls $NANOCHAT_BASE_DIR/chatsft_checkpoints/$RUN/model_*.pt >/dev/null 2>&1; then
  mkdir -p /data/sft && gcloud --no-user-output-enabled storage rsync -r $B/sft/v1 /data/sft/v1 && wc -l /data/sft/v1/*.jsonl
  cd $NC && uv run --no-sync torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m scripts.chat_sft_fa -- --run=dummy --model-tag=$RUN --chatcore-every=-1 --sft-dir=/data/sft/v1 --device-batch-size=16 > /data/logs/sft-$RUN.log 2>&1
  echo "sft exit=$?"; ls $NANOCHAT_BASE_DIR/chatsft_checkpoints/$RUN/ || { echo "!! no SFT checkpoint"; exit 1; }
fi
gcloud --no-user-output-enabled storage rsync -r $NANOCHAT_BASE_DIR/chatsft_checkpoints/$RUN $B/checkpoints/${RUN}_sft; cp /data/logs/sft-$RUN.log /tmp/ && gcloud --no-user-output-enabled storage cp /data/logs/sft-$RUN.log $B/checkpoints/${RUN}_sft/
# ---- stage 3: Persian evals (ParsiNLU MC / entailment / QQP) ----
bash $T/fetch_eval.sh
cd $NC && uv run --no-sync python -m scripts.eval_fa -i sft -g $RUN -o /data/eval/results_$RUN.json > /data/logs/eval-$RUN.log 2>&1; tail -n 8 /data/logs/eval-$RUN.log
gcloud --no-user-output-enabled storage cp /data/eval/results_$RUN.json /data/logs/eval-$RUN.log $B/pilot/$RUN/eval/
# ---- stage 4: export bundle for the Mac (nanochat-native checkpoint + tokenizer; llama.cpp has no nanochat arch, so no GGUF) ----
E=/data/export/$RUN; mkdir -p $E/tokenizer $E/chatsft_checkpoints/$RUN $E/base_checkpoints/$RUN
cp $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl $NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt $E/tokenizer/
last=$(ls $NANOCHAT_BASE_DIR/chatsft_checkpoints/$RUN | grep -oE 'model_[0-9]{6}' | sort | tail -n 1 | grep -oE '[0-9]+')
cp $NANOCHAT_BASE_DIR/chatsft_checkpoints/$RUN/model_$last.pt $NANOCHAT_BASE_DIR/chatsft_checkpoints/$RUN/meta_$last.json $E/chatsft_checkpoints/$RUN/
cp $FINAL $NANOCHAT_BASE_DIR/base_checkpoints/$RUN/meta_$(printf %06d $ITERS).json $E/base_checkpoints/$RUN/
grep -aoE '^step [0-9]+/[0-9]+ .*loss: [0-9.]+' /data/logs/train-$RUN.log | sed -E 's/step ([0-9]+)\/[0-9]+ .*loss: ([0-9.]+)/\1,\2/' > $E/loss.csv
cp /data/eval/results_$RUN.json $E/ 2>/dev/null; cp $T/pilot_README.md $E/README.md; du -sh $E
gcloud --no-user-output-enabled storage rsync -r $E $B/pilot/$RUN && echo "== exported to $B/pilot/$RUN"
touch /data/pilot_done; echo "== $(date -u +%FT%TZ) PIPELINE DONE"
Z=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
NAME=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/name)
sleep 60; gcloud compute instances stop "$NAME" --zone "$Z" --discard-local-ssd=true --quiet && echo "stopping self ($NAME)"
