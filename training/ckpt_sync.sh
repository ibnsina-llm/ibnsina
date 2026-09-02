#!/bin/bash
# Checkpoint <-> GCS sync for nanochat runs. Runs on the GPU box alongside training.
#   ckpt_sync.sh pull  RUN   -> fetch the latest checkpoint set for RUN from GCS into $NANOCHAT_BASE_DIR/checkpoints/RUN ; prints the step or -1
#   ckpt_sync.sh watch RUN   -> loop: whenever a new model_XXXXXX.pt appears (or every 30 min), rsync the run dir + training log to GCS
# GCS layout: gs://B/checkpoints/<RUN>/{model_,optim_,meta_}*  and gs://B/checkpoints/<RUN>/train.log (+ loss.csv for the phone view)
set -u
B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}; BASE=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}
MODE=$1; RUN=$2; DIR=$BASE/base_checkpoints/$RUN; LOG=${3:-/data/logs/train-$RUN.log}
case $MODE in
  pull)
    mkdir -p "$DIR"; gcloud --no-user-output-enabled storage rsync -r "$B/checkpoints/$RUN" "$DIR" 2>/dev/null || true
    step=$(ls "$DIR" 2>/dev/null | grep -oE 'model_[0-9]{6}' | sort | tail -n 1 | grep -oE '[0-9]+' | sed 's/^0*//'); echo "${step:--1}" ;;
  watch)
    last=""; t=$(date +%s)
    while true; do
      newest=$(ls "$DIR" 2>/dev/null | grep -E '^meta_[0-9]{6}\.json$' | sort | tail -n 1)
      if { [ -n "$newest" ] && [ "$newest" != "$last" ]; } || [ $(( $(date +%s) - t )) -ge 1800 ]; then
        # loss.csv: step,loss extracted from the training log (nanochat prints "step 00042/... | loss: 3.1234 | ...")
        [ -f "$LOG" ] && grep -aoE 'step [0-9]+/[0-9]+ .*loss: [0-9.]+' "$LOG" | sed -E 's/step ([0-9]+)\/[0-9]+ .*loss: ([0-9.]+)/\1,\2/' > "$DIR/loss.csv" 2>/dev/null
        [ -f "$LOG" ] && cp "$LOG" "$DIR/train.log"
        gcloud --no-user-output-enabled storage rsync -r "$DIR" "$B/checkpoints/$RUN" 2>/dev/null && echo "$(date -u +%FT%TZ) synced $RUN ($newest)" || echo "$(date -u +%FT%TZ) sync failed"
        # prune local checkpoints beyond the newest KEEP_LAST (default 3) once they are in GCS (GCS keeps everything)
        for old in $(ls "$DIR" | grep -oE 'model_[0-9]{6}' | sort | head -n -${KEEP_LAST:-3} | grep -oE '[0-9]{6}'); do
          gcloud --no-user-output-enabled storage ls "$B/checkpoints/$RUN/model_$old.pt" >/dev/null 2>&1 && rm -f "$DIR"/model_$old.pt "$DIR"/optim_${old}_rank*.pt "$DIR"/meta_$old.json && echo "$(date -u +%FT%TZ) pruned local step $old"
        done
        last=$newest; t=$(date +%s)
      fi
      sleep 60
    done ;;
  sync)
    [ -f "$LOG" ] && cp "$LOG" "$DIR/train.log" 2>/dev/null; gcloud --no-user-output-enabled storage rsync -r "$DIR" "$B/checkpoints/$RUN" && echo "$(date -u +%FT%TZ) final sync $RUN" ;;
  *) echo "usage: ckpt_sync.sh pull|watch|sync RUN [LOG]"; exit 2;;
esac
