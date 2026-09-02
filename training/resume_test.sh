#!/bin/bash
# Kill-and-resume proof. Runs RUN via train_run.sh, SIGKILLs it (like a preemption) once the log passes step KILL_AT,
# resumes it with `train_run.sh RUN` (no args = recorded args + latest checkpoint), waits for EXIT, prints the proof.
# usage: resume_test.sh RUN KILL_AT [train args...]     -> /data/logs/resume-proof-RUN.md
set -u; RUN=$1; KILL=$2; shift 2
LOG=/data/logs/train-$RUN.log; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/data/nc}; CK=$NANOCHAT_BASE_DIR/base_checkpoints/$RUN
B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}
last_step(){ grep -oE 'step[: ]+0*[0-9]+' "$LOG" 2>/dev/null | grep -oE '[0-9]+$' | tail -n 1; }
bash /data/pipeline/training/train_run.sh "$RUN" "$@"
s=0; t=0; while [ "$s" -lt "$KILL" ]; do sleep 10; s=$(last_step); s=${s:-0}; t=$((t+10)); [ $t -ge 1800 ] && { echo "!! never reached step $KILL (last $s)"; tail -n 5 "$LOG"; exit 1; }; done
echo "$(date -u +%FT%TZ) [resume_test] KILL -9 at logged step $s (target $KILL)" | tee -a "$LOG"
tmux kill-session -t "train-$RUN" 2>/dev/null; pkill -9 -f "scripts.base_train"; sleep 5
echo "[resume_test] checkpoints on disk at kill: $(ls "$CK" 2>/dev/null | grep -E '^model_' | tr '\n' ' ')" | tee -a "$LOG"
bash /data/pipeline/training/train_run.sh "$RUN"
t=0; until grep -aq '^EXIT=' "$LOG"; do sleep 15; t=$((t+15)); [ $t -ge 3000 ] && { echo "!! resume did not finish in 50 min"; break; }; done
sleep 20   # let the final sync land
{ echo "# kill-and-resume proof — run $RUN ($(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M JST'))"; echo; echo '```'
  grep -anE '\[train_run\]|\[resume_test\]|EXIT=|[Rr]esum' "$LOG" | cut -c1-220; echo '```'; echo; echo "training-step lines around the kill (step $KILL):"; echo '```'
  grep -E 'step[: ]+0*[0-9]+' "$LOG" | awk -v k="$KILL" '{ if (match($0, /step[: ]+0*[0-9]+/)) { n=substr($0,RSTART,RLENGTH); gsub(/[^0-9]/,"",n); n=n+0; if (n>=k-12 && n<=k+12) print substr($0,1,160) } }'; echo '```'; echo
  echo "local checkpoints: $(ls "$CK" 2>/dev/null | grep -E '^model_' | tr '\n' ' ')"; echo "GCS $B/checkpoints/$RUN/: $(gcloud storage ls $B/checkpoints/$RUN/ 2>/dev/null | sed 's#.*/##' | tr '\n' ' ')"
} | tee /data/logs/resume-proof-$RUN.md
