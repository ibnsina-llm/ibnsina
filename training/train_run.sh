#!/bin/bash
# Launch (or RESUME) a nanochat base_train run on this GPU box, under tmux, with the GCS checkpoint sync loop.
#   train_run.sh RUN --depth=6 --num-iterations=300 ...   first call: records the args in /data/runs/RUN.args
#   train_run.sh RUN                                       later calls (after kill / preemption / reboot): reuses the recorded args and
#                                                          resumes from the latest checkpoint found locally or in gs://.../checkpoints/RUN
# Rules baked in: preempted runs resume (never restart); --save-every is the caller's job (<= 1000 steps); wandb off (loss log goes to GCS instead).
set -u
RUN=$1; shift
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/data/nc}; export PATH=/root/.local/bin:$PATH
mkdir -p /data/runs /data/logs; ARGS=/data/runs/$RUN.args; LOG=/data/logs/train-$RUN.log
[ $# -gt 0 ] && { printf '%s\n' "$@" > "$ARGS"; echo "NANOCHAT_ARCH=${NANOCHAT_ARCH:-gpt}" > /data/runs/$RUN.env; }
[ -f /data/runs/$RUN.env ] && { set -a; source /data/runs/$RUN.env; set +a; }; export NANOCHAT_ARCH=${NANOCHAT_ARCH:-gpt}
[ -f "$ARGS" ] || { echo "no recorded args for $RUN"; exit 2; }
mapfile -t A < "$ARGS"
echo "$RUN" > /data/active_run
step=$(bash /data/pipeline/training/ckpt_sync.sh pull "$RUN" | tail -n 1)
# one-shot override: /data/runs/RUN.resume_once holding a step number forces the next resume from that checkpoint (then is deleted)
if [ -f /data/runs/$RUN.resume_once ]; then step=$(cat /data/runs/$RUN.resume_once); rm -f /data/runs/$RUN.resume_once; echo "$(date -u +%FT%TZ) [train_run] resume_once -> step $step" | tee -a "$LOG"; fi
RES=""; [ "$step" -ge 0 ] && RES="--resume-from-step=$step"
NGPU=$(nvidia-smi -L | wc -l)
echo "$(date -u +%FT%TZ) [train_run] run=$RUN ngpu=$NGPU resume_step=$step args: ${A[*]}" | tee -a "$LOG"
tmux kill-session -t "sync-$RUN" 2>/dev/null; tmux new-session -d -s "sync-$RUN" "bash /data/pipeline/training/ckpt_sync.sh watch $RUN $LOG >> /data/logs/sync-$RUN.log 2>&1"
tmux kill-session -t "train-$RUN" 2>/dev/null
tmux new-session -d -s "train-$RUN" "cd /data/nanochat && NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR NANOCHAT_ARCH=$NANOCHAT_ARCH PATH=$PATH uv run --no-sync torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- --run=dummy --model-tag=$RUN --core-metric-every=-1 $RES ${A[*]} >> $LOG 2>&1; echo \"EXIT=\$? \$(date -u +%FT%TZ)\" >> $LOG; rm -f /data/active_run; bash /data/pipeline/training/ckpt_sync.sh sync $RUN $LOG >> /data/logs/sync-$RUN.log 2>&1"
echo "launched tmux train-$RUN (log $LOG)"
