#!/bin/bash
# Launch (or RESUME) a CPT run on this box — mirrors training/train_run.sh conventions, nohup instead of tmux.
#   launch_cpt.sh RUN [train_cpt.py args...]  first call: records args in $CPT_HOME/runs/RUN.args AND the FULL
#                                             guard+config env in RUN.env (the 1.5B lesson: resumes MUST inherit
#                                             the guard suite — the env file is sourced on every relaunch)
#   launch_cpt.sh RUN                         later calls (kill / preemption / reboot): recorded env+args, --resume
#   launch_cpt.sh sync  RUN                   one-shot GCS mirror (this is also the trainer's post-save hook)
#   launch_cpt.sh watch RUN                   sidecar loop: mirror on a new COMPLETE marker or every 30 min
# Rules baked in: preempted runs resume (never restart); a QUARANTINED run refuses to launch until the owner
# reviews (rm ckpt/RUN/QUARANTINED or FORCE=1); no literal project/bucket anywhere — env vars only (scrub rule).
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}
CPT_HOME=${CPT_HOME:-/data/cpt}
RUNS=$CPT_HOME/runs; LOGS=$CPT_HOME/logs; mkdir -p "$RUNS" "$LOGS"

do_sync(){ # two passes: everything EXCEPT COMPLETE markers, then the markers — a marker in GCS implies a full dir
  local RUN=$1 SRC=$CPT_HOME/ckpt/$1 DST=$B/cpt_ckpt/$1
  [ -d "$SRC" ] || return 0
  gcloud --no-user-output-enabled storage rsync -r -x '.*COMPLETE$' "$SRC" "$DST" 2>/dev/null || { echo "$(date -u +%FT%TZ) sync pass1 failed"; return 1; }
  gcloud --no-user-output-enabled storage rsync -r "$SRC" "$DST" 2>/dev/null || { echo "$(date -u +%FT%TZ) sync pass2 failed"; return 1; }
  [ -f "$LOGS/train-$RUN.log" ] && gcloud --no-user-output-enabled storage cp "$LOGS/train-$RUN.log" "$DST/train.log" 2>/dev/null
  [ -f "$LOGS/metrics-$RUN.jsonl" ] && gcloud --no-user-output-enabled storage cp "$LOGS/metrics-$RUN.jsonl" "$DST/metrics.jsonl" 2>/dev/null
  [ -f "$LOGS/router-$RUN.jsonl" ] && gcloud --no-user-output-enabled storage cp "$LOGS/router-$RUN.jsonl" "$DST/router.jsonl" 2>/dev/null
  [ -d "$CPT_HOME/quarantine/$RUN" ] && gcloud --no-user-output-enabled storage rsync -r "$CPT_HOME/quarantine/$RUN" "$B/cpt_ckpt/quarantine/$RUN" 2>/dev/null
  echo "$(date -u +%FT%TZ) synced $RUN"
}

case ${1:-} in
  sync) do_sync "${2:?RUN}"; exit $? ;;
  watch)
    RUN=${2:?RUN}; SRC=$CPT_HOME/ckpt/$RUN; last=""; t=$(date +%s)
    while :; do
      newest=$(ls "$SRC" 2>/dev/null | grep -E '^step_[0-9]{8}$' | sort | tail -n 1)
      if { [ -n "$newest" ] && [ -f "$SRC/$newest/COMPLETE" ] && [ "$newest" != "$last" ]; } || [ $(( $(date +%s) - t )) -ge 1800 ]; then
        do_sync "$RUN"; last=$newest; t=$(date +%s)
      fi
      sleep 60
    done ;;
  "") echo "usage: launch_cpt.sh RUN [args...] | sync RUN | watch RUN"; exit 2 ;;
esac

RUN=$1; shift
ENVF=$RUNS/$RUN.env; ARGS=$RUNS/$RUN.args; LOG=$LOGS/train-$RUN.log; CK=$CPT_HOME/ckpt/$RUN
[ -f "$CK/QUARANTINED" ] && [ "${FORCE:-0}" != 1 ] && { echo "refusing: $RUN is QUARANTINED ($(cat "$CK/QUARANTINED")) — owner review, then rm $CK/QUARANTINED or FORCE=1"; exit 4; }

if [ $# -gt 0 ]; then    # first call: freeze args + the FULL env suite (guards included, always — no partial envs)
  printf '%s\n' "$@" > "$ARGS"
  cat > "$ENVF" <<EOF
CPT_MODEL_ID=${CPT_MODEL_ID:-Qwen/Qwen3-30B-A3B-Base}
CPT_DATA_DIR=${CPT_DATA_DIR:-$CPT_HOME/data/train_v1_1_qwen3}
CPT_SEQ_LEN=${CPT_SEQ_LEN:-4096}
CPT_MICRO_BSZ=${CPT_MICRO_BSZ:-2}
CPT_GRAD_ACCUM=${CPT_GRAD_ACCUM:-8}
CPT_LR=${CPT_LR:-2e-5}
CPT_MIN_LR_FRAC=${CPT_MIN_LR_FRAC:-0.10}
CPT_WARMUP=${CPT_WARMUP:-500}
CPT_TOTAL_TOKENS=${CPT_TOTAL_TOKENS:-15000000000}
CPT_WEIGHT_DECAY=${CPT_WEIGHT_DECAY:-0.1}
CPT_GRAD_CLIP=${CPT_GRAD_CLIP:-1.0}
CPT_SAVE_EVERY=${CPT_SAVE_EVERY:-100}
CPT_KEEP_LAST=${CPT_KEEP_LAST:-3}
CPT_SPIKE_GUARD=${CPT_SPIKE_GUARD:-1.0}
CPT_GRAD_GUARD=${CPT_GRAD_GUARD:-4.0}
CPT_GUARD_VALVE=${CPT_GUARD_VALVE:-100}
CPT_GUARD_WARMUP=${CPT_GUARD_WARMUP:-5}
CPT_QUARANTINE_AFTER=${CPT_QUARANTINE_AFTER:-8}
CPT_RANK_DEV=${CPT_RANK_DEV:-2.0}
CPT_PROBE_EVERY=${CPT_PROBE_EVERY:-100}
CPT_ROUTER_ENT_MIN=${CPT_ROUTER_ENT_MIN:-0.5}
CPT_ROUTER_MAXFRAC=${CPT_ROUTER_MAXFRAC:-0.25}
CPT_ROUTER_PATIENCE=${CPT_ROUTER_PATIENCE:-3}
CPT_ROUTER_AUX=${CPT_ROUTER_AUX:-0.0}
CPT_ATTN=${CPT_ATTN:-sdpa}
CPT_SEED=${CPT_SEED:-1337}
CPT_LOG_EVERY=${CPT_LOG_EVERY:-1}
EOF
fi
[ -f "$ENVF" ] || { echo "no recorded env for $RUN (first call needs args)"; exit 2; }
[ -f "$ARGS" ] || { echo "no recorded args for $RUN"; exit 2; }
set -a; . "$ENVF"; set +a
mapfile -t A < "$ARGS"
export CPT_HOME CPT_SYNC_SCRIPT=${CPT_SYNC_SCRIPT:-$DIR/launch_cpt.sh}

# fresh boot after preemption: pull the GCS mirror when the local checkpoint dir is empty
mkdir -p "$CK"
if [ -z "$(ls "$CK" 2>/dev/null)" ] && command -v gcloud >/dev/null; then
  gcloud --no-user-output-enabled storage rsync -r "$B/cpt_ckpt/$RUN" "$CK" 2>/dev/null || true
fi

pkill -f "launch_cpt.sh watch $RUN" 2>/dev/null; pkill -f "train_cpt.py --run $RUN" 2>/dev/null; sleep 2
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '); [ "$NGPU" -ge 1 ] || NGPU=1
echo "$RUN" > "$CPT_HOME/active_run"
echo "$(date -u +%FT%TZ) [launch_cpt] run=$RUN ngpu=$NGPU args: ${A[*]:-}" | tee -a "$LOG"
command -v gcloud >/dev/null && nohup bash "$0" watch "$RUN" >> "$LOGS/sync-$RUN.log" 2>&1 &
nohup bash -c "torchrun --standalone --nproc_per_node=$NGPU '$DIR/train_cpt.py' --run '$RUN' --resume ${A[*]:-} >> '$LOG' 2>&1; echo \"EXIT=\$? \$(date -u +%FT%TZ)\" >> '$LOG'; rm -f '$CPT_HOME/active_run'; bash '$0' sync '$RUN' >> '$LOGS/sync-$RUN.log' 2>&1" >/dev/null 2>&1 &
echo "launched $RUN (log $LOG)"
