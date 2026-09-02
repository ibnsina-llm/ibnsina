#!/bin/bash
# CPT lane gates (docs/plan/cpt_stack_plan.md §4). G1 also runs OFF-box: small proxy model, CPU or 1 GPU,
# synthetic shards — proves the loop/guard/checkpoint/resume logic without touching the 30B or the corpus.
#   gates.sh g1 [MODEL_ID]     smoke: 20 fresh steps + resume-to-30 on synthetic data (G1_FSDP=1 for the
#                              on-box 8-GPU form with the real model id; G1_SEQ overrides seq len)
#   gates.sh g2 RUN            kill -9 mid-run, resume, assert step + loss-EMA continuity
#                              (env: G2_KILL_AT=60 G2_STEPS=150 G2_SAVE=20 G2_TOL=0.05)
#   gates.sh g3 RUN [MINUTES]  1-2 h guarded run (default 90): assert zero trips/alerts/quarantine, probes
#                              present, >=2 complete checkpoints, print mean throughput vs the 18k decision
#                              threshold, check the GCS mirror (G3_ARGS holds first-launch train args)
# Budget rule: nothing beyond G3-scale spend until G3 prints PASS on every line (plan §4).
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
CPT_HOME=${CPT_HOME:-/data/cpt}; LOGS=$CPT_HOME/logs
PY=${PYTHON:-python3}
FAILED=0
pass(){ echo "PASS: $*"; }
fail(){ echo "FAIL: $*"; FAILED=1; }
last_step(){ grep -aE '^step [0-9]+/' "$1" 2>/dev/null | tail -n 1 | sed -E 's/^step ([0-9]+).*/\1/'; }
ema_at(){ grep -aE "^step $2/" "$1" 2>/dev/null | tail -n 1 | sed -E 's/.*ema ([0-9.]+).*/\1/'; }

g1(){
  M=${1:-Qwen/Qwen3-0.6B}  # TODO-VERIFY proxy id; a small MoE (e.g. Qwen/Qwen1.5-MoE-A2.7B) also exercises the router probe
  G=${G1_DIR:-$CPT_HOME/g1}; D=$G/data; SL=${G1_SEQ:-256}
  rm -rf "$G"; mkdir -p "$D" "$LOGS"
  $PY - "$D" "$SL" <<'PYEOF'
import json, os, sys
import numpy as np
d, sl = sys.argv[1], int(sys.argv[2])
a = np.random.default_rng(0).integers(1, 1000, size=(512, sl + 1), dtype=np.uint32)  # ids < any real vocab
np.save(os.path.join(d, "shard_0000_00.npy"), a)
json.dump({"tokenizer": "synthetic", "eos_id": 0, "seq_len": sl, "row_len": sl + 1, "dtype": "uint32",
           "shards": [{"file": "shard_0000_00.npy", "n_seqs": 512, "n_tokens": 512 * (sl + 1)}],
           "total_seqs": 512, "total_tokens": 512 * (sl + 1)}, open(os.path.join(d, "manifest.json"), "w"))
PYEOF
  COMMON=(--model-id "$M" --data-dir "$D" --ckpt-dir "$G/ckpt" --quarantine-dir "$G/quarantine"
          --log-dir "$LOGS" --run g1 --seq-len "$SL" --micro-bsz 1 --grad-accum 2 --warmup 5 --lr 1e-4
          --save-every 10 --probe-every 5 --log-every 1 --resume)
  if [ "${G1_FSDP:-0}" = 1 ]; then
    NG=$(nvidia-smi -L | wc -l | tr -d ' '); L="torchrun --standalone --nproc_per_node=$NG"
  else
    L="$PY"; COMMON+=(--no-fsdp)
  fi
  echo "== G1 1/2: 20 fresh steps ($M)"
  $L "$DIR/train_cpt.py" "${COMMON[@]}" --max-steps 20 > "$LOGS/g1-a.log" 2>&1; RC=$?
  [ $RC -eq 0 ] && pass "exit 0" || fail "trainer exit $RC (tail: $(tail -n 3 "$LOGS/g1-a.log" | tr '\n' ' '))"
  [ "$(grep -ac '^step ' "$LOGS/g1-a.log")" -eq 20 ] && pass "20 step lines" || fail "step lines != 20 ($(grep -ac '^step ' "$LOGS/g1-a.log"))"
  grep -aq 'CPT_DONE' "$LOGS/g1-a.log" && pass "CPT_DONE" || fail "no CPT_DONE"
  grep -aq 'CPT guards armed' "$LOGS/g1-a.log" && pass "guards armed banner" || fail "guards not armed"
  [ -f "$G/ckpt/step_00000020/COMPLETE" ] && pass "COMPLETE marker at step 20" || fail "no COMPLETE marker"
  { grep -aq 'CPT_PROBE' "$LOGS/g1-a.log" || grep -aq 'probe n/a' "$LOGS/g1-a.log"; } \
    && pass "router probe ran (or logged n/a on a dense proxy)" || fail "no probe output at all"
  echo "== G1 2/2: resume to step 30"
  $L "$DIR/train_cpt.py" "${COMMON[@]}" --max-steps 30 > "$LOGS/g1-b.log" 2>&1 || fail "resume run exit != 0"
  grep -aq 'resume: step 20' "$LOGS/g1-b.log" && pass "resumed from step 20" || fail "no 'resume: step 20' line"
  [ "$(last_step "$LOGS/g1-b.log")" = 30 ] && pass "ran 21..30" || fail "last step $(last_step "$LOGS/g1-b.log") != 30"
  # on-box (G1_FSDP=1, real model + real corpus sample via COMMON overrides): ALSO check plan §4 by hand —
  # initial loss ~ base-model ppl on held-out Persian (not random-init), loss decreasing, >=10% mem headroom/GPU
}

g2(){
  RUN=${1:?usage: gates.sh g2 RUN}; KILL_AT=${G2_KILL_AT:-60}; TOL=${G2_TOL:-0.05}
  LOG=$LOGS/train-$RUN.log; CK=$CPT_HOME/ckpt/$RUN
  bash "$DIR/launch_cpt.sh" "$RUN" --max-steps "${G2_STEPS:-150}" --save-every "${G2_SAVE:-20}" --deterministic
  t=0
  while s=$(last_step "$LOG"); [ "${s:-0}" -lt "$KILL_AT" ]; do
    sleep 15; t=$((t + 15))
    [ $t -ge 2700 ] && { fail "never reached step $KILL_AT (last ${s:-0}; tail: $(tail -n 3 "$LOG" 2>/dev/null | tr '\n' ' '))"; return 1; }
  done
  echo "$(date -u +%FT%TZ) [g2] kill -9 at logged step $s" | tee -a "$LOG"
  pkill -9 -f "train_cpt.py --run $RUN"; pkill -9 -f "torchrun.*--run $RUN"; sleep 5
  K=$(ls "$CK" | grep -E '^step_[0-9]{8}$' | while read -r d; do [ -f "$CK/$d/COMPLETE" ] && echo "${d#step_}"; done | sort | tail -n 1 | sed 's/^0*//')
  [ -n "$K" ] || { fail "no complete checkpoint at kill time"; return 1; }
  EPRE=$(ema_at "$LOG" "$K")
  echo "[g2] resuming from checkpoint step $K (pre-kill ema $EPRE)"
  bash "$DIR/launch_cpt.sh" "$RUN"
  t=0
  until grep -aq '^EXIT=' "$LOG"; do sleep 15; t=$((t + 15)); [ $t -ge 3600 ] && { fail "resume did not finish in 60 min"; break; }; done
  grep -aq "resume: step $K" "$LOG" && pass "resumed from step $K" || fail "no 'resume: step $K' line"
  B2=$(grep -ac 'CPT guards armed' "$LOG")
  [ "$B2" -ge 2 ] && pass "guard suite re-armed on resume ($B2 banners)" || fail "guards banner not repeated on resume — RUN.env not inherited?"
  FIRST=$(awk -v k="$K" '/\[g2\] kill/{seen=1} seen && /^step [0-9]+\//{sub(/^step /, ""); sub(/\/.*/, ""); print; exit}' "$LOG")
  [ "$FIRST" = "$((K + 1))" ] && pass "step continuity: first post-resume step $FIRST == K+1" || fail "first post-resume step '$FIRST' != $((K + 1))"
  CHK=$((K + 10)); EPOST=$(ema_at "$LOG" "$CHK")   # tail -1 = the post-resume pass over step K+10
  if [ -n "${EPRE:-}" ] && [ -n "${EPOST:-}" ]; then
    awk -v a="$EPRE" -v b="$EPOST" -v t="$TOL" 'BEGIN{d = a - b; if (d < 0) d = -d; exit !(d <= t)}' \
      && pass "loss continuity: ema $EPRE (step $K) vs $EPOST (step $CHK) within $TOL" \
      || fail "loss continuity: |$EPRE - $EPOST| > $TOL"
  else
    fail "could not extract EMAs (pre '$EPRE' post '$EPOST')"
  fi
  SKIP2=$(grep -ac 'SKIPPED' "$LOG"); [ "$SKIP2" -eq 0 ] && pass "no guard skips during G2" || echo "note: $SKIP2 skipped steps (inspect before calling G2 clean)"
}

g3(){
  RUN=${1:?usage: gates.sh g3 RUN [MINUTES]}; MIN=${2:-90}
  LOG=$LOGS/train-$RUN.log; CK=$CPT_HOME/ckpt/$RUN
  # first launch needs real args: G3_ARGS='--save-every 100 ...' (data dir etc. come from the CPT_* env suite)
  if [ -n "${G3_ARGS:-}" ]; then bash "$DIR/launch_cpt.sh" "$RUN" $G3_ARGS; else bash "$DIR/launch_cpt.sh" "$RUN"; fi
  END=$(( $(date +%s) + MIN * 60 ))
  while [ "$(date +%s)" -lt $END ]; do
    grep -aq '^EXIT=' "$LOG" 2>/dev/null && { fail "trainer exited early: $(grep -a '^EXIT=' "$LOG" | tail -n 1) (quarantine lines: $(grep -ac 'CPT_QUARANTINE' "$LOG"))"; break; }
    sleep 60
  done
  grep -aq '^EXIT=' "$LOG" || { echo "[g3] $MIN min elapsed -> graceful SIGTERM (final checkpoint)"; pkill -TERM -f "train_cpt.py --run $RUN"; sleep 90; }
  T=$(grep -ac 'CPT_GUARD trip' "$LOG"); [ "$T" -eq 0 ] && pass "zero guard trips" || fail "$T guard trips: $(grep -a 'CPT_GUARD trip' "$LOG" | head -n 3)"
  A=$(grep -ac 'CPT_ALERT' "$LOG"); [ "$A" -eq 0 ] && pass "zero router alerts" || fail "$A router alerts"
  Q=$(grep -ac 'CPT_QUARANTINE' "$LOG"); [ "$Q" -eq 0 ] && pass "no quarantine" || fail "QUARANTINED — owner review before ANYTHING restarts"
  P=$(grep -ac 'CPT_PROBE' "$LOG"); [ "$P" -ge 3 ] && pass "$P router probes logged" || fail "only $P probes (probe cadence too slow or probe broken)"
  NM=$(find "$CK" -name COMPLETE 2>/dev/null | wc -l | tr -d ' ')
  [ "$NM" -ge 2 ] && pass "$NM complete checkpoints on disk" || fail "only $NM complete checkpoints"
  TPS=$(grep -aE '^step ' "$LOG" | sed -E 's/.*tps ([0-9.]+).*/\1/' | awk '{s += $1; n++} END{if (n) printf "%.0f", s / n}')
  echo "throughput: mean ${TPS:-?} tok/s (record this in the Gate-3 note)"
  if [ -n "${TPS:-}" ] && awk -v x="$TPS" 'BEGIN{exit !(x + 0 < 18000)}'; then
    echo "WARN: below the 18k tok/s decision threshold — plan §2: accept wall-clock or port to torchtitan AFTER the lane passes"
  fi
  if command -v gcloud >/dev/null; then
    GB=${CORPUS_BUCKET:-gs://YOUR-BUCKET}
    GM=$(gcloud storage ls "$GB/cpt_ckpt/$RUN/step_*/COMPLETE" 2>/dev/null | wc -l | tr -d ' ')
    [ "$GM" -ge 1 ] && pass "GCS mirror holds $GM complete checkpoints" || fail "GCS mirror empty — sync sidecar dead?"
  fi
}

case ${1:-} in
  g1) shift; g1 "$@" ;;
  g2) shift; g2 "$@" ;;
  g3) shift; g3 "$@" ;;
  *) echo "usage: gates.sh g1 [MODEL_ID] | g2 RUN | g3 RUN [MINUTES]"; exit 2 ;;
esac
[ "$FAILED" -eq 0 ] && echo "== GATE RESULT: PASS" || echo "== GATE RESULT: FAIL"
exit $FAILED
