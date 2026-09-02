#!/bin/bash
# synth_v1 bulk driver: serial waves (gen batch -> judge batch -> assemble) until STOP SY-B (~1B kept
# tokens), budget hard stop, or a keep-rate floor breach. Run detached on corpus-pipeline2:
#   setsid nohup bash /data/synth_v1/code/run_synth.sh > /data/synth_v1/bulk/driver.log 2>&1 < /dev/null &
# Monitor: /data/synth_v1/bulk/driver.log, ledger.json, state.json, reports/wave-*.json
set -u
PY=/opt/pipe/bin/python3
CODE=/data/synth_v1/code
BULK=/data/synth_v1/bulk
export PYTHONPATH="$CODE"
mkdir -p "$BULK"
FAILS=0
while :; do
  W=$($PY "$CODE/gen_bulk.py" --print-next-wave)
  echo "==== $(date -u +%FT%TZ) wave $W: gen ===="
  $PY "$CODE/gen_bulk.py" --wave "$W" --run; rc=$?
  if [ $rc -eq 43 ]; then echo "BUDGET HARD STOP"; touch "$BULK/BUDGET_STOP"; break; fi
  if [ $rc -ne 0 ]; then FAILS=$((FAILS+1)); echo "gen failed rc=$rc (fail #$FAILS)"; [ $FAILS -ge 3 ] && break; sleep 600; continue; fi
  FAILS=0
  echo "==== $(date -u +%FT%TZ) wave $W: judge ===="
  $PY "$CODE/judge_bulk.py" --wave "$W" --run; rc=$?
  if [ $rc -eq 43 ]; then echo "BUDGET HARD STOP"; touch "$BULK/BUDGET_STOP"; break; fi
  if [ $rc -ne 0 ]; then FAILS=$((FAILS+1)); echo "judge failed rc=$rc (fail #$FAILS)"; [ $FAILS -ge 3 ] && break; sleep 600; continue; fi
  FAILS=0
  echo "==== $(date -u +%FT%TZ) wave $W: assemble ===="
  $PY "$CODE/assemble_shards.py" --wave "$W"; rc=$?
  case $rc in
    0) FAILS=0 ;;
    42) echo "STOP SY-B reached — pausing for Sina's review"; touch "$BULK/SYB_READY"; break ;;
    43) echo "BUDGET HARD STOP"; touch "$BULK/BUDGET_STOP"; break ;;
    44) echo "KEEP-RATE FLOOR BREACH — stopping for inspection"; touch "$BULK/KEEPRATE_STOP"; break ;;
    *) FAILS=$((FAILS+1)); echo "assemble failed rc=$rc (fail #$FAILS)"; [ $FAILS -ge 3 ] && break; sleep 600 ;;
  esac
done
echo "driver exited at $(date -u +%FT%TZ)"
