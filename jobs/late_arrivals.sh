#!/bin/bash
# Runs Phase 0 on late-arriving raw/ datasets as their _manifest.json (status ok) appears. Low CPU priority (nice) so web lanes keep the cores.
source /etc/profile.d/pipe.sh; export PATH=/opt/pipe/bin:$PATH; cd /data/pipeline || exit 1
LOG=/data/logs/p0-late.log; mkdir -p /data/logs; B=$CORPUS_BUCKET
say(){ echo "$(date -u +%FT%TZ) $*" >> $LOG; }
ready(){ gcloud storage cat "$B/$1/_manifest.json" 2>/dev/null | python3 -c "import json,sys;sys.exit(0 if json.load(sys.stdin).get('status')=='ok' else 1)" 2>/dev/null; }
done_(){ gcloud storage ls "$B/clean/$1/_DONE.json" >/dev/null 2>&1; }
# dataset:raw-prefix
QUEUE="open_web_math:raw/english_math/open-web-math starcoder_ts:raw/code/starcoder-typescript starcoder_py:raw/code/starcoder-python-slice ganjoor:raw/literature/ganjoor"
say ">>> late-arrivals lane started"
while true; do
  left=0
  for pair in $QUEUE; do
    ds=${pair%%:*}; pre=${pair##*:}
    done_ $ds && continue
    if ready $pre; then
      say ">>> p0_run $ds"; nice -n 10 python3 pipeline/p0_run.py $ds --workers 8 >> $LOG 2>&1 || say "!!! FAILED: $ds"
    else left=$((left+1)); fi
  done
  [ $left -eq 0 ] && break
  sleep 600
done
say "=== late-arrivals lane finished"; gcloud storage cp -q $LOG $B/clean/_logs/ 2>/dev/null
