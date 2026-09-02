#!/bin/bash
# Phase-1 auto-kick gate (runs in tmux on corpus-pipeline).
# Fires when: mC4 + FineWeb-2 domain filters report status ok AND phase 0 has processed every filtered shard of
# culturax_fa, mc4_fa, fineweb2_fa. Then finalizes those datasets (_DONE.json) and runs jobs/phase1.sh.
source /etc/profile.d/pipe.sh; export PATH=/opt/pipe/bin:$PATH; cd /data/pipeline || exit 1
LOG=/data/logs/p1-gate.log; mkdir -p /data/logs
B=$CORPUS_BUCKET
say(){ echo "$(date -u +%FT%TZ) $*" >> $LOG; }
filter_ok(){ gcloud storage cat $B/raw_filtered/web/$1/_manifest.json 2>/dev/null | python3 -c "import json,sys;m=json.load(sys.stdin);sys.exit(0 if m.get('status')=='ok' else 1)" 2>/dev/null; }
expected(){ case $1 in culturax-fa) echo 96;; mc4-fa) echo 1026;; fineweb-2-fa) echo 25;; *) echo 999999;; esac; }
n_filtered(){ n=$(gcloud storage ls "$B/raw_filtered/web/$1/" 2>/dev/null | grep -c -E '\.(parquet|json\.gz|jsonl\.gz)$'); e=$(expected $1); [ "${n:-0}" -ge "$e" ] && echo $n || echo $e; }
n_processed(){ gcloud storage cat $B/clean/$1/_processed.json 2>/dev/null | python3 -c "import json,sys;print(len(json.load(sys.stdin)['processed']))" 2>/dev/null || echo 0; }
say ">>> gate started"
while true; do
  ok=1
  for pair in "mc4-fa:mc4_fa" "fineweb-2-fa:fineweb2_fa" "culturax-fa:culturax_fa"; do
    raw=${pair%%:*}; ds=${pair##*:}
    nf=$(n_filtered $raw); np=$(n_processed $ds)
    if ! filter_ok $raw; then say "  $raw: filter not finished ($nf shards uploaded, $np processed)"; ok=0
    elif [ "${np:-0}" -lt "$(expected $raw)" ] || [ "${np:-0}" -lt "$nf" ]; then say "  $raw: filter ok, phase0 $np/$nf"; ok=0
    else say "  $raw: complete ($np/$nf)"; fi
  done
  if [ $ok -eq 1 ]; then break; fi
  sleep 300
done
say ">>> all web datasets complete — finalizing and starting Phase 1"
tmux kill-session -t p0-web 2>/dev/null
python3 pipeline/p0_run.py culturax_fa mc4_fa fineweb2_fa --final --workers 4 >> $LOG 2>&1
python3 pipeline/p0_report.py >> $LOG 2>&1
bash jobs/phase1.sh
say "=== gate done"; gcloud storage cp -q $LOG $B/clean/_logs/ 2>/dev/null
