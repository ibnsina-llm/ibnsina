#!/bin/bash
source /etc/profile.d/corpus.sh
F="/opt/corpus-venv/bin/python3 /data/dl/filter_parallel.py --src $CORPUS_BUCKET/raw/web/mc4-fa/multilingual --dst $CORPUS_BUCKET/raw_filtered/web/mc4-fa --name mc4-fa-filtered --workers 8 --source-raw $CORPUS_BUCKET/raw/web/mc4-fa/multilingual"
while tmux has-session -t mc4filter 2>/dev/null; do sleep 60; done
for i in 1 2 3; do
  $F 2>&1 | tee -a /data/logs/mc4filter.log
  n=$(gcloud storage ls $CORPUS_BUCKET/raw_filtered/web/mc4-fa/ 2>/dev/null | grep -c json.gz); [ "$n" -ge 1026 ] && break; sleep 30
done
$F --audit 2>&1 | tee -a /data/logs/mc4filter.log
echo "$(date -u +%FT%TZ) mc4_retry finished (shards=$n)" >> /data/logs/mc4filter.log
