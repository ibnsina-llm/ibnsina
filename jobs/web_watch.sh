#!/bin/bash
# Polls raw_filtered/ every 10 min and processes any NEW web shards incrementally (culturax_fa, mc4_fa, fineweb2_fa).
source /etc/profile.d/pipe.sh; export PATH=/opt/pipe/bin:$PATH; cd /data/pipeline || exit 1
LOG=/data/logs/p0-web.log; mkdir -p /data/logs
while true; do
  for ds in culturax_fa mc4_fa fineweb2_fa; do
    python3 pipeline/p0_run.py $ds --workers ${WEB_WORKERS:-24} >> $LOG 2>&1 || echo "$(date -u +%FT%TZ) !!! FAILED: $ds" >> $LOG
  done
  gcloud storage cp -q $LOG $CORPUS_BUCKET/clean/_logs/ 2>/dev/null
  sleep 600
done
