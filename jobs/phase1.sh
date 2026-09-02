#!/bin/bash
[ -f "$(dirname "$0")/private_sources.env" ] && source "$(dirname "$0")/private_sources.env"
# Phase 1 (dedup) on corpus-pipeline. usage: jobs/phase1.sh [extra p1_dedup.py args, e.g. --skip-minhash --force]
source /etc/profile.d/pipe.sh; export PATH=/opt/pipe/bin:$PATH
cd /data/pipeline || exit 1
ulimit -n 1048576 2>/dev/null || ulimit -n 65536   # datatrove bucket stage opens every signature file (4k+) per task
LOG=/data/logs/p1-dedup.log; mkdir -p /data/logs
WEB="matina culturax_fa mc4_fa fineweb2_fa"                                          # priority order: first wins
CURATED="fawiki fawikisource poems history chap_textbooks ganjoor ${PRIVATE_SOURCES:-}"  # never dropped; missing ones are skipped
echo "$(date -u +%FT%TZ) >>> p1_dedup --web $WEB --curated $CURATED --workers 60 $*" >> $LOG
python3 pipeline/p1_dedup.py --web $WEB --curated $CURATED --workers 60 "$@" >> $LOG 2>&1 || echo "$(date -u +%FT%TZ) !!! FAILED: p1_dedup $*" >> $LOG
echo "$(date -u +%FT%TZ) === phase 1 finished" >> $LOG; gcloud storage cp -q $LOG $CORPUS_BUCKET/clean/_logs/ 2>/dev/null
