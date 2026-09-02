#!/bin/bash
[ -f "$(dirname "$0")/private_sources.env" ] && source "$(dirname "$0")/private_sources.env"
# Phase 0 lanes on corpus-pipeline. usage: jobs/phase0.sh <small|enwiki|fineweb|stackoverflow|web DATASET...>
source /etc/profile.d/pipe.sh; export PATH=/opt/pipe/bin:$PATH
cd /data/pipeline || exit 1
LANE=$1; shift; LOG=/data/logs/p0-$LANE.log; mkdir -p /data/logs
run(){ echo "$(date -u +%FT%TZ) >>> p0_run $*" >> $LOG; python3 pipeline/p0_run.py "$@" >> $LOG 2>&1 || echo "$(date -u +%FT%TZ) !!! FAILED: $*" >> $LOG; gcloud storage cp -q $LOG $CORPUS_BUCKET/clean/_logs/ 2>/dev/null; }
case $LANE in
  small)   run poems history fawikisource chap_textbooks konkur ${PRIVATE_SOURCES:-} en_books opus100 --workers 8
           run opus_ted2020 opus_wikimatrix opus_tep opus_globalvoices opus_xlent opus_mizan opus_hplt opus_ccaligned opus_opensubtitles opus_ccmatrix --workers 8
           run ncert openstax --workers 12
           run fawiki --workers 12 ;;
  enwiki)  run enwiki --workers 24 ;;
  fineweb) run fineweb_edu --workers 8 ;;
  stackoverflow) echo "$(date -u +%FT%TZ) >>> so_extract" >> $LOG; python3 pipeline/so_extract.py --workers 8 >> $LOG 2>&1 || echo "!!! FAILED so_extract" >> $LOG ;;
  web)     run "$@" --workers 28 ;;
  *)       run "$LANE" "$@" ;;
esac
echo "$(date -u +%FT%TZ) === lane $LANE finished" >> $LOG; gcloud storage cp -q $LOG $CORPUS_BUCKET/clean/_logs/ 2>/dev/null
