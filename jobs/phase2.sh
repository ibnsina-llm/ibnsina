#!/bin/bash
# Phase 2 (quality scoring) lanes on corpus-pipeline. usage: jobs/phase2.sh <sample|label|train|score|report|all|nollm> [extra p2_quality.py args]
# `label`/`all` need google-genai in /opt/pipe (pip install google-genai) and roles/aiplatform.user on the VM service account.
source /etc/profile.d/pipe.sh; export PATH=/opt/pipe/bin:$PATH
cd /data/pipeline || exit 1
LANE=$1; shift; LOG=/data/logs/p2-$LANE.log; mkdir -p /data/logs
run(){ echo "$(date -u +%FT%TZ) >>> p2_quality $*" >> $LOG; python3 pipeline/p2_quality.py "$@" >> $LOG 2>&1 || echo "$(date -u +%FT%TZ) !!! FAILED: $*" >> $LOG; gcloud storage cp -q $LOG $CORPUS_BUCKET/scored/_logs/ 2>/dev/null; }
case $LANE in
  label|all) python3 -c "import google.genai" 2>/dev/null || echo "$(date -u +%FT%TZ) !!! google-genai missing: /opt/pipe/bin/pip install google-genai" >> $LOG
             run "$LANE" --workers 28 "$@" ;;
  nollm)     run all --no-llm --workers 28 "$@" ;;      # fallback path: keyword weak labels, no Gemini
  *)         run "$LANE" --workers 28 "$@" ;;
esac
echo "$(date -u +%FT%TZ) === lane $LANE finished" >> $LOG; gcloud storage cp -q $LOG $CORPUS_BUCKET/scored/_logs/ 2>/dev/null
