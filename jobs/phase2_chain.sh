#!/bin/bash
# Waits for Phase 1 (deduped/_DONE.json) then runs Phase 2 end-to-end (sample -> label -> train -> score -> report).
source /etc/profile.d/pipe.sh; export PATH=/opt/pipe/bin:$PATH; cd /data/pipeline || exit 1
LOG=/data/logs/p2-chain.log; mkdir -p /data/logs
say(){ echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say ">>> waiting for $CORPUS_BUCKET/deduped/_DONE.json"
until gcloud storage ls $CORPUS_BUCKET/deduped/_DONE.json >/dev/null 2>&1; do sleep 300; done
say ">>> Phase 1 done — starting Phase 2 (all)"
bash jobs/phase2.sh all >> $LOG 2>&1 || say "!!! phase2 all failed — see /data/logs/p2-*.log"
say "=== phase 2 chain finished"; gcloud storage cp -q $LOG $CORPUS_BUCKET/scored/_logs/ 2>/dev/null
