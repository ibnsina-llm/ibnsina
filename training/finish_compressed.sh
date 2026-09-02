#!/bin/bash
# Escape-ladder rung 1b (run died past ~50k): finish with a compressed warmdown from the last clean checkpoint and ship.
# usage: finish_compressed.sh RUN LAST_CLEAN_STEP NEW_TOTAL_ITERS      e.g. finish_compressed.sh big 52000 60000
# - stops the pipeline loop, trainer and sync watcher; quarantines every checkpoint newer than LAST_CLEAN (local + GCS, reversible)
# - sets --num-iterations=NEW_TOTAL_ITERS in RUN.args, NANOCHAT_WARMDOWN_START=LAST_CLEAN in RUN.env, resume_once=LAST_CLEAN
# - relaunches the pipeline with ITERS=NEW_TOTAL_ITERS so the SFT/eval/export stages pick up the new final checkpoint
set -eu; RUN=$1; LAST=$2; NEW=$3; B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}/checkpoints; D=/data/nc/base_checkpoints/$RUN; Q=/data/nc/base_checkpoints/${RUN}_quarantine_$(date -u +%Y%m%dT%H%M)
pl=$(pgrep -f "^bash /data/pipeline/training/big_pipeline" || true); [ -n "$pl" ] && kill $pl; tmux kill-session -t train-$RUN 2>/dev/null || true; tmux kill-session -t sync-$RUN 2>/dev/null || true; sleep 5
mkdir -p $Q; for f in $D/*_0*; do st=$(basename $f | sed -E 's/^[a-z]+_0*([0-9]+).*/\1/'); [ "$st" -gt "$LAST" ] && mv "$f" $Q/; done; echo "local quarantined: $(ls $Q | wc -l) files -> $Q"
for st in $(gsutil ls $B/$RUN/ | grep -oE "meta_[0-9]+" | sed -E 's/meta_0*//' | sort -n); do [ "$st" -gt "$LAST" ] && gsutil -m -q mv "$B/$RUN/*_$(printf %06d $st)*" $B/${RUN}_quarantine/ ; done; echo "gcs latest: $(gsutil ls $B/$RUN/ | grep -a meta | tail -n 1 | xargs basename)"
sed -i "s/^--num-iterations=.*/--num-iterations=$NEW/" /data/runs/$RUN.args
E=/data/runs/$RUN.env; grep -v "^NANOCHAT_WARMDOWN_START=" $E > $E.tmp || true; echo "NANOCHAT_WARMDOWN_START=$LAST" >> $E.tmp; mv $E.tmp $E
echo $LAST > /data/runs/$RUN.resume_once
echo "$(date -u +%FT%TZ) [curator] COMPRESSED FINISH: resume $LAST, warmdown $LAST -> $NEW, num_iterations=$NEW (ladder rung 1b)" >> /data/logs/train-$RUN.log
cd /data/pipeline/training && ITERS=$NEW nohup bash /data/pipeline/training/big_pipeline.sh >/dev/null 2>&1 &
sleep 5; echo "pipeline relaunched with ITERS=$NEW (procs: $(pgrep -fc '^bash /data/pipeline/training/big_pipeline'))"
