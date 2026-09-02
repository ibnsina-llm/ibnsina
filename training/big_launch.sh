#!/bin/bash
# Launch the IbnSina 1.5B box and start big_pipeline.sh on it (run from the Mac). Frees the CPU quota, launches via the ladder,
# waits for READY, drops the /data/big marker and starts the pipeline (reboot-safe from then on).
# usage: DATA_SHARDS=<n train shards of the mix> [MIX=train_v1_1_open] big_launch.sh [NAME=big-8g]
set -u; P=${GCP_PROJECT:-YOUR-GCP-PROJECT}; Z=us-central1-a; NAME=${1:-big-8g}; HERE=$(cd "$(dirname "$0")" && pwd); MIX=${MIX:-train_v1_1_open}; : "${DATA_SHARDS:?set DATA_SHARDS to the train shard count of the mix}"
for vm in corpus-downloader2 sbx-test-linux wordpress-learningloop0b-vm pilot-8g; do
  st=$(gcloud compute instances describe $vm --project=$P --zone=$Z --format='value(status)' 2>/dev/null); [ "$st" = "RUNNING" ] && { echo "stopping $vm"; gcloud compute instances stop $vm --project=$P --zone=$Z --discard-local-ssd=true --quiet >/dev/null 2>&1 || gcloud compute instances stop $vm --project=$P --zone=$Z --quiet >/dev/null 2>&1; }
done
LADDER="${LADDER:-a100spot8,a100od}" EXTRA_META="data-shards=$DATA_SHARDS,mix=$MIX" MAX_WAIT_MIN=${MAX_WAIT_MIN:-30} bash "$HERE/gpu_launch.sh" "$NAME" h100spot "$HERE/gpu_startup.sh" 2000 | tee /tmp/big_launch.out
grep -q LAUNCHED /tmp/big_launch.out || { echo "!! launch failed"; exit 1; }
for i in $(seq 1 60); do sleep 60; r=$(gcloud compute ssh $NAME --project=$P --zone=$Z --quiet --command='sudo test -f /data/READY && echo READY' 2>/dev/null); [ "$r" = "READY" ] && break; echo "[$i min] waiting for READY"; done
[ "$r" = "READY" ] || { echo "!! box not READY after 60 min"; exit 1; }
gcloud compute ssh $NAME --project=$P --zone=$Z --quiet --command='sudo bash -c "gcloud --no-user-output-enabled storage rsync -r ${CORPUS_BUCKET:-gs://YOUR-BUCKET}/code/pipeline/training /data/pipeline/training; echo big > /data/big; nohup bash /data/pipeline/training/big_pipeline.sh >/dev/null 2>&1 < /dev/null & sleep 5; echo started; tail -n 2 /data/logs/big_pipeline.log"' 2>/dev/null
echo "$(TZ=Asia/Tokyo date '+%H:%M JST') big pipeline started on $NAME"
