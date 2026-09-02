#!/bin/bash
# T2 pilot: free the CPU quota (a3-highgpu-8g needs 208 of 209 CPUS) and launch the 8-GPU box.
# Ladder per the training rules: H100 spot (zones a->b->c->f) -> A100-80GB x8 spot -> A100-80GB x8 on-demand. Never blocks > MAX_WAIT_MIN.
# usage: pilot_launch.sh [NAME]      (run from the Mac; needs the pipeline-agent gcloud account)
set -u; P=${GCP_PROJECT:-YOUR-GCP-PROJECT}; Z=us-central1-a; NAME=${1:-pilot-8g}; HERE=$(cd "$(dirname "$0")" && pwd)
for vm in corpus-downloader2 sbx-test-linux wordpress-learningloop0b-vm; do
  st=$(gcloud compute instances describe $vm --project=$P --zone=$Z --format='value(status)' 2>/dev/null)
  [ "$st" = "RUNNING" ] && { echo "stopping $vm"; gcloud compute instances stop $vm --project=$P --zone=$Z --quiet >/dev/null 2>&1 || echo "  !! stop failed"; }
done
gcloud compute instances describe smoke-l4 --project=$P --zone=$Z >/dev/null 2>&1 && { echo "deleting smoke-l4"; gcloud compute instances delete smoke-l4 --project=$P --zone=$Z --quiet >/dev/null 2>&1; }
echo "CPUS in use: $(gcloud compute regions describe us-central1 --project=$P --format="value(quotas[0].usage,quotas[0].limit)" 2>/dev/null)"
LADDER="${LADDER:-a100spot8,a100od}" EXTRA_META="data-shards=189" MAX_WAIT_MIN=${MAX_WAIT_MIN:-30} bash "$HERE/gpu_launch.sh" "$NAME" h100spot "$HERE/gpu_startup.sh" 1000
