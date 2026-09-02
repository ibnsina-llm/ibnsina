#!/bin/bash
# GPU instance launcher implementing the training-track rules:
#   * zones rotate us-central1 a -> b -> c -> f; the winning zone is logged to gs://.../checkpoints/_launches.log
#   * spot first; if all zones refuse, fall back down the ladder given in --ladder (e.g. "h100spot,a100spot,a100od")
#   * every instance: --max-run-duration=24h --instance-termination-action=STOP, label auto-kill=true, corpus-worker SA
# usage: gpu_launch.sh NAME TIER [STARTUP_SCRIPT] [DISK_GB]      TIER in: l4spot | a100spot | a100_80spot | h100spot (a3-highgpu-8g) | a100spot8 (a2-ultragpu-8g spot) | a100od (a2-ultragpu-8g on-demand)
# env: LADDER="h100spot,a100spot,a100od" (fallback order after TIER), MAX_WAIT_MIN=30, EXTRA_META="data-shards=4" (extra instance metadata)
set -u
NAME=$1; TIER=$2; STARTUP=${3:-}; DISK=${4:-500}
P=${GCP_PROJECT:-YOUR-GCP-PROJECT}; SA=corpus-worker@$P.iam.gserviceaccount.com; B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}
IMG="--image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 --image-project=deeplearning-platform-release"
ZONES="us-central1-a us-central1-b us-central1-c us-central1-f"
spec(){ case $1 in
  l4spot)     echo "--machine-type=g2-standard-8 --provisioning-model=SPOT";;
  a100spot)   echo "--machine-type=a2-highgpu-1g --provisioning-model=SPOT";;
  a100spot8)  echo "--machine-type=a2-ultragpu-8g --provisioning-model=SPOT";;
  a100_80spot) echo "--machine-type=a2-ultragpu-1g --provisioning-model=SPOT";;
  h100spot)   echo "--machine-type=a3-highgpu-8g --provisioning-model=SPOT";;
  a100od)     echo "--machine-type=a2-ultragpu-8g --provisioning-model=STANDARD";;
  h100spot4)  echo "--machine-type=a3-highgpu-4g --provisioning-model=SPOT";;
  a100spot4)  echo "--machine-type=a2-ultragpu-4g --provisioning-model=SPOT";;
  a100od4)    echo "--machine-type=a2-ultragpu-4g --provisioning-model=STANDARD";;
  *) echo "unknown tier $1" >&2; exit 2;; esac; }
zones_for(){ case $1 in h100spot|h100spot4) echo "us-central1-a us-central1-b us-central1-c";; a100spot8|a100od|a100spot4|a100od4) echo "us-central1-a us-central1-c";; *) echo "$ZONES";; esac; }  # where the machine type exists (checked 2026-08-29)
try_tier(){ local tier=$1 zone
  for zone in $(zones_for $tier); do
    # STOP-on-termination is required with --max-run-duration for every provisioning model; 8-GPU shapes carry local SSDs, which must be declared discardable
    local extra="--instance-termination-action=STOP"; case $tier in h100spot|a100spot8|a100od|h100spot4|a100spot4|a100od4) extra="$extra --discard-local-ssds-at-termination-timestamp=true";; esac
    echo "$(date -u +%FT%TZ) trying $tier in $zone"
    if gcloud compute instances create $NAME --project=$P --zone=$zone $(spec $tier) $extra --max-run-duration=24h \
         --maintenance-policy=TERMINATE $IMG --boot-disk-size=${DISK}GB --boot-disk-type=pd-balanced \
         --network-interface=nic-type=GVNIC --service-account=$SA --scopes=cloud-platform \
         --metadata=install-nvidia-driver=True${EXTRA_META:+,$EXTRA_META}${STARTUP:+ --metadata-from-file=startup-script=$STARTUP} \
         --labels=purpose=persian-corpus,role=train,auto-kill=true,tier=$tier --quiet >/tmp/gpu_launch.out 2>&1; then
      echo "$(date -u +%FT%TZ) LAUNCHED $NAME tier=$tier zone=$zone" | tee -a /tmp/gpu_launches.log
      echo "$(date -u +%FT%TZ) $NAME tier=$tier zone=$zone" | gcloud storage cp - $B/checkpoints/_launches.log 2>/dev/null || true
      echo "$zone"; return 0
    fi
    grep -qiE "ZONE_RESOURCE_POOL_EXHAUSTED|resources available|stockout|QUOTA" /tmp/gpu_launch.out && echo "  refused: $(grep -oiE 'ZONE_RESOURCE_POOL_EXHAUSTED|not have enough resources|QUOTA_EXCEEDED|Quota .* exceeded' /tmp/gpu_launch.out | head -1)" || { echo "  error: $(tail -n 2 /tmp/gpu_launch.out)"; }
  done
  return 1; }
START=$(date +%s)
for tier in $TIER ${LADDER:+$(echo $LADDER | tr ',' ' ')}; do
  try_tier $tier && exit 0
  echo "$(date -u +%FT%TZ) all zones refused tier=$tier"
  [ $(( ($(date +%s) - START) / 60 )) -ge ${MAX_WAIT_MIN:-30} ] && { echo "!! blocked on capacity for ${MAX_WAIT_MIN:-30}+ min — STOP and tell the curator"; exit 3; }
done
echo "!! no tier succeeded ($TIER ${LADDER:-})"; exit 3
