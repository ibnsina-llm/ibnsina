#!/bin/bash
# GCE startup-script for the training boxes. Runs as root on EVERY boot (idempotent):
# code from GCS -> nanochat@92d63d4 + uv env -> tokenizer -> data shards (metadata attr data-shards, default 4) -> resume /data/active_run if any.
exec >> /var/log/persian-startup.log 2>&1
echo "== $(date -u +%FT%TZ) boot $(hostname)"
export HOME=/root PATH=/root/.local/bin:/usr/local/bin:$PATH NANOCHAT_BASE_DIR=/data/nc
B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}; mkdir -p /data/logs /data/runs /data/pipeline $NANOCHAT_BASE_DIR
gcloud --no-user-output-enabled storage rsync -r $B/code/pipeline /data/pipeline
[ -d /data/nanochat/.git ] || { git clone -q https://github.com/karpathy/nanochat /data/nanochat && (cd /data/nanochat && git checkout -q 92d63d4); }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
(cd /data/nanochat && [ -x .venv/bin/python ] || uv sync --extra gpu)
[ -f $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl ] || { mkdir -p $NANOCHAT_BASE_DIR/tokenizer; gcloud --no-user-output-enabled storage cp "$B/tokenizer/v1_32k/tokenizer.pkl" "$B/tokenizer/v1_32k/token_bytes.pt" $NANOCHAT_BASE_DIR/tokenizer/; }
N=$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/data-shards || echo 4)
bash /data/pipeline/training/fetch_data.sh "$N"
nvidia-smi -L || echo "!! no GPU visible"
if [ -f /data/active_run ]; then RUN=$(cat /data/active_run); echo "resuming active run $RUN"; bash /data/pipeline/training/train_run.sh "$RUN"; fi
[ -f /data/pilot ] && { echo "pilot marker: (re)starting pilot_pipeline"; nohup bash /data/pipeline/training/pilot_pipeline.sh >/dev/null 2>&1 & }
[ -f /data/big ] && { echo "big marker: (re)starting big_pipeline"; nohup bash /data/pipeline/training/big_pipeline.sh >/dev/null 2>&1 & }
echo "== $(date -u +%FT%TZ) ready"; touch /data/READY
