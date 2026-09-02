#!/bin/bash
# Build $NANOCHAT_BASE_DIR/base_data_climbmix (nanochat's data dir; sorted files, LAST file = val) from train_v1_open:
#   shard_00000..shard_{N-1} = train shards, shard_{N} = val/shard_00000. Data is nanochat-native already (parquet, `text` column, 2000-doc row groups).
# usage: fetch_data.sh N        (N=4 for the smoke test, 189 = all of train_v1_open)
set -u; N=$1; B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}; BASE=${NANOCHAT_BASE_DIR:-/data/nc}; D=$BASE/base_data_climbmix
MIX=${MIX:-$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/mix 2>/dev/null || echo train_v1_open)}
mkdir -p "$D" /data/mix/train /data/mix/val
if [ "$N" -ge 100 ]; then gcloud --no-user-output-enabled storage rsync -r "$B/$MIX/train" /data/mix/train; fi
for i in $(seq 0 $((N-1))); do f=$(printf 'shard_%05d.parquet' $i)
  [ -s /data/mix/train/$f ] || gcloud --no-user-output-enabled storage cp "$B/$MIX/train/$f" /data/mix/train/$f
  ln -sfn /data/mix/train/$f "$D/$f"; done
[ -s /data/mix/val/shard_00000.parquet ] || gcloud --no-user-output-enabled storage cp "$B/$MIX/val/shard_00000.parquet" /data/mix/val/shard_00000.parquet
ln -sfn /data/mix/val/shard_00000.parquet "$D/$(printf 'shard_%05d.parquet' $N)"
for f in "$D"/shard_*.parquet; do i=$(basename "$f" .parquet | cut -d_ -f2 | sed 's/^0*//'); [ "${i:-0}" -gt "$N" ] && rm -f "$f"; done
echo "data dir $D: $(ls "$D" | wc -l) files (last = val)"
