#!/bin/bash
# Polls the bigcode/starcoderdata gate with the VM's HF token; once accepted, fetches typescript/ (all) + python/ (first 10 shards) -> raw/code/
source /data/dl/lib.sh starcoder
T=$(cat ~/.cache/huggingface/token)
while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $T" -I https://huggingface.co/datasets/bigcode/starcoderdata/resolve/main/typescript/train-00000-of-00027.parquet)
  [ "$code" = "200" ] || [ "$code" = "302" ] && break
  echo "$(date -u +%FT%TZ) starcoderdata gate not accepted yet (http $code)" >> /data/logs/starcoder.log; sleep 600
done
log "gate accepted — fetching"
hf_ds starcoder-typescript code "StarCoderData typescript/ (bigcode/starcoderdata, The Stack v1 dedup, permissive licenses)" "per-file permissive licenses (The Stack v1); see dataset card" 0 bigcode/starcoderdata "typescript/*"
hf_ds starcoder-python-slice code "StarCoderData python/ shards 00000-00009 of 59 (bigcode/starcoderdata)" "per-file permissive licenses (The Stack v1); see dataset card" 0 bigcode/starcoderdata "python/train-0000[0-9]-of-00059.parquet"
log "=== starcoder finished"; sync_log
