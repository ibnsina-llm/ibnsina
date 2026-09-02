#!/bin/bash
# FineWeb-2 fas_Arab: local files already verified against the Hub -> manifest + upload to raw/, then domain filter (2 passes) + audit.
source /data/dl/lib.sh fw2
if [ ! -e /data/.done/fineweb-2-fa ]; then
  finalize fineweb-2-fa web "FineWeb-2 Persian (HuggingFaceFW/fineweb-2 data/fas_Arab/train, 25 parquet)" "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2" "ODC-By 1.0" "hf_download (sizes verified vs Hub tree after preemption)" ok "" 0
fi
F="/opt/corpus-venv/bin/python3 /data/dl/filter_parallel.py --src $CORPUS_BUCKET/raw/web/fineweb-2-fa/data/fas_Arab/train --dst $CORPUS_BUCKET/raw_filtered/web/fineweb-2-fa --name fineweb-2-fa-filtered --workers 6"
log "filtering fineweb-2-fa -> raw_filtered"
for pass in 1 2; do $F 2>&1 | tee -a $LOG; done
$F --audit 2>&1 | tee -a $LOG
log "=== fw2 finished"; sync_log
