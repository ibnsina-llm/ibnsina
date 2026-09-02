#!/bin/bash
# On corpus-downloader: fetch FineWeb-2 fas_Arab train (25 parquet, ~102 GB) -> raw/web/fineweb-2-fa/, then domain-filter -> raw_filtered/web/fineweb-2-fa/
source /data/dl/lib.sh fw2
hf_ds fineweb-2-fa web "FineWeb-2 Persian (HuggingFaceFW/fineweb-2 data/fas_Arab/train, 25 parquet)" "ODC-By 1.0" 0 HuggingFaceFW/fineweb-2 "data/fas_Arab/train/*"
log "filtering fineweb-2-fa -> raw_filtered"
for pass in 1 2; do /opt/corpus-venv/bin/python3 /data/dl/filter_parallel.py --src $CORPUS_BUCKET/raw/web/fineweb-2-fa/data/fas_Arab/train --dst $CORPUS_BUCKET/raw_filtered/web/fineweb-2-fa --name fineweb-2-fa-filtered --workers 6 2>&1 | tee -a $LOG; done; /opt/corpus-venv/bin/python3 /data/dl/filter_parallel.py --audit --src $CORPUS_BUCKET/raw/web/fineweb-2-fa/data/fas_Arab/train --dst $CORPUS_BUCKET/raw_filtered/web/fineweb-2-fa --name fineweb-2-fa-filtered --workers 6 2>&1 | tee -a $LOG
log "=== fw2 finished"; sync_log
