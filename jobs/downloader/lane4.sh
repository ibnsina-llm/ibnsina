#!/bin/bash
source /data/dl/lib.sh lane4
log "=== lane start; free disk: $(df -h /data | tail -1 | awk "{print \$4}")"
hf_ds culturax-fa web "CulturaX Persian (uonlp/CulturaX fa, 96 parquet parts)" "uonlp/CulturaX terms (gated; mC4 ODC-BY + OSCAR CC0 derived)" 0 uonlp/CulturaX "fa/*"
hf_ds a4411f5aff parallel "FLORES-200 fas_Arab dev/devtest (eval only — never mix into pretraining)" "CC BY-SA 4.0" 1 facebook/flores "**/fas_Arab/*"
log "=== lane finished; free disk: $(df -h /data | tail -1 | awk "{print \$4}")"
sync_log
