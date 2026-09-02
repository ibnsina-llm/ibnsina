#!/bin/bash
source /data/dl/lib.sh flores
hf_ds flores200-pes parallel "FLORES-200 pes_Arab dev/devtest (EVAL ONLY — never train on it)" "CC BY-SA 4.0" 0 facebook/flores "data/language/pes_Arab/*"
log "=== flores finished"; sync_log
