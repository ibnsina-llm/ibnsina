#!/bin/bash
# Relaunch all downloader-VM jobs (idempotent: each skips finished work). Run at boot via cron @reboot or by hand.
source /etc/profile.d/corpus.sh
sleep 20
has(){ tmux has-session -t $1 2>/dev/null; }
has mc4filter || tmux new-session -d -s mc4filter "/opt/corpus-venv/bin/python3 /data/dl/filter_parallel.py --src $CORPUS_BUCKET/raw/web/mc4-fa/multilingual --dst $CORPUS_BUCKET/raw_filtered/web/mc4-fa --name mc4-fa-filtered --workers 8 --source-raw $CORPUS_BUCKET/raw/web/mc4-fa/multilingual 2>&1 | tee -a /data/logs/mc4filter.log"
has fw2 || tmux new-session -d -s fw2 "bash /data/dl/fw2_finalize.sh"
# pes2o paused until mc4 + fw2 are done: has pes2o || tmux new-session -d -s pes2o "bash /data/dl/pes2o_gh_flores.sh"
has mc4retry || tmux new-session -d -s mc4retry "bash /data/dl/mc4_retry.sh"
echo "$(date -u +%FT%TZ) restart_jobs: $(tmux ls | cut -d: -f1 | tr "\n" " ")" >> /data/logs/restart_jobs.log
