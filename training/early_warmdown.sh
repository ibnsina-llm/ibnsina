#!/bin/bash
# Escape-ladder rung 1: start the LR warmdown now.  usage: early_warmdown.sh RUN START_STEP
# Sets NANOCHAT_WARMDOWN_START in /data/runs/RUN.env and restarts the trainer (the pipeline loop resumes from the latest checkpoint).
set -eu; RUN=$1; START=$2; E=/data/runs/$RUN.env
grep -v "^NANOCHAT_WARMDOWN_START=" $E > $E.tmp || true; echo "NANOCHAT_WARMDOWN_START=$START" >> $E.tmp; mv $E.tmp $E
echo "$(date -u +%FT%TZ) [curator] early warmdown from step $START (NANOCHAT_WARMDOWN_START) — escape ladder rung 1" >> /data/logs/train-$RUN.log
tmux kill-session -t train-$RUN 2>/dev/null || true; echo "trainer restarted with warmdown from $START (env: $(tr '\n' ' ' < $E))"
