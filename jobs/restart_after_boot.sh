#!/bin/bash
# corpus-pipeline: relaunch every lane after a (spot) reboot. All lanes are idempotent via GCS _DONE.json / _processed.json.
source /etc/profile.d/pipe.sh; export PATH=/opt/pipe/bin:$PATH
sleep 30; cd /data/pipeline || exit 1
has(){ tmux has-session -t $1 2>/dev/null; }
has p0-small         || tmux new-session -d -s p0-small "bash jobs/phase0.sh small"
has p0-enwiki        || tmux new-session -d -s p0-enwiki "bash jobs/phase0.sh enwiki"
has p0-stackoverflow || tmux new-session -d -s p0-stackoverflow "bash jobs/phase0.sh stackoverflow"
has p0-web           || tmux new-session -d -s p0-web "WEB_WORKERS=24 bash jobs/web_watch.sh"
has p0-fawiki         || tmux new-session -d -s p0-fawiki "bash jobs/phase0.sh fawiki --workers 8"
has p0-late          || tmux new-session -d -s p0-late "bash jobs/late_arrivals.sh"
has p2-chain         || tmux new-session -d -s p2-chain "bash jobs/phase2_chain.sh"
has p1-gate          || tmux new-session -d -s p1-gate "bash jobs/phase1_gate.sh"
echo "$(date -u +%FT%TZ) restart_after_boot: $(tmux ls | cut -d: -f1 | tr '\n' ' ')" >> /data/logs/restart.log
