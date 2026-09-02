#!/bin/bash
# sft_v2 full run driver (on corpus-pipeline2). usage: run_all.sh bulk|diff — generation then judging for the group; renders S-B when both groups are done.
set -u; G=$1; cd /data/pipeline/sft_v2; export PYTHONUNBUFFERED=1; B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}
BULK="general_instruction,reasoning_math_cot_fa,persian_native,formatting_control"
DIFF="persian_writing,translation_fa_en,iran_knowledge,everyday_advice,taarof_register,respect_and_contested,multiturn_repair,uncertainty,refusals,language_discipline,toolcall,anti_sycophancy"
echo "== $(date -u +%FT%TZ) gen $G start"
if [ "$G" = bulk ]; then
  /opt/pipe/bin/python3 -X faulthandler gen.py --categories $BULK --gemini-model gemini-2.5-flash-lite --kimi-share 0 --gemini-concurrency 24 --budget-usd 60 2>&1 | grep --line-buffered -v AFC
else
  /opt/pipe/bin/python3 -X faulthandler gen.py --categories $DIFF --gemini-model gemini-2.5-flash --kimi-share 0.12 --kimi-cap-usd 30 --gemini-concurrency 12 --kimi-concurrency 3 --budget-usd 100 2>&1 | grep --line-buffered -v AFC
fi
echo "== $(date -u +%FT%TZ) gen $G done"
CATS=$([ "$G" = bulk ] && echo $BULK || echo $DIFF)
/opt/pipe/bin/python3 -X faulthandler judge.py --categories $CATS --judge-model gemini-2.5-flash --concurrency 16 2>&1 | grep --line-buffered -v AFC
echo "== $(date -u +%FT%TZ) judge $G done"; touch /data/sft_v2/.done_$G
if [ -f /data/sft_v2/.done_bulk ] && [ -f /data/sft_v2/.done_diff ]; then
  /opt/pipe/bin/python3 judge.py --categories $BULK,$DIFF --judge-model gemini-2.5-flash >/dev/null 2>&1   # re-summarise all categories into S-B.md (already-judged rows are skipped)
  /opt/pipe/bin/python3 render_report.py /data/sft_v2/report /data/sft_v2/report/S-B.html
  gcloud --no-user-output-enabled storage rsync -r /data/sft_v2/report $B/sft/v2/report; gcloud --no-user-output-enabled storage rsync -r /data/sft_v2/dpo_pairs $B/sft/v2/dpo_pairs
  cp /data/sft_v2/ledger.json /data/sft_v2/report/; echo "== $(date -u +%FT%TZ) S-B READY"
fi
