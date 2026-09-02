#!/bin/bash
# ParsiNLU test sets (CC-BY-NC-SA-4.0 — evaluation only) from github.com/persiannlp/parsinlu -> /data/eval/parsinlu/
set -u; D=/data/eval/parsinlu; mkdir -p $D; R=https://raw.githubusercontent.com/persiannlp/parsinlu/master/data
for f in multiple-choice/test.jsonl entailment/test.csv qqp/test.jsonl; do o=$D/$(echo $f | tr '/' '_'); [ -s $o ] || curl -sL --retry 3 -o $o $R/$f; echo "$o $(wc -l < $o) lines"; done
# PersianMedQA test split (HF parquet; needs a token in /data/secrets/hf.token or $HF_TOKEN)
T=${HF_TOKEN:-$(cat /data/secrets/hf.token 2>/dev/null)}; P=/data/eval/persianmedqa; mkdir -p $P
if [ -n "$T" ] && [ ! -s $P/test.parquet ]; then
  U=$(curl -s -H "Authorization: Bearer $T" https://huggingface.co/api/datasets/MohammadJRanjbar/PersianMedQA/parquet/default/test | python3 -c "import json,sys; print(json.load(sys.stdin)[0])" 2>/dev/null)
  [ -n "$U" ] && curl -sL -H "Authorization: Bearer $T" -o $P/test.parquet "$U" && echo "$P/test.parquet $(stat -c %s $P/test.parquet) bytes"
else [ -s $P/test.parquet ] && echo "persianmedqa present" || echo "persianmedqa: no HF token yet — skipped"; fi
