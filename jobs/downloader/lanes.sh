#!/bin/bash
# Batch-1 downloader for the Persian corpus. usage: lanes.sh <lane1|lane2|lane3>
# Layout: /data/<id>/ -> gs://$CORPUS_BUCKET/raw/<category>/<id>/ (+ _manifest.json). Resumable: skips ids with /data/.done/<id>.
source /etc/profile.d/corpus.sh
export PATH=/opt/corpus-venv/bin:$PATH
LANE=$1; LOG=/data/logs/$LANE.log; DONE=/data/.done; mkdir -p $DONE /data/logs
MAN=/data/dl/manifest.py
log(){ echo "$(date -u +%FT%TZ) [$LANE] $*" | tee -a $LOG; }
sync_log(){ gcloud storage cp -q $LOG $CORPUS_BUCKET/raw/_logs/claude-batch1-$LANE.log 2>/dev/null || true; }

# finalize ID CATEGORY NAME URL LICENSE TOOL STATUS ERROR IN_SOURCES_JSON
finalize(){
  local id=$1 cat=$2 name=$3 url=$4 lic=$5 tool=$6 status=$7 err=$8 insrc=${9:-1}
  local dir=/data/$id
  find $dir -name '*.aria2' -delete 2>/dev/null; rm -rf $dir/.cache 2>/dev/null
  if [ -z "$(find $dir -type f -not -name '_manifest.json' 2>/dev/null | head -1)" ]; then
    log "$id: no files downloaded -> failed"; status=failed
    rm -rf $dir; return 1
  fi
  $MAN $dir --id "$id" --category "$cat" --name "$name" --url "$url" --license "$lic" --tool "$tool" --status "$status" --error "$err" --in-sources-json "$insrc" 2>&1 | tee -a $LOG
  local dst=$CORPUS_BUCKET/raw/$cat/$id
  log "$id: uploading to $dst"
  if gcloud storage rsync -r $dir $dst 2>&1 | tail -1 | tee -a $LOG; then
    local lbytes=$(du -sb $dir | cut -f1) rbytes=$(gcloud storage du -s $dst 2>/dev/null | awk '{print $1}')
    if [ "$lbytes" = "$rbytes" ]; then
      log "$id: upload verified ($lbytes bytes) -> removing local copy"; rm -rf $dir; touch $DONE/$id
    else
      log "$id: SIZE MISMATCH local=$lbytes remote=$rbytes -> keeping local copy"
    fi
  else
    log "$id: upload FAILED -> keeping local copy"
  fi
  sync_log
}

# http ID CATEGORY NAME LICENSE CONNS IN_SOURCES_JSON URL [URL...]   (aria2c, resumable; 404s tolerated)
http(){
  local id=$1 cat=$2 name=$3 lic=$4 conns=$5 insrc=$6; shift 6
  [ -e $DONE/$id ] && { log "$id: already done, skip"; return; }
  local dir=/data/$id; mkdir -p $dir; local fails=0 first=$1
  log "$id: start ($# urls)"
  for u in "$@"; do
    aria2c -c -x$conns -s$conns -k1M --file-allocation=none --console-log-level=warn --summary-interval=0 --retry-wait=10 --max-tries=8 --timeout=120 --auto-file-renaming=false --allow-overwrite=true --check-certificate=false -d $dir "$u" >>$LOG 2>&1 || { fails=$((fails+1)); log "$id: FAILED $u"; }
  done
  local st=ok; [ $fails -gt 0 ] && st=partial
  finalize "$id" "$cat" "$name" "$first" "$lic" "aria2c" $st "$fails of $# urls failed" $insrc
}

# hf ID CATEGORY NAME LICENSE IN_SOURCES_JSON REPO INCLUDE
hf_ds(){
  local id=$1 cat=$2 name=$3 lic=$4 insrc=$5 repo=$6 inc=$7
  [ -e $DONE/$id ] && { log "$id: already done, skip"; return; }
  local dir=/data/$id; mkdir -p $dir
  log "$id: start hf download $repo --include $inc"
  local st=ok
  hf download "$repo" --repo-type dataset --include "$inc" --local-dir $dir --max-workers 8 >>$LOG 2>&1 || { st=partial; log "$id: hf download returned non-zero (retrying once)"; sleep 30; hf download "$repo" --repo-type dataset --include "$inc" --local-dir $dir --max-workers 8 >>$LOG 2>&1 && st=ok; }
  rm -rf $dir/.cache
  finalize "$id" "$cat" "$name" "https://huggingface.co/datasets/$repo" "$lic" "hf_download" $st "" $insrc
}

# gitrepos ID CATEGORY NAME LICENSE IN_SOURCES_JSON URL [URL...]  (shallow clone, strip .git, tar.zst)
gitrepos(){
  local id=$1 cat=$2 name=$3 lic=$4 insrc=$5; shift 5
  [ -e $DONE/$id ] && { log "$id: already done, skip"; return; }
  local dir=/data/$id; mkdir -p $dir; local fails=0 first=$1
  for u in "$@"; do
    local r=$(basename $u .git); local org=$(basename $(dirname $u))
    if git clone -q --depth=1 "$u" $dir/${org}__$r >>$LOG 2>&1; then
      rm -rf $dir/${org}__$r/.git; (cd $dir && tar --zstd -cf ${org}__$r.tar.zst ${org}__$r && rm -rf ${org}__$r)
    else fails=$((fails+1)); log "$id: clone FAILED $u"; fi
  done
  local st=ok; [ $fails -gt 0 ] && st=partial
  finalize "$id" "$cat" "$name" "$first" "$lic" "git_clone_depth1+tar.zst" $st "$fails of $# repos failed" $insrc
}

log "=== lane start; free disk: $(df -h /data | tail -1 | awk '{print $4}')"
case $LANE in
lane1)  # Persian web + big HTTP files
  http seed007 web "CC-100 fa (CommonCrawl/CCNet Persian)" "CC-100 terms (CommonCrawl-derived; research)" 8 1 \
    https://data.statmt.org/cc-100/fa.txt.xz
  http enwiki-latest english_edu "English Wikipedia pages-articles-multistream (latest dump)" "CC BY-SA 4.0 / GFDL" 2 0 \
    https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
  http 0210b42bc0 code "Stack Overflow data dump (Posts) — for fa-tagged / code Q&A extraction" "CC BY-SA 4.0" 4 1 \
    https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z
  ;;
lane2)  # Hugging Face bulk
  hf_ds mc4-fa web "mC4 Persian (allenai/c4 multilingual/c4-fa, 1024 train + 2 val shards)" "ODC-BY" 0 allenai/c4 "multilingual/c4-fa*"
  hf_ds fineweb-edu-sample-10bt english_edu "FineWeb-Edu sample-10BT (HuggingFaceFW/fineweb-edu)" "ODC-By 1.0" 0 HuggingFaceFW/fineweb-edu "sample/10BT/*"
  ;;
lane3)  # small/medium: wiki fa, parallel, textbooks, code repos
  http seed011 wikipedia "Persian Wikipedia pages-meta-current (latest dump)" "CC BY-SA 4.0 / GFDL" 2 1 \
    https://dumps.wikimedia.org/fawiki/latest/fawiki-latest-pages-meta-current.xml.bz2
  http seed001 literature "Persian Wikisource pages-meta-current (latest dump) — PD classics" "CC BY-SA 4.0 / GFDL; source texts public domain" 2 1 \
    https://dumps.wikimedia.org/fawikisource/latest/fawikisource-latest-pages-meta-current.xml.bz2
  http 0ec785d2ba parallel "OPUS TED2020 en-fa" "CC BY-NC-ND 4.0" 4 1 https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-fa.txt.zip
  http 6aed3c18ac parallel "OPUS WikiMatrix en-fa" "CC BY-SA 4.0" 4 1 https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/en-fa.txt.zip
  http fa6017852a parallel "OPUS OpenSubtitles v2024 en-fa" "per-subtitle; research use" 4 1 https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-fa.txt.zip
  http d60aa4894a parallel "OPUS CCAligned en-fa" "various (web-captured)" 4 1 https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/en-fa.txt.zip
  http opus-xlent parallel "OPUS XLEnt v1.2 en-fa" "CC BY-SA 3.0" 4 0 https://object.pouta.csc.fi/OPUS-XLEnt/v1.2/moses/en-fa.txt.zip
  http 2207986fe4 parallel "OPUS CCMatrix v1 en-fa (== NLLB en-fa)" "various (CCNet web crawl)" 4 1 https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/en-fa.txt.zip
  http opus-mizan parallel "OPUS MIZAN en-fa" "non-commercial" 4 0 https://object.pouta.csc.fi/OPUS-MIZAN/v1/moses/en-fa.txt.zip
  http opus-hplt parallel "OPUS HPLT v2 en-fa" "CC0" 4 0 https://object.pouta.csc.fi/OPUS-HPLT/v2/moses/en-fa.txt.zip
  http e4f7513ffd parallel "OPUS GlobalVoices en-fa" "CC BY 3.0" 4 1 https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/en-fa.txt.zip
  http 94faccd9f9 parallel "OPUS TEP (Tehran English-Persian) en-fa" "see OPUS page" 4 1 https://object.pouta.csc.fi/OPUS-TEP/v1/moses/en-fa.txt.zip
  http fb62356885 parallel "OPUS-100 en-fa (HF parquet)" "CC BY 2.0" 4 1 \
    https://huggingface.co/datasets/Helsinki-NLP/opus-100/resolve/main/en-fa/train-00000-of-00001.parquet \
    https://huggingface.co/datasets/Helsinki-NLP/opus-100/resolve/main/en-fa/validation-00000-of-00001.parquet \
    https://huggingface.co/datasets/Helsinki-NLP/opus-100/resolve/main/en-fa/test-00000-of-00001.parquet
  http 8244ced715 parallel "Tatoeba exports (sentences_detailed + links + pes-eng links)" "CC BY 2.0" 4 1 \
    https://downloads.tatoeba.org/exports/sentences_detailed.tar.bz2 \
    https://downloads.tatoeba.org/exports/links.tar.bz2 \
    https://downloads.tatoeba.org/exports/per_language/pes/pes-eng_links.tsv.bz2 \
    https://downloads.tatoeba.org/exports/per_language/pes/pes_sentences_detailed.tsv.bz2
  http seed004 math "chap.sch.ir official textbooks — grade 8 set 1404-1405 (C101..C113)" "Iranian gov. textbook site; no terms page" 2 1 \
    $(for c in C101 C102 C103 C104 C105 C106 C107 C108 C109 C110 C111 C112 C113; do echo http://www.chap.sch.ir/sites/default/files/lbooks/1404-1405/8/$c.pdf; done)
  http 27aeb7f157 math "OpenStax Calculus Vol 1-3 (CloudFront PDFs)" "CC BY-NC-SA 4.0" 4 1 \
    https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/CalculusVolume1-OP.pdf \
    https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/CalculusVolume2-OP.pdf \
    https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/CalculusVolume3-OP.pdf
  http 55d37ef61d english_edu "NCERT full-book zips: Physics/Chem/Math/Bio classes 11-12" "NCERT (free, non-commercial edu)" 2 1 \
    $(for c in leph1dd leph2dd lech1dd lech2dd lemh1dd lemh2dd lebo1dd keph1dd keph2dd kech1dd kech2dd kemh1dd kebo1dd; do echo https://ncert.nic.in/textbook/pdf/$c.zip; done)
  http seed015 english_edu "Project Gutenberg catalog + sample" "Public domain" 2 1 \
    https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz \
    https://www.gutenberg.org/cache/epub/236/pg236.txt
  gitrepos 26697ded2e code "Persian NLP/code GitHub repos (hazm, hezar, persian-tools, divar-ir)" "MIT / Apache-2.0 / per-repo" 1 \
    https://github.com/roshan-research/hazm https://github.com/hezar-ai/hezar https://github.com/persian-tools/persian-tools \
    https://github.com/persian-tools/py-persian-tools https://github.com/divar-ir/ai-doc-gen https://github.com/divar-ir/kenar-docs
  ;;
esac
log "=== lane finished; free disk: $(df -h /data | tail -1 | awk '{print $4}')"
sync_log
