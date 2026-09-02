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

