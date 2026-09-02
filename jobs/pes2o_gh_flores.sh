#!/bin/bash
# On corpus-downloader, after FineWeb-2: PeS2o v2 (EN STEM, ~93 GB), GitHub persian-nlp topic repos (shallow), FLORES-200 fas_Arab (eval only).
source /data/dl/lib.sh pes2o_gh
# 1) FLORES (tiny, eval only)
hf_ds flores200-pes parallel "FLORES-200 pes_Arab dev/devtest (EVAL ONLY — never train on it)" "CC BY-SA 4.0" 0 facebook/flores "data/language/pes_Arab/*"
# 2) GitHub persian-nlp topic: enumerate via API (2 pages), shallow-clone all
urls=$(for p in 1 2; do curl -s "https://api.github.com/search/repositories?q=topic:persian-nlp&per_page=100&page=$p" | python3 -c "import json,sys;[print(r['clone_url']) for r in json.load(sys.stdin).get('items',[])]"; sleep 3; done | sort -u)
log "github persian-nlp topic: $(echo "$urls" | wc -l) repos"
gitrepos gh-persian-nlp-topic code "GitHub topic persian-nlp — all repos (shallow, .git stripped)" "per-repo" 0 $urls
# 3) PeS2o v2
hf_ds pes2o-v2 english_edu "PeS2o v2 (allenai/peS2o data/v2, arXiv/S2ORC STEM)" "ODC-By 1.0" 0 allenai/peS2o "data/v2/*"
log "=== pes2o_gh_flores finished"; sync_log
