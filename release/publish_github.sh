#!/bin/bash
# Release day: create the public repo github.com/ibnsina-llm/ibnsina and push this repository. DOES NOTHING without --yes (Sina's sign-off).
# usage: release/publish_github.sh --yes
set -u; ORG=ibnsina-llm; REPO=ibnsina; cd "$(dirname "$0")/.."
VIS=public; [ "${2:-}" = "--private" ] && VIS=private
[ "${1:-}" = "--yes" ] || { echo "dry run: would create $ORG/$REPO ($VIS, Apache-2.0) and push a single squashed snapshot of the current tree (no history). Re-run with --yes [--private] after sign-off."; exit 0; }
grep -rnIE "hf_[A-Za-z0-9]{20,}|sk-or-v1-[0-9a-f]{20,}" --exclude-dir=.git . && { echo "!! secret-looking string in repo — aborting"; exit 1; }
[ -f LICENSE ] || curl -sL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE && git add LICENSE && git commit -qm "Apache-2.0 licence" || true
gh repo view $ORG/$REPO >/dev/null 2>&1 || gh repo create $ORG/$REPO --$VIS --description "IbnSina — open Persian-first language models: corpus pipeline, tokenizer, training, SFT recipe, release tooling" --disable-wiki
git remote get-url public >/dev/null 2>&1 || git remote add public https://github.com/$ORG/$REPO.git
# public history = one snapshot commit of the current tree (no working history, no old file contents)
SNAP=$(git commit-tree "$(git write-tree)" -m "IbnSina: initial public release") && git push --force public "$SNAP:refs/heads/main" && echo "pushed snapshot $SNAP to https://github.com/$ORG/$REPO"
