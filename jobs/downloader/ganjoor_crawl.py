#!/opt/corpus-venv/bin/python3
"""Ganjoor API walk -> raw/literature/ganjoor/<poet>.jsonl.gz (+ _manifest.json). Resume-safe, polite (~4 req/s).

pass 1: /api/ganjoor/poets -> recursive /api/ganjoor/cat/{id}?poems=true -> poem index (id, poet, cat path, title, url)
pass 2: /api/ganjoor/poem/{id}?catInfo=false&catPoems=false&rhymes=false&recitations=false&images=false -> plainText
Output doc: {"id","poet","poet_id","cat","title","url","text"}  (text = Ganjoor plainText: one hemistich per line)
"""
import gzip, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

API = "https://api.ganjoor.net/api/ganjoor"
BUCKET = os.environ.get("CORPUS_BUCKET", os.environ.get("CORPUS_BUCKET", "gs://YOUR-BUCKET"))
OUT = Path("/data/ganjoor"); OUT.mkdir(parents=True, exist_ok=True)
INDEX = OUT / "poems_index.jsonl"; DONE = OUT / "poets_done.txt"; LOG = Path("/data/logs/ganjoor.log")
RPS, WORKERS = 4.0, 4
_lock = threading.Lock(); _last = [0.0]
S = requests.Session(); S.headers["User-Agent"] = "persian-corpus-research-crawler (contact: curator)"


def log(*a):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def get(path, params=None, tries=6):
    for i in range(tries):
        with _lock:  # global rate limit
            wait = _last[0] + 1.0 / RPS - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = S.get(f"{API}{path}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            log(f"http {r.status_code} on {path} (try {i+1})")
        except Exception as e:
            log(f"error {type(e).__name__} on {path} (try {i+1})")
        time.sleep(5 * (i + 1))
    return None


def walk_cat(cat_id, path, poet, out):
    c = get(f"/cat/{cat_id}", {"poems": "true", "mainSections": "false"})
    if not c or not c.get("cat"):
        return
    cat = c["cat"]; title = cat.get("title", "")
    here = path + [title] if title else path
    for p in cat.get("poems") or []:
        out.append({"id": p["id"], "poet": poet["name"], "poet_id": poet["id"], "cat": " > ".join(here), "title": p.get("title", ""),
                    "url": "https://ganjoor.net" + (p.get("urlSlug") and p.get("fullUrl") or "") if p.get("fullUrl") else ""})
    for ch in cat.get("children") or []:
        walk_cat(ch["id"], here, poet, out)


def pass1():
    if INDEX.exists() and (OUT / "_index_done").exists():
        return [json.loads(l) for l in INDEX.open()]
    poets = get("/poets") or []
    log(f"pass1: {len(poets)} poets")
    idx = []
    with INDEX.open("w") as f:
        for i, poet in enumerate(poets, 1):
            rows = []
            walk_cat(poet["rootCatId"], [], poet, rows)
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); idx.append(r)
            if i % 20 == 0:
                log(f"pass1: {i}/{len(poets)} poets, {len(idx):,} poems indexed")
    (OUT / "_index_done").touch()
    log(f"pass1 done: {len(idx):,} poems")
    return idx


def fetch_poem(row):
    p = get(f"/poem/{row['id']}", {"catInfo": "false", "catPoems": "false", "rhymes": "false", "recitations": "false", "images": "false"})
    if not p:
        return None
    text = (p.get("plainText") or "").strip()
    if not text and p.get("verses"):
        text = "\n".join(v.get("text", "") for v in p["verses"])
    if not text:
        return None
    url = p.get("fullUrl") or row.get("url") or ""
    return {"id": row["id"], "poet": row["poet"], "poet_id": row["poet_id"], "cat": row["cat"], "title": p.get("title") or row["title"],
            "url": ("https://ganjoor.net" + url) if url.startswith("/") else url, "text": text}


def pass2(idx):
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    by_poet = {}
    for r in idx:
        by_poet.setdefault((r["poet_id"], r["poet"]), []).append(r)
    total = sum(len(v) for v in by_poet.values()); n = 0; t0 = time.time()
    for (pid, pname), rows in sorted(by_poet.items(), key=lambda kv: -len(kv[1])):
        fn = OUT / f"poet_{pid}.jsonl.gz"
        if str(pid) in done:
            n += len(rows); continue
        docs = []
        with ThreadPoolExecutor(WORKERS) as ex:
            for d in ex.map(fetch_poem, rows):
                if d:
                    docs.append(d)
        with gzip.open(fn, "wt", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        subprocess.run(["gcloud", "--no-user-output-enabled", "storage", "cp", str(fn), f"{BUCKET}/raw/literature/ganjoor/{fn.name}"], check=False)
        with DONE.open("a") as f:
            f.write(f"{pid}\n")
        n += len(rows); rate = n / max(1, time.time() - t0)
        log(f"pass2: {pname} {len(docs)}/{len(rows)} poems | {n:,}/{total:,} total | eta {(total-n)/max(rate,1e-6)/3600:.1f} h")
    log("pass2 done")


def manifest(idx):
    files = sorted(OUT.glob("poet_*.jsonl.gz")); total = sum(f.stat().st_size for f in files); docs = 0; chars = 0
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                docs += 1; chars += len(json.loads(line)["text"])
    m = {"id": "ganjoor", "name": "Ganjoor — full API walk (api.ganjoor.net), all published poets", "url": "https://api.ganjoor.net",
         "category": "literature", "license": "classical poems public domain; site metadata per ganjoor.net terms", "files": [{"path": f.name, "bytes": f.stat().st_size} for f in files],
         "file_count": len(files), "total_bytes": total, "docs": docs, "chars": chars, "poems_indexed": len(idx), "est_tokens": int(chars * 0.30),
         "est_method": "chars x 0.30 (bpe64k proxy ratio for Persian)", "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tool": "ganjoor_crawl.py",
         "vm": os.uname().nodename, "status": "ok", "error": None}
    (OUT / "_manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))
    subprocess.run(["gcloud", "--no-user-output-enabled", "storage", "cp", str(OUT / "_manifest.json"), f"{BUCKET}/raw/literature/ganjoor/_manifest.json"], check=False)
    log(f"manifest: {len(files)} files, {docs:,} poems, {chars:,} chars")


if __name__ == "__main__":
    idx = pass1()
    pass2(idx)
    manifest(idx)
    Path("/data/.done").mkdir(exist_ok=True); (Path("/data/.done") / "ganjoor").touch()
