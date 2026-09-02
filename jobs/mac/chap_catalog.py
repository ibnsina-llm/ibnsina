#!/usr/bin/env python3
"""chap.sch.ir official textbook catalog -> ~/Downloads/corpus_harvest/chap/ -> gs://.../curated/harvest/chap/
Runs on the Mac (chap.sch.ir blocks GCP IPs). Polite (~1.5 req/s for pages, 5 parallel PDF downloads; >6 concurrent connections get an HTML busy page), resumable.

Scope: general-education tracks only — primary 1-6, lower secondary 7-9, upper secondary (math-physics, experimental,
humanities, Islamic studies) 10-12 + pre-university. Current school year (1404-1405) with last year (1403-1404) as fallback.
"""
import hashlib, json, re, subprocess, sys, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

BASE = "http://www.chap.sch.ir"
OUT = Path.home() / "Downloads" / "corpus_harvest" / "chap"
GCS = "gs://YOUR-BUCKET/curated/harvest/chap"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 Chrome/128 Safari/537.36 (research corpus fetch)"
YEARS = {"1394": "1404-1405", "1389": "1403-1404"}
# tid -> label (leaf taxonomy nodes of the general tracks)
TIDS = {
    "8": "primary/grade01", "9": "primary/grade02", "10": "primary/grade03", "12": "primary/grade04", "13": "primary/grade05", "32": "primary/grade06",
    "555": "lower-secondary/grade07", "558": "lower-secondary/grade08", "556": "lower-secondary/grade09",
    "29": "upper-secondary/math-physics/grade10", "30": "upper-secondary/math-physics/grade11", "711": "upper-secondary/math-physics/grade12", "31": "upper-secondary/math-physics/pre-university",
    "40": "upper-secondary/experimental/grade10", "41": "upper-secondary/experimental/grade11", "712": "upper-secondary/experimental/grade12", "42": "upper-secondary/experimental/pre-university",
    "37": "upper-secondary/humanities/grade10", "38": "upper-secondary/humanities/grade11", "709": "upper-secondary/humanities/grade12", "39": "upper-secondary/humanities/pre-university",
    "43": "upper-secondary/islamic-studies/grade10", "44": "upper-secondary/islamic-studies/grade11", "710": "upper-secondary/islamic-studies/grade12", "45": "upper-secondary/islamic-studies/pre-university",
}
LOG = OUT / "_crawl.log"


def log(*a):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True); LOG.open("a").write(line + "\n")


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            log(f"retry {i+1} {url}: {type(e).__name__}"); time.sleep(5 * (i + 1))
    return ""


def index_books():
    """tid x year -> set of book ids (cached in _index.json)."""
    idx_path = OUT / "_index.json"
    if idx_path.exists():
        return json.loads(idx_path.read_text())
    idx = {}
    for tid, label in TIDS.items():
        for ytid, yname in YEARS.items():
            html = get(f"{BASE}/advanced_search?tid={tid}&field_year_tid={ytid}"); time.sleep(0.7)
            ids = sorted(set(re.findall(r'/books/(\d+)', html)))
            for b in ids:
                idx.setdefault(b, {"tid": tid, "label": label, "years": []})["years"].append(yname)
            log(f"index {label} {yname}: {len(ids)} books")
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    return idx


def book_meta(bid):
    html = get(f"{BASE}/books/{bid}")
    t = re.search(r"<title>(.*?)</title>", html, re.S)
    title = (t.group(1).split("|")[0].strip() if t else "")
    pdfs = sorted(set(re.findall(r'(?:https?://[^"\']+)?(/sites/default/files/lbooks/[^"\'\s]+\.pdf)', html)))
    return {"id": bid, "title": title, "pdfs": [BASE + p for p in pdfs]}


def looks_complete(dest: Path):
    """%PDF header and a %%EOF marker in the last 4 KB. (Content-Length can't be trusted: a HEAD that hits the
    server's HTML 'busy' page reports that page's length.)"""
    if not dest.exists() or dest.stat().st_size < 1000:
        return False
    with dest.open("rb") as f:
        head = f.read(5); f.seek(max(0, dest.stat().st_size - 4096)); tail = f.read()
    return head.startswith(b"%PDF") and b"%%EOF" in tail


def download(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if looks_complete(dest):
        return "cached"
    last = "failed"
    for i in range(5):
        r = subprocess.run(["curl", "-sL", "-C", "-", "--max-time", "1800", "--retry", "3", "-A", UA, "-o", str(dest), url])
        if looks_complete(dest):
            return "ok"
        if dest.exists() and dest.stat().st_size >= 5 and not dest.open("rb").read(5).startswith(b"%PDF"):
            dest.unlink(missing_ok=True); last = "not-pdf"      # HTML busy/404 page -> drop and retry
        else:
            last = "truncated" if r.returncode == 0 else "curl-error"   # keep the partial; -C - resumes it
        time.sleep(20 * (i + 1))
    return last


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = index_books()
    log(f"{len(idx)} unique books across {len(TIDS)} grade/track nodes")
    meta_path = OUT / "_books.json"
    books = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for i, (bid, info) in enumerate(idx.items(), 1):
        if bid in books:
            continue
        m = book_meta(bid); m.update(info); books[bid] = m; time.sleep(0.7)
        if i % 25 == 0:
            meta_path.write_text(json.dumps(books, ensure_ascii=False, indent=1)); log(f"meta {i}/{len(idx)}")
    meta_path.write_text(json.dumps(books, ensure_ascii=False, indent=1))
    jobs, seen = [], set()
    for bid, m in books.items():
        for u in m["pdfs"]:
            if u in seen:  # the same PDF can be listed under several catalogue entries/years -> download once
                continue
            seen.add(u)
            code = u.rsplit("/", 1)[-1]; year = u.split("/lbooks/")[1].split("/")[0] if "/lbooks/" in u else "unknown"
            jobs.append((u, OUT / m["label"] / f"{year}_{code}", bid))
    log(f"{len(jobs)} PDFs to fetch")
    results = {}
    with ThreadPoolExecutor(5) as ex:
        for (u, dest, bid), st in zip(jobs, ex.map(lambda j: download(j[0], j[1]), jobs)):
            results[u] = st
            if st not in ("ok", "cached"):
                log(f"!! {st}: {u}")
    ok = [u for u, s in results.items() if s in ("ok", "cached")]
    log(f"downloaded {len(ok)}/{len(jobs)} PDFs")
    files = []
    for bid, m in books.items():
        for u in m["pdfs"]:
            code = u.rsplit("/", 1)[-1]; year = u.split("/lbooks/")[1].split("/")[0]
            p = OUT / m["label"] / f"{year}_{code}"
            if p.exists() and results.get(u) in ("ok", "cached"):
                files.append({"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                              "book_id": bid, "title": m["title"], "grade": m["label"], "year": year, "url": u})
    manifest = {"id": "chap-catalog", "name": "chap.sch.ir official textbooks — general-education tracks, grades 1-12 + pre-university (1404-1405, fallback 1403-1404)",
                "url": f"{BASE}/advanced_search", "category": "textbooks", "license": "Iranian Ministry of Education textbook site (no terms page); curator-approved",
                "files": files, "file_count": len(files), "total_bytes": sum(f["bytes"] for f in files), "books": len(books),
                "failed": {u: s for u, s in results.items() if s not in ("ok", "cached")}, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tool": "jobs/mac/chap_catalog.py (curator's Mac; chap.sch.ir blocks GCP IPs)", "status": "ok" if not any(s not in ("ok", "cached") for s in results.values()) else "partial"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    log(f"manifest: {len(files)} files, {manifest['total_bytes']/1e9:.2f} GB, status {manifest['status']}")
    r = subprocess.run(["gcloud", "--no-user-output-enabled", "storage", "rsync", "-r", "-x", r"_probe|_crawl\.log", str(OUT), GCS], capture_output=True, text=True)
    log("upload:", "ok" if r.returncode == 0 else r.stderr[-300:])


if __name__ == "__main__":
    main()
