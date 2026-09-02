#!/usr/bin/env python3
"""Parallel, per-shard-uploading domain-blocklist filter (drop-in for Agent A's sequential filter_news.py).

Same semantics: a row is dropped iff pdomain(url) — last two host labels — is in raw/_control/blocklist.txt.
Same shard names, same _manifest.json shape. Each shard is uploaded as soon as it is done, so
downstream (phase 0 incremental) can start immediately. Resume-safe: shards already in --dst are skipped.

usage: filter_parallel.py --src gs://B/raw/web/mc4-fa/multilingual --dst gs://B/raw_filtered/web/mc4-fa --name mc4-fa-filtered --workers 8
"""
from __future__ import annotations
import argparse, gzip, json, multiprocessing as mp, os, shutil, subprocess, sys, time
from pathlib import Path
from urllib.parse import urlparse

BUCKET = os.environ.get("CORPUS_BUCKET", os.environ.get("CORPUS_BUCKET", "gs://YOUR-BUCKET"))
BLOCKLIST_GCS = f"{BUCKET}/raw/_control/blocklist.txt"
WORK = Path("/data/filtered_parallel")
DOMS: set = set()


def sh(*a, timeout=None):
    return subprocess.run([str(x) for x in a], capture_output=True, text=True, timeout=timeout)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_blocklist():
    r = sh("gcloud", "storage", "cat", BLOCKLIST_GCS, timeout=120)
    return {l.strip().lower() for l in r.stdout.splitlines() if l.strip() and not l.startswith("#")}


def pdomain(url):
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def gcs_names(prefix, exts):
    r = sh("gcloud", "storage", "ls", f"{prefix.rstrip('/')}/", timeout=300)
    return sorted(l.strip() for l in r.stdout.splitlines() if l.strip().endswith(exts))


def _init(doms):
    global DOMS
    DOMS = doms


def filter_gz(url, name, dst):
    tmp = WORK / (name + ".part")
    kept = drop = 0
    with subprocess.Popen(["gcloud", "storage", "cat", url], stdout=subprocess.PIPE) as p, \
         gzip.GzipFile(fileobj=p.stdout, mode="rb") as gz, gzip.open(tmp, "wb", compresslevel=3) as out:
        for raw in gz:
            try:
                u = json.loads(raw).get("url")
            except Exception:
                continue
            if u and pdomain(u) in DOMS:
                drop += 1
            else:
                out.write(raw); kept += 1
    return tmp, kept, drop


def filter_parquet(url, name, dst):
    import pyarrow as pa, pyarrow.parquet as pq
    src = WORK / (name + ".in"); tmp = WORK / (name + ".part")
    with subprocess.Popen(["gcloud", "storage", "cat", url], stdout=subprocess.PIPE) as p, open(src, "wb") as f:
        shutil.copyfileobj(p.stdout, f, 1 << 22)
    pf = pq.ParquetFile(src); writer = None; kept = drop = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        if "url" not in tbl.column_names:
            idx = list(range(tbl.num_rows))
        else:
            idx = [i for i, u in enumerate(tbl.column("url").to_pylist()) if not (u and pdomain(u) in DOMS)]
        drop += tbl.num_rows - len(idx)
        if not idx:
            continue
        sub = tbl.take(pa.array(idx, type=pa.int64()))
        if writer is None:
            writer = pq.ParquetWriter(tmp, sub.schema, compression="snappy")
        writer.write_table(sub); kept += len(idx)
    if writer is not None:
        writer.close()
    src.unlink(missing_ok=True)
    return (tmp if writer is not None else None), kept, drop


def do_shard(args):
    url, dst = args
    name = url.rsplit("/", 1)[-1]
    last = None
    for attempt in range(4):  # transient gcloud stream/upload errors: re-download + re-filter
        try:
            tmp, kept, drop = (filter_parquet if name.endswith(".parquet") else filter_gz)(url, name, dst)
            if tmp is not None:
                final = WORK / name; tmp.rename(final)
                for up in range(4):
                    r = sh("gcloud", "storage", "cp", "-q", str(final), f"{dst}/{name}", timeout=3600)
                    if r.returncode == 0:
                        break
                    last = f"upload failed: {r.stderr.strip()[-200:]}"; time.sleep(10 * (up + 1))
                else:
                    time.sleep(15 * (attempt + 1)); continue
                final.unlink(missing_ok=True)
            return name, kept, drop, None
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            for f in (WORK / (name + ".part"), WORK / (name + ".in")):
                f.unlink(missing_ok=True)
            time.sleep(15 * (attempt + 1))
    return name, 0, 0, last


def count_rows(url):
    """Rows in a GCS shard (json.gz lines / parquet rows) — streams, no local copy."""
    name = url.rsplit("/", 1)[-1]
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq
        tmp = WORK / (name + ".cnt")
        with subprocess.Popen(["gcloud", "storage", "cat", url], stdout=subprocess.PIPE) as p, open(tmp, "wb") as f:
            shutil.copyfileobj(p.stdout, f, 1 << 22)
        n = pq.ParquetFile(tmp).metadata.num_rows; tmp.unlink(missing_ok=True); return n
    n = 0
    with subprocess.Popen(["gcloud", "storage", "cat", url], stdout=subprocess.PIPE) as p, gzip.GzipFile(fileobj=p.stdout, mode="rb") as gz:
        for _ in gz:
            n += 1
    return n


def audit_pair(args):
    src_url, dst_url = args
    name = src_url.rsplit("/", 1)[-1]
    for attempt in range(3):
        try:
            raw = count_rows(src_url); kept = count_rows(dst_url) if dst_url else 0
            return name, raw, kept, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"; time.sleep(10 * (attempt + 1))
    return name, 0, 0, err


def audit(a, doms):
    """Exact, run-independent totals: for every source shard, rows_raw and rows_kept (from the uploaded filtered shard)."""
    exts = (".json.gz", ".jsonl.gz", ".parquet")
    src = gcs_names(a.src, exts); dst = {u.rsplit("/", 1)[-1]: u for u in gcs_names(a.dst, exts)}
    missing = [u.rsplit("/", 1)[-1] for u in src if u.rsplit("/", 1)[-1] not in dst]
    log(f"audit: {len(src)} source shards, {len(dst)} filtered shards present, {len(missing)} missing")
    pairs = [(u, dst.get(u.rsplit("/", 1)[-1])) for u in src]
    raw_t = kept_t = 0; errors = {}; per = {}; t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, (name, raw, kept, err) in enumerate(pool.imap_unordered(audit_pair, pairs), 1):
            if err:
                errors[name] = err
            else:
                raw_t += raw; kept_t += kept; per[name] = [raw, kept]
            if i % 50 == 0 or i == len(pairs):
                log(f"audit {i}/{len(pairs)} raw={raw_t:,} kept={kept_t:,} eta {((len(pairs)-i)/max(i/(time.time()-t0),1e-9))/60:.0f} min")
    ok = not missing and not errors
    manifest = {"id": a.name, "status": "ok" if ok else "partial", "filter": "domain-blocklist v1 (curator-signed 2026-08-28)",
                "blocklist_domains": len(doms), "rows_raw": raw_t, "rows_kept": kept_t, "rows_dropped": raw_t - kept_t,
                "kept_ratio": round(kept_t / max(raw_t, 1), 4), "source_raw": a.source_raw or a.src, "files": len(dst), "files_total": len(src),
                "missing_files": missing, "errors": errors, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"), "vm": os.uname().nodename,
                "note": "counts from a full audit pass over source and filtered shards (run-independent); filtered by filter_parallel.py, identical rule to filter_news.py"}
    (WORK / "_manifest.json").write_text(json.dumps(manifest, indent=2)); (WORK / "_audit_per_shard.json").write_text(json.dumps(per))
    sh("gcloud", "storage", "cp", "-q", str(WORK / "_manifest.json"), f"{a.dst}/_manifest.json", timeout=120)
    sh("gcloud", "storage", "cp", "-q", str(WORK / "_audit_per_shard.json"), f"{a.dst}/_audit_per_shard.json", timeout=120)
    log(f"== audit {a.name}: status={manifest['status']} raw={raw_t:,} kept={kept_t:,} dropped={raw_t-kept_t:,} missing={len(missing)} errors={len(errors)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--source-raw", default="")
    ap.add_argument("--audit", action="store_true", help="no filtering: recount every shard and write an exact manifest")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.audit:
        audit(a, load_blocklist()); return
    doms = load_blocklist(); log(f"blocklist: {len(doms)} domains")
    exts = (".json.gz", ".jsonl.gz", ".parquet")
    src = gcs_names(a.src, exts); done = {u.rsplit("/", 1)[-1] for u in gcs_names(a.dst, exts)}
    todo = [u for u in src if u.rsplit("/", 1)[-1] not in done]
    log(f"{len(src)} source shards, {len(done)} already in dst, {len(todo)} to do, {a.workers} workers")
    counts = {"kept": 0, "dropped": 0, "files_done": len(done), "files_total": len(src), "errors": {}}
    cpath = Path(f"/data/filter_counts_{a.name}.json")
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init, initargs=(doms,)) as pool:
        for i, (name, kept, drop, err) in enumerate(pool.imap_unordered(do_shard, [(u, a.dst) for u in todo]), 1):
            if err:
                counts["errors"][name] = err; log(f"!! {name}: {err}")
            else:
                counts["kept"] += kept; counts["dropped"] += drop; counts["files_done"] += 1
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0; rate = i / el
                log(f"{i}/{len(todo)} shards ({counts['files_done']}/{counts['files_total']} total) kept={counts['kept']:,} dropped={counts['dropped']:,} eta {((len(todo)-i)/max(rate,1e-9))/60:.0f} min")
                cpath.write_text(json.dumps(counts, indent=1))
    ok = not counts["errors"]
    manifest = {"id": a.name, "status": "partial", "filter": "domain-blocklist v1 (curator-signed 2026-08-28)",
                "blocklist_domains": len(doms), "rows_kept": counts["kept"], "rows_dropped": counts["dropped"],
                "kept_ratio": round(counts["kept"] / max(counts["kept"] + counts["dropped"], 1), 4),
                "source_raw": a.source_raw or a.src, "files": counts["files_done"], "errors": counts["errors"],
                "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"), "vm": os.uname().nodename,
                "note": "PROVISIONAL counts from this run only — the --audit pass rewrites this manifest with exact totals"}
    (WORK / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    sh("gcloud", "storage", "cp", "-q", str(WORK / "_manifest.json"), f"{a.dst}/_manifest.json", timeout=120)
    log(f"== {a.name} {'DONE' if ok else 'DONE WITH ERRORS'} → {a.dst}  kept={counts['kept']:,} dropped={counts['dropped']:,} in {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
