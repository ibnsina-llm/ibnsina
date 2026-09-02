#!/opt/pipe/bin/python3
"""Phase 0 — extract + normalize one dataset:  p0_run.py DATASET [--workers N] [--keep-input]

raw/, raw_filtered/, curated/  ->  clean/{dataset}/part-*.jsonl.gz  (+ rejects/{dataset}/, clean/{dataset}/_stats.json, _DONE.json)
Doc schema: {id, text, source, url, lang, meta{category,...}}
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os, shutil, subprocess, sys, time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (BUCKET, DATA, LangID, ShardWriter, Stats, doc_id, domain_of, drop_boilerplate_lines, gcs_download, gcs_list,
                    gcs_upload_dir, gsutil, log, normalize, script_ratio, short_line_keys)
import readers as R

# ----------------------------------------------------------------------------- dataset registry
# sources: list of (gcs_prefix, glob). lang: fa | en | mixed | code. category feeds the final mix.
D = {}
def reg(name, sources, reader, lang, category, min_chars=50, boilerplate=True, files_per_shard=1, reader_kw=None, pre=None, langid=True, incremental=False):
    D[name] = dict(sources=sources, reader=reader, lang=lang, category=category, min_chars=min_chars, boilerplate=boilerplate,
                   files_per_shard=files_per_shard, reader_kw=reader_kw or {}, pre=pre, langid=langid, incremental=incremental)

B = BUCKET
# --- Persian
reg("fawiki",        [(f"{B}/raw/wikipedia/seed011", "*.xml.bz2")], "wikiextractor", "fa", "wikipedia", pre="wikiextractor", min_chars=100)
reg("fawikisource",  [(f"{B}/raw/literature/seed001", "*.xml.bz2")], "wikiextractor", "fa", "literature", pre="wikiextractor", min_chars=100)
reg("poems",         [(f"{B}/curated/poems", "*.txt")], "text", "fa", "literature", reader_kw={"chunk_lines": 400}, boilerplate=False, langid=False)
reg("history",       [(f"{B}/curated/history", "*.txt")], "text", "fa", "literature", reader_kw={"chunk_lines": 400}, boilerplate=False)
reg("chap_textbooks",[(f"{B}/raw/math/seed004", "*.pdf")], "pdf", "fa", "math", reader_kw={"ocr_lang": "fas+eng"}, boilerplate=False, langid=False)
reg("konkur",        [(f"{B}/curated/harvest/konkur", "*.pdf")], "pdf", "fa", "math", reader_kw={"force_ocr": True, "ocr_lang": "fas+eng"}, boilerplate=False, langid=False)
_priv = Path(__file__).parent / "sources_private.py"   # optional, gitignored: extra reg(...) lines kept out of public docs
if _priv.exists(): exec(_priv.read_text(), globals())
reg("konkur_answers", [(f"{B}/curated/harvest/konkur/answers", "*.pdf")], "pdf", "fa", "math", reader_kw={"ocr_lang": "fas+eng"}, boilerplate=False, langid=False)
# --- web (Persian) — ONLY from raw_filtered/ (Agent A's domain filter output)
reg("culturax_fa",   [(f"{B}/raw_filtered/web/culturax-fa", "**/*.parquet")], "parquet", "fa", "web",
    reader_kw={"meta_cols": ["source", "timestamp"]}, incremental=True)
reg("mc4_fa",        [(f"{B}/raw_filtered/web/mc4-fa", "**/*.json.gz"), (f"{B}/raw_filtered/web/mc4-fa", "**/*.jsonl.gz")], "jsonl", "fa", "web",
    reader_kw={"meta_keys": ["timestamp"]}, incremental=True)
reg("fineweb2_fa",   [(f"{B}/raw_filtered/web/fineweb-2-fa", "**/*.parquet"), (f"{B}/raw_filtered/web/fineweb2-fa", "**/*.parquet")], "parquet", "fa", "web",
    reader_kw={"meta_cols": ["dump", "date", "language_score", "minhash_cluster_size"]}, incremental=True)
# --- English
reg("fineweb_edu",   [(f"{B}/raw/english_edu/fineweb-edu-sample-10bt", "**/*.parquet")], "parquet", "en", "english_edu",
    reader_kw={"meta_cols": ["score", "int_score", "dump", "token_count"]})
reg("enwiki",        [(f"{B}/raw/english_edu/enwiki-latest", "*.xml.bz2")], "wikiextractor", "en", "wikipedia", pre="wikiextractor", min_chars=100)
reg("openstax",      [(f"{B}/curated/harvest/openstax", "*.pdf"), (f"{B}/raw/math/27aeb7f157", "*.pdf")], "pdf", "en", "english_edu",
    reader_kw={"ocr_lang": "eng"}, boilerplate=False)
reg("ncert",         [(f"{B}/raw/english_edu/55d37ef61d", "*.zip")], "zip_pdfs", "en", "english_edu", reader_kw={"ocr_lang": "eng"}, boilerplate=False)
reg("en_books",      [(f"{B}/raw/literature/628d6cd57e", "*.txt"), (f"{B}/raw/literature/c605ed915a", "*.txt")], "text", "en", "literature_en",
    reader_kw={"chunk_lines": 400}, boilerplate=False)
# --- parallel fa-en
OPUS = {"0ec785d2ba": "TED2020", "2207986fe4": "CCMatrix", "6aed3c18ac": "WikiMatrix", "94faccd9f9": "TEP", "d60aa4894a": "CCAligned",
        "e4f7513ffd": "GlobalVoices", "fa6017852a": "OpenSubtitles", "opus-hplt": "HPLT", "opus-mizan": "MIZAN", "opus-xlent": "XLEnt"}
for sid, cname in OPUS.items():
    reg(f"opus_{cname.lower()}", [(f"{B}/raw/parallel/{sid}", "*.zip")], "opus_zip", "mixed", "parallel", reader_kw={"corpus": cname},
        boilerplate=False, langid=False, min_chars=20)
reg("opus100",       [(f"{B}/raw/parallel/fb62356885", "*")], "opus100", "mixed", "parallel", boilerplate=False, langid=False, min_chars=20)
# --- late arrivals (downloader VM, 2026-08-28 evening)
reg("ganjoor",       [(f"{B}/raw/literature/ganjoor", "*.jsonl.gz")], "jsonl", "fa", "literature", boilerplate=False,
    reader_kw={"text_key": "text", "url_key": "url", "meta_keys": ["poet", "cat", "title"]}, min_chars=20)
reg("open_web_math", [(f"{B}/raw/english_math/open-web-math", "**/*.parquet")], "parquet", "en", "math", reader_kw={"meta_cols": ["date"]})
reg("starcoder_ts",  [(f"{B}/raw/code/starcoder-typescript", "**/*.parquet")], "parquet", "code", "code", boilerplate=False, langid=False, min_chars=20,
    reader_kw={"text_col": "content", "url_col": "max_stars_repo_name", "meta_cols": ["max_stars_repo_path", "max_stars_count"]})
reg("starcoder_py",  [(f"{B}/raw/code/starcoder-python-slice", "**/*.parquet")], "parquet", "code", "code", boilerplate=False, langid=False, min_chars=20,
    reader_kw={"text_col": "content", "url_col": "max_stars_repo_name", "meta_cols": ["max_stars_repo_path", "max_stars_count"]})
reg("codeparrot_py", [(f"{B}/raw/code/codeparrot-clean-python", "*.json.gz")], "jsonl", "code", "code", boilerplate=False, langid=False, min_chars=20,
    reader_kw={"text_key": "content", "url_key": "repo_name", "meta_keys": ["path", "license"]})
reg("pes2o",         [(f"{B}/raw/english_edu/pes2o-v2", "**/*.json.gz")], "jsonl", "en", "english_edu", reader_kw={"meta_keys": ["source", "added"]})
reg("gh_persian_nlp",[(f"{B}/raw/code/gh-persian-nlp-topic", "*.tar.zst")], "tar_zst_code", "code", "code", boilerplate=False, langid=False, min_chars=20)
reg("chap_catalog",  [(f"{B}/curated/harvest/chap", "**/*.pdf")], "pdf", "fa", "textbooks", reader_kw={"ocr_lang": "fas+eng"}, boilerplate=False, langid=False)
# --- code
reg("gh_repos",      [(f"{B}/raw/code/26697ded2e", "*.tar.zst")], "tar_zst_code", "code", "code", boilerplate=False, langid=False, min_chars=20)

READERS = {
    "parquet": R.read_parquet, "jsonl": R.read_jsonl, "cc100": R.read_cc100_xz, "wikiextractor": R.read_wikiextractor,
    "text": R.read_text_file, "pdf": R.read_pdf, "zip_pdfs": R.read_zip_of_pdfs, "opus_zip": R.read_opus_zip,
    "opus100": R.read_opus100_parquet, "tar_zst_code": R.read_tar_zst_code,
    "html_or_text": lambda p, **kw: (R.read_html_file(p) if p.endswith(".html") else R.read_text_file(p)),
}


# ----------------------------------------------------------------------------- input staging
def list_inputs(spec):
    files = []
    for prefix, pattern in spec["sources"]:
        for url, size in gcs_list(prefix, pattern):
            base = url.split("/")[-1]
            if base.startswith("_") or base.startswith(".") or base.endswith((".json", ".sha256", ".meta4", ".png", ".md")) and spec["reader"] not in ("opus100",):
                continue
            if spec["reader"] == "opus100" and base.endswith(".json"):
                continue
            files.append((url, size))
    return files


def stage_inputs(name, spec, in_dir: Path, files=None) -> list[str]:
    """Download inputs with per-file size verification + retries (spot-preemption safe: truncated files are re-fetched)."""
    from concurrent.futures import ThreadPoolExecutor
    files = list_inputs(spec) if files is None else files
    if not files:
        return []
    in_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{name}] staging {len(files)} files, {sum(s for _, s in files)/1e9:.2f} GB")

    def fetch(item):
        url, size = item; dest = in_dir / url.split("/")[-1]
        if dest.exists() and dest.stat().st_size == size:
            return dest
        dest.unlink(missing_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part"); err = ""
        for attempt in range(4):
            r = subprocess.run(["gcloud", "--no-user-output-enabled", "storage", "cp", url, str(tmp)], capture_output=True, text=True)
            if r.returncode == 0 and tmp.exists() and (size == 0 or tmp.stat().st_size == size):
                tmp.rename(dest); return dest
            err = r.stderr[-200:]; tmp.unlink(missing_ok=True); time.sleep(5 * (attempt + 1))
        log(f"[{name}] FAILED to stage {url}: {err}")
        return None

    with ThreadPoolExecutor(8) as ex:
        got = [p for p in ex.map(fetch, files) if p is not None]
    if len(got) < len(files):
        log(f"[{name}] staged {len(got)}/{len(files)} files — the rest will be retried next run")
    return sorted(str(p) for p in got)


def run_wikiextractor(name, files, in_dir: Path, workers: int) -> list[str]:
    out = in_dir / "extracted"
    if not (out / "_DONE").exists():
        shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
        for f in files:
            log(f"[{name}] wikiextractor on {Path(f).name} ({workers} procs)")
            subprocess.run([sys.executable, "-m", "wikiextractor.WikiExtractor", "--json", "--no-templates", "--processes", str(workers), "-o", str(out),
                            "-b", "256M", f], check=True, stdout=subprocess.DEVNULL, stderr=open(in_dir / "wikiextractor.log", "ab"))
        (out / "_DONE").touch()
    return sorted(str(p) for p in out.rglob("wiki_*"))


def _read_unit(spec, unit):
    """unit is a path or (path, [row_groups]) for parquet."""
    reader = READERS[spec["reader"]]
    if isinstance(unit, tuple):
        return reader(unit[0], row_groups=unit[1], **spec["reader_kw"])
    return reader(unit, **spec["reader_kw"])


def _unit_name(unit):
    return f"{Path(unit[0]).name}#rg{unit[1][0]}" if isinstance(unit, tuple) else Path(unit).name


# ----------------------------------------------------------------------------- pass 1: boilerplate counting
def count_shard(shard, spec):
    keys = []
    for f in shard:
        for d in _read_unit(spec, f):
            dom = domain_of(d.get("url"), spec["_name"])
            keys.extend(short_line_keys(d["text"], dom))
    if not keys:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
    u, c = np.unique(np.asarray(keys, dtype=np.uint64), return_counts=True)
    return u, c


def banned_lines(shards, spec, workers, threshold=100) -> set:
    with mp.Pool(workers) as pool:
        parts = pool.map(partial(count_shard, spec=spec), shards, chunksize=1)
    keys = np.concatenate([p[0] for p in parts]); cnt = np.concatenate([p[1] for p in parts])
    if keys.size == 0:
        return set()
    order = np.argsort(keys, kind="stable"); keys = keys[order]; cnt = cnt[order]
    uniq, idx = np.unique(keys, return_index=True)
    sums = np.add.reduceat(cnt, idx)
    return set(int(k) for k in uniq[sums > threshold])


# ----------------------------------------------------------------------------- pass 2: process
_LID = None
def _lid():
    global _LID
    if _LID is None:
        _LID = LangID()
    return _LID


def process_shard(args, spec, banned, out_dir, rej_dir):
    idx, shard = args
    name, lang, cat = spec["_name"], spec["lang"], spec["category"]
    reader = READERS[spec["reader"]]
    pre = spec.get("_prefix", "")
    w = ShardWriter(out_dir, pre + "part", idx); rw = ShardWriter(rej_dir, pre + "rej", idx)
    st = Stats(); lid = _lid() if (spec["langid"] and lang in ("fa", "en")) else None
    for f in shard:
        n = 0
        try:
            docs = _read_unit(spec, f)
            for d in docs:
                n += 1; st.inc("docs_in")
                raw = d.get("text") or ""
                text = normalize(raw, "fa" if lang in ("fa", "mixed") else "en")
                if spec["boilerplate"]:
                    text, dropped = drop_boilerplate_lines(text, domain_of(d.get("url"), name), banned); st.inc("bp_lines", dropped)
                did = doc_id(name, _unit_name(f), n)
                reason, detected, prob = None, None, None
                if len(text) < spec["min_chars"]:
                    reason = "too_short"
                elif lid is not None:
                    detected, prob = lid.predict(text)
                    if detected != lang and prob >= 0.6:
                        fa_share, la_share = script_ratio(text)
                        # rescue: fastText says something else but the script is overwhelmingly right (e.g. fa vs. 'ckb'/'ur'/'ar' on short docs)
                        if not ((lang == "fa" and fa_share > 0.9 and detected in ("ar", "ur", "ckb", "ps", "mzn", "azb")) or
                                (lang == "en" and la_share > 0.95 and detected in ("de", "nl", "fr", "es", "it", "pt"))):
                            reason = f"lang_mismatch:{detected}"
                if reason:
                    st.inc("reject:" + reason.split(":")[0]); st.inc("reject_detail:" + reason)
                    rw.write({"id": did, "source": name, "reason": reason, "lang_detected": detected, "prob": prob, "url": d.get("url"),
                              "text": text[:2000]})
                    continue
                doc = {"id": did, "text": text, "source": name, "url": d.get("url"), "lang": lang if lang != "mixed" else "fa+en",
                       "meta": {"category": cat, **(d.get("meta") or {})}}
                if detected is not None:
                    doc["meta"]["lid"] = [detected, round(prob, 3)]
                w.write(doc); st.inc("docs_out"); st.inc("chars_out", len(text))
        except Exception as e:
            st.inc("reject:reader_error"); log(f"[{name}] reader error on {f}: {type(e).__name__}: {e}")
    w.close(); rw.close(); st.inc("bytes_out", w.bytes)
    return st


def make_shards(files, per_shard, workers, spec=None):
    if spec and spec["reader"] == "parquet":  # row-group units so a few big parquet files still fan out across workers
        import pyarrow.parquet as pq
        units = []
        for f in files:
            n = pq.ParquetFile(f).metadata.num_row_groups
            step = max(1, n // 6)
            units += [(f, list(range(i, min(n, i + step)))) for i in range(0, n, step)]
        return [[u] for u in units]
    if per_shard <= 1 and len(files) < workers * 2:
        return [[f] for f in files]
    per = max(per_shard, (len(files) + workers * 4 - 1) // (workers * 4)) if per_shard > 1 else 1
    return [files[i:i + per] for i in range(0, len(files), per)]


def _cat_json(url, default):
    r = subprocess.run(["gcloud", "storage", "cat", url], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else default


def run_dataset(name, workers, keep_input=False, force=False, final=False):
    spec = dict(D[name]); spec["_name"] = name
    in_dir, out_dir, rej_dir = DATA / "in" / name, DATA / "clean" / name, DATA / "rejects" / name
    remote_clean, remote_rej = f"{BUCKET}/clean/{name}", f"{BUCKET}/rejects/{name}"
    incremental = spec.get("incremental", False)
    if not force and gcs_list(remote_clean, "_DONE.json"):
        log(f"[{name}] already done in GCS, skipping (use --force to redo)"); return
    t0 = time.time()
    ledger, batch = {"processed": [], "batches": []}, ""
    all_files = list_inputs(spec)
    if incremental and not force:
        ledger = _cat_json(f"{remote_clean}/_processed.json", ledger)
        seen = set(ledger["processed"])
        all_files = [(u, sz) for u, sz in all_files if u.split("/")[-1] not in seen]
        batch = f"b{len(ledger['batches']):03d}"
        if not all_files:
            log(f"[{name}] incremental: nothing new ({len(seen)} shards already processed)")
            if final and seen:
                (DATA / "tmp_done.json").write_text(json.dumps({"dataset": name, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
                gsutil("cp", "-q", str(DATA / "tmp_done.json"), f"{remote_clean}/_DONE.json")
            return
        log(f"[{name}] incremental batch {batch}: {len(all_files)} new shards ({len(seen)} done before)")
    files = stage_inputs(name, spec, in_dir, all_files)
    if not files:
        log(f"[{name}] no input files found — nothing to do"); return
    staged = {Path(f).name for f in files}
    all_files = [(u, sz) for u, sz in all_files if u.split("/")[-1] in staged]
    if spec["pre"] == "wikiextractor":
        files = run_wikiextractor(name, files, in_dir, workers)
    shards = make_shards(files, spec["files_per_shard"], workers, spec)
    log(f"[{name}] {len(files)} files -> {len(shards)} shards/units, {workers} workers")
    banned = set()
    if spec["boilerplate"]:
        banned = banned_lines(shards, spec, workers)
        log(f"[{name}] boilerplate: {len(banned)} (domain,line) keys repeated >100x" + (" (within this batch)" if batch else ""))
    shutil.rmtree(out_dir, ignore_errors=True); shutil.rmtree(rej_dir, ignore_errors=True)
    out_dir.mkdir(parents=True); rej_dir.mkdir(parents=True)
    spec["_prefix"] = f"{batch}-" if batch else ""
    total = Stats()
    with mp.Pool(workers) as pool:
        for st in pool.imap_unordered(partial(process_shard, spec=spec, banned=banned, out_dir=out_dir, rej_dir=rej_dir), list(enumerate(shards)), chunksize=1):
            total.merge(st)
    stats = total.to_dict(); stats.update({"dataset": name, "lang": spec["lang"], "category": spec["category"], "input_files": len(files),
                                           "boilerplate_keys": len(banned), "seconds": round(time.time() - t0), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if batch:
        (out_dir / f"_stats.{batch}.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        # aggregate over batches
        agg = _cat_json(f"{remote_clean}/_stats.json", None) or {"docs_in": 0, "docs_out": 0, "chars_out": 0, "bytes_out_jsonl": 0, "lines_dropped_boilerplate": 0, "rejects": {}, "input_files": 0, "seconds": 0}
        for k in ("docs_in", "docs_out", "chars_out", "bytes_out_jsonl", "lines_dropped_boilerplate", "input_files", "seconds"):
            agg[k] = agg.get(k, 0) + stats[k]
        for k, v in stats["rejects"].items():
            agg["rejects"][k] = agg["rejects"].get(k, 0) + v
        agg.update({"dataset": name, "lang": spec["lang"], "category": spec["category"], "batches": len(ledger["batches"]) + 1, "finished_at": stats["finished_at"], "incremental": True})
        (out_dir / "_stats.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))
        ledger["processed"] += [u.split("/")[-1] for u, _ in all_files]; ledger["batches"].append({"batch": batch, "shards": len(all_files), "docs_out": stats["docs_out"], "at": stats["finished_at"]})
        (out_dir / "_processed.json").write_text(json.dumps(ledger, indent=1))
    else:
        (out_dir / "_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        gsutil("rm", "-r", "-q", remote_clean, check=False); gsutil("rm", "-r", "-q", remote_rej, check=False)
    log(f"[{name}] docs_in={stats['docs_in']:,} docs_out={stats['docs_out']:,} chars={stats['chars_out']:,} rejects={stats['rejects']} in {stats['seconds']}s")
    gcs_upload_dir(out_dir, remote_clean)
    if any(rej_dir.iterdir()):
        gcs_upload_dir(rej_dir, remote_rej)
    if not batch or final:
        (out_dir / "_DONE.json").write_text(json.dumps({"dataset": name, "docs_out": stats["docs_out"], "finished_at": stats["finished_at"]}))
        gsutil("cp", "-q", str(out_dir / "_DONE.json"), f"{remote_clean}/_DONE.json")
    shutil.rmtree(out_dir, ignore_errors=True); shutil.rmtree(rej_dir, ignore_errors=True)
    if not keep_input:
        shutil.rmtree(in_dir, ignore_errors=True)
    log(f"[{name}] uploaded to {remote_clean} — done" + (f" (batch {batch})" if batch else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*"); ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 2))
    ap.add_argument("--keep-input", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--final", action="store_true", help="incremental datasets: also write _DONE.json after this batch")
    a = ap.parse_args()
    if a.list:
        for k, v in D.items(): print(f"{k:16s} {v['lang']:5s} {v['category']:14s} {v['sources'][0][0]}")
        sys.exit()
    for ds in a.datasets:
        if ds not in D:
            log(f"unknown dataset {ds}; known: {', '.join(D)}"); sys.exit(2)
        run_dataset(ds, a.workers, a.keep_input, a.force, a.final)
