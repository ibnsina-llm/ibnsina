#!/opt/pipe/bin/python3
"""Phase 1 — dedup Persian WEB datasets against each other and against CURATED references.

RESUME is ADDITIVE: a dataset added later is appended after all previously-cached files (their ranks — and the finished
signature stage — stay valid); only the new files get signatures, then buckets/cluster/filter recompute. Limitations:
(1) an appended WEB dataset has the lowest priority (existing web wins ties); (2) the exact-dup pass is not recomputed for
already-written web shards, so a web doc identical to a doc in a *later-added* curated set survives exact dedup (minhash
still sees it, but curated is never dropped and lower rank wins, so both copies remain). Use --force for a clean run.

  p1_dedup.py --web matina culturax_fa mc4_fa fineweb2_fa --curated fawiki fawikisource poems ... [--workers 30]
              [--skip-minhash] [--tokens-per-char 0.30] [--stock-cluster] [--local] [--force] [--keep-work]

clean/{ds}/part-*.jsonl.gz  ->  deduped/{ds}/part-*.jsonl.gz, rejects/dedup/{ds}/rej-*.jsonl.gz, deduped/_report.json, _DONE.json
Doc schema passes through untouched: {id, text, source, url, lang, meta{category,...}}. Rejects: {id, source, reason, url, text[:500]}.

PRIORITY. Every input file gets a global position: curated files first (--curated order), then web files in --web order
(first wins, e.g. matina > culturax_fa > mc4_fa > fineweb2_fa), docs in file order. Both steps keep the EARLIEST doc of a
duplicate group and drop the later ones. Curated docs are never dropped, so a web doc matching a curated doc is dropped.
  A. exact    xxh64 of normalised text (lower-case, ZWNJ joined, punctuation stripped, whitespace collapsed). numpy only:
              one uint64 array for all ~1e8 docs, stable argsort, first occurrence of each hash wins. Reason exact_dup_of:<ds>.
  B. minhash  datatrove MinHash LSH on 5-word shingles, 14 buckets x 8 hashes, 64-bit xxhash. LSH threshold (1/14)^(1/8)=0.72;
              P(detect | jaccard 0.8) = 1-(1-0.8^8)^14 = 0.92, P(0.7) = 0.56, P(0.6) = 0.21 — the FineWeb setting for ~0.8.
              The reader gets a paths file written in priority order with tasks == number of files, so datatrove's file_id
              (= task rank) IS the priority rank.
              datatrove's MinhashDedupCluster does NOT keep the lowest (file_id, doc_id): its union-find is union-BY-SIZE
              (smaller set joins the larger; only ties keep the lower id), so a big cluster can survive through a low-priority
              copy and even a curated doc can be scheduled for removal. Default here: datatrove stages 1+2
              (MinhashDedupSignature, MinhashDedupBuckets) produce the duplicate pairs; a numpy min-label connected-components
              pass replaces stage 3 — the survivor is always the lowest (file_id, doc_id) of its cluster and curated files get
              no removals — and writes the same {file_id:06d}.remove files (sorted <u4 doc indices) that datatrove's
              MinhashDedupFilter (stage 4) consumes. --stock-cluster runs datatrove's MinhashDedupCluster instead (curated
              .remove files are deleted afterwards; no minhash overlap matrix in that mode). Reason minhash_dup.

datatrove API verified against tag v0.4.0 and main (0.10.0) of github.com/huggingface/datatrove: MinhashConfig(n_grams,
num_buckets, hashes_per_bucket, hash_config=HashConfig(precision)); MinhashDedupSignature(output_folder, config, language);
MinhashDedupBuckets(input_folder, output_folder, config) [tasks % num_buckets == 0]; MinhashDedupCluster(input_folder,
output_folder, config) [tasks == 1]; MinhashDedupFilter(input_folder, exclusion_writer); JsonlReader(data_folder, paths_file=,
adapter=(self, data, path, id_in_file), add_file_path=); JsonlWriter(output_folder, output_filename="${tag}", adapter=(self,
doc)); LocalPipelineExecutor(pipeline, tasks, workers, logging_dir, depends); get_shard_from_paths_file hands line i to rank i
when tasks == number of lines. Sig files bucket_{b:03d}/{rank:05d}.minhash.sig; pairs *.dups of <4I (file1, doc1, file2, doc2);
removals {file:06d}.remove of <I. datatrove[processing] installs no spaCy (Persian is assigned SpaCyTokenizer("fa")), so a
whitespace WordTokenizer instance is passed as `language=` (load_word_tokenizer returns WordTokenizer instances unchanged).
"""
from __future__ import annotations
import argparse, gzip, json, multiprocessing as mp, re, shutil, sys, time
from functools import partial
from pathlib import Path

import numpy as np
import orjson
import xxhash

sys.path.insert(0, str(Path(__file__).parent))
from common import BUCKET, DATA, gcs_list, gcs_upload_dir, gsutil, log

try:
    from datatrove.utils.word_tokenizers import WordTokenizer
except ImportError:  # datatrove is only needed for step B; --skip-minhash works without it
    WordTokenizer = object

WORK = DATA / "dedup"
NGRAMS, NUM_BUCKETS, HASHES_PER_BUCKET, PRECISION = 5, 14, 8, 64
_PUNCT = re.compile(r"[^\w\s]|_")


# ----------------------------------------------------------------------------- normalisation
def tokens(text: str) -> list[str]:
    """Words with ZWNJ joined and punctuation/symbols (incl. Persian ، ؛ ؟) stripped — shared by both steps."""
    return _PUNCT.sub(" ", text.replace("‌", "")).split()


def norm_exact(text: str) -> str:
    return " ".join(tokens(text.lower()))


class WsTok(WordTokenizer):
    """Whitespace word tokenizer for MinhashDedupSignature (datatrove applies simplify_text before calling it)."""
    def word_tokenize(self, text): return tokens(text)
    def sent_tokenize(self, text): return [text]
    def span_tokenize(self, text): return [(0, len(text))]


# ----------------------------------------------------------------------------- inputs
def stage(datasets, role, local):
    """rsync clean/{ds} from GCS (unless --local); returns [(ds, role, Path)] in priority order. Missing datasets are skipped."""
    files = []
    for ds in datasets:
        d = DATA / "clean" / ds
        if not local:
            if not gcs_list(f"{BUCKET}/clean/{ds}", "*part-*.jsonl.gz"):
                log(f"[{ds}] not in GCS (yet) — skipped; note it as ABSENT in the STOP 2 report"); continue
            d.mkdir(parents=True, exist_ok=True); gsutil("rsync", "-r", "-q", f"{BUCKET}/clean/{ds}", str(d))
        parts = sorted(d.glob("*part-*.jsonl.gz"))  # incremental web datasets use b###-part-* names
        if not parts:
            log(f"[{ds}] no clean shards under {d} — skipped"); continue
        log(f"[{ds}] {len(parts)} shards ({role})"); files += [(ds, role, p) for p in parts]
    return files


# ----------------------------------------------------------------------------- step A: exact
def hash_file(i, path, cache):
    """(xxh64 of normalised text, char length) per doc of one shard; cached as npz so a restart skips the read."""
    c = cache / f"{i:05d}.npz"
    if c.exists():
        z = np.load(c); return z["h"], z["n"]
    h, n = [], []
    with gzip.open(path, "rb") as f:
        for line in f:
            t = orjson.loads(line).get("text") or ""
            h.append(xxhash.xxh64_intdigest(norm_exact(t))); n.append(len(t))
    h, n = np.asarray(h, dtype=np.uint64), np.asarray(n, dtype=np.int32)
    np.savez(c.with_suffix(".tmp.npz"), h=h, n=n); c.with_suffix(".tmp.npz").rename(c)
    return h, n


def exact_resolve(H, n_curated_docs):
    """First occurrence (priority order) of each hash wins. Returns dup_of: global index of the winner, -1 if kept."""
    N = len(H); dup_of = np.full(N, -1, dtype=np.int64)
    if N == 0:
        return dup_of
    order = np.argsort(H, kind="stable"); Hs = H[order]
    first = np.empty(N, dtype=bool); first[0] = True; first[1:] = Hs[1:] != Hs[:-1]
    run_start = np.maximum.accumulate(np.where(first, np.arange(N), 0))  # sorted position of each run's first element
    dup_of[order[~first]] = order[run_start[~first]]
    dup_of[:n_curated_docs] = -1  # curated never dropped (they precede all web docs, so only curated-curated repeats hit this)
    return dup_of


def rewrite_file(job, out_root, rej_root, names):
    """Copy the exact-kept lines of a web shard verbatim; dropped docs go to rejects with reason exact_dup_of:<dataset>."""
    ds, path, keep, dup_ds = job
    out, rej = out_root / ds / path.name, rej_root / ds / f"rej-exact-{path.name}"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True); rej.parent.mkdir(parents=True, exist_ok=True)
    tmp_o, tmp_r = out.with_suffix(".tmp"), rej.with_suffix(".tmp")
    with gzip.open(path, "rb") as f, gzip.open(tmp_o, "wb", 4) as fo, gzip.open(tmp_r, "wb", 4) as fr:
        for i, line in enumerate(f):
            if keep[i]:
                fo.write(line); continue
            d = orjson.loads(line)
            fr.write(orjson.dumps({"id": d.get("id"), "source": ds, "reason": f"exact_dup_of:{names[dup_ds[i]]}", "url": d.get("url"),
                                   "text": (d.get("text") or "")[:500]}) + b"\n")
    tmp_r.unlink() if keep.all() else tmp_r.rename(rej)
    tmp_o.rename(out)


# ----------------------------------------------------------------------------- step B: minhash
def min_label_components(a, b, n):
    """Connected components of edges (a, b) over nodes [0, n); returns lab with lab[x] = smallest node id in x's component.
    Invariant lab[x] <= x, so each component's root is its minimum. Per round: hook the larger root of every straddling
    edge onto the smaller, then pointer-jump until idempotent. O(log n) rounds in practice for LSH clusters."""
    lab = np.arange(n, dtype=np.int64)
    while True:
        la, lb = lab[a], lab[b]
        m = la != lb
        if not m.any():
            return lab
        a, b, la, lb = a[m], b[m], la[m], lb[m]
        np.minimum.at(lab, np.maximum(la, lb), np.minimum(la, lb))
        while True:
            nxt = lab[lab]
            if np.array_equal(nxt, lab):
                break
            lab = nxt


def cluster_min_priority(buk, rem, n_curated_files):
    """Replacement for MinhashDedupCluster: keep the lowest (file_id, doc_id) of every cluster, never remove curated docs.
    Writes {file_id:06d}.remove (sorted <u4 doc indices) for web files; returns (file_id of removed, file_id of survivor)."""
    E = [np.fromfile(p, dtype="<u4").reshape(-1, 4).astype(np.int64) for p in sorted(buk.glob("*.dups"))]
    E = np.concatenate(E) if E else np.zeros((0, 4), np.int64)
    key = np.concatenate([E[:, 0] << 32 | E[:, 1], E[:, 2] << 32 | E[:, 3]])  # (file_id, doc_id) -> int64, priority-ordered
    nodes, inv = np.unique(key, return_inverse=True)
    lab = min_label_components(inv[:len(E)], inv[len(E):], len(nodes))
    files, docs, rep_files = nodes >> 32, nodes & 0xFFFFFFFF, nodes[lab] >> 32
    removed = (lab != np.arange(len(nodes))) & (files >= n_curated_files)
    rem.mkdir(parents=True, exist_ok=True)
    for f in np.unique(files[removed]):
        lo, hi = np.searchsorted(files, [f, f + 1])
        docs[lo:hi][removed[lo:hi]].astype("<u4").tofile(rem / f"{int(f):06d}.remove")
    log(f"[minhash] {len(E):,} pairs, {len(nodes):,} docs in clusters, {int(removed.sum()):,} web docs to remove")
    return files[removed], rep_files[removed]


def _read_adapter(self, data, path, id_in_file):
    """clean-schema line -> Document. _ds/_fn (dataset, shard stem) drive the output filename template; stripped on write."""
    ds, fn = path.split("/")[-2:]
    meta = {k: v for k, v in data.items() if k not in ("text", "id")}
    return {"text": data.get("text") or "", "id": str(data.get("id", f"{path}/{id_in_file}")),
            "metadata": meta | {"_ds": ds, "_fn": fn.removesuffix(".jsonl.gz")}}


def _write_adapter(self, doc):
    return {"id": doc.id, "text": doc.text, **{k: v for k, v in doc.metadata.items() if not k.startswith("_")}}


def _reject_adapter(self, doc):
    return {"id": doc.id, "source": doc.metadata.get("source"), "reason": "minhash_dup", "url": doc.metadata.get("url"), "text": doc.text[:500]}


def run_minhash(paths, n_curated_files, workers, stock, out, rej):
    """datatrove stages 1, 2, (3), 4 over `paths` (relative to DATA, priority order). Returns (removed, survivor) file_ids or None."""
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.dedup.minhash import (MinhashConfig, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter,
                                                  MinhashDedupSignature)
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter
    from datatrove.utils.hashing import HashConfig

    cfg = MinhashConfig(n_grams=NGRAMS, num_buckets=NUM_BUCKETS, hashes_per_bucket=HASHES_PER_BUCKET, hash_config=HashConfig(precision=PRECISION))
    mh = WORK / "mh"; sig, buk, rem, logs = mh / "sigs", mh / "buckets", mh / "remove", mh / "logs"
    mh.mkdir(parents=True, exist_ok=True); (mh / "paths.txt").write_text("".join(p + "\n" for p in paths))
    n = len(paths)
    reader = lambda: JsonlReader(str(DATA), paths_file=str(mh / "paths.txt"), adapter=_read_adapter, add_file_path=False)
    s1 = LocalPipelineExecutor([reader(), MinhashDedupSignature(str(sig), config=cfg, language=WsTok())],
                               tasks=n, workers=workers, logging_dir=str(logs / "sig"))
    bt = NUM_BUCKETS * max(1, workers // NUM_BUCKETS)  # 30 workers -> 28 tasks = 2 hash ranges per bucket
    s2 = LocalPipelineExecutor([MinhashDedupBuckets(str(sig), str(buk), config=cfg)],
                               tasks=bt, workers=min(bt, workers), logging_dir=str(logs / "buckets"), depends=s1)
    log(f"[minhash] stages 1+2: {n} files, {bt} bucket tasks, {workers} workers"); s2.run()
    ov = None
    if stock:
        LocalPipelineExecutor([MinhashDedupCluster(str(buk), str(rem), config=cfg)], tasks=1, logging_dir=str(logs / "cluster")).run()
        for f in range(n_curated_files):
            (rem / f"{f:06d}.remove").unlink(missing_ok=True)  # curated never dropped
    elif (rem / "_overlap.npy").exists():
        ov = np.load(rem / "_overlap.npy")
    else:
        ov = np.stack(cluster_min_priority(buk, rem, n_curated_files))
        np.save(rem / "_overlap.tmp.npy", ov); (rem / "_overlap.tmp.npy").rename(rem / "_overlap.npy")
    s4 = LocalPipelineExecutor([reader(),
                                MinhashDedupFilter(str(rem), exclusion_writer=JsonlWriter(str(rej), output_filename="${_ds}/rej-minhash-${_fn}.jsonl.gz",
                                                                                          adapter=_reject_adapter)),
                                JsonlWriter(str(out), output_filename="${_ds}/${_fn}.jsonl.gz", adapter=_write_adapter)],
                               tasks=n, workers=workers, logging_dir=str(logs / "filter"))
    log("[minhash] stage 4: filter + write"); s4.run()
    return ov


# ----------------------------------------------------------------------------- driver
def main(a):
    t0 = time.time()
    if not a.force and not a.local and gcs_list(f"{BUCKET}/deduped", "_DONE.json"):
        log("deduped/_DONE.json exists in GCS — nothing to do (use --force)"); return
    files = stage(a.curated, "curated", a.local) + stage(a.web, "web", a.local)  # global priority order
    n_cf = sum(r == "curated" for _, r, _ in files)
    if n_cf == len(files):
        log("no web shards — nothing to do"); return
    manifest = [[str(p), p.stat().st_size] for _, _, p in files]; mf = WORK / "_manifest.json"
    if mf.exists() and not a.force and json.loads(mf.read_text()) != manifest:
        # ADDITIVE RESUME: finished stages are keyed by file rank, so existing files must keep their ranks.
        # New files (new datasets) are appended after them; only their signatures are computed, then the cheap
        # downstream stages (buckets -> cluster -> filter) re-run over everything. Removed/changed files -> abort.
        old = json.loads(mf.read_text()); old_paths = [m[0] for m in old]; by_path = {str(p): (ds, role, p) for ds, role, p in files}
        cur_sizes = {str(p): p.stat().st_size for _, _, p in files}
        changed = [pth for pth, sz in old if pth not in cur_sizes or cur_sizes[pth] != sz]
        if changed:
            log(f"{len(changed)} previously-cached input files are missing or changed (e.g. {changed[0]}). Re-run with --force to reset the work dir ({WORK})."); sys.exit(3)
        new_files = [f for f in files if str(f[2]) not in set(old_paths)]
        files = [by_path[pth] for pth in old_paths] + new_files          # old order first, then appended
        manifest = [[str(p), p.stat().st_size] for _, _, p in files]
        log(f"additive resume: {len(old_paths)} cached files keep their ranks; {len(new_files)} new files appended "
            f"({', '.join(sorted({f[0] for f in new_files}))}); buckets/cluster/filter will be recomputed")
        for sub in ("mh/buckets", "mh/remove", "mh/logs/buckets", "mh/logs/cluster", "mh/logs/filter", "out", "exact_resolve"):
            shutil.rmtree(WORK / sub, ignore_errors=True)
        mf.write_text(json.dumps(manifest))
    if a.force or not mf.exists():
        log("fresh run (or --force): resetting work dir"); shutil.rmtree(WORK, ignore_errors=True)
        WORK.mkdir(parents=True); mf.write_text(json.dumps(manifest))
    names = list(dict.fromkeys(ds for ds, _, _ in files)); K = len(names)
    file_ds = np.array([names.index(ds) for ds, _, _ in files]); role = {ds: r for ds, r, _ in files}
    ex, rej, out = WORK / "exact", WORK / "rejects", WORK / "out"

    # --- A. exact
    cache = WORK / "hashes"; cache.mkdir(exist_ok=True)
    log(f"[exact] hashing {len(files)} shards with {a.workers} workers")
    with mp.Pool(a.workers) as pool:
        hn = pool.starmap(partial(hash_file, cache=cache), [(i, p) for i, (_, _, p) in enumerate(files)], chunksize=1)
    lens = [n for _, n in hn]; counts = np.array([len(h) for h, _ in hn]); off = np.concatenate([[0], np.cumsum(counts)])
    dup_of = exact_resolve(np.concatenate([h for h, _ in hn]), int(off[n_cf])); del hn
    dup_ds = np.where(dup_of >= 0, file_ds[np.searchsorted(off, dup_of, side="right") - 1], -1).astype(np.int16); del dup_of
    keep = [dup_ds[off[i]:off[i + 1]] < 0 for i in range(len(files))]
    ov_exact = np.zeros((K, K), np.int64)
    for i in range(n_cf, len(files)):
        d = dup_ds[off[i]:off[i + 1]]; ov_exact[file_ds[i]] += np.bincount(d[d >= 0], minlength=K)
    log(f"[exact] {int(counts.sum()):,} docs, {int((dup_ds >= 0).sum()):,} web docs are exact dups")
    jobs = [(ds, p, keep[i], dup_ds[off[i]:off[i + 1]]) for i, (ds, r, p) in enumerate(files) if r == "web"]
    with mp.Pool(a.workers) as pool:
        pool.map(partial(rewrite_file, out_root=ex, rej_root=rej, names=names), jobs, chunksize=1)

    # --- B. minhash (curated shards are read from clean/, web shards from the exact-deduped copies)
    paths = [str((p if r == "curated" else ex / ds / p.name).relative_to(DATA)) for ds, r, p in files]
    mh_rm, mh_ch, ov_mh = np.zeros(len(files), np.int64), np.zeros(len(files), np.int64), None
    if a.skip_minhash:
        for i, (ds, _, p) in enumerate(files):
            dst = out / ds / p.name; dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(DATA / paths[i], dst)
    else:
        ov = run_minhash(paths, n_cf, a.workers, a.stock_cluster, out, rej)
        if ov is not None:
            ov_mh = np.zeros((K, K), np.int64); np.add.at(ov_mh, (file_ds[ov[0]], file_ds[ov[1]]), 1)
        for i in range(n_cf, len(files)):
            r = WORK / "mh" / "remove" / f"{i:06d}.remove"
            if r.exists():  # VERIFY: datatrove doc index == line index (the reader only skips empty-text lines; none after phase 0)
                idx = np.fromfile(r, dtype="<u4"); lk = lens[i][keep[i]]
                mh_rm[i] = len(idx); mh_ch[i] = int(lk[idx[idx < len(lk)]].sum())

    # --- report
    per = {}
    for k, ds in enumerate(names):
        fi = np.flatnonzero(file_ds == k)
        din, cin = int(counts[fi].sum()), sum(int(lens[i].sum()) for i in fi)
        erm, ech = sum(int((~keep[i]).sum()) for i in fi), sum(int(lens[i][~keep[i]].sum()) for i in fi)
        mrm, mch = int(mh_rm[fi].sum()), int(mh_ch[fi].sum())
        per[ds] = {"role": role[ds], "shards": len(fi), "docs_in": din, "exact_removed": erm, "minhash_removed": mrm,
                   "docs_out": din - erm - mrm, "chars_in": cin, "chars_out": cin - ech - mch}
    matrix = lambda m: None if m is None else {A: {B: int(m[i, j]) for j, B in enumerate(names) if m[i, j]}
                                               for i, A in enumerate(names) if role[A] == "web"}
    web = [ds for ds in names if role[ds] == "web"]; wc = sum(per[d]["chars_out"] for d in web)
    report = {"datasets": per, "priority": names, "overlap": {"exact": matrix(ov_exact), "minhash": matrix(ov_mh)},
              "web_totals": {"docs_in": sum(per[d]["docs_in"] for d in web), "docs_out": sum(per[d]["docs_out"] for d in web),
                             "chars_out": wc, "tokens_per_char": a.tokens_per_char, "tokens_est": int(wc * a.tokens_per_char)},
              "minhash": {"skipped": a.skip_minhash, "n_grams": NGRAMS, "num_buckets": NUM_BUCKETS, "hashes_per_bucket": HASHES_PER_BUCKET,
                          "precision": PRECISION, "lsh_threshold": round((1 / NUM_BUCKETS) ** (1 / HASHES_PER_BUCKET), 3),
                          "cluster": "datatrove_union_by_size" if a.stock_cluster else "min_priority"},
              "seconds": round(time.time() - t0), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.mkdir(parents=True, exist_ok=True); (out / "_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for ds in names:
        log(f"[{ds}] " + " ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}" for k, v in per[ds].items()))
    log(f"web: docs_out={report['web_totals']['docs_out']:,} chars_out={wc:,} tokens_est={report['web_totals']['tokens_est']:,}")

    # --- upload
    if not a.local:
        gsutil("rm", "-r", "-q", f"{BUCKET}/deduped", check=False); gsutil("rm", "-r", "-q", f"{BUCKET}/rejects/dedup", check=False)
        gcs_upload_dir(out, f"{BUCKET}/deduped")
        if any(rej.rglob("*.jsonl.gz")):
            gcs_upload_dir(rej, f"{BUCKET}/rejects/dedup")
        (out / "_DONE.json").write_text(json.dumps({"docs_out": {d: per[d]["docs_out"] for d in names}, "finished_at": report["finished_at"]}))
        gsutil("cp", "-q", str(out / "_DONE.json"), f"{BUCKET}/deduped/_DONE.json")
        log(f"uploaded to {BUCKET}/deduped and {BUCKET}/rejects/dedup")
        if not a.keep_work:
            shutil.rmtree(WORK, ignore_errors=True)
    log(f"phase 1 done in {report['seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 1 dedup (exact + minhash) of Persian web datasets against curated references")
    ap.add_argument("--web", nargs="+", required=True, help="web datasets, highest priority first (first wins)")
    ap.add_argument("--curated", nargs="*", default=[], help="reference datasets: never dropped, web docs matching them are")
    ap.add_argument("--workers", type=int, default=30); ap.add_argument("--tokens-per-char", type=float, default=0.30)
    ap.add_argument("--skip-minhash", action="store_true", help="exact dedup only")
    ap.add_argument("--stock-cluster", action="store_true", help="use datatrove's MinhashDedupCluster (union-by-size) instead of min-priority")
    ap.add_argument("--local", action="store_true", help="no GCS: read $PIPE_DATA/clean, leave outputs in $PIPE_DATA/dedup/out")
    ap.add_argument("--force", action="store_true", help="ignore GCS _DONE.json and reset the local work dir")
    ap.add_argument("--keep-work", action="store_true", help="keep $PIPE_DATA/dedup after upload")
    main(ap.parse_args())
