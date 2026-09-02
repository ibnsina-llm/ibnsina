#!/opt/pipe/bin/python3
"""Phase 3 — build the final training mix (train_v1_open) from scored/, deduped/ and clean/.

  p3_mix.py [--target 48e9] [--name train_v1_open] [--workers 60] [--shard-chars 1e9] [--dry-run]

Slices (share of --target tokens), sources, selection rules and epochs are in SLICES below; every knob is written to
mix_manifest.json. Token estimates use each dataset's bpe64k tokens/char from clean/_report_phase0.json (fallbacks by lang).
Selection is deterministic: a doc is kept iff xxh64(id) mod 1e6 < p*1e6 for its slice/rule, so reruns are reproducible.
Val: 0.5% of every slice by doc id (hash mod 1000 < 5) -> val/ (never trained on). Output: parquet shards (text, source,
id, slice, epoch) of ~--shard-chars characters, globally document-shuffled (bucket by hash, shuffle within bucket).
License gate: only sources with train_v1_open=true in licenses.json.
"""
from __future__ import annotations
import argparse, gzip, json, multiprocessing as mp, os, random, shutil, subprocess, sys, time
from pathlib import Path

import orjson, xxhash
import pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from common import BUCKET, DATA, gcs_list, gsutil, log

LIC = json.loads((Path(__file__).parent / "licenses.json").read_text())
_PRIV = Path(__file__).parent / "licenses_private.json"          # optional, gitignored: extra sources kept out of public docs
if _PRIV.exists(): LIC.update(json.loads(_PRIV.read_text()))
FALLBACK_TPC = {"fa": 0.25, "en": 0.23, "code": 0.29, "mixed": 0.28}

# ----------------------------------------------------------------------------- the mix (all knobs)
SLICES = {
    "fa_web":     {"share": 0.60, "layer": "scored", "sources": ["culturax_fa", "mc4_fa", "fineweb2_fa"], "rule": "web_score",
                   "web": {"keep_all_min": 1.5, "fill_lo": 1.0, "fill_nonnews_tokens": 6.0e9, "fill_news_tokens": 2.0e9, "news_prob": 0.5}},
    "en_edu":     {"share": 0.15, "layer": "clean", "sources": ["fineweb_edu", "openstax"], "rule": "random"},
    "code_pyts":  {"share": 0.05, "layer": "clean", "sources": ["starcoder_py", "starcoder_ts"], "rule": "random"},
    "code_other": {"share": 0.05, "layer": "clean", "sources": ["stackoverflow", "gh_persian_nlp", "gh_repos"], "rule": "random"},
    "math":       {"share": 0.05, "layer": "clean", "sources": ["open_web_math", "chap_textbooks", "chap_catalog"], "rule": "random",
                   "epochs": {"chap_textbooks": 3, "chap_catalog": 3}, "always": ["chap_textbooks", "chap_catalog"]},
    "fa_lit":     {"share": 0.05, "layer": "deduped", "sources": ["ganjoor", "poems", "fawikisource", "history"], "rule": "all",
                   "epochs": {"ganjoor": 4, "poems": 4, "fawikisource": 4, "history": 4}},
    "wiki":       {"share": 0.03, "layer": "mixed", "sources": ["fawiki", "enwiki"], "rule": "wiki",
                   "epochs": {"fawiki": 4, "enwiki": 1}, "layers": {"fawiki": "deduped", "enwiki": "clean"}},
    "parallel":   {"share": 0.02, "layer": "clean", "sources": ["opus_hplt", "opus_ccaligned", "opus_ccmatrix", "opus_opensubtitles", "opus_xlent",
                                                                "opus_wikimatrix", "opus100", "opus_globalvoices"], "rule": "random"},
    "synth":      {"share": 0.04, "layer": "clean", "sources": ["synth_v1"], "rule": "all"},   # rule=all: taken in full; share is bookkeeping only
}
_PS = Path(__file__).parent / "sources_private.json"              # optional, gitignored: {"slice": {"sources": [...], "epochs": {...}, "always": [...]}}
if _PS.exists():
    for _sl, _ext in json.loads(_PS.read_text()).items():
        SLICES[_sl]["sources"] += [x for x in _ext.get("sources", []) if x not in SLICES[_sl]["sources"]]
        SLICES[_sl].setdefault("epochs", {}).update(_ext.get("epochs", {})); SLICES[_sl].setdefault("always", [])
        SLICES[_sl]["always"] += _ext.get("always", [])
VAL_PERMILLE = 5


def h_id(s: str) -> int:
    return xxhash.xxh64_intdigest(s.encode())


# ----------------------------------------------------------------------------- availability
def read_json(url, default=None):
    r = subprocess.run(["gcloud", "storage", "cat", url], capture_output=True, text=True)
    try:
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else default
    except Exception:
        return default


def availability():
    """per source: layer, chars, docs, tokens/char, est tokens; web from scored/_bands.json."""
    rep = {r["dataset"]: r for r in (read_json(f"{BUCKET}/clean/_report_phase0.json", []) or [])}
    bands = read_json(f"{BUCKET}/scored/_bands.json", {}) or {}
    out = {}
    for sl, cfg in SLICES.items():
        for ds in cfg["sources"]:
            layer = cfg.get("layers", {}).get(ds, cfg["layer"])
            st = read_json(f"{BUCKET}/clean/{ds}/_stats.json", None)
            if st is None:
                log(f"[avail] {ds}: no clean stats — skipped"); continue
            tpc = rep.get(ds, {}).get("tok_per_char_bpe64k") or FALLBACK_TPC.get(st.get("lang", "fa"), 0.25)
            if layer == "scored" and ds in bands:
                chars = sum(bands[ds]["chars"]); docs = sum(bands[ds]["docs"])
            elif layer == "deduped":
                dr = read_json(f"{BUCKET}/deduped/_report.json", {}) or {}
                d = (dr.get("datasets") or {}).get(ds, {}); chars = d.get("chars_out", st["chars_out"]); docs = d.get("docs_out", st["docs_out"])
            else:
                chars, docs = st["chars_out"], st["docs_out"]
            out[ds] = {"slice": sl, "layer": layer, "chars": chars, "docs": docs, "tpc": tpc, "tokens": chars * tpc,
                       "license_class": LIC.get(ds, {}).get("class", "unknown"), "train_v1_open": LIC.get(ds, {}).get("train_v1_open", False),
                       "bands": bands.get(ds)}
    return out


# ----------------------------------------------------------------------------- plan: per-source keep probabilities
def plan(avail, target, name):
    P = {}; notes = []
    lit_avail = sum(v["tokens"] * SLICES["fa_lit"]["epochs"].get(ds, 1) for ds, v in avail.items() if v["slice"] == "fa_lit" and v["train_v1_open"])
    lit_short = max(0.0, SLICES["fa_lit"]["share"] * target - lit_avail)
    if lit_short > 0:
        notes.append(f"fa_lit has only {lit_avail/1e9:.2f}B tokens (with epochs) vs {SLICES['fa_lit']['share']*target/1e9:.1f}B target; "
                     f"{lit_short/1e9:.2f}B reallocated to fa_web")
    for sl, cfg in SLICES.items():
        budget = cfg["share"] * target + (lit_short if sl == "fa_web" else 0)
        srcs = {ds: v for ds, v in avail.items() if v["slice"] == sl and v["train_v1_open"]}
        skipped = [ds for ds, v in avail.items() if v["slice"] == sl and not v["train_v1_open"]]
        for ds in skipped:
            notes.append(f"{ds} excluded by license class {avail[ds]['license_class']}")
        if cfg["rule"] == "web_score":
            w = cfg["web"]; b = {ds: v["bands"] for ds, v in srcs.items()}
            def mass(lo, hi, news=None):
                t = 0.0
                for ds, v in srcs.items():
                    bb = v["bands"]; k0, k1 = int(lo / 0.25), int(hi / 0.25)
                    c = sum(bb["chars"][k0:k1]); n = sum(bb["news_chars"][k0:k1])
                    t += (c if news is None else (n if news else c - n)) * v["tpc"]
                return t
            core = mass(w["keep_all_min"], 5.0)
            fill_nonnews_avail = mass(w["fill_lo"], w["keep_all_min"], news=False); fill_news_avail = mass(w["fill_lo"], w["keep_all_min"], news=True)
            want_nonnews = max(0.0, min(w["fill_nonnews_tokens"], budget - core)); want_news = max(0.0, min(w["fill_news_tokens"], budget - core - want_nonnews))
            p_nonnews = min(1.0, want_nonnews / max(1, fill_nonnews_avail)); p_news = min(1.0, want_news / max(1, fill_news_avail))
            for ds in srcs:
                P[ds] = {"rule": "web_score", "keep_all_min": w["keep_all_min"], "fill_lo": w["fill_lo"], "news_prob": w["news_prob"],
                         "p_fill_nonnews": p_nonnews, "p_fill_news": p_news, "epochs": 1}
            notes.append(f"fa_web: core(score>={w['keep_all_min']})={core/1e9:.1f}B, fill non-news p={p_nonnews:.3f} (~{want_nonnews/1e9:.1f}B), "
                         f"fill news p={p_news:.3f} (~{want_news/1e9:.1f}B), budget {budget/1e9:.1f}B")
        elif cfg["rule"] == "all":
            for ds, v in srcs.items():
                P[ds] = {"rule": "all", "p": 1.0, "epochs": cfg.get("epochs", {}).get(ds, 1)}
        elif cfg["rule"] == "wiki":
            for ds, v in srcs.items():
                e = cfg["epochs"].get(ds, 1); P[ds] = {"rule": "all" if ds == "fawiki" else "random", "p": 1.0, "epochs": e}
            fa = avail.get("fawiki", {}).get("tokens", 0) * cfg["epochs"].get("fawiki", 1)
            en_budget = max(0.0, budget - fa); en_avail = avail.get("enwiki", {}).get("tokens", 1)
            if "enwiki" in P: P["enwiki"]["p"] = min(1.0, en_budget / en_avail)
            notes.append(f"wiki: fawiki x{cfg['epochs'].get('fawiki',1)} = {fa/1e9:.2f}B, enwiki p={P.get('enwiki',{}).get('p',0):.3f} (~{en_budget/1e9:.2f}B)")
        else:  # random pool: 'always' sources fully (with epochs), the rest share the remaining budget uniformly
            always = set(cfg.get("always", []))
            fixed = sum(v["tokens"] * cfg.get("epochs", {}).get(ds, 1) for ds, v in srcs.items() if ds in always)
            pool = sum(v["tokens"] for ds, v in srcs.items() if ds not in always)
            p = min(1.0, max(0.0, budget - fixed) / max(1, pool))
            for ds, v in srcs.items():
                P[ds] = {"rule": "all" if ds in always else "random", "p": 1.0 if ds in always else p, "epochs": cfg.get("epochs", {}).get(ds, 1)}
            notes.append(f"{sl}: budget {budget/1e9:.2f}B, fixed {fixed/1e9:.2f}B, pool p={p:.3f}")
    return P, notes


def keep_doc(doc, pl):
    """(keep, is_val). Deterministic on doc id."""
    did = doc["id"]; hv = h_id(did)
    is_val = (hv % 1000) < VAL_PERMILLE
    u = (hv // 1000) % 1_000_000 / 1_000_000.0
    if pl["rule"] == "all":
        return True, is_val
    if pl["rule"] == "random":
        return u < pl["p"], is_val
    m = doc.get("meta") or {}; s = float(m.get("edu_score", 0.0)); news = float(m.get("news_prob", 0.0)) >= pl["news_prob"]
    if s >= pl["keep_all_min"]:
        return True, is_val
    if s >= pl["fill_lo"]:
        return (u < (pl["p_fill_news"] if news else pl["p_fill_nonnews"])), is_val
    return False, is_val


# ----------------------------------------------------------------------------- phase A: select into buckets
def _lines(path, url):
    """gzip lines; on a truncated/corrupt local shard, re-fetch it from GCS once and start over."""
    for attempt in range(2):
        try:
            with gzip.open(path, "rb") as f:
                yield from f
            return
        except (EOFError, OSError, gzip.BadGzipFile) as e:
            if attempt:
                raise
            log(f"[select] corrupt local shard {path} ({type(e).__name__}) — re-fetching"); Path(path).unlink(missing_ok=True)
            gsutil("cp", "-q", url, path)


def select_file(args):
    path, ds, sl, pl, nb, tmp, wid, url = args
    ws = {}; counts = {"docs": 0, "chars": 0, "val_docs": 0, "val_chars": 0, "seen": 0}
    def w(kind, k):
        key = (kind, k)
        if key not in ws:
            ws[key] = open(tmp / kind / f"b{k:04d}_{wid:03d}_{Path(path).stem[:40]}.jsonl", "ab")
        return ws[key]
    if True:
        for line in _lines(path, url):
            d = orjson.loads(line); counts["seen"] += 1
            keep, is_val = keep_doc(d, pl)
            if not keep:
                continue
            row = {"text": d["text"], "source": ds, "id": d["id"], "slice": sl}
            n = len(d["text"])
            if is_val:
                w("val", h_id(d["id"] + "#v") % 4).write(orjson.dumps({**row, "epoch": 0}) + b"\n"); counts["val_docs"] += 1; counts["val_chars"] += n
                continue
            for e in range(pl["epochs"]):
                k = h_id(f"{d['id']}#e{e}") % nb
                w("train", k).write(orjson.dumps({**row, "epoch": e}) + b"\n"); counts["docs"] += 1; counts["chars"] += n
    for fh in ws.values():
        fh.close()
    return ds, counts


# ----------------------------------------------------------------------------- phase B: shuffle buckets -> parquet
def write_bucket(args):
    kind, k, tmp, out, seed = args
    parts = sorted((tmp / kind).glob(f"b{k:04d}_*.jsonl"))
    rows = []
    for p in parts:
        with open(p, "rb") as f:
            for line in f:
                rows.append(orjson.loads(line))
    if not rows:
        return kind, k, 0, 0
    random.Random(seed * 100003 + k).shuffle(rows)
    tbl = pa.table({c: [r[c] for r in rows] for c in ("text", "source", "id", "slice", "epoch")})
    dest = out / kind / f"shard_{k:05d}.parquet"; dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, dest, compression="zstd", row_group_size=2000)
    for p in parts:
        p.unlink()
    return kind, k, len(rows), sum(len(r["text"]) for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=48e9); ap.add_argument("--name", default="train_v1_open")
    ap.add_argument("--workers", type=int, default=60); ap.add_argument("--shard-chars", type=float, default=1e9)
    ap.add_argument("--seed", type=int, default=20260829); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--keep-tmp", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    avail = availability(); P, notes = plan(avail, a.target, a.name)
    est = {}
    for ds, pl in P.items():
        v = avail[ds]
        if pl["rule"] == "web_score":
            b = v["bands"]; k = int(pl["keep_all_min"] / 0.25); k0 = int(pl["fill_lo"] / 0.25)
            core = sum(b["chars"][k:]); nn = sum(b["chars"][k0:k]) - sum(b["news_chars"][k0:k]); nw = sum(b["news_chars"][k0:k])
            est[ds] = (core + nn * pl["p_fill_nonnews"] + nw * pl["p_fill_news"]) * v["tpc"]
        else:
            est[ds] = v["tokens"] * pl["p"] * pl["epochs"]
    for n in notes: log("[plan] " + n)
    for ds, pl in sorted(P.items(), key=lambda kv: -est[kv[0]]):
        log(f"[plan] {ds:18s} {avail[ds]['slice']:10s} {pl['rule']:9s} p={pl.get('p', pl.get('p_fill_nonnews', 1)):.3f} x{pl['epochs']}  ≈{est[ds]/1e9:6.2f}B  ({avail[ds]['license_class']})")
    log(f"[plan] total ≈ {sum(est.values())/1e9:.1f}B tokens (target {a.target/1e9:.0f}B)")
    if a.dry_run:
        return
    # stage inputs
    tmp = DATA / "mix" / "tmp"; out = DATA / "mix" / a.name
    shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree(out, ignore_errors=True)
    (tmp / "train").mkdir(parents=True); (tmp / "val").mkdir(parents=True); out.mkdir(parents=True)
    files = []
    for ds, pl in P.items():
        layer = avail[ds]["layer"]; d = DATA / layer / ds
        d.mkdir(parents=True, exist_ok=True); log(f"[stage] {layer}/{ds}")
        gsutil("rsync", "-r", "-q", f"{BUCKET}/{layer}/{ds}", str(d))   # always: rsync repairs stale/partial local shards by size
        files += [(str(p), ds, avail[ds]["slice"], pl, f"{BUCKET}/{layer}/{ds}/{p.name}") for p in sorted(d.glob("*part-*.jsonl.gz"))]
    total_chars = sum(est[ds] / avail[ds]["tpc"] for ds in P)
    nb = max(8, int(total_chars / a.shard_chars) + 1)
    log(f"[select] {len(files)} input shards -> {nb} train buckets (~{a.shard_chars/1e9:.1f}G chars each), {a.workers} workers")
    per = {}
    with mp.Pool(a.workers) as pool:
        for i, (ds, c) in enumerate(pool.imap_unordered(select_file, [(f, ds, sl, pl, nb, tmp, i % 1000, url) for i, (f, ds, sl, pl, url) in enumerate(files)], chunksize=1), 1):
            agg = per.setdefault(ds, {"docs": 0, "chars": 0, "val_docs": 0, "val_chars": 0, "seen": 0})
            for k2 in agg: agg[k2] += c[k2]
            if i % 500 == 0: log(f"[select] {i}/{len(files)}")
    log("[shuffle] writing parquet shards")
    shards = {"train": [], "val": []}
    jobs = [("train", k, tmp, out, a.seed) for k in range(nb)] + [("val", k, tmp, out, a.seed) for k in range(4)]
    with mp.Pool(min(a.workers, 24)) as pool:  # each bucket is ~1G of text in memory
        for kind, k, n, ch in pool.imap_unordered(write_bucket, jobs, chunksize=1):
            if n: shards[kind].append({"shard": f"{kind}/shard_{k:05d}.parquet", "docs": n, "chars": ch})
    # manifest
    tok = lambda ds, ch: ch * avail[ds]["tpc"]
    sources = {ds: {**{k: v for k, v in avail[ds].items() if k != "bands"}, "plan": P[ds], "selected_docs": per.get(ds, {}).get("docs", 0),
                    "selected_chars": per.get(ds, {}).get("chars", 0), "selected_tokens_est": tok(ds, per.get(ds, {}).get("chars", 0)),
                    "val_docs": per.get(ds, {}).get("val_docs", 0), "val_chars": per.get(ds, {}).get("val_chars", 0),
                    "license": LIC.get(ds, {}).get("license"), "license_note": LIC.get(ds, {}).get("note")} for ds in P}
    by_slice = {}
    for ds, s in sources.items():
        b = by_slice.setdefault(s["slice"], {"tokens_est": 0.0, "docs": 0, "sources": []}); b["tokens_est"] += s["selected_tokens_est"]; b["docs"] += s["selected_docs"]; b["sources"].append(ds)
    tot = sum(b["tokens_est"] for b in by_slice.values())
    for b in by_slice.values(): b["share"] = round(b["tokens_est"] / max(1, tot), 4)
    manifest = {"name": a.name, "target_tokens": a.target, "total_tokens_est": tot, "seed": a.seed, "val_permille": VAL_PERMILLE,
                "tokenizer_proxy": "bpe64k trained on clean/ sample (clean/_tokenizer/bpe64k.json); real nanochat tokenizer will differ ~±15%",
                "slices_config": SLICES, "plan_notes": notes, "by_slice": by_slice, "sources": sources, "shards": shards,
                "excluded_by_license": [ds for ds, v in avail.items() if not v["train_v1_open"]],
                "absent": ["matina (license/gate)", "oscar-2301 (gate)", "tlpc (gate)", "konkur/konkur_answers (proprietary, excluded)"],
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0)}
    (out / "mix_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    log(f"[done] {tot/1e9:.2f}B tokens, {sum(s['docs'] for s in shards['train']):,} train docs in {len(shards['train'])} shards; val {sum(s['docs'] for s in shards['val']):,} docs; {(time.time()-t0)/60:.0f} min")
    gsutil("rsync", "-r", "-q", str(out), f"{BUCKET}/{a.name}")
    if not a.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"[done] uploaded to {BUCKET}/{a.name}")


if __name__ == "__main__":
    main()
