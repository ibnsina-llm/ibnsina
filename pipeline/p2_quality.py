#!/opt/pipe/bin/python3
"""Phase 2 — FineWeb-Edu-style educational-value scoring for PERSIAN web docs.

  p2_quality.py {sample|label|train|score|report|all} [--datasets culturax_fa mc4_fa fineweb2_fa] [--workers N] [--no-llm] ...

deduped/{dataset}/part-*.jsonl.gz  ->  scored/{dataset}/part-*.jsonl.gz   (same shard names; NO drops in this phase)
  adds meta.edu_score (E[score] over fastText probs, float 0-5), meta.edu_int (argmax), meta.news_prob,
       meta.heur = {mean_line_len, symbol_ratio, digit_ratio, doc_chars, short_line_ratio}
p2/ artifacts: sample_100k.jsonl.gz, sample_10k.jsonl.gz, labels_10k.jsonl, edu_fa.bin, news_fa.bin, train_metrics.json, score_agg/, report.md
English fineweb_edu already carries meta.score from the FineWeb-Edu classifier and is NOT re-scored here.

Vertex AI auth = Application Default Credentials of the VM service account — the SA needs roles/aiplatform.user.
`/opt/pipe/bin/pip install google-genai` before `label`.  `--no-llm` is a documented FALLBACK: keyword weak labels (see weak_label()).
"""
from __future__ import annotations
import argparse, asyncio, gzip, heapq, json, math, multiprocessing as mp, os, random, re, shutil, sys, time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from common import BUCKET, DATA, SHORT_LINE, gcs_download, gcs_list, gcs_upload_dir, gsutil, log

WEB_FA = ["culturax_fa", "mc4_fa", "fineweb2_fa"]
P2 = DATA / "p2"
FT_HEAD = 4000        # chars of text fed to fastText (train AND score, so the distributions match)
PROMPT_HEAD = 1500    # chars of text shown to Gemini
NBINS, N_EXCERPT = 20, 20
HEUR_KEYS = ["mean_line_len", "symbol_ratio", "digit_ratio", "doc_chars", "short_line_ratio"]
PRICE_IN, PRICE_OUT = 0.30, 2.50   # VERIFY: $/1M tokens for gemini-2.5-flash on Vertex (Aug 2026 list); only used for the cost log line

RUBRIC = """You grade PERSIAN (Farsi) web pages for EDUCATIONAL VALUE on a 0-5 scale, in the spirit of the FineWeb-Edu classifier.
0 = no educational value: spam, ads, product listings, navigation/menus, boilerplate, comment threads, gibberish, adult content.
1 = some basic information on a topic but mostly non-educational: news, blogs, forums, marketing, opinion, very shallow.
2 = addresses educational topics but is incoherent, incomplete, poorly structured, or mixed with junk/ads/navigation.
3 = clearly useful educational content at school or university level: explains a concept, method or fact coherently, mostly free of junk.
4 = highly relevant and well organized, like a textbook section, lecture note, tutorial or reference article; minimal noise.
5 = outstanding: expert, complete, well-structured teaching material (definitions, examples, exercises), clearly written for learners.
Separately decide whether the page is NEWS/JOURNALISM (reporting current events in press-agency or newspaper style); this is independent of the score.
Judge the content, not fluency alone; do not reward length; the text may be truncated. Reply with JSON only:
{"score": <int 0-5>, "news": <true|false>, "reason": "<at most 20 words>"}"""

# --- FALLBACK (--no-llm): educational vocabulary for weak labels. Much noisier than Gemini labels; documented, not recommended.
EDU_KW = re.compile("|".join(["درس", "آموزش", "دانشگاه", "فصل", "تعریف", "قضیه", "مثال", "تمرین", "مسئله", "فرمول", "معادله", "نظریه",
                              "مفهوم", "دانش‌آموز", "دانشجو", "استاد", "پژوهش", "اثبات", "تابع", "سلول"]))
NEWS_KW = re.compile("|".join(["خبرگزاری", "به گزارش", "اعلام کرد", "خبرنگار", "روزنامه", "ایسنا", "ایرنا", "تسنیم", "گفت:"]))


# ----------------------------------------------------------------------------- io / staging / markers
def read_jsonl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rb") as f:
        for line in f:
            if line.strip():
                try: yield orjson.loads(line)
                except Exception: pass


def stage(ds: str) -> list[str]:
    """gs://.../deduped/{ds}/part-*.jsonl.gz -> /data/deduped/{ds}/ (files already present are kept)."""
    d = DATA / "deduped" / ds
    urls = [u for u, _ in gcs_list(f"{BUCKET}/deduped/{ds}", "*part-*.jsonl.gz")]
    missing = [u for u in urls if not (d / u.split("/")[-1]).exists()]
    if missing:
        d.mkdir(parents=True, exist_ok=True); log(f"[{ds}] staging {len(missing)}/{len(urls)} shards"); gsutil("cp", "-q", "-n", *missing, str(d))
    files = sorted(str(p) for p in d.glob("*part-*.jsonl.gz")) if d.exists() else []
    if not files: log(f"[{ds}] no deduped shards locally or under {BUCKET}/deduped/{ds}")
    return files


def put(*names):
    for n in names: gsutil("cp", "-q", str(P2 / n), f"{BUCKET}/p2/{n}")


def fetch(name) -> Path:
    """Local p2 artifact; pulled from GCS only if it exists there (a miss just means 'not produced yet')."""
    p = P2 / name
    if not p.exists() and gcs_list(f"{BUCKET}/p2", name):
        gcs_download(f"{BUCKET}/p2/{name}", p)
    return p


def is_done(step) -> bool: return bool(gcs_list(f"{BUCKET}/p2", f"_DONE_{step}.json"))


def mark_done(step, info):
    (P2 / f"_DONE_{step}.json").write_text(json.dumps({"step": step, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info}, ensure_ascii=False))
    put(f"_DONE_{step}.json")


# ----------------------------------------------------------------------------- text features
_SYM, _DIG, _WS = re.compile(r"[^\w\s‌]"), re.compile(r"\d"), re.compile(r"\s+")


def heuristics(text: str) -> dict:
    n = len(text) or 1; lines = [l for l in text.split("\n") if l.strip()] or [""]
    return {"mean_line_len": round(sum(map(len, lines)) / len(lines), 1), "symbol_ratio": round(len(_SYM.findall(text)) / n, 4),
            "digit_ratio": round(len(_DIG.findall(text)) / n, 4), "doc_chars": len(text),
            "short_line_ratio": round(sum(len(l.strip()) < SHORT_LINE for l in lines) / len(lines), 3)}


def ft_text(text: str) -> str:
    """fastText input: head of doc, lowercased, all whitespace (incl. newlines) -> single spaces. Whitespace tokenization is fastText's own."""
    return _WS.sub(" ", text[:FT_HEAD].lower()).strip()


def weak_label(text: str) -> dict:
    """FALLBACK for --no-llm: density of educational vocabulary (hits per 1000 chars) + layout heuristics -> pseudo score 0-5."""
    h = heuristics(text); dens = len(EDU_KW.findall(text)) * 1000 / max(1, h["doc_chars"])
    if h["symbol_ratio"] > 0.12 or h["short_line_ratio"] > 0.6 or h["doc_chars"] < 300 or dens == 0: score = 0
    else: score = 1 + sum(dens >= t for t in (0.6, 1.5, 3.0, 6.0))
    return {"score": score, "news": bool(NEWS_KW.search(text[:2000])), "reason": f"weak:kw_density={dens:.2f}", "weak": True}


def _dist(m, s: str) -> np.ndarray:
    """fastText probability vector over __label__0..5 (absent labels -> 0)."""
    out = np.zeros(6)
    if s:
        labels, probs = m.predict(s, k=6)      # verified: returns (tuple[str], np.ndarray), descending prob
        for l, p in zip(labels, probs): out[int(l[9:])] = p
    return out


# ----------------------------------------------------------------------------- 1. sample (bottom-k random keys == exact uniform sample over all docs)
def _sample_keys(args, k, seed):
    """Pass 1: per shard, keep the k smallest random keys as (-key, line_no). Tiny to pickle; no doc text moves."""
    ds, path = args; rng = random.Random(f"{seed}:{ds}:{Path(path).name}"); heap, n = [], 0
    with gzip.open(path, "rb") as f:
        for n, _ in enumerate(f, 1):
            it = (-rng.random(), n)
            if len(heap) < k: heapq.heappush(heap, it)
            elif it[0] > heap[0][0]: heapq.heapreplace(heap, it)
    return ds, path, heap, n


def _sample_fetch(args):
    """Pass 2: re-read one shard and return only the globally selected lines."""
    path, wanted = args; out = []
    with gzip.open(path, "rb") as f:
        for i, line in enumerate(f, 1):
            if i in wanted:
                try: out.append((wanted[i], orjson.loads(line)))
                except Exception: pass
    return out


def cmd_sample(a):
    shards = [(ds, f) for ds in a.datasets for f in stage(ds)]
    if not shards: log("[sample] nothing to sample"); return
    log(f"[sample] {len(shards)} shards, k={a.n_sample:,}, {a.workers} workers"); top, seen, t0 = [], Counter(), time.time()
    with mp.Pool(a.workers) as pool:
        for i, (ds, path, heap, n) in enumerate(pool.imap_unordered(partial(_sample_keys, k=a.n_sample, seed=a.seed), shards, chunksize=1), 1):
            seen[ds] += n
            for key, idx in heap:      # global bottom-k over the union of per-shard bottom-k's
                if len(top) < a.n_sample: heapq.heappush(top, (key, path, idx))
                elif key > top[0][0]: heapq.heapreplace(top, (key, path, idx))
            if i % 50 == 0: log(f"[sample] {i}/{len(shards)} shards, {sum(seen.values()):,} docs seen")
        wanted = {}
        for key, path, idx in top: wanted.setdefault(path, {})[idx] = key
        rows = [r for part in pool.imap_unordered(_sample_fetch, list(wanted.items()), chunksize=1) for r in part]
    rows.sort(key=lambda t: -t[0])   # ascending random key -> random order; the first n_label are the labeling subset
    for name, sub in (("sample_100k.jsonl.gz", rows), ("sample_10k.jsonl.gz", rows[:a.n_label])):
        with gzip.open(P2 / name, "wb", compresslevel=4) as f:
            for _, d in sub: f.write(orjson.dumps(d) + b"\n")
    info = {"docs_seen": dict(seen), "sampled": len(rows), "label_subset": min(a.n_label, len(rows)), "seconds": round(time.time() - t0)}
    put("sample_100k.jsonl.gz", "sample_10k.jsonl.gz"); mark_done("sample", info); log(f"[sample] {info}")


# ----------------------------------------------------------------------------- 2. label (Gemini 2.5 Flash on Vertex AI, async)
async def _label_all(a, docs, out_path):
    from google import genai
    from google.genai import errors, types
    from pydantic import BaseModel   # pydantic is a dependency of google-genai

    class Label(BaseModel):
        score: int; news: bool; reason: str

    # VERIFY: google-genai now documents `enterprise=True` (Gemini Enterprise Agent Platform, "formerly Vertex AI"); `vertexai=True` is
    # kept as the legacy alias of the same flag. Both use ADC. Switch to enterprise=True if a future release drops the alias.
    client = genai.Client(vertexai=True, project=a.project, location=a.location, http_options=types.HttpOptions(
        timeout=120_000,   # ms
        retry_options=types.HttpRetryOptions(attempts=5, initial_delay=1.0, max_delay=30.0, http_status_codes=[408, 429, 500, 502, 503, 504])))
    cfg = types.GenerateContentConfig(
        system_instruction=RUBRIC, temperature=0.0, max_output_tokens=256, response_mime_type="application/json", response_schema=Label,
        thinking_config=types.ThinkingConfig(thinking_budget=a.thinking_budget))   # VERIFY: 0 disables thinking on 2.5 Flash (not on 3.x Pro)
    sem = asyncio.Semaphore(a.concurrency); st = Counter(); t0 = time.time()

    async def one(d):
        prompt = f"Persian web page (first {PROMPT_HEAD} chars):\n<<<\n{d['text'][:PROMPT_HEAD]}\n>>>"
        async with sem:
            for attempt in range(4):   # app-level retries on top of the SDK's transport retries (bad JSON, timeouts, exhausted 429s)
                try:
                    r = await client.aio.models.generate_content(model=a.model, contents=prompt, config=cfg)
                    lab = r.parsed if isinstance(r.parsed, Label) else Label.model_validate_json(r.text)
                    u = r.usage_metadata
                    st["tok_in"] += u.prompt_token_count or 0; st["tok_out"] += (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
                    return {"id": d["id"], "source": d["source"], "score": max(0, min(5, lab.score)), "news": lab.news, "reason": lab.reason[:200]}
                except errors.APIError as e:
                    if e.code in (400, 401, 403, 404): raise
                    err = f"http{e.code}"
                except Exception as e:   # empty/blocked response (r.text None), invalid JSON, network timeouts
                    err = type(e).__name__
                await asyncio.sleep(min(30, 2 ** attempt) + random.random())
        st["fail"] += 1; log(f"[label] giving up on {d['id']}: {err}"); return None

    with open(out_path, "a", encoding="utf-8") as out:
        for fut in asyncio.as_completed([asyncio.create_task(one(d)) for d in docs]):
            res = await fut
            if res: out.write(json.dumps(res, ensure_ascii=False) + "\n"); st["done"] += 1
            if (st["done"] + st["fail"]) % 200 == 0:
                out.flush(); el = time.time() - t0; cost = (st["tok_in"] * PRICE_IN + st["tok_out"] * PRICE_OUT) / 1e6
                log(f"[label] {st['done']}/{len(docs)} ok, {st['fail']} failed, {(st['done'] + st['fail']) / el:.1f} docs/s, "
                    f"tokens in/out {st['tok_in']:,}/{st['tok_out']:,}, est ${cost:.2f}")
    return dict(st)


def cmd_label(a):
    src, outp = fetch("sample_10k.jsonl.gz"), P2 / "labels_10k.jsonl"
    done_ids = {r["id"] for r in read_jsonl(outp)} if outp.exists() else set()
    docs = [d for d in read_jsonl(src) if d["id"] not in done_ids]
    if a.no_llm:
        log(f"[label] --no-llm FALLBACK: keyword weak labels for {len(docs)} docs (no Gemini calls)")
        with open(outp, "a", encoding="utf-8") as f:
            for d in docs: f.write(json.dumps({"id": d["id"], "source": d["source"], **weak_label(d["text"])}, ensure_ascii=False) + "\n")
        st = {"done": len(docs), "fail": 0, "weak": True}
    else:
        log(f"[label] {len(docs)} docs to label ({len(done_ids)} resumed from {outp.name}); {a.model} @ {a.location}, {a.concurrency} in flight")
        st = asyncio.run(_label_all(a, docs, outp))
        st["est_usd"] = round((st.get("tok_in", 0) * PRICE_IN + st.get("tok_out", 0) * PRICE_OUT) / 1e6, 2)
    put("labels_10k.jsonl"); log(f"[label] {st}")
    if st.get("fail", 0) > 0.02 * max(1, len(docs)): log("[label] >2% failures — NOT marking done; rerun `label` to resume"); return
    mark_done("label", {**st, "labels": len(done_ids) + st.get("done", 0), "model": None if a.no_llm else a.model})


# ----------------------------------------------------------------------------- 3. train (fastText supervised: edu 0-5 + news binary)
def cmd_train(a):
    import fasttext; fasttext.FastText.eprint = lambda *x, **k: None
    texts = {d["id"]: ft_text(d["text"]) for d in read_jsonl(fetch("sample_10k.jsonl.gz"))}
    rows = [r for r in read_jsonl(fetch("labels_10k.jsonl")) if texts.get(r["id"])]
    rng = random.Random(a.seed); rng.shuffle(rows); held, train = rows[:a.n_heldout], rows[a.n_heldout:]
    weak = sum(bool(r.get("weak")) for r in rows)
    metrics = {"n_train": len(train), "n_heldout": len(held), "weak_labels": weak, "label_hist": np.bincount([r["score"] for r in rows], minlength=6).tolist()}
    log(f"[train] {len(train)} train / {len(held)} held-out, label hist {metrics['label_hist']}" + (f", WEAK labels: {weak}" if weak else ""))
    for name, lab in (("edu", lambda r: f"__label__{r['score']}"), ("news", lambda r: "__label__" + ("news" if r["news"] else "other"))):
        trp, hdp = P2 / f"ft_{name}_train.txt", P2 / f"ft_{name}_heldout.txt"
        trp.write_text("\n".join(f"{lab(r)} {texts[r['id']]}" for r in train) + "\n"); hdp.write_text("\n".join(f"{lab(r)} {texts[r['id']]}" for r in held) + "\n")
        # verified at fasttext.cc/docs/en/supervised-tutorial.html: train_supervised(input, epoch, lr, wordNgrams, dim, loss, minCount); test() -> (N, P@1, R@1)
        m = fasttext.train_supervised(input=str(trp), epoch=20, lr=0.5, wordNgrams=2, dim=100, minCount=2, loss="softmax", thread=a.workers, verbose=0)
        n, p1, _ = m.test(str(hdp)); met = {"heldout_n": n, "precision_at_1": round(p1, 4)}
        if name == "edu":
            y = np.array([r["score"] for r in held]); dist = np.array([_dist(m, texts[r["id"]]) for r in held]); ev, am = dist @ np.arange(6), dist.argmax(1)
            met.update(accuracy=round(float((am == y).mean()), 4), acc_within_1=round(float((np.abs(am - y) <= 1).mean()), 4),
                       mae_argmax=round(float(np.abs(am - y).mean()), 4), mae_expected=round(float(np.abs(ev - y).mean()), 4),
                       mae_majority_baseline=round(float(np.abs(np.bincount(y, minlength=6).argmax() - y).mean()), 4))
        m.save_model(str(P2 / f"{name}_fa.bin")); metrics[name] = met; log(f"[train] {name}: {met}")
    (P2 / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    put("edu_fa.bin", "news_fa.bin", "train_metrics.json"); mark_done("train", metrics)


# ----------------------------------------------------------------------------- 4. score (all deduped Persian web shards, one model load per worker)
_M = {}
def _models():
    if not _M:
        import fasttext; fasttext.FastText.eprint = lambda *x, **k: None
        _M["edu"], _M["news"] = fasttext.load_model(str(P2 / "edu_fa.bin")), fasttext.load_model(str(P2 / "news_fa.bin"))
    return _M


def _score_shard(args, out_root, seed):
    ds, path = args; m = _models(); rng = random.Random(f"{seed}:{path}")
    out = Path(out_root) / ds / Path(path).name; tmp = out.with_suffix(".part"); out.parent.mkdir(parents=True, exist_ok=True)
    hist = np.zeros(NBINS, dtype=np.int64); S = np.zeros((len(HEUR_KEYS), 5)); bands = [[] for _ in range(6)]; n = 0   # S cols: Σx Σy Σx² Σy² Σxy
    with gzip.open(tmp, "wb", compresslevel=4) as f:
        for d in read_jsonl(path):
            text = d.get("text") or ""; s = ft_text(text); h = heuristics(text)
            dist = _dist(m["edu"], s); es = float(dist @ np.arange(6)); ei = int(dist.argmax())
            news = float(dict(zip(*m["news"].predict(s, k=2))).get("__label__news", 0.0)) if s else 0.0
            d.setdefault("meta", {}).update(edu_score=round(es, 3), edu_int=ei, news_prob=round(news, 3), heur=h)
            f.write(orjson.dumps(d) + b"\n"); n += 1; hist[min(NBINS - 1, int(es / 5 * NBINS))] += 1
            x = np.array([math.log1p(h["mean_line_len"]), h["symbol_ratio"], h["digit_ratio"], math.log1p(h["doc_chars"]), h["short_line_ratio"]])
            S += np.stack([x, np.full(5, es), x * x, np.full(5, es * es), x * es], 1)
            it = (-rng.random(), d["id"], text[:300], round(es, 2)); b = bands[ei]     # bottom-N_EXCERPT keys per band == uniform sample
            if len(b) < N_EXCERPT: heapq.heappush(b, it)
            elif it[0] > b[0][0]: heapq.heapreplace(b, it)
    tmp.rename(out)
    return {"n": n, "hist": hist.tolist(), "S": S.tolist(), "bands": bands}


def _merge(agg, r):
    agg["n"] += r["n"]; agg["hist"] = [x + y for x, y in zip(agg["hist"], r["hist"])]; agg["S"] += np.array(r["S"])
    for b, rb in zip(agg["bands"], r["bands"]):
        for it in rb:
            if len(b) < N_EXCERPT: heapq.heappush(b, it)
            elif it[0] > b[0][0]: heapq.heapreplace(b, it)


def cmd_score(a):
    fetch("edu_fa.bin"); fetch("news_fa.bin"); out_root = DATA / "scored"; agg_dir = P2 / "score_agg"; agg_dir.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        remote = f"{BUCKET}/scored/{ds}"
        if not a.force and gcs_list(remote, "_DONE.json"): log(f"[{ds}] already scored in GCS, skipping (use --force to redo)"); continue
        files = stage(ds)
        if not files: continue
        t0 = time.time(); od = out_root / ds; shutil.rmtree(od, ignore_errors=True); od.mkdir(parents=True)
        agg = {"dataset": ds, "n": 0, "hist": [0] * NBINS, "S": np.zeros((len(HEUR_KEYS), 5)), "bands": [[] for _ in range(6)]}
        log(f"[{ds}] scoring {len(files)} shards with {a.workers} workers")
        with mp.Pool(a.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(partial(_score_shard, out_root=out_root, seed=a.seed), [(ds, f) for f in files], chunksize=1), 1):
                _merge(agg, r)
                if i % 25 == 0: log(f"[{ds}] {i}/{len(files)} shards, {agg['n']:,} docs")
        agg["S"] = agg["S"].tolist(); agg["seconds"] = round(time.time() - t0)
        (agg_dir / f"{ds}.json").write_text(json.dumps(agg, ensure_ascii=False))
        stats = {"dataset": ds, "docs": agg["n"], "shards": len(files), "edu_hist_20bins": agg["hist"], "seconds": agg["seconds"],
                 "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (od / "_stats.json").write_text(json.dumps(stats, indent=2))
        gsutil("rm", "-r", "-q", remote, check=False); gcs_upload_dir(od, remote)
        (od / "_DONE.json").write_text(json.dumps(stats)); gsutil("cp", "-q", str(od / "_DONE.json"), f"{remote}/_DONE.json")
        gsutil("cp", "-q", str(agg_dir / f"{ds}.json"), f"{BUCKET}/p2/score_agg/{ds}.json")
        shutil.rmtree(od, ignore_errors=True)
        if not a.keep_input: shutil.rmtree(DATA / "deduped" / ds, ignore_errors=True)
        log(f"[{ds}] scored {agg['n']:,} docs in {agg['seconds']}s -> {remote}")
    mark_done("score", {"datasets": a.datasets})


# ----------------------------------------------------------------------------- 5. report (from the per-dataset aggregates written by `score`)
def cmd_report(a):
    L = ["# Phase 2 — educational-value scoring of Persian web docs", ""]
    try: L += ["## fastText classifier (held-out metrics; labels are ordinal 0-5)", "", "```json", fetch("train_metrics.json").read_text().strip(), "```", ""]
    except Exception: L += ["_train_metrics.json not found_", ""]
    aggs = []
    for ds in a.datasets:
        try: aggs.append(json.loads(fetch(f"score_agg/{ds}.json").read_text()))
        except Exception: log(f"[report] no score aggregates for {ds} — run `score` first")
    L += [f"## edu_score histogram ({NBINS} bins over [0,5])", "", "| bin | " + " | ".join(f"{g['dataset']} (n={g['n']:,})" for g in aggs) + " |", "|---|" + "---|" * len(aggs)]
    for i in range(NBINS):
        L.append(f"| {i * 5 / NBINS:.2f}-{(i + 1) * 5 / NBINS:.2f} | " + " | ".join(f"{g['hist'][i]:,} ({100 * g['hist'][i] / max(1, g['n']):.1f}%)" for g in aggs) + " |")
    L += ["", "## Pearson r of edu_score vs heuristics (mean_line_len and doc_chars as log1p)", "", "| dataset | " + " | ".join(HEUR_KEYS) + " |", "|---|" + "---|" * len(HEUR_KEYS)]
    for g in aggs:
        S, n = np.array(g["S"]), g["n"]
        r = (n * S[:, 4] - S[:, 0] * S[:, 1]) / np.sqrt(np.maximum(1e-12, (n * S[:, 2] - S[:, 0] ** 2) * (n * S[:, 3] - S[:, 1] ** 2)))
        L.append(f"| {g['dataset']} | " + " | ".join(f"{v:+.3f}" for v in r) + " |")
    L += ["", f"## {N_EXCERPT} random docs per integer score band (300-char excerpts)"]
    for band in range(6):
        items = sorted((tuple(it) for g in aggs for it in g["bands"][band]), key=lambda t: -t[0])[:N_EXCERPT]
        L += ["", f"### edu_int = {band}  ({len(items)} shown)", ""]
        L += [f"- **{did}** (edu_score {es}): {_WS.sub(' ', ex).replace('|', '¦')}" for _, did, ex, es in items]
    rp = P2 / "report.md"; rp.write_text("\n".join(L) + "\n", encoding="utf-8")
    gsutil("cp", "-q", str(rp), f"{BUCKET}/scored/_report.md"); put("report.md"); log(f"[report] {rp} -> {BUCKET}/scored/_report.md")


# ----------------------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "label", "train", "score", "report", "all"])
    ap.add_argument("--datasets", nargs="*", default=WEB_FA); ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 2))
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR-GCP-PROJECT")); ap.add_argument("--location", default="us-central1"); ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--concurrency", type=int, default=16, help="Gemini requests in flight")   # VERIFY: Vertex dynamic shared quota; raise if no 429s
    ap.add_argument("--thinking-budget", type=int, default=0, help="0 = no thinking (cheapest); -1 = model decides")
    ap.add_argument("--n-sample", type=int, default=100_000); ap.add_argument("--n-label", type=int, default=10_000); ap.add_argument("--n-heldout", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=20260828); ap.add_argument("--no-llm", action="store_true", help="FALLBACK: keyword weak labels instead of Gemini")
    ap.add_argument("--keep-input", action="store_true"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); P2.mkdir(parents=True, exist_ok=True)
    steps = {"sample": cmd_sample, "label": cmd_label, "train": cmd_train, "score": cmd_score, "report": cmd_report}
    for step in (list(steps) if a.cmd == "all" else [a.cmd]):
        if a.cmd == "all" and step != "report" and not a.force and is_done(step): log(f"[{step}] done marker in {BUCKET}/p2, skipping"); continue
        steps[step](a)
