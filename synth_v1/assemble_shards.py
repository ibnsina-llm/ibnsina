#!/opt/pipe/bin/python3
"""synth_v1 wave assembly: auto-checks -> keep top third per domain (floor overall>=7) -> MinHash dedup
(5-gram, 14x8, persistent index) -> decontamination vs eval sets (normalized 13-gram overlap) ->
normalize_fa -> parquet shard -> GCS + manifest + state. Nothing counts as kept before this gate.
Exit codes: 0 next wave, 42 STOP at kept-token target (2B per SY-B ② ruling), 43 budget hard stop, 44 keep-rate floor breach."""
import argparse, collections, json, pickle, re, sys, time
from pathlib import Path
import numpy as np, xxhash
from bulk_common import (BULK, CODE, GCS, FLASH, HARD_STOP_USD, SYB_KEPT_TOKENS, Ledger, budget_events,
                         gcs, jdump, jload, read_jsonl)
sys.path.insert(0, "/data/pipeline")
try:
    from pipeline.common import normalize_fa
except Exception:
    normalize_fa = lambda s: s
KEEP_MIN, KEEP_RATE_FLOOR = 7, 0.15
EVALSETS = Path("/data/sft_v2/evalsets")
SELF_REF = re.compile(r"(به عنوان یک مدل زبانی|as an ai|language model|gemini|openai|chatgpt|claude|anthropic)", re.I)


def norm_words(s):
    return re.sub(r"\W+", " ", normalize_fa(s)).strip().lower().split()


def ngram_set(words, n=13):
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


class MinHashIndex:
    """5-gram word shingles, 112 xxhash-seeded min-hashes, 14 bands x 8 rows; first-wins on band collision."""
    def __init__(self, path):
        self.path = Path(path)
        self.bands = pickle.loads(self.path.read_bytes()) if self.path.exists() else [dict() for _ in range(14)]
    def signature(self, words):
        sh = list({" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))})
        h = np.array([xxhash.xxh64_intdigest(s) for s in sh], dtype=np.uint64)
        mins = np.empty(112, dtype=np.uint64)
        for i in range(112):
            mins[i] = np.min(h * np.uint64(2 * i + 1) + np.uint64(i * 0x9E3779B97F4A7C15 & (2**64 - 1)))
        return mins
    def add_if_new(self, words):
        sig = self.signature(words)
        keys = [xxhash.xxh64_intdigest(sig[b * 8:(b + 1) * 8].tobytes()) for b in range(14)]
        if any(k in self.bands[b] for b, k in enumerate(keys)): return False
        for b, k in enumerate(keys): self.bands[b][k] = 1
        return True
    def save(self): self.path.write_bytes(pickle.dumps(self.bands, protocol=4))


def auto_flags(text):
    flags = []
    words = text.split()
    if not (1200 <= len(text) <= 60000): flags.append("length")
    fa = sum(1 for ch in text[:4000] if "؀" <= ch <= "ۿ")
    if fa < 0.35 * min(len(text), 4000): flags.append("not_persian")
    for n in (8, 12):
        grams = collections.Counter(tuple(words[j:j + n]) for j in range(max(0, len(words) - n)))
        if grams and grams.most_common(1)[0][1] >= 3: flags.append("repetition_loop"); break
    if SELF_REF.search(text): flags.append("self_reference")
    return flags


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--wave", type=int, required=True); a = ap.parse_args()
    wdir = BULK / f"waves/wave-{a.wave:04d}"
    state = jload(BULK / "state.json", {"wave": 1, "kept_tokens": 0, "usd_notified": 0})
    ledger = Ledger()
    docs = read_jsonl(wdir / "docs.jsonl")
    judged = {r["id"]: r["scores"] for r in read_jsonl(wdir / "judged.jsonl")}
    scens = read_jsonl(wdir / "scenarios.jsonl")
    n_cand = collections.Counter(s["domain"] for s in scens)
    # eval-set 13-grams (loaded once per run)
    grams = set()
    for f in sorted(EVALSETS.glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try: grams |= ngram_set(norm_words(json.loads(l)["q"]))
                except Exception: pass
    print(f"[assemble w{a.wave}] {len(docs)} docs, {len(judged)} judged, eval 13-grams {len(grams):,}", flush=True)
    # selection: per domain, top third of CANDIDATES by (overall, correctness), floor overall >= KEEP_MIN
    by_dom = collections.defaultdict(list)
    drop = collections.Counter()
    for d in docs:
        s = judged.get(d["id"])
        if not s or s["overall"] < 0: drop["unjudged"] += 1; continue
        fl = auto_flags(d["text"])
        if fl: drop["auto_" + fl[0]] += 1; continue
        if s["overall"] < KEEP_MIN: drop["below_floor"] += 1; continue
        d["scores"] = s; by_dom[d["domain"]].append(d)
    mh = MinHashIndex(BULK / "minhash.pkl")
    kept, rate = [], {}
    for dom, ds in by_dom.items():
        ds.sort(key=lambda d: (-d["scores"]["overall"], -d["scores"].get("correctness", 0), d["id"]))
        n_keep = min(len(ds), max(1, n_cand[dom] // 3))
        chosen = ds[:n_keep]
        for d in chosen:
            words = norm_words(d["text"])
            if not mh.add_if_new(words): drop["minhash_dup"] += 1; continue
            if ngram_set(words) & grams: drop["decontaminated"] += 1; continue
            kept.append(d)
        rate[dom] = round(len([k for k in kept if k["domain"] == dom]) / max(1, n_cand[dom]), 3)
    mh.save()
    # parquet shard
    import pyarrow as pa, pyarrow.parquet as pq
    shard_dir = BULK / "shards"; shard_dir.mkdir(exist_ok=True)
    cols = {"id": [k["id"] for k in kept], "text": [normalize_fa(k["text"]) for k in kept],
            "source": ["synth_v1"] * len(kept), "domain": [k["domain"] for k in kept],
            "subdomain": [k["subdomain"] for k in kept], "doc_type": [k["doc_type"] for k in kept],
            "teacher": [k["teacher"] for k in kept], "topic": [k["topic"] for k in kept],
            "judge_overall": [k["scores"]["overall"] for k in kept],
            "tokens_billed": [k["usage"]["out"] for k in kept], "license": ["Apache-2.0"] * len(kept)}
    fp = shard_dir / f"wave-{a.wave:04d}.parquet"
    pq.write_table(pa.table(cols), fp, compression="zstd")
    gcs("cp", str(fp), f"{GCS}/shards/{fp.name}")
    wave_tokens = sum(k["usage"]["out"] for k in kept)
    state["kept_tokens"] += wave_tokens
    # per-wave report (feeds STOP SY-B)
    hist = collections.Counter((d["domain"], d["teacher"].split("-")[-1], judged[d["id"]]["overall"])
                               for d in docs if d["id"] in judged and judged[d["id"]]["overall"] >= 0)
    rep = {"wave": a.wave, "candidates": len(docs), "kept": len(kept), "kept_tokens": wave_tokens,
           "keep_rate_by_domain": rate, "drops": dict(drop),
           "score_hist": {f"{d}|{t}|{s}": n for (d, t, s), n in sorted(hist.items())},
           "usd_total": round(ledger.usd(), 2)}
    (BULK / "reports").mkdir(exist_ok=True); jdump(BULK / f"reports/wave-{a.wave:04d}.json", rep)
    # manifest (Apache-2.0)
    man = jload(BULK / "manifest.json", {"license": "Apache-2.0", "dataset": "synth_v1", "format": "parquet",
        "token_counts_are": "Gemini billed output tokens", "decontaminated_against": sorted(f.stem for f in EVALSETS.glob("*.jsonl")),
        "dedup": "MinHash 5-gram 14x8 + judge top-third selection", "waves": {}})
    man["waves"][f"{a.wave:04d}"] = {"rows": len(kept), "tokens": wave_tokens, "keep_rate": rate, "built": time.strftime("%FT%TZ", time.gmtime())}
    man["total_rows"] = sum(w["rows"] for w in man["waves"].values())
    man["total_tokens"] = sum(w["tokens"] for w in man["waves"].values())
    jdump(BULK / "manifest.json", man); gcs("cp", str(BULK / "manifest.json"), f"{GCS}/shards/manifest.json")
    state["wave"] = a.wave + 1
    usd = budget_events(ledger, state); jdump(BULK / "state.json", state)
    print(f"[assemble w{a.wave}] kept {len(kept)}/{len(docs)} ({wave_tokens/1e6:.1f}M tokens); cum {state['kept_tokens']/1e9:.3f}B; ${usd:.2f}", flush=True)
    overall_rate = len(kept) / max(1, len(docs))
    if usd > HARD_STOP_USD: sys.exit(43)
    if state["kept_tokens"] >= SYB_KEPT_TOKENS: print("STOP SY-C: 2B kept tokens reached (owner ruling ②)"); sys.exit(42)
    if overall_rate < KEEP_RATE_FLOOR: print(f"keep rate {overall_rate:.2f} < floor {KEEP_RATE_FLOOR} — STOP, show Sina"); sys.exit(44)


if __name__ == "__main__":
    main()
