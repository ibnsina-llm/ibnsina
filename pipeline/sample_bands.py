#!/opt/pipe/bin/python3
"""Seeded random full-text samples per edu band for curator review -> /data/samples/ibnsina-samples/<band>/*.txt + README.md
Replays p3_mix's selection rule (same hash, same p) so `in_mix` is exact for the v1 build."""
import gzip, json, random, re, sys
from pathlib import Path
from urllib.parse import urlparse
import orjson, xxhash
import pyarrow.parquet as pq

SEED = 20260829
OUT = Path("/data/samples/ibnsina-samples")
SCORED = Path("/data/scored"); TRAIN = Path("/data/mix/train_v1_open/train")
MIX = json.loads(Path("/data/mix/train_v1_open/mix_manifest.json").read_text())
WEB = MIX["sources"]["culturax_fa"]["plan"]           # same rule for all three web sets
P_NONNEWS, P_NEWS, KEEP_ALL, FILL_LO, NEWS_P = WEB["p_fill_nonnews"], WEB["p_fill_news"], WEB["keep_all_min"], WEB["fill_lo"], WEB["news_prob"]
BANDS = {  # name -> (lo, hi, news filter, n)
    "band_2.75_up": (2.75, 9.0, None, 20), "band_2.0_2.75": (2.0, 2.75, None, 20), "band_1.5_2.0": (1.5, 2.0, None, 20),
    "band_1.0_1.5_nonnews": (1.0, 1.5, False, 25), "band_1.0_1.5_news": (1.0, 1.5, True, 15), "band_dropped_junk": (0.0, 1.0, None, 10),
}
rnd = random.Random(SEED)


def in_mix(doc):
    hv = xxhash.xxh64_intdigest(doc["id"].encode())
    if hv % 1000 < 5:
        return False  # val holdout
    u = (hv // 1000) % 1_000_000 / 1_000_000.0
    s = float(doc["meta"].get("edu_score", 0)); news = float(doc["meta"].get("news_prob", 0)) >= NEWS_P
    if s >= KEEP_ALL: return True
    if s >= FILL_LO: return u < (P_NEWS if news else P_NONNEWS)
    return False


def band_of(doc):
    s = float(doc["meta"].get("edu_score", 0)); news = float(doc["meta"].get("news_prob", 0)) >= NEWS_P
    for name, (lo, hi, nf, _) in BANDS.items():
        if lo <= s < hi and (nf is None or nf == news):
            return name
    return None


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:60]


def write_doc(band, doc, extra=""):
    (OUT / band).mkdir(parents=True, exist_ok=True)
    s = float(doc.get("meta", {}).get("edu_score", 0)); src = doc["source"]; url = doc.get("url") or ""
    dom = urlparse(url).hostname or "-"
    fn = OUT / band / f"{band}_{s:.2f}_{src}_{safe(doc['id'].split(':', 1)[-1])}.txt"
    head = f"score={s:.2f} | source={src} | domain={dom} | news_prob={float(doc.get('meta', {}).get('news_prob', 0)):.2f} | in_mix={'yes' if band.startswith('band') and in_mix(doc) else ('no' if band.startswith('band') else 'yes')}{extra} | id={doc['id']}\nurl={url}\n\n"
    fn.write_text(head + doc["text"], encoding="utf-8")


def sample_web():
    # reservoir per band over a seeded random subset of shards from each web set
    need = {b: cfg[3] for b, cfg in BANDS.items()}; res = {b: [] for b in BANDS}; seen = {b: 0 for b in BANDS}
    for ds in ("culturax_fa", "mc4_fa", "fineweb2_fa"):
        shards = sorted((SCORED / ds).glob("*part-*.jsonl.gz")); rnd.shuffle(shards)
        for p in shards[:5]:
            with gzip.open(p, "rb") as f:
                for line in f:
                    d = orjson.loads(line); b = band_of(d)
                    if b is None: continue
                    if b == "band_1.0_1.5_news" and not in_mix(d):   # news filler that actually made it in
                        continue
                    seen[b] += 1
                    if len(res[b]) < need[b]: res[b].append(d)
                    else:
                        j = rnd.randrange(seen[b])
                        if j < need[b]: res[b][j] = d
    for b, docs in res.items():
        for d in docs: write_doc(b, d)
        print(f"{b}: {len(docs)} docs (from {seen[b]:,} candidates)")


def sample_lit():
    want = {"ganjoor": 5, "fawikisource": 2}; got = {k: [] for k in want}; seen = {k: 0 for k in want}
    shards = sorted(TRAIN.glob("shard_*.parquet")); rnd.shuffle(shards)
    for p in shards[:6]:
        t = pq.read_table(p, columns=["text", "source", "id", "epoch"]); src = t.column("source").to_pylist()
        for i, s in enumerate(src):
            if s in want and t.column("epoch")[i].as_py() == 0:
                seen[s] += 1
                if len(got[s]) < want[s]: got[s].append(i)
                else:
                    j = rnd.randrange(seen[s])
                    if j < want[s]: got[s][j] = i
        for s, idxs in got.items():
            for i in idxs:
                write_doc("literature_speeches", {"id": t.column("id")[i].as_py(), "source": s, "text": t.column("text")[i].as_py(), "meta": {}, "url": ""}, extra=f" | from={p.name}")
        got = {k: [] for k in want}  # write per shard to keep it simple; totals may slightly exceed targets
    print("literature_speeches:", len(list((OUT / 'literature_speeches').glob('*.txt'))), "files")


def readme():
    w = WEB
    (OUT / "README.md").write_text(f"""# ibnsina-samples — curator review set for train_v1_open

Seeded random samples (seed {SEED}) of FULL documents from the scored Persian web (`scored/`) and from the finished train shards.
First line of every file: score | source dataset | domain | news_prob | in_mix (replays the exact v1 selection rule) | id.

## How the Persian-web slice of train_v1_open was selected
| band (edu_score) | rule in v1 | tokens |
|---|---|---|
| >= {w['keep_all_min']} | **all docs kept** | ~21.5 B |
| [{w['fill_lo']}, {w['keep_all_min']}) non-news (news_prob < {w['news_prob']}) | sampled at p = {w['p_fill_nonnews']:.3f} | ~6 B |
| [{w['fill_lo']}, {w['keep_all_min']}) news (news_prob >= {w['news_prob']}) | sampled at p = {w['p_fill_news']:.3f} | ~2 B |
| < {w['fill_lo']} | dropped | (14.9 B junk + 13.7 B mostly-news below 1.0) |

edu_score is the fastText regressor trained on 10k Gemini-2.5-Flash labels (0-5 educational value, FineWeb-Edu rubric adapted to Persian).
Labels were bimodal (0/1/3), so read scores as ~0 junk, ~1 news/general web, >=2 informative/educational.

## Folders
- band_2.75_up/ (20), band_2.0_2.75/ (20), band_1.5_2.0/ (20): fully included bands.
- band_1.0_1.5_nonnews/ (25): partially included (p={w['p_fill_nonnews']:.2f}); header says which ones made it.
- band_1.0_1.5_news/ (15): news filler that DID make it into the mix (p={w['p_fill_news']:.2f}).
- band_dropped_junk/ (10): score < 1.0, dropped.
- literature_speeches/: Ganjoor and fa-wikisource docs exactly as they sit in the train shards (post-normalization).

Normalization applied to all Persian text: Arabic kaf/yeh -> Persian, ZWNJ cleanup, Persian/Arabic digits -> ASCII, whitespace collapse,
boilerplate lines (<25 chars, repeated >100x within a domain) removed. Poems: one hemistich per line (Ganjoor plainText).
Mix manifest: gs://YOUR-BUCKET/train_v1_open/mix_manifest.json
""", encoding="utf-8")


if __name__ == "__main__":
    import shutil
    shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
    sample_web(); sample_lit(); readme()
    print("done:", sum(1 for _ in OUT.rglob("*.txt")), "files")
