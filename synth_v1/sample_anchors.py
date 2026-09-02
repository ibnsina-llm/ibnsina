#!/opt/pipe/bin/python3
"""synth_v1 anchor sampler — pulls ~200 high-scoring educational Persian passages from the scored layer in GCS
into /data/synth_v1/anchors/anchors.jsonl on the VM. Anchor TEXT never leaves the VM and is never committed
(corpus text stays private); only this script is in the repo.
  sample_anchors.py [--n 200] [--sources fineweb2_fa,culturax_fa] [--shards-per-source 15]
                    [--min-edu 3.0] [--max-news 0.3] [--min-chars 500] [--max-chars 1500]
Runs on corpus-pipeline2; streams shards with `gcloud storage cat` (no GCS python client in /opt/pipe)."""
import argparse, gzip, io, json, random, subprocess
from pathlib import Path

BUCKET = "gs://YOUR-BUCKET/scored"


def list_shards(src):
    out = subprocess.run(["gcloud", "storage", "ls", f"{BUCKET}/{src}/"], capture_output=True, text=True, check=True).stdout
    return [l.strip() for l in out.splitlines() if l.strip().endswith(".jsonl.gz") and not l.rsplit("/", 1)[-1].startswith("_")]


def cut_passage(text, lo, hi):
    """Whole doc if it fits; otherwise cut the opening at a paragraph/sentence boundary inside [lo, hi]."""
    t = text.strip()
    if len(t) < lo: return None
    if len(t) <= hi + 100: return t
    window = t[:hi]
    for sep in ("\n\n", ".\n", "؟\n", "!\n", ". ", "؟ "):
        i = window.rfind(sep)
        if i >= lo: return window[: i + 1].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sources", default="fineweb2_fa,culturax_fa")
    ap.add_argument("--shards-per-source", type=int, default=15)
    ap.add_argument("--min-edu", type=float, default=3.0, help="FineWeb-Edu-style classifier score (0-5) floor")
    ap.add_argument("--max-news", type=float, default=0.3)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--out", default="/data/synth_v1/anchors/anchors.jsonl")
    ap.add_argument("--seed", type=int, default=20260831)
    a = ap.parse_args()
    rnd = random.Random(a.seed); pool = []; seen = set()
    for src in a.sources.split(","):
        shards = list_shards(src)
        take = rnd.sample(shards, min(a.shards_per_source, len(shards)))
        print(f"[{src}] {len(shards)} shards, scanning {len(take)}", flush=True)
        kept = scanned = 0
        for sh in take:
            p = subprocess.run(["gcloud", "storage", "cat", sh], capture_output=True, check=True)
            for line in io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(p.stdout)), encoding="utf-8"):
                scanned += 1
                try: d = json.loads(line)
                except Exception: continue
                m = d.get("meta", {})
                if m.get("edu_score", 0) < a.min_edu or m.get("news_prob", 0) > a.max_news: continue
                passage = cut_passage(d.get("text", ""), a.min_chars, a.max_chars)
                if not passage: continue
                key = passage[:80]
                if key in seen: continue        # crude near-dup guard
                seen.add(key)
                pool.append({"id": d["id"], "source": src, "edu_score": round(float(m["edu_score"]), 3),
                             "news_prob": round(float(m.get("news_prob", 0)), 3), "chars": len(passage), "text": passage})
                kept += 1
        print(f"[{src}] scanned {scanned} docs -> {kept} candidates", flush=True)
    if not pool: raise SystemExit("no candidates found — loosen --min-edu or add shards")
    pool.sort(key=lambda r: -r["edu_score"])
    top = pool[: max(a.n * 4, a.n)]                 # top band by edu score, then sample inside it for diversity
    sample = rnd.sample(top, min(a.n, len(top)))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in sample: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sc = sorted(r["edu_score"] for r in sample)
    print(f"wrote {len(sample)} anchors -> {out}; edu_score min/med/max = {sc[0]:.2f}/{sc[len(sc)//2]:.2f}/{sc[-1]:.2f}; "
          f"avg chars {sum(r['chars'] for r in sample) // len(sample)}; sources {sorted({r['source'] for r in sample})}")


if __name__ == "__main__":
    main()
