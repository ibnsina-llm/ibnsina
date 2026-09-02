#!/opt/pipe/bin/python3
"""Collect the question texts of every evaluation set for decontamination -> /data/sft_v2/evalsets/<name>.jsonl ({"q": text}).
ParsiNLU (GitHub raw, all splits), PerCoR (HF parquet), Khayyam (HF parquet if access granted), TARAZ / GhazalBench (add their release URLs to SOURCES when located)."""
import csv, io, json, os, sys, urllib.request
from pathlib import Path
OUT = Path("/data/sft_v2/evalsets"); OUT.mkdir(parents=True, exist_ok=True)
R = "https://raw.githubusercontent.com/persiannlp/parsinlu/master/data"
SOURCES = {"taraz": os.environ.get("TARAZ_URL", ""), "ghazalbench": os.environ.get("GHAZALBENCH_URL", "")}


def _hf_token():
    for p in (os.environ.get("HF_TOKEN"), "/data/secrets/hf.token", os.path.expanduser("~/.cache/huggingface/token")):
        if p and (p.startswith("hf_") or os.path.exists(p)): return p if p.startswith("hf_") else open(p).read().strip()
    return None


def get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_hf_token()}"} if "huggingface.co" in url and _hf_token() else {})
    with urllib.request.urlopen(req, timeout=180) as r: return r.read()


def write(name, qs):
    qs = [q.strip() for q in qs if q and q.strip()]
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for q in qs: f.write(json.dumps({"q": q}, ensure_ascii=False) + "\n")
    print(f"{name}: {len(qs)} questions")


def parsinlu():
    qs = []
    for split in ("train", "valid", "test", "test_ck", "test_lit", "test_ml"):
        try: qs += [json.loads(l)["question"] for l in get(f"{R}/multiple-choice/{split}.jsonl").decode().splitlines() if l.strip()]
        except Exception as e: print(" skip mc", split, e)
    for split in ("train", "dev", "test"):
        try:
            for row in csv.DictReader(io.StringIO(get(f"{R}/entailment/{split}.csv").decode())): qs += [row.get("sent1", ""), row.get("sent2", "")]
        except Exception as e: print(" skip entailment", split, e)
        try: qs += [v for l in get(f"{R}/qqp/{split}.jsonl").decode().splitlines() if l.strip() for v in (json.loads(l).get("q1", ""), json.loads(l).get("q2", ""))]
        except Exception as e: print(" skip qqp", split, e)
    for split in ("train", "dev", "eval"):
        try: qs += [json.loads(l).get("question", "") for l in get(f"{R}/reading_comprehension/{split}.jsonl").decode().splitlines() if l.strip()]
        except Exception: pass
    write("parsinlu", qs)


def hf_parquet(repo, config="default", split="train", cols=("question",)):
    import pyarrow.parquet as pq
    urls = json.loads(get(f"https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}")); out = []
    for u in urls:
        t = pq.read_table(io.BytesIO(get(u)))
        for c in cols:
            if c in t.column_names: out += [str(x) for x in t.column(c).to_pylist()]
    return out


def main():
    parsinlu()
    try:
        qs = []
        for split in ("train", "validation", "test"):
            try: qs += hf_parquet("MCINext/PerCoR", "default", split, ("question", "premise", "context", "text"))
            except Exception: pass
        write("percor", qs)
    except Exception as e: print("percor failed:", e)
    try:
        qs = hf_parquet("raia-center/khayyam-challenge", "default", "train", ("question", "Question")); write("khayyam", qs)
    except Exception as e: print("khayyam not accessible (gated?):", str(e)[:80])
    # TARAZ (Georgetown-IR-Lab/TARAZ): ISN-SAQ + non-taboo CSVs, PerCul JSONs, annotation prompts — take every Persian string field
    T = "https://raw.githubusercontent.com/Georgetown-IR-Lab/TARAZ/main"; qs = []
    for f in ("ISN/isn_farsi_saq.csv", "ISN/isn_iran_nontaboo.csv", "data/prompts/Iran_prompts.csv"):
        try:
            for row in csv.DictReader(io.StringIO(get(f"{T}/{f}").decode())): qs += [v for v in row.values() if isinstance(v, str) and len(v) >= 15]
        except Exception as e: print(" taraz skip", f, str(e)[:60])
    for f in ("PerCul/percul.json", "PerCul/percul_cat.json", "data/annotations/Iran_data.json"):
        try:
            d = json.loads(get(f"{T}/{f}").decode()); stack = [d]
            while stack:
                x = stack.pop()
                if isinstance(x, dict): stack += list(x.values())
                elif isinstance(x, list): stack += x
                elif isinstance(x, str) and len(x) >= 15 and any("\u0600" <= ch <= "\u06ff" for ch in x): qs.append(x)
        except Exception as e: print(" taraz skip", f, str(e)[:60])
    write("taraz", qs)
    # GhazalBench (arXiv 2603.09979) has no public release; it is built from Hafez ghazals 1-100 -> proxy: those couplets from our Ganjoor data
    import gzip, glob, re
    qs = []
    for shard in glob.glob("/data/clean/ganjoor/*.jsonl.gz"):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for l in fh:
                d = json.loads(l); key = (d.get("id", "") + " " + str(d.get("url", "")) + " " + str(d.get("meta", {}))).lower()
                if "hafez" in key and "ghazal" in key:
                    m = re.search(r"ghazal[^0-9]*(\d+)", key)
                    if m and 1 <= int(m.group(1)) <= 100:
                        qs += [ln for ln in d.get("text", "").splitlines() if len(ln) >= 15]
    write("ghazalbench_proxy_hafez1-100", qs)
    # PersianMedQA (arXiv 2506.00250; HF MohammadJRanjbar/PersianMedQA, needs an HF token) — every Persian string field of every split, schema-agnostic
    qs = []
    try:
        for sp in ("train", "validation", "test"):
            for url in json.loads(get(f"https://huggingface.co/api/datasets/MohammadJRanjbar/PersianMedQA/parquet/default/{sp}")):
                import pyarrow.parquet as pq
                t = pq.read_table(io.BytesIO(get(url)))
                for c in t.column_names:
                    col = t.column(c).to_pylist()
                    if col and isinstance(col[0], str): qs += [x for x in col if x and len(x) >= 15 and any("\u0600" <= ch <= "\u06ff" for ch in x)]
        write("persianmedqa", qs)
    except Exception as e: print("persianmedqa: not accessible yet (needs HF token / access):", str(e)[:80])


if __name__ == "__main__":
    main()
