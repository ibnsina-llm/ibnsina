#!/opt/pipe/bin/python3
"""sft_v2 assembly: kept/<cat>.jsonl + human files -> decontaminate against eval sets -> normalize_fa -> holdout 500 -> train.jsonl / sft_eval.jsonl / manifest.json -> gs://.../sft/v2/
  assemble.py --work /data/sft_v2 --human sft_v2/human --evalsets /data/sft_v2/evalsets [--no-upload]"""
import argparse, collections, hashlib, json, random, re, sys, time
from pathlib import Path
import yaml
sys.path.insert(0, "/data/pipeline")
from pipeline.common import normalize_fa
HERE = Path(__file__).resolve().parent; B = "gs://YOUR-BUCKET/sft/v2"


def norm_q(s):
    return re.sub(r"\W+", " ", normalize_fa(s)).strip().lower()


def ngrams(s, n=13):
    w = norm_q(s).split(); return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def norm_msgs(msgs):
    out = []
    for m in msgs:
        c = m["content"]
        out.append({"role": m["role"], "content": normalize_fa(c) if isinstance(c, str) else [{"type": p["type"], "text": normalize_fa(p["text"]) if p["type"] == "text" else p["text"]} for p in c]})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", default="/data/sft_v2"); ap.add_argument("--human", default=str(HERE / "human")); ap.add_argument("--evalsets", default="/data/sft_v2/evalsets")
    ap.add_argument("--holdout", type=int, default=500); ap.add_argument("--seed", type=int, default=20260829); ap.add_argument("--no-upload", action="store_true"); a = ap.parse_args()
    tax = yaml.safe_load(open(HERE / "sft_taxonomy.yaml", encoding="utf-8")); W = Path(a.work); rnd = random.Random(a.seed)
    # eval sets -> exact set + 13-gram set
    exact, grams, counts = set(), set(), {}
    for f in sorted(Path(a.evalsets).glob("*.jsonl")):
        n = 0
        for l in f.read_text(encoding="utf-8").splitlines():
            if not l.strip(): continue
            q = json.loads(l)["q"]; exact.add(norm_q(q)); grams |= ngrams(q); n += 1
        counts[f.stem] = n
    print("eval sets:", counts, "| exact", len(exact), "| 13-grams", len(grams))
    rows, per_cat, dropped = [], collections.Counter(), collections.Counter()
    src = {c["name"]: W / "kept" / f"{c['name']}.jsonl" for c in tax["categories"]}
    for cat in tax["categories"]:
        files = [src[cat["name"]]] + sorted(Path(a.human).glob(f"{cat['name']}*.jsonl"))
        for f in files:
            if not f.exists(): continue
            for l in f.read_text(encoding="utf-8").splitlines():
                if not l.strip(): continue
                r = json.loads(l)
                if "messages" not in r: continue
                if any("[IDENTITY-PLACEHOLDER]" in (m["content"] if isinstance(m["content"], str) else "") for m in r["messages"]): dropped["placeholder"] += 1; continue
                users = [m["content"] for m in r["messages"] if m["role"] == "user" and isinstance(m["content"], str)]
                if any(norm_q(u) in exact for u in users) or any(ngrams(u) & grams for u in users): dropped["decontaminated"] += 1; continue
                rows.append({"id": r.get("id") or hashlib.blake2b(l.encode(), digest_size=8).hexdigest(), "category": cat["name"], "subtype": r.get("subtype"), "teacher": r.get("teacher", "human" if "human" in str(f) else None),
                             "source_file": f.name, "license": "generated (Apache-2.0 release)" if "human" not in str(f) else "human-written", "messages": norm_msgs(r["messages"])})
                per_cat[cat["name"]] += 1
    rnd.shuffle(rows)
    # stratified holdout
    by_cat = collections.defaultdict(list)
    for r in rows: by_cat[r["category"]].append(r)
    hold, train = [], []
    for c, rs in by_cat.items():
        k = max(1, round(a.holdout * len(rs) / max(1, len(rows)))) if len(rs) > 10 else 0
        hold += rs[:k]; train += rs[k:]
    rnd.shuffle(train); out = W / "v2"; out.mkdir(exist_ok=True)
    for name, rs in (("train.jsonl", train), ("sft_eval.jsonl", hold)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"version": "sft_v2", "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "train": len(train), "sft_eval": len(hold), "per_category": dict(per_cat), "dropped": dict(dropped), "evalsets": counts,
           "teacher_split": dict(collections.Counter(r["teacher"] for r in rows)), "tokenizer": tax["tokenizer"], "chat_format": tax["chat_format"], "seed": a.seed,
           "targets": {c["name"]: c["target"] for c in tax["categories"]}, "licenses": {"generated": "Apache-2.0 (IbnSina release)", "persian_native sources": ["pn_summary MIT", "SynTran-fa MIT", "Wikipedia CC-BY-SA", "P3 Apache-2.0"], "smoltalk": "Apache-2.0"}}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1)); print(json.dumps({k: man[k] for k in ("train", "sft_eval", "dropped")}))
    for c, n in sorted(per_cat.items()): print(f"  {c:26s} {n:6d} / target {man['targets'][c]}")
    if not a.no_upload:
        import os; os.system(f"gcloud --no-user-output-enabled storage rsync -r {out} {B} && echo uploaded to {B}")


if __name__ == "__main__":
    main()
