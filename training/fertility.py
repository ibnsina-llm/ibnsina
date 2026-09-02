#!/opt/pipe/bin/python3
"""T0 deliverable — fertility report: tokens per word and bytes per token on held-out text (val/ shards, never in training)
for our nanochat tokenizer vs Qwen3.5 vs Gemma 3. Writes /data/tok/fertility_report.md (+ .json).

  /opt/pipe/bin/python3 fertility.py --ours /data/tok/tokenizer --chars-per-slice 20e6
"""
import argparse, json, pickle, re, time
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq

VAL = Path("/data/mix/train_v1_open/val")
GROUPS = {"Persian web": ("fa_web",), "Persian literature/poetry": ("fa_lit",), "Persian Wikipedia": ("wiki", "fawiki"), "English edu": ("en_edu",),
          "English Wikipedia": ("wiki", "enwiki"), "Code (py/ts)": ("code_pyts",), "Code (Stack Overflow)": ("code_other",), "Math (EN)": ("math",), "Parallel fa-en": ("parallel",)}
WORD = re.compile(r"\S+")


def load_ours(d):
    enc = pickle.load(open(Path(d) / "tokenizer.pkl", "rb"))
    return lambda s: enc.encode_ordinary(s)


def load_hf(name):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    return lambda s: t.encode(s, add_special_tokens=False), t.vocab_size if hasattr(t, "vocab_size") else len(t)


def held_out(chars_per_slice):
    got = defaultdict(list); need = {g: chars_per_slice for g in GROUPS}
    for p in sorted(VAL.glob("shard_*.parquet")):
        b = pq.read_table(p, columns=["text", "source", "slice"])
        for t, s, sl in zip(b.column("text").to_pylist(), b.column("source").to_pylist(), b.column("slice").to_pylist()):
            for g, key in GROUPS.items():
                if sl == key[0] and (len(key) == 1 or s == key[1]) and sum(map(len, got[g])) < need[g]:
                    got[g].append(t)
    return got


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ours", default="/data/tok/tokenizer"); ap.add_argument("--chars-per-slice", type=float, default=20e6)
    a = ap.parse_args()
    texts = held_out(int(a.chars_per_slice))
    toks = {"ours (nanochat BPE 32k)": (load_ours(a.ours), 32768)}
    for label, name in (("Qwen3.5 (Qwen/Qwen3.5-9B)", "Qwen/Qwen3.5-9B"), ("Gemma 3 (google/gemma-3, 262k)", "unsloth/gemma-3-1b-it")):
        try:
            f, v = load_hf(name); toks[label] = (f, v)
        except Exception as e:
            print(f"!! could not load {name}: {type(e).__name__}: {str(e)[:120]}")
    res = {}
    for g, docs in texts.items():
        words = sum(len(WORD.findall(t)) for t in docs); nbytes = sum(len(t.encode()) for t in docs); nchars = sum(map(len, docs))
        res[g] = {"docs": len(docs), "words": words, "bytes": nbytes, "chars": nchars, "tok": {}}
        for label, (f, v) in toks.items():
            t0 = time.time(); n = sum(len(f(t)) for t in docs)
            res[g]["tok"][label] = {"tokens": n, "tokens_per_word": n / max(1, words), "bytes_per_token": nbytes / max(1, n), "chars_per_token": nchars / max(1, n), "sec": round(time.time() - t0, 1)}
        print(g, {k: round(v["tokens_per_word"], 3) for k, v in res[g]["tok"].items()}, flush=True)
    labels = list(toks)
    md = ["# Tokenizer fertility — held-out text (val split of train_v1_open, never trained on)", "",
          "Tokens per whitespace-separated word (lower = more efficient) and bytes per token (higher = more efficient). Same held-out documents for every tokenizer.", "",
          "| text | words | " + " | ".join(f"{l}<br>tok/word" for l in labels) + " | " + " | ".join(f"{l}<br>bytes/tok" for l in labels) + " |",
          "|---|---:|" + "---:|" * (2 * len(labels))]
    for g, r in res.items():
        md.append(f"| {g} | {r['words']:,} | " + " | ".join(f"{r['tok'][l]['tokens_per_word']:.3f}" for l in labels) + " | " + " | ".join(f"{r['tok'][l]['bytes_per_token']:.2f}" for l in labels) + " |")
    md += ["", "Vocab sizes: " + ", ".join(f"{l}: {toks[l][1]:,}" for l in labels), "",
           "Reading guide: a 32k vocab cannot match a 150k–262k vocab on raw compression; the point is that Persian is not penalised — compare the Persian rows to the English rows within each column, and ours vs. the big vocabs on Persian specifically."]
    Path("/data/tok/fertility_report.md").write_text("\n".join(md)); Path("/data/tok/fertility_report.json").write_text(json.dumps(res, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
