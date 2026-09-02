#!/opt/pipe/bin/python3
"""STOP 1 report: per-dataset table from clean/*/_stats.json + token estimates.

Token estimate = chars_out x (tokens/char measured on a sample of that dataset) with
  (a) a 64k byte-level BPE trained once on a mixed sample of the clean outputs (proxy for nanochat's own tokenizer), and
  (b) cl100k_base for reference.
Writes clean/_report_phase0.md and clean/_report_phase0.json; tokenizer cached at clean/_tokenizer/bpe64k.json.
"""
from __future__ import annotations
import gzip, io, json, random, subprocess, sys, time
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).parent))
from common import BUCKET, DATA, gcs_list, gsutil, log

SAMPLE_CHARS = 3_000_000            # per dataset, for the tokens/char ratio
TRAIN_CHARS = 120_000_000           # mixed sample for the BPE
WORK = DATA / "report"; WORK.mkdir(parents=True, exist_ok=True)


def datasets_done():
    out = {}
    for url, _ in gcs_list(f"{BUCKET}/clean", "*/_stats.json"):
        name = url.split("/")[-2]
        for attempt in range(3):  # a stats file can be mid-rewrite by the incremental web watcher
            txt = subprocess.run(["gcloud", "storage", "cat", url], capture_output=True, text=True).stdout
            try:
                out[name] = json.loads(txt); break
            except Exception:
                time.sleep(5)
        else:
            log(f"[report] WARNING: could not read {url} — dataset skipped in this run")
    return out


def sample_text(name, max_chars, seed=0):
    """Read docs from up to 3 random shards until max_chars collected."""
    shards = [u for u, _ in gcs_list(f"{BUCKET}/clean/{name}", "*part-*.jsonl.gz")]
    if not shards:
        return ""
    rnd = random.Random(seed); rnd.shuffle(shards)
    buf, n = [], 0
    for u in shards[:3]:
        raw = subprocess.run(["gcloud", "storage", "cat", u], capture_output=True).stdout
        for line in io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw)), encoding="utf-8", errors="ignore"):
            try:
                t = orjson.loads(line)["text"]
            except Exception:
                continue
            buf.append(t); n += len(t)
            if n >= max_chars:
                return "\n".join(buf)
    return "\n".join(buf)


def get_tokenizer(stats):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    cache = WORK / "bpe64k.json"; remote = f"{BUCKET}/clean/_tokenizer/bpe64k.json"
    if not cache.exists():
        gsutil("cp", remote, str(cache), check=False)
    if cache.exists() and cache.stat().st_size > 0:
        return Tokenizer.from_file(str(cache))
    # mixed sample: weight roughly like the target mix (fa web/wiki/lit 70%, en 20%, code 10%)
    weights = {}
    for name, s in stats.items():
        cat, lang = s.get("category", ""), s.get("lang", "")
        weights[name] = 0.7 if lang == "fa" else 0.2 if lang == "en" else 0.1 if lang == "code" else 0.05
    tot = sum(weights.values()); texts = []
    for name, w in weights.items():
        t = sample_text(name, int(TRAIN_CHARS * w / tot), seed=1)
        if t:
            texts.append(t); log(f"[tok] sample {name}: {len(t):,} chars")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=65536, min_frequency=2, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    log(f"[tok] training 64k BPE on {sum(len(t) for t in texts)/1e6:.0f}M chars")
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(cache)); gsutil("cp", str(cache), remote)
    return tok


def main():
    import tiktoken
    stats = datasets_done()
    if not stats:
        log("no finished datasets"); return
    tok = get_tokenizer(stats); enc = tiktoken.get_encoding("cl100k_base")
    rows = []
    for name in sorted(stats):
        s = stats[name]; t = sample_text(name, SAMPLE_CHARS, seed=2)
        if t:
            r_bpe = len(tok.encode(t).ids) / len(t); r_cl = len(enc.encode(t, disallowed_special=())) / len(t)
        else:
            r_bpe = r_cl = 0.0
        rej = s.get("rejects", {}); top = ", ".join(f"{k} {v:,}" for k, v in list(rej.items())[:3]) or "-"
        rows.append({"dataset": name, "category": s.get("category"), "lang": s.get("lang"), "docs_in": s.get("docs_in", 0), "docs_out": s.get("docs_out", 0),
                     "chars_out": s.get("chars_out", 0), "tok_per_char_bpe64k": round(r_bpe, 3), "est_tokens_bpe64k": int(s.get("chars_out", 0) * r_bpe),
                     "est_tokens_cl100k": int(s.get("chars_out", 0) * r_cl), "bp_lines_dropped": s.get("lines_dropped_boilerplate", 0), "top_rejects": top,
                     "seconds": s.get("seconds")})
        log(f"[report] {name}: {rows[-1]['est_tokens_bpe64k']/1e9:.2f}B tokens (bpe64k), {rows[-1]['est_tokens_cl100k']/1e9:.2f}B (cl100k)")
    tot_b = sum(r["est_tokens_bpe64k"] for r in rows); tot_c = sum(r["est_tokens_cl100k"] for r in rows)
    md = ["# Phase 0 report — extract + normalize", f"_generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_", "",
          "| dataset | category | lang | docs in | docs out | kept | chars | tok/char (bpe64k) | est tokens (bpe64k) | est tokens (cl100k) | boilerplate lines dropped | top rejects |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        kept = f"{100*r['docs_out']/max(1,r['docs_in']):.1f}%"
        md.append(f"| {r['dataset']} | {r['category']} | {r['lang']} | {r['docs_in']:,} | {r['docs_out']:,} | {kept} | {r['chars_out']/1e9:.2f}G | {r['tok_per_char_bpe64k']} | "
                  f"{r['est_tokens_bpe64k']/1e9:.2f}B | {r['est_tokens_cl100k']/1e9:.2f}B | {r['bp_lines_dropped']:,} | {r['top_rejects']} |")
    md += ["", f"**Total est. tokens: {tot_b/1e9:.1f}B (bpe64k) / {tot_c/1e9:.1f}B (cl100k)**", "",
           "bpe64k = 64k byte-level BPE trained on a mixed sample of these outputs (proxy for the nanochat tokenizer); cl100k over-counts Persian ~2x."]
    (WORK / "report_phase0.md").write_text("\n".join(md)); (WORK / "report_phase0.json").write_text(json.dumps(rows, indent=1))
    gsutil("cp", str(WORK / "report_phase0.md"), f"{BUCKET}/clean/_report_phase0.md"); gsutil("cp", str(WORK / "report_phase0.json"), f"{BUCKET}/clean/_report_phase0.json")
    print("\n".join(md))


if __name__ == "__main__":
    main()
