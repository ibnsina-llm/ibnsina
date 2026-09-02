#!/opt/pipe/bin/python3
"""One-off: expose the synth_v1 shard set to the mixer as a clean/-layer source.

Reads the judge-filtered synth_v1 parquet shards (id, text, ... columns), writes
clean/synth_v1/part-NNNN.jsonl.gz in the pipeline's clean-layer format plus _stats.json,
and uploads both to $CORPUS_BUCKET/clean/synth_v1/. Deterministic order (sorted shards).

  synth_to_clean.py [--shards /data/synth_v1/bulk/shards] [--out /data/clean/synth_v1] [--parts 8]
"""
import argparse, glob, gzip, json, os, sys, subprocess
from pathlib import Path

import orjson
import pyarrow.parquet as pq

BUCKET = os.environ.get("CORPUS_BUCKET", "gs://YOUR-BUCKET")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="/data/synth_v1/bulk/shards")
    ap.add_argument("--out", default="/data/clean/synth_v1")
    ap.add_argument("--parts", type=int, default=8)
    a = ap.parse_args()
    paths = sorted(glob.glob(str(Path(a.shards) / "wave-*.parquet")))
    if not paths:
        sys.exit(f"no shards under {a.shards}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    writers = [gzip.open(out / f"part-{i:04d}.jsonl.gz", "wb", compresslevel=6) for i in range(a.parts)]
    docs = chars = 0
    for p in paths:
        for r in pq.read_table(p).to_pylist():
            row = {"id": r["id"], "text": r["text"],
                   "meta": {"domain": r.get("domain"), "doc_type": r.get("doc_type"),
                            "teacher": r.get("teacher"), "judge_overall": r.get("judge_overall")}}
            writers[docs % a.parts].write(orjson.dumps(row) + b"\n")
            docs += 1; chars += len(r["text"])
        print(f"[synth_to_clean] {Path(p).name}: cum docs={docs:,} chars={chars/1e9:.2f}B", flush=True)
    for w in writers:
        w.close()
    stats = {"dataset": "synth_v1", "lang": "fa", "docs_out": docs, "chars_out": chars,
             "note": "synth_v1 judge-filtered shards converted for the mixer (synth_to_clean.py)"}
    (out / "_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[synth_to_clean] done: {docs:,} docs, {chars/1e9:.2f}B chars -> {out}")
    subprocess.run(["gsutil", "-m", "-q", "rsync", "-r", str(out), f"{BUCKET}/clean/synth_v1"], check=True)
    print(f"[synth_to_clean] uploaded to {BUCKET}/clean/synth_v1")


if __name__ == "__main__":
    main()
