#!/usr/bin/env python3
"""Corpus packer for the CPT lane (cpt/train_cpt.py consumes the output). CPU-only, multiprocess.

In : parquet shards with a `text` column (pipeline convention: train_v1_1_open / synth_v1 text shards).
Out: OUT/shard_FFFF_PP.npy  — uint32 array, shape (n_rows, seq_len+1).
     Docs are tokenized with the BASE model's tokenizer (Qwen3 for the 30B lane — NOT our v2 tokenizer,
     per cpt_stack_plan.md §5), concatenated into one stream with a single EOS between docs, and sliced
     into fixed rows of seq_len+1 tokens. Rows cross document boundaries (standard packed CPT); EOS is
     the only separator, no BOS. train_cpt.py feeds a whole row as input_ids AND labels (HF shifts
     internally -> exactly seq_len targets per row).
     OUT/manifest.json — tokenizer id, eos_id, seq_len, row_len, dtype, per-shard {file,n_seqs,n_tokens},
     totals + dropped tail count. train_cpt.py refuses a manifest whose row_len != its seq_len+1.
Parallelism: one task per input parquet; the sub-row tail of each input file is DROPPED (counted in the
manifest) — simpler than cross-file carryover and loses < row_len tokens per input file.

usage: pack_data.py --in-glob '/data/cpt/mix/*.parquet' --out-dir /data/cpt/data/train_v1_1_qwen3 \
                    --tokenizer Qwen/Qwen3-30B-A3B-Base [--seq-len 4096] [--workers N]
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

_TOK = None  # per-worker tokenizer (loaded once in the pool initializer)
_C = None    # per-worker config dict


def _init(tok_id, conf):
    global _TOK, _C
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    _C = conf


def _pack_file(task):
    """Tokenize+pack one parquet file -> zero or more .npy shards named by input index."""
    idx, path = task
    import pyarrow.parquet as pq
    row_len = _C["seq_len"] + 1
    eos = _TOK.eos_token_id
    assert eos is not None, "tokenizer has no eos_token_id"  # TODO-VERIFY: Qwen3 base eos = <|endoftext|>
    out_dir = Path(_C["out_dir"])
    state = {"buf": [], "buf_n": 0, "part": 0}
    shards = []
    docs = 0
    toks = 0

    def flush(final=False):
        cat = np.concatenate(state["buf"]) if state["buf"] else np.empty(0, dtype=np.uint32)
        n_rows = len(cat) // row_len
        cap = _C["rows_per_shard"]
        while n_rows >= cap or (final and n_rows > 0):
            take = min(n_rows, cap)
            f = f"shard_{idx:04d}_{state['part']:02d}.npy"
            np.save(out_dir / f, cat[: take * row_len].reshape(take, row_len))
            shards.append({"file": f, "n_seqs": int(take), "n_tokens": int(take * row_len)})
            state["part"] += 1
            cat = cat[take * row_len:]
            n_rows = len(cat) // row_len
        state["buf"] = [cat]
        state["buf_n"] = len(cat)
        return len(cat)  # remaining (sub-row) tail

    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(columns=["text"], batch_size=_C["batch_docs"]):
        texts = [t for t in rb.column("text").to_pylist() if t]
        if not texts:
            continue
        for ids in _TOK(texts, add_special_tokens=False)["input_ids"]:
            if not ids:
                continue
            state["buf"].append(np.array(ids + [eos], dtype=np.uint32))
            state["buf_n"] += len(ids) + 1
            docs += 1
            toks += len(ids) + 1
        if state["buf_n"] >= _C["rows_per_shard"] * row_len:
            flush()
    dropped = flush(final=True)
    return {"input": os.path.basename(path), "docs": docs, "tokens": toks,
            "dropped_tail": int(dropped), "shards": shards}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-glob", required=True, help="input parquet files with a `text` column")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tokenizer", required=True,
                   help="BASE model tokenizer path/id (e.g. Qwen/Qwen3-30B-A3B-Base) — NOT our v2 tokenizer")
    p.add_argument("--seq-len", type=int, default=int(os.environ.get("CPT_SEQ_LEN", 4096)))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--rows-per-shard", type=int, default=16384)  # 16384 x 4097 x 4B ~ 268 MB/shard
    p.add_argument("--batch-docs", type=int, default=256)        # docs per tokenizer batch call
    args = p.parse_args()

    files = sorted(glob.glob(args.in_glob))
    assert files, f"no inputs match {args.in_glob}"
    os.makedirs(args.out_dir, exist_ok=True)
    conf = {"seq_len": args.seq_len, "out_dir": args.out_dir,
            "rows_per_shard": args.rows_per_shard, "batch_docs": args.batch_docs}
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init, initargs=(args.tokenizer, conf)) as pool:
        results = pool.map(_pack_file, list(enumerate(files)))

    from transformers import AutoTokenizer
    eos = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True).eos_token_id
    shards = sorted((s for r in results for s in r["shards"]), key=lambda s: s["file"])
    man = {"tokenizer": args.tokenizer, "eos_id": int(eos), "seq_len": args.seq_len,
           "row_len": args.seq_len + 1, "dtype": "uint32", "shards": shards,
           "total_seqs": sum(s["n_seqs"] for s in shards),
           "total_tokens": sum(s["n_tokens"] for s in shards),
           "docs": sum(r["docs"] for r in results),
           "dropped_tail_tokens": sum(r["dropped_tail"] for r in results),
           "inputs": [r["input"] for r in results],
           "created": time.strftime("%FT%TZ", time.gmtime())}
    json.dump(man, open(Path(args.out_dir) / "manifest.json", "w"), indent=1)
    print(f"packed {man['docs']} docs -> {man['total_seqs']} rows x {man['row_len']} "
          f"({man['total_tokens'] / 1e9:.3f}B tokens, dropped {man['dropped_tail_tokens']} tail tokens) "
          f"in {time.time() - t0:.0f}s across {len(files)} inputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
