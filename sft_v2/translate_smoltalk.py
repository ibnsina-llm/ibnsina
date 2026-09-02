#!/opt/pipe/bin/python3
"""Extra general_instruction candidates: multi-turn SmolTalk conversations (Apache-2.0) translated to Persian with Gemini 2.5 Flash.
Writes candidate rows into /data/sft_v2/candidates/general_instruction.jsonl (judged like everything else).
  translate_smoltalk.py --n 3000 [--concurrency 8]"""
import argparse, asyncio, json, random, sys
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, "/data/pipeline/training"); sys.path.insert(0, "/data/pipeline")
from sft_data import hf_parquet_urls, fetch, translate_all
OUT = Path("/data/sft_v2/candidates"); CACHE = Path("/data/sft/v1/_cache")


def candidates(n, seed=20260829):
    urls = hf_parquet_urls("HuggingFaceTB/smol-smoltalk", "default", "train"); rows = []
    for si, u in enumerate(urls[:3]):
        t = pq.read_table(fetch(u, CACHE / f"smoltalk.{si:03d}.parquet"), columns=["messages"])
        for i, msgs in enumerate(t.column("messages").to_pylist()):
            if not (4 <= len(msgs) <= 8) or msgs[0]["role"] != "user": continue
            total = sum(len(m["content"]) for m in msgs)
            if not (300 <= total <= 3000) or any("```" in m["content"] or "http" in m["content"] for m in msgs): continue
            if any(m["role"] != ("user" if j % 2 == 0 else "assistant") for j, m in enumerate(msgs)): continue
            rows.append({"id": f"smoltalk-{si}-{i}", "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]})
    random.Random(seed).shuffle(rows); return rows[:n]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--model", default="gemini-2.5-flash"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); fp = OUT / "general_instruction.jsonl"
    done = set()
    if fp.exists():
        for l in fp.read_text(encoding="utf-8").splitlines():
            if l.strip(): done.add(json.loads(l)["scenario_id"])
    cands = [c for c in candidates(a.n) if c["id"] not in done]; print(f"{len(cands)} multi-turn conversations to translate ({len(done)} already present)", flush=True)
    for start in range(0, len(cands), 200):
        fa = asyncio.run(translate_all(cands[start:start + 200], a.model, a.concurrency))
        with open(fp, "a", encoding="utf-8") as f:
            for r in fa:
                sid = r["id"].replace(":fa", "")
                f.write(json.dumps({"id": sid + "-0", "scenario_id": sid, "category": "general_instruction", "subtype": "smoltalk_multiturn_translated", "persona": None, "register": None, "length": None,
                                    "turns": len(r["messages"]) // 2, "domain": "smoltalk", "teacher": a.model + "/translation", "seeds_used": [], "prompt_hash": "smoltalk", "messages": r["messages"]}, ensure_ascii=False) + "\n")
        print(f"  {start + len(fa)} written", flush=True)


if __name__ == "__main__":
    main()
