#!/opt/pipe/bin/python3
"""T2 SFT data v1 — Persian instruction data for the pilot's chat stage.
  * FarsInstruct (ParsiAI/FarsInstruct, Apache-2.0) — only subsets whose UPSTREAM license is permissive (see SUBSETS/EXCLUDED)
  * 500 SmolTalk conversations (HuggingFaceTB/smol-smoltalk, Apache-2.0) translated to Persian with Gemini 2.5 Flash (Vertex)
Text goes through the same normalize_fa() as pretraining. Output: /data/sft/v1/{train,val,smoltalk_fa}.jsonl + manifest.json -> gs://.../sft/v1/
Run on the pipeline VM:  /opt/pipe/bin/python3 /data/pipeline/training/sft_data.py
"""
import argparse, asyncio, hashlib, io, json, os, random, sys, time, urllib.request
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, "/data/pipeline")
from pipeline.common import normalize_fa

OUT = Path("/data/sft/v1"); B = "gs://YOUR-BUCKET/sft/v1"; FI = "ParsiAI/FarsInstruct"
SUBSETS = {  # config -> (upstream license, train cap, val cap, approx rows)
    "pn_sum": ("MIT (HooshvareLab/pn_summary)", 40000, 500, 974949), "syntran": ("MIT (SLPL/syntran-fa)", 40000, 500, 481060),
    "wiki_sum": ("CC-BY-SA-4.0 (Wikipedia)", 40000, 500, 450904), "p3_qa_translated": ("Apache-2.0 (bigscience P3, translated in FarsInstruct)", 40000, 500, 723070)}
EXCLUDED = {"parsinlu_fa_en, parsinlu_en_fa, parsinlu_sentiment, parsinlu_qpp, parsinlu_entailment, parsinlu_multiple_choice, parsinlu_comp": "CC-BY-NC-SA-4.0 (ParsiNLU)",
            "farstail": "CC-BY-NC-SA-4.0", "xl_wic": "CC-BY-NC-4.0",
            "persian_qa, exappc, snapp_sentiment, digimag, digi_sentiment, persian_ner, peyma, persian_news, pars_absa": "upstream license not verified -> not used in an Apache-2.0 release"}
SEED = 20260829


def hf_parquet_urls(repo, config, split):
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}") as r:
        return json.load(r)


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0: return dest
    tmp = dest.with_suffix(".tmp")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b: break
            f.write(b)
    tmp.rename(dest); return dest


def u01(key):
    return int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16) / 2 ** 64


def farsinstruct_rows(config, split, cap, approx):
    """deterministic sample of `cap` rows from one FarsInstruct config/split, streamed shard by shard"""
    holdout = lambda i: u01(f"{config}:valhold:{i}") < 0.002   # 0.2% of train is reserved as validation for configs without a validation split
    try:
        urls = hf_parquet_urls(FI, config, split); from_train = False
    except urllib.error.HTTPError as e:
        if split != "validation" or e.code != 404: raise
        print(f"  {config}: no validation split on the hub -> using the held-out 0.2% hash slice of train", flush=True)
        urls = hf_parquet_urls(FI, config, "train"); from_train = True
    p_keep = 1.0 if from_train else min(1.0, cap / max(1, approx if split == "train" else cap * 4) * 1.5)
    rows, seen, n = [], set(), 0
    for si, url in enumerate(urls):
        f = fetch(url, OUT / "_cache" / f"{config}.{'train' if from_train else split}.{si:03d}.parquet"); pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["inputs", "outputs", "template"])
            for inp, outp, tpl in zip(t.column("inputs").to_pylist(), t.column("outputs").to_pylist(), t.column("template").to_pylist()):
                n += 1
                if split == "train" and holdout(n): continue          # never train on the held-out slice
                if from_train and not holdout(n): continue            # validation-from-train: only the held-out slice
                if u01(f"{config}:{split}:{n}") > p_keep: continue
                inp, outp = normalize_fa((inp or "").strip()), normalize_fa((outp or "").strip())
                if not (20 <= len(inp) <= 4000 and 1 <= len(outp) <= 3000): continue
                h = hashlib.blake2b(inp.encode(), digest_size=8).hexdigest()
                if h in seen: continue
                seen.add(h); rows.append({"id": f"farsinstruct:{config}:{split}:{n}", "source": f"farsinstruct/{config}", "template": tpl,
                                          "messages": [{"role": "user", "content": inp}, {"role": "assistant", "content": outp}]})
        if len(rows) >= cap * 1.2: break
    random.Random(SEED).shuffle(rows); return rows[:cap], n


def smoltalk_candidates(n_want):
    urls = hf_parquet_urls("HuggingFaceTB/smol-smoltalk", "default", "train")
    f = fetch(urls[0], OUT / "_cache" / "smoltalk.000.parquet"); t = pq.read_table(f, columns=["messages"])
    cands = []
    for i, msgs in enumerate(t.column("messages").to_pylist()):
        if not (2 <= len(msgs) <= 4) or msgs[0]["role"] != "user": continue
        total = sum(len(m["content"]) for m in msgs)
        if not (150 <= total <= 1200) or any("```" in m["content"] or "http" in m["content"] for m in msgs): continue
        if any(m["role"] != ("user" if j % 2 == 0 else "assistant") for j, m in enumerate(msgs)): continue
        cands.append({"id": f"smoltalk:{i}", "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]})
    random.Random(SEED).shuffle(cands); return cands[:n_want]


async def translate_all(convs, model, concurrency):
    from google import genai
    from google.genai import types
    from pydantic import BaseModel
    class Msg(BaseModel):
        role: str
        content: str
    class Conv(BaseModel):
        messages: list[Msg]
    client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR-GCP-PROJECT"), location="us-central1")
    cfg = types.GenerateContentConfig(temperature=0.3, max_output_tokens=4096, response_mime_type="application/json", response_schema=Conv,
                                      thinking_config=types.ThinkingConfig(thinking_budget=0),
                                      system_instruction="You translate chat conversations from English into natural, fluent, contemporary Persian (Farsi) as a native speaker would write. "
                                      "Keep exactly the same number of messages, the same roles and the same order. Translate everything the user and assistant say; keep proper nouns, "
                                      "numbers, units, code identifiers and formatting as they are. Do not add, drop or summarise content. Return JSON only.")
    sem = asyncio.Semaphore(concurrency); out = [None] * len(convs); stats = {"ok": 0, "bad": 0}
    async def one(i, c):
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.aio.models.generate_content(model=model, contents=json.dumps({"messages": c["messages"]}, ensure_ascii=False), config=cfg)
                    d = json.loads(r.text); msgs = d["messages"]
                    assert len(msgs) == len(c["messages"]) and all(m["role"] == o["role"] and m["content"].strip() for m, o in zip(msgs, c["messages"]))
                    out[i] = {"id": c["id"] + ":fa", "source": "smoltalk_fa", "messages": [{"role": m["role"], "content": normalize_fa(m["content"].strip())} for m in msgs], "en": c["messages"]}
                    stats["ok"] += 1; return
                except Exception as e:  # noqa
                    err = f"{type(e).__name__}: {str(e)[:100]}"; await asyncio.sleep(2 * (attempt + 1))
            stats["bad"] += 1; print(f"  drop {c['id']}: {err}", flush=True)
    await asyncio.gather(*(one(i, c) for i, c in enumerate(convs)))
    print(f"translated ok={stats['ok']} dropped={stats['bad']}", flush=True)
    return [o for o in out if o]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gemini-2.5-flash"); ap.add_argument("--smoltalk", type=int, default=550); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-upload", action="store_true"); a = ap.parse_args()
    (OUT / "_cache").mkdir(parents=True, exist_ok=True); t0 = time.time(); man = {"seed": SEED, "sources": {}, "excluded": EXCLUDED, "gemini_model": a.model}
    train, val = [], []
    for cfg, (lic, cap, vcap, approx) in SUBSETS.items():
        tr, n = farsinstruct_rows(cfg, "train", cap, approx); va, _ = farsinstruct_rows(cfg, "validation", vcap, approx)
        train += tr; val += va; man["sources"][f"farsinstruct/{cfg}"] = {"license": lic, "train": len(tr), "val": len(va), "scanned": n}
        print(f"{cfg}: train {len(tr)} val {len(va)} (scanned {n:,}) {time.time()-t0:.0f}s", flush=True)
    cands = smoltalk_candidates(a.smoltalk); print(f"smoltalk candidates: {len(cands)}", flush=True)
    fa = asyncio.run(translate_all(cands, a.model, a.concurrency))
    st_val, st_train = fa[:50], fa[50:]
    man["sources"]["smoltalk_fa"] = {"license": "Apache-2.0 (HuggingFaceTB/smol-smoltalk), translated with " + a.model, "train": len(st_train), "val": len(st_val)}
    random.Random(SEED).shuffle(train)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val + st_val), ("smoltalk_fa.jsonl", st_train)):
        with open(OUT / name, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man["totals"] = {"train": len(train), "val": len(val) + len(st_val), "smoltalk_fa": len(st_train), "seconds": round(time.time() - t0)}
    (OUT / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1)); print(json.dumps(man["totals"]))
    if not a.no_upload:
        os.system(f"gcloud --no-user-output-enabled storage cp {OUT}/train.jsonl {OUT}/val.jsonl {OUT}/smoltalk_fa.jsonl {OUT}/manifest.json {B}/ && echo uploaded to {B}")


if __name__ == "__main__":
    main()
