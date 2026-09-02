#!/opt/pipe/bin/python3
"""Frontier comparison on PersianMedQA (test split) using the paper's zero-shot protocol (Ranjbar Kalahroodi et al. 2025):
English instruction, Persian question + 4 numbered options, temperature 0, answer = option number only. Runs any OpenRouter model id
(and gemini-* via Vertex). Resumable per model; writes /data/eval/frontier/<model>.jsonl and a summary with per-field accuracy.
  frontier_persianmedqa.py --models anthropic/claude-opus-5,openai/gpt-5.6 [--subset 1000] [--concurrency 8] --yes
DOES NOTHING without --yes (real-money spend)."""
import argparse, asyncio, collections, json, os, re, sys, time, urllib.error, urllib.request
from pathlib import Path
import pyarrow.parquet as pq

PROMPT = ("You are a medical expert tasked with answering multiple-choice medical questions.\n\n"
          "Question: {q}\n1: {o1}\n2: {o2}\n3: {o3}\n4: {o4}\n\n"
          "Important Notes:\n- Select the best answer from the provided choices.\n- Your output must be only the option number (1, 2, 3, or 4).\n"
          "- Do not add explanations or extra text.\n- Base your answers on authoritative medical knowledge.")
OUT = Path("/data/eval/frontier"); DATA = "/data/eval/persianmedqa/test.parquet"


def load(subset, seed):
    import ast, random
    rows = []
    for r in pq.read_table(DATA, columns=["question_id", "question", "answer", "correct answer", "field"]).to_pylist():
        try: d = json.loads(r["answer"])
        except Exception:
            try: d = ast.literal_eval(r["answer"])
            except Exception: continue
        if str(r["correct answer"]).strip() not in ("1", "2", "3", "4"): continue
        if not isinstance(d, dict) or any(k not in d for k in ("1", "2", "3", "4")): continue   # a few rows lack a full option set
        rows.append({"id": str(r["question_id"]), "q": r["question"], "o": [str(d[k]) for k in ("1", "2", "3", "4")], "a": str(r["correct answer"]).strip(), "field": str(r["field"])})
    if subset and subset < len(rows):
        rnd = random.Random(seed); by = collections.defaultdict(list)
        for r in rows: by[r["field"]].append(r)
        pick = []
        for f, rs in by.items():
            rnd.shuffle(rs); pick += rs[:max(1, round(subset * len(rs) / len(rows)))]
        rows = sorted(pick, key=lambda r: r["id"])[:subset]
    return rows


def parse_answer(text):
    t = (text or "").strip(); m = re.search(r"\b([1-4])\b", t.splitlines()[-1] if t else "") or re.search(r"[1-4]", t); return m.group(1) if m and m.lastindex else (m.group(0) if m else None)


REASONING_MANDATORY = set()   # model ids whose endpoint refuses reasoning: {enabled: false}; they get the flag omitted and room to think


BASE_URL = "https://openrouter.ai/api/v1"; NO_REASONING_FIELD = False


async def call_openrouter(model, key, prompt, sem, max_tokens=16):
    def body_for():
        b = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens}
        if model in REASONING_MANDATORY: b["max_tokens"] = max(max_tokens, 2048)   # thinking tokens count against the cap
        elif not NO_REASONING_FIELD: b["reasoning"] = {"enabled": False}
        return json.dumps(b).encode()
    def do():
        req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body_for(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/ibnsina-llm", "X-Title": "IbnSina PersianMedQA eval"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 400 and "Reasoning is mandatory" in msg and model not in REASONING_MANDATORY:
                REASONING_MANDATORY.add(model); print(f"[{model}] reasoning is mandatory on this endpoint -> thinking left on, max_tokens 2048", flush=True)
                with urllib.request.urlopen(urllib.request.Request(req.full_url, data=body_for(), headers=req.headers), timeout=300) as r: return json.loads(r.read())
            raise
    async with sem:
        for attempt in range(4):
            try:
                d = await asyncio.to_thread(do); u = d.get("usage", {})
                return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0), float(u.get("cost") or 0)
            except Exception as e:
                if attempt == 3: raise
                await asyncio.sleep(5 * 2 ** attempt)


async def call_vertex(client, model, prompt, sem):
    from google.genai import types
    # Gemini 3.x/2.5 Pro think by default and reject thinking_budget=0; leave thinking on, allow room for it, parse the final digit
    cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=2048)
    async with sem:
        r = await client.aio.models.generate_content(model=model, contents=prompt, config=cfg)
    return r.text, r.usage_metadata.prompt_token_count or 0, r.usage_metadata.candidates_token_count or 0, 0.0


async def run_model(model, rows, a, key, client):
    tag = a.tag_prefix + model.replace("/", "__"); fp = OUT / f"{tag}.jsonl"; done = {}
    if fp.exists():
        for l in fp.read_text(encoding="utf-8").splitlines():
            if l.strip(): r = json.loads(l); done[r["id"]] = r
    if a.retry_unparsed:
        bad = {k for k, v in done.items() if v["pred"] is None}
        for k in bad: done.pop(k)
        print(f"[{model}] retrying {len(bad)} unparsed rows with max_tokens {a.retry_max_tokens}", flush=True)
        if bad: fp.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in done.values()), encoding="utf-8")
    todo = [r for r in rows if r["id"] not in done]; sem = asyncio.Semaphore(a.concurrency); lock = asyncio.Lock(); t0 = time.time(); spent = [0.0]
    print(f"[{model}] {len(done)} done, {len(todo)} to run", flush=True)
    async def one(r):
        prompt = PROMPT.format(q=r["q"], o1=r["o"][0], o2=r["o"][1], o3=r["o"][2], o4=r["o"][3])
        try:
            text, pin, pout, cost = await (call_vertex(client, model, prompt, sem) if model.startswith("gemini") else call_openrouter(model, key, prompt, sem, a.retry_max_tokens if a.retry_unparsed else 16))
            rec = {"id": r["id"], "field": r["field"], "gold": r["a"], "raw": (text or "")[:40], "pred": parse_answer(text), "tokens": [pin, pout], "cost": cost}
        except Exception as e:
            rec = {"id": r["id"], "field": r["field"], "gold": r["a"], "raw": f"ERROR {type(e).__name__}", "pred": None, "tokens": [0, 0], "cost": 0.0}
        async with lock:
            done[r["id"]] = rec; spent[0] += rec["cost"]
            with open(fp, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    queue = asyncio.Queue()
    for r in todo: queue.put_nowait(r)
    async def worker():
        while True:
            try: r = queue.get_nowait()
            except asyncio.QueueEmpty: return
            await one(r)
    await asyncio.gather(*(worker() for _ in range(a.concurrency)))
    recs = [done[r["id"]] for r in rows if r["id"] in done]
    acc = sum(1 for x in recs if x["pred"] == x["gold"]) / max(1, len(recs)); errs = sum(1 for x in recs if x["pred"] is None)
    by_field = collections.defaultdict(lambda: [0, 0])
    for x in recs: by_field[x["field"]][1] += 1; by_field[x["field"]][0] += (x["pred"] == x["gold"])
    print(f"[{model}] acc {100*acc:.2f}% on {len(recs)} (unparsed {errs}) cost≈${sum(x['cost'] for x in recs):.2f} {time.time()-t0:.0f}s", flush=True)
    return {"model": model, "n": len(recs), "accuracy": acc, "unparsed": errs, "cost_usd": sum(x["cost"] for x in recs), "by_field": {f: {"acc": c / n, "n": n} for f, (c, n) in by_field.items()}}


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--subset", type=int, default=0); ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--yes", action="store_true")
    ap.add_argument("--retry-unparsed", action="store_true", help="re-query rows with no parsable answer, with a larger token cap"); ap.add_argument("--retry-max-tokens", type=int, default=1024)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible endpoint (e.g. a local vLLM/llama-server)"); ap.add_argument("--tag-prefix", default="", help="prefix for the output filename tag"); ap.add_argument("--no-reasoning-field", action="store_true", help="never send the reasoning field (local servers reject it)")
    a = ap.parse_args()
    global BASE_URL, NO_REASONING_FIELD; BASE_URL = a.base_url; NO_REASONING_FIELD = a.no_reasoning_field
    rows = load(a.subset, a.seed); models = a.models.split(",")
    if not a.yes: print(f"dry run: {len(rows)} questions × {len(models)} models = {len(rows)*len(models)} calls; nothing sent. Re-run with --yes."); return
    OUT.mkdir(parents=True, exist_ok=True); key = Path("/data/secrets/openrouter.key").read_text().strip(); client = None
    if any(m.startswith("gemini") for m in models):
        from google import genai; client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR-GCP-PROJECT"), location=os.environ.get("VERTEX_LOCATION", "global"))
    results = [await run_model(m, rows, a, key, client) for m in models]
    summ = OUT / "summary.json"; prev = json.loads(summ.read_text()) if summ.exists() else {}
    for r in results: prev[r["model"]] = r
    summ.write_text(json.dumps(prev, ensure_ascii=False, indent=1)); print(json.dumps({m: round(100 * v["accuracy"], 2) for m, v in prev.items()}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
