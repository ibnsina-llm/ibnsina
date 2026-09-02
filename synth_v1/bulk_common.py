import os
#!/opt/pipe/bin/python3
"""Shared plumbing for the synth_v1 bulk pipeline: cost ledger (batch-priced), tolerant JSONL IO, resumable
wave state, Vertex batch submit/poll/parse, prompt building, and a generic online async runner as fallback.
House patterns from sft_v2/gen.py + judge.py."""
import asyncio, hashlib, json, random, re, subprocess, time
from pathlib import Path

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR-GCP-PROJECT")
GCS = os.environ.get("CORPUS_BUCKET", "gs://YOUR-BUCKET") + "/synth_v1"
BULK = Path("/data/synth_v1/bulk")
CODE = Path(__file__).resolve().parent
FLASH, LITE = "gemini-3.7-flash", "gemini-3.5-flash-lite"
# $/M tokens in/out — Vertex list 2026-08-31 (flash intro price doubles 2027-01-01); @batch = -50%
PRICES = {FLASH: (0.75, 3.75), LITE: (0.30, 2.50), FLASH + "@batch": (0.375, 1.875), LITE + "@batch": (0.15, 1.25)}
HARD_STOP_USD, NOTIFY_EVERY_USD, SYB_KEPT_TOKENS = 20000.0, 5000.0, 2_000_000_000  # SY-B ② ruling 2026-09-01: gate now 2B kept
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def read_jsonl(p):
    rows = []
    if not Path(p).exists(): return rows
    with open(p, encoding="utf-8") as f:
        for l in f:
            if not l.strip(): continue
            try: rows.append(json.loads(l))
            except Exception: pass                       # torn line — redone on resume
    return rows


def append_jsonl(p, rows):
    with open(p, "a", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jload(p, default):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def jdump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


class Ledger:
    def __init__(self, path=None):
        self.path = Path(path or BULK / "ledger.json")
        self.d = jload(self.path, {})
    def add(self, model, pin, pout, pth=0, n=1):
        e = self.d.setdefault(model, {"in": 0, "out": 0, "thoughts": 0, "calls": 0})
        e["in"] += pin; e["out"] += pout; e["thoughts"] += pth; e["calls"] += n
        self.path.write_text(json.dumps(self.d, indent=1))
    def usd(self):
        return sum(v["in"] / 1e6 * PRICES[m][0] + (v["out"] + v.get("thoughts", 0)) / 1e6 * PRICES[m][1]
                   for m, v in self.d.items() if m in PRICES)


def budget_events(ledger, state):
    """Notify at every $5k crossing (marker file + loud log line); return current spend."""
    usd = ledger.usd()
    while usd >= state.get("usd_notified", 0) + NOTIFY_EVERY_USD:
        state["usd_notified"] = state.get("usd_notified", 0) + NOTIFY_EVERY_USD
        marker = BULK / f"NOTIFY_{int(state['usd_notified'])}"
        marker.write_text(f"cumulative ledger crossed ${state['usd_notified']:.0f} at {time.strftime('%FT%TZ', time.gmtime())}\n")
        print(f"!!!! BUDGET NOTIFY: ledger crossed ${state['usd_notified']:.0f} (now ${usd:.2f}) — tell Sina !!!!", flush=True)
    return usd


def gcs(*args, capture=True):
    return subprocess.run(["gcloud", "storage", *args], capture_output=capture, text=True, check=True)


_CLIENTS = {}


def genai_client(location="global"):
    # Cached, module-lifetime clients. A temporary `genai.Client(...).batches.create(...)` is refcount-dropped
    # right after the attribute fetch; its finalizer closes the shared httpx transport mid-request
    # ("Cannot send a request, as the client has been closed") — bit wave 1 twice on 2026-08-31.
    if location not in _CLIENTS:
        from google import genai
        _CLIENTS[location] = genai.Client(vertexai=True, project=PROJECT, location=location)
    return _CLIENTS[location]


def submit_batch(model, src_uri, dest_uri, location="global", retries=8):
    from google.genai.types import CreateBatchJobConfig
    for attempt in range(retries):
        try:
            job = genai_client(location).batches.create(model=model, src=src_uri, config=CreateBatchJobConfig(dest=dest_uri))
            return {"name": job.name, "location": location, "model": model, "src": src_uri, "dest": dest_uri, "state": job_state(job)}
        except Exception as e:
            msg = str(e)
            if attempt == retries - 1 or not any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "quota", "Quota", "503", "502", "rate")): raise
            print(f"  submit retry ({model}): {msg[:120]}", flush=True); time.sleep(180)


def job_state(job):
    s = str(getattr(job, "state", ""))
    return s.split(".")[-1]                              # JobState.JOB_STATE_X -> JOB_STATE_X


def poll_batch(info):
    j = genai_client(info["location"]).batches.get(name=info["name"])
    return job_state(j), (str(getattr(j, "error", "")) or "")[:300]


def batch_line(rid, system, user, model, temperature, max_tokens, json_mime=False):
    gc = {"temperature": temperature, "max_output_tokens": max_tokens}
    if json_mime: gc["response_mime_type"] = "application/json"
    if model == FLASH: gc["thinking_config"] = {"thinking_budget": 0}
    req = {"contents": [{"role": "user", "parts": [{"text": user}]}], "generation_config": gc}
    if system: req["system_instruction"] = {"parts": [{"text": system}]}
    return {"id": rid, "request": req}


def parse_batch_line(line):
    """-> (rid, text, usage{in,out,thoughts}, error_str)"""
    try: d = json.loads(line)
    except Exception: return None, "", None, "unparseable_line"
    rid = d.get("id")
    resp = d.get("response")
    if not resp:
        return rid, "", None, json.dumps(d.get("status") or d.get("error") or "no_response")[:200]
    um = resp.get("usageMetadata") or resp.get("usage_metadata") or {}
    usage = {"in": um.get("promptTokenCount") or um.get("prompt_token_count") or 0,
             "out": um.get("candidatesTokenCount") or um.get("candidates_token_count") or 0,
             "thoughts": um.get("thoughtsTokenCount") or um.get("thoughts_token_count") or 0}
    try:
        parts = resp["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except Exception:
        return rid, "", usage, "no_candidates"
    return rid, text, usage, None


def list_output_jsonls(dest_uri):
    try: out = gcs("ls", "-r", dest_uri).stdout
    except Exception: return []
    return [l.strip() for l in out.splitlines() if l.strip().endswith(".jsonl")]


def parse_doc(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t); t = re.sub(r"\n```\s*$", "", t).strip()
    m = re.match(r"^#\s+(.+)", t)
    if not m or len(t) < 700: return None
    return m.group(1).strip(), t


def load_templates(prompts_dir=None):
    d = Path(prompts_dir or CODE / "prompts")
    return {k: (d / f"{k}.md").read_text(encoding="utf-8") for k in ("gen_system", "gen_user", "topics_user", "judge")}


def build_gen_prompt(tax, templates, anchors, sc):
    rnd = random.Random(sc["id"])
    anchor = anchors[rnd.randrange(len(anchors))]
    dom = next(d for d in tax["domains"] if d["name"] == sc["domain"])
    extra = []
    if dom.get("gen_extra_fa"): extra.append("- " + dom["gen_extra_fa"])
    if sc.get("aspect"): extra.append("- زاویهٔ نگاه این سند: " + sc["aspect"])
    user = (templates["gen_user"].replace("{ANCHOR}", anchor["text"]).replace("{DOMAIN}", sc["domain"])
            .replace("{SUBDOMAIN}", sc["subdomain"]).replace("{TOPIC}", sc["topic"])
            .replace("{DOC_TYPE}", sc["doc_type"]).replace("{DOC_TYPE_NOTE}", tax["doc_types"][sc["doc_type"]])
            .replace("{AUDIENCE}", dom["audience_fa"]).replace("{LENGTH_TOKENS}", str(sc["length"]))
            .replace("{LENGTH_WORDS}", str(int(sc["length"] / 1.7))).replace("{VARIATION}", str(sc["variation"]))
            .replace("{EXTRA}", "\n".join(extra)))
    return templates["gen_system"], user


def build_judge_prompt(templates, d):
    return (templates["judge"].replace("{DOMAIN}", d["domain"]).replace("{SUBDOMAIN}", d["subdomain"])
            .replace("{DOC_TYPE}", d["doc_type"]).replace("{TOPIC}", d["topic"]).replace("{DOCUMENT}", d["text"]))


def routing(tax):
    r = tax["teachers"]["routing"]
    return lambda domain: r.get(domain, r["default"]) if r.get(domain, r["default"]) in (FLASH, LITE) else r["default"]


async def online_run(items, model, temperature, max_tokens, json_mime, concurrency, on_result, cap_fn=None):
    """Fallback online runner: items = [(rid, system, user)]; on_result(rid, text, usage) or on_result(rid, None, err)."""
    from google.genai import types
    client = genai_client()
    sem = asyncio.Semaphore(concurrency)
    q = asyncio.Queue()
    for it in items: q.put_nowait(it)
    async def one(rid, system, user):
        if cap_fn and cap_fn(): return
        kw = dict(temperature=temperature, max_output_tokens=max_tokens)
        if system: kw["system_instruction"] = system
        if json_mime: kw["response_mime_type"] = "application/json"
        if model == FLASH: kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        cfg = types.GenerateContentConfig(**kw)
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.aio.models.generate_content(model=model, contents=user, config=cfg); break
                except Exception as e:
                    if attempt == 3 or not any(x in str(e) for x in ("429", "RESOURCE_EXHAUSTED", "503", "502", "timed out", "Timeout", "overloaded", "DEADLINE")):
                        on_result(rid, None, f"{type(e).__name__}: {str(e)[:120]}"); return
                    await asyncio.sleep(5 * 2 ** attempt)
        u = r.usage_metadata
        on_result(rid, r.text or "", {"in": u.prompt_token_count or 0, "out": u.candidates_token_count or 0,
                                      "thoughts": getattr(u, "thoughts_token_count", 0) or 0})
    async def worker():
        while True:
            try: it = q.get_nowait()
            except asyncio.QueueEmpty: return
            await one(*it)
    await asyncio.gather(*(worker() for _ in range(concurrency)))
