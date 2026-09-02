#!/opt/pipe/bin/python3
"""sft_v2 generation — two teachers (Gemini on Vertex, Kimi K3 via OpenRouter), scenario sampling from the taxonomy's diversity axes,
2 sibling candidates per scenario (for judge-selected DPO pairs), resumable, with a token/cost ledger.
  gen.py --taxonomy sft_taxonomy.yaml --seeds seeds/ --out /data/sft_v2/candidates [--categories a,b] [--per-category N] [--kimi-share 0.3] [--gemini-model gemini-2.5-flash]
Every prompt template lives in sft_v2/prompts/ (the recipe is publishable)."""
import argparse, asyncio, hashlib, json, os, random, re, sys, time, urllib.request
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
PRICES = {"gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40), "moonshotai/kimi-k3": (3.0, 15.0)}  # $/M tokens in/out (list prices, estimate)


def load_seeds(seed_dir, cat):
    p = Path(seed_dir) / f"{cat}.jsonl"
    if not p.exists(): return []
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if "messages" in r]


def render_seed(r):
    return json.dumps({"messages": r["messages"]}, ensure_ascii=False)


def sample_scenario(cat, axes, rnd):
    sub = None
    if cat.get("subtypes"):
        st = cat["subtypes"]
        if isinstance(st[0], dict):
            sub = rnd.choices([s["name"] for s in st], weights=[s.get("share", 1) for s in st])[0]
        else:
            sub = rnd.choice(st)
    turns = 1 if rnd.random() > cat.get("multi_turn_share", 0) else rnd.choice([2, 3, 4])
    if cat["name"] == "multiturn_repair": turns = max(turns, 3)
    domain = rnd.choice(cat.get("domains") or cat.get("rules") or [sub or cat["name"]])
    return {"subtype": sub, "turns": turns, "domain": domain, "persona": rnd.choice(axes["persona"]), "register": rnd.choice(axes["register"]),
            "length": rnd.choice(axes["length"]), "place": rnd.choice(axes["city_or_region"]), "variation": rnd.randrange(10 ** 6)}


def build_prompt(cat, sc, seeds, templates, rnd):
    sub_note = ""
    if sc["subtype"] and cat.get("subtypes") and isinstance(cat["subtypes"][0], dict):
        sub_note = next((s.get("note", "") for s in cat["subtypes"] if s["name"] == sc["subtype"]), "")
    rubric = cat["rubric"].strip() + (f"\nزیرگونه «{sc['subtype']}»: {sub_note}" if sub_note else "")
    content_note = ("For tool calls, assistant \"content\" is a LIST of parts: {\"type\":\"text\",\"text\":...}, {\"type\":\"python\",\"text\":...}, {\"type\":\"python_output\",\"text\":...}; otherwise a plain string."
                    if cat["name"] == "toolcall" else "Assistant \"content\" is a plain string.")
    system = templates["gen_system"].replace("{RUBRIC}", rubric).replace("{TURNS}", str(sc["turns"])).replace("{CONTENT_NOTE}", content_note)
    ex = rnd.sample(seeds, min(3, len(seeds)))
    extra = ""
    if cat["name"] == "toolcall": extra = ("- tools (call EXACTLY these as Python; nothing else exists): (1) calculator = any plain Python arithmetic expression, e.g. (1250000*0.23)+45000 or round(196/12, 2); "
                                          "(2) jalali_to_gregorian(jy, jm, jd) -> 'YYYY-MM-DD' and gregorian_to_jalali(gy, gm, gd) -> 'YYYY-MM-DD' (no other date functions, no arithmetic on date strings); "
                                          "(3) search('query') -> 1-3 short snippets. Every python part is immediately followed by a python_output part with exactly what Python would print; "
                                          "the pipeline re-executes calculator and date calls and rejects wrong outputs, so the final Persian text must use those exact values.")
    if cat["name"] == "persian_native": extra = "- include a realistic Persian source text (news/encyclopedic, 2–6 sentences) inside the user's message; the assistant must stay faithful to it"
    user = (templates["gen_user"].replace("{SEEDS}", "\n".join(render_seed(r) for r in ex)).replace("{CATEGORY}", cat["name"])
            .replace("{SUBTYPE_LINE}", f" / subtype: {sc['subtype']}" if sc["subtype"] else "").replace("{DOMAIN}", str(sc["domain"]))
            .replace("{PERSONA}", sc["persona"]).replace("{PLACE}", sc["place"]).replace("{REGISTER}", sc["register"]).replace("{LENGTH}", sc["length"])
            .replace("{TURNS}", str(sc["turns"])).replace("{VARIATION}", str(sc["variation"])).replace("{EXTRA}", extra))
    return system, user, [r.get("id") for r in ex]


def parse_conversation(text, expect_turns):
    start = text.find("{")
    if start < 0: return None
    try:
        d, _ = json.JSONDecoder().raw_decode(text[start:])   # first complete JSON object (models sometimes emit two, or trailing prose)
    except Exception:
        return None
    msgs = d.get("messages") if isinstance(d, dict) else None
    if not msgs or not isinstance(msgs, list): return None
    if isinstance(msgs[-1], dict) and msgs[-1].get("role") == "user" and len(msgs) > 1: msgs = msgs[:-1]
    for i, mm in enumerate(msgs):
        if mm.get("role") != ("user" if i % 2 == 0 else "assistant"): return None
        c = mm.get("content")
        if isinstance(c, str):
            if not c.strip(): return None
        elif isinstance(c, list):
            if not all(isinstance(p, dict) and p.get("type") in ("text", "python", "python_output") and isinstance(p.get("text"), str) for p in c): return None
        else:
            return None
    if msgs and msgs[-1].get("role") == "user": msgs = msgs[:-1]        # models often leave a dangling user turn
    if len(msgs) % 2: return None
    n_turns = len(msgs) // 2
    if n_turns < 1 or abs(n_turns - expect_turns) > 1: return None       # ±1 turn tolerated; the judge grades the rest
    return msgs


class Ledger:
    def __init__(self, path):
        self.path = Path(path); self.d = json.loads(self.path.read_text()) if self.path.exists() else {}
    def add(self, model, pin, pout):
        e = self.d.setdefault(model, {"in": 0, "out": 0, "calls": 0}); e["in"] += pin; e["out"] += pout; e["calls"] += 1
        self.path.write_text(json.dumps(self.d, indent=1))
    def usd_for(self, model):
        v = self.d.get(model); return 0.0 if not v else v["in"] / 1e6 * PRICES.get(model, (1, 5))[0] + v["out"] / 1e6 * PRICES.get(model, (1, 5))[1]
    def usd(self):
        return sum(v["in"] / 1e6 * PRICES.get(m, (1, 5))[0] + v["out"] / 1e6 * PRICES.get(m, (1, 5))[1] for m, v in self.d.items())


async def call_gemini(client, model, system, user, temperature, sem):
    from google.genai import types
    cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature, max_output_tokens=4096, response_mime_type="application/json",
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    async with sem:
        r = await client.aio.models.generate_content(model=model, contents=user, config=cfg)
    u = r.usage_metadata
    return r.text, (u.prompt_token_count or 0), (u.candidates_token_count or 0)


async def call_openrouter(model, key, system, user, temperature, sem):
    body = json.dumps({"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": temperature,
                       "max_tokens": 4096, "response_format": {"type": "json_object"}, "reasoning": {"enabled": False}}).encode()
    def do():
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/ibnsina-llm", "X-Title": "IbnSina sft_v2"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())
    async with sem:
        d = await asyncio.to_thread(do)
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default=str(HERE / "sft_taxonomy.yaml")); ap.add_argument("--seeds", default=str(HERE / "seeds")); ap.add_argument("--out", default="/data/sft_v2/candidates")
    ap.add_argument("--categories", default=""); ap.add_argument("--per-category", type=int, default=0, help="scenarios per category (0 = target*oversample/2)")
    ap.add_argument("--kimi-share", type=float, default=None); ap.add_argument("--gemini-model", default=None); ap.add_argument("--bulk-gemini-model", default=None, help="cheaper model for group=bulk")
    ap.add_argument("--gemini-concurrency", type=int, default=12); ap.add_argument("--kimi-concurrency", type=int, default=3); ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--budget-usd", type=float, default=100.0); ap.add_argument("--kimi-cap-usd", type=float, default=30.0); a = ap.parse_args()
    tax = yaml.safe_load(open(a.taxonomy, encoding="utf-8")); axes = tax["diversity_axes"]
    templates = {k: (HERE / "prompts" / f"{k}.md").read_text(encoding="utf-8") for k in ("gen_system", "gen_user")}
    gem_model = a.gemini_model or tax["teachers"]["gemini"]["model"]; kimi_model = tax["teachers"]["kimi"]["model"]
    kimi_share = tax["teachers"]["kimi"]["share"] if a.kimi_share is None else a.kimi_share
    from google import genai
    client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT") or exit("set GOOGLE_CLOUD_PROJECT (Vertex AI project)"), location="us-central1")
    key_path = Path("/data/secrets/openrouter.key"); or_key = key_path.read_text().strip() if key_path.exists() else None
    if kimi_share > 0 and not or_key: print("!! no OpenRouter key -> kimi share set to 0"); kimi_share = 0
    gsem, ksem = asyncio.Semaphore(a.gemini_concurrency), asyncio.Semaphore(a.kimi_concurrency)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); ledger = Ledger(out.parent / f"ledger_gen_{hashlib.blake2b(a.categories.encode(), digest_size=4).hexdigest()}.json")  # one ledger per run (parallel runs must not clobber each other)
    cats = [c for c in tax["categories"] if not c.get("human_required") and (not a.categories or c["name"] in a.categories.split(","))]
    for cat in cats:
        seeds = load_seeds(a.seeds, cat["name"])
        if len(seeds) < 3: print(f"[{cat['name']}] skipped: only {len(seeds)} seeds"); continue
        n_scen = a.per_category or max(1, cat["target"] * cat.get("oversample", tax["generation"]["oversample"]) // 2)
        fp = out / f"{cat['name']}.jsonl"; done = set()
        if fp.exists():
            bad_lines = 0
            for l in fp.read_text(encoding="utf-8").splitlines():
                if not l.strip(): continue
                try: done.add(json.loads(l)["scenario_id"])
                except Exception: bad_lines += 1          # torn line (e.g. reboot mid-write) — skipped; the scenario simply gets regenerated
            if bad_lines: print(f"[{cat['name']}] skipped {bad_lines} malformed candidate line(s)", flush=True)
        rnd = random.Random(f"{a.seed}:{cat['name']}"); model = (a.bulk_gemini_model if cat["group"] == "bulk" and a.bulk_gemini_model else gem_model)
        todo = []
        for i in range(n_scen):
            sc = sample_scenario(cat, axes, rnd); sid = f"{cat['name']}-{i:06d}"
            teacher = "kimi" if (kimi_share > 0 and rnd.random() < kimi_share) else "gemini"
            if sid in done: continue
            todo.append((sid, sc, teacher))
        print(f"[{cat['name']}] {len(done)} done, {len(todo)} scenarios to generate with {model} (kimi share {kimi_share})", flush=True)
        stats = {"ok": 0, "bad": 0}; lock = asyncio.Lock(); t0 = time.time()

        async def one(sid, sc, teacher):
            system, user, used = build_prompt(cat, sc, seeds, templates, random.Random(sid))
            rows = []
            for k in range(2):  # two siblings per scenario -> judge picks best/worst for DPO
                try:
                    use_kimi = teacher == "kimi" and k == 1 and ledger.usd_for(kimi_model) < a.kimi_cap_usd
                    for attempt in range(4):
                        try:
                            if use_kimi:
                                text, pin, pout = await call_openrouter(kimi_model, or_key, system, user, 0.9, ksem); m = kimi_model
                            else:
                                text, pin, pout = await call_gemini(client, model, system, user, 0.9 if k == 0 else 1.0, gsem); m = model
                            break
                        except Exception as e:
                            if attempt == 3 or not any(x in str(e) for x in ("429", "RESOURCE_EXHAUSTED", "503", "502", "timed out", "Timeout", "overloaded")): raise
                            await asyncio.sleep(5 * 2 ** attempt)
                    ledger.add(m, pin, pout)
                    msgs = parse_conversation(text, sc["turns"])
                    if msgs is None: stats["bad"] += 1; continue
                    rows.append({"id": f"{sid}-{k}", "scenario_id": sid, "category": cat["name"], "subtype": sc["subtype"], "persona": sc["persona"], "register": sc["register"], "length": sc["length"],
                                 "turns": sc["turns"], "domain": sc["domain"], "teacher": m, "seeds_used": used, "prompt_hash": hashlib.blake2b((system + user).encode(), digest_size=8).hexdigest(), "messages": msgs})
                    stats["ok"] += 1
                except Exception as e:
                    stats["bad"] += 1; print(f"  {sid}-{k} {type(e).__name__}: {str(e)[:120]}", flush=True); await asyncio.sleep(3)
            async with lock:
                with open(fp, "a", encoding="utf-8") as f:
                    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n = stats["ok"] + stats["bad"]
                if n % 200 == 0: print(f"  [{cat['name']}] {stats} ${ledger.usd():.2f} {time.time()-t0:.0f}s", flush=True)

        # bounded worker pool: each worker runs a scenario's two sibling calls back-to-back, so scenarios complete continuously
        # (a flat gather over thousands of coroutines makes every scenario do call 1 before any does call 2 -> no rows for hours)
        queue = asyncio.Queue()
        for t in todo: queue.put_nowait(t)
        async def worker():
            while True:
                try: t = queue.get_nowait()
                except asyncio.QueueEmpty: return
                await one(*t)
        await asyncio.gather(*(worker() for _ in range(a.gemini_concurrency + a.kimi_concurrency)))
        # top-up: scenarios whose siblings failed to parse leave the pool short; add fresh scenarios until >= 95% of target x oversample
        want = cat["target"] * cat.get("oversample", tax["generation"]["oversample"]); rnd_extra = random.Random(f"{a.seed}:{cat['name']}:topup")
        for rnd_i in range(3):
            rows_now = sum(1 for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()) if fp.exists() else 0
            if rows_now >= int(0.95 * want) or a.per_category: break
            base = 500000 + rnd_i * 100000; n_extra = (want - rows_now) // 2 + 1; extra = []
            for i in range(n_extra):
                sc = sample_scenario(cat, axes, rnd_extra); sid = f"{cat['name']}-{base + i:06d}"
                teacher = "kimi" if (kimi_share > 0 and rnd_extra.random() < kimi_share) else "gemini"
                if sid not in done: extra.append((sid, sc, teacher))
            print(f"[{cat['name']}] top-up round {rnd_i}: {rows_now}/{want} rows -> +{len(extra)} scenarios", flush=True)
            for t in extra: queue.put_nowait(t)
            await asyncio.gather(*(worker() for _ in range(a.gemini_concurrency + a.kimi_concurrency)))
        print(f"[{cat['name']}] done {stats}; ledger ${ledger.usd():.2f}", flush=True)
        if ledger.usd() > 2 * a.budget_usd: print(f"!! spend ${ledger.usd():.2f} > 2x budget ${a.budget_usd} — NOTIFY (continuing per rules)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
