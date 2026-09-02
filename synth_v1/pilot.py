import os
#!/opt/pipe/bin/python3
"""synth_v1 SY-A pilot — ~50 docs per domain via Vertex Gemini (gemini-3.7-flash primary + a labelled
Flash-Lite slice for comparison), then a Flash judge pass, then readable per-domain .md rendering and
_pilot_report.md with measured $/B-token projections (generation-only and generation+judge at 3x
oversample / keep-top-third).
  pilot.py --stage all [--per-domain 50] [--lite-per-domain 10] [--out /data/synth_v1/pilot]
House patterns reused from sft_v2/gen.py + judge.py: cost ledger, bounded worker pool over a queue,
tolerant JSONL readers, resumable done-sets. Hard cap: stops issuing calls past --cap-usd (default 18)."""
import argparse, asyncio, json, random, re, time
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
FLASH, LITE = "gemini-3.7-flash", "gemini-3.5-flash-lite"
# $/M tokens in/out — Vertex list prices as looked up 2026-08-31.
# gemini-3.7-flash is intro-priced through 2026-12-31 (doubles to 1.50/7.50 on 2027-01-01); batch mode is -50%.
PRICES = {FLASH: (0.75, 3.75), LITE: (0.30, 2.50)}
PRICES_2027 = {FLASH: (1.50, 7.50), LITE: (0.30, 2.50)}


def read_jsonl(p):
    rows = []
    if not Path(p).exists(): return rows
    for l in Path(p).read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        try: rows.append(json.loads(l))
        except Exception: pass                     # torn line — the item is simply redone on resume
    return rows


class Ledger:
    def __init__(self, path):
        self.path = Path(path); self.d = json.loads(self.path.read_text()) if self.path.exists() else {}
    def add(self, model, pin, pout, pth=0):
        e = self.d.setdefault(model, {"in": 0, "out": 0, "thoughts": 0, "calls": 0})
        e["in"] += pin; e["out"] += pout; e["thoughts"] += pth; e["calls"] += 1
        self.path.write_text(json.dumps(self.d, indent=1))
    def usd(self, prices=PRICES):
        return sum(v["in"] / 1e6 * prices[m][0] + (v["out"] + v.get("thoughts", 0)) / 1e6 * prices[m][1]
                   for m, v in self.d.items() if m in prices)


async def call(client, model, system, user, temperature, sem, max_tokens=5000, json_mime=False):
    from google.genai import types
    kw = dict(temperature=temperature, max_output_tokens=max_tokens)
    if system: kw["system_instruction"] = system
    if json_mime: kw["response_mime_type"] = "application/json"
    if model == FLASH: kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)   # verified live 2026-08-31: works, 0 thought tokens
    cfg = types.GenerateContentConfig(**kw)
    async with sem:
        for attempt in range(4):
            try:
                r = await client.aio.models.generate_content(model=model, contents=user, config=cfg)
                break
            except Exception as e:
                if attempt == 3 or not any(x in str(e) for x in ("429", "RESOURCE_EXHAUSTED", "503", "502", "timed out", "Timeout", "overloaded", "DEADLINE")):
                    raise
                await asyncio.sleep(5 * 2 ** attempt)
    u = r.usage_metadata
    return r.text or "", (u.prompt_token_count or 0), (u.candidates_token_count or 0), (getattr(u, "thoughts_token_count", 0) or 0)


def parse_doc(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t); t = re.sub(r"\n```\s*$", "", t).strip()
    m = re.match(r"^#\s+(.+)", t)
    if not m or len(t) < 700: return None
    return m.group(1).strip(), t


def parse_json_array(text):
    i = text.find("[")
    if i >= 0:
        arr, _ = json.JSONDecoder().raw_decode(text[i:])
        if isinstance(arr, list): return arr
    d = json.loads(text)                            # tolerate {"topics": [...]}
    for v in d.values():
        if isinstance(v, list): return v
    raise ValueError("no JSON array in reply")


async def pool(todo, worker_fn, n_workers):
    q = asyncio.Queue()
    for t in todo: q.put_nowait(t)
    async def worker():
        while True:
            try: t = q.get_nowait()
            except asyncio.QueueEmpty: return
            await worker_fn(t)
    await asyncio.gather(*(worker() for _ in range(n_workers)))


# ---------------- stage: topics ----------------
async def stage_topics(tax, tpl, client, ledger, out, sem, n_topics, cap):
    fp = out / "topics.json"
    topics = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
    jobs = [(dom, sub) for dom in tax["domains"] for sub in dom["subdomains"]
            if not topics.get(dom["name"], {}).get(sub["name"])]
    async def one(job):
        dom, sub = job
        if ledger.usd() > cap: return
        user = (tpl.replace("{DOMAIN}", dom["name"]).replace("{SUBDOMAIN}", sub["name"])
                   .replace("{STRATEGY}", sub["seed_topic_strategy"]).replace("{N}", str(n_topics)))
        try:
            text, pin, pout, pth = await call(client, FLASH, None, user, 1.0, sem, max_tokens=3000, json_mime=True)
            ledger.add(FLASH, pin, pout, pth)
            topics.setdefault(dom["name"], {})[sub["name"]] = [str(t).strip() for t in parse_json_array(text)][:n_topics]
        except Exception as e:
            print(f"  [topics] {dom['name']}/{sub['name']} {type(e).__name__}: {str(e)[:140]}", flush=True)
    await pool(jobs, one, 6)
    fp.write_text(json.dumps(topics, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(v) for d in topics.values() for v in d.values())
    print(f"[topics] {n} topics across {sum(len(v) for v in topics.values())} subdomains; ${ledger.usd():.2f}", flush=True)
    return topics


# ---------------- scenarios ----------------
def build_scenarios(tax, topics, per_domain, lite_n, seed):
    rows = []
    for dom in tax["domains"]:
        rnd = random.Random(f"{seed}:{dom['name']}")
        subs = [s for s in dom["subdomains"] if topics.get(dom["name"], {}).get(s["name"])]
        sw = [s["weight"] for s in subs]
        mix = dom["doc_type_mix"]; dts = list(mix); dtw = [mix[k] for k in dts]
        for i in range(per_domain):
            sub = rnd.choices(subs, weights=sw)[0]
            tl = topics[dom["name"]][sub["name"]]
            rows.append({"id": f"{dom['name']}-{i:03d}", "domain": dom["name"], "subdomain": sub["name"],
                         "topic": tl[rnd.randrange(len(tl))], "doc_type": rnd.choices(dts, weights=dtw)[0],
                         "length": rnd.choice(tax["targets"]["tokens_per_doc"]["sample_grid"]),
                         "teacher": LITE if i >= per_domain - lite_n else FLASH,
                         "variation": rnd.randrange(10 ** 6)})
    return rows


# ---------------- stage: gen ----------------
async def stage_gen(tax, templates, client, ledger, out, anchors, scens, gsem, lsem, cap):
    fp = out / "docs.jsonl"
    done = {r["id"] for r in read_jsonl(fp)}
    todo = [s for s in scens if s["id"] not in done]
    print(f"[gen] {len(done)} done, {len(todo)} to generate", flush=True)
    lock = asyncio.Lock(); stats = {"ok": 0, "bad": 0}; t0 = time.time()
    dt_notes = tax["doc_types"]; doms = {d["name"]: d for d in tax["domains"]}
    async def one(sc):
        if ledger.usd() > cap:
            return
        rnd = random.Random(sc["id"]); anchor = anchors[rnd.randrange(len(anchors))]
        dom = doms[sc["domain"]]
        user = (templates["gen_user"].replace("{ANCHOR}", anchor["text"]).replace("{DOMAIN}", sc["domain"])
                .replace("{SUBDOMAIN}", sc["subdomain"]).replace("{TOPIC}", sc["topic"])
                .replace("{DOC_TYPE}", sc["doc_type"]).replace("{DOC_TYPE_NOTE}", dt_notes[sc["doc_type"]])
                .replace("{AUDIENCE}", dom["audience_fa"]).replace("{LENGTH_TOKENS}", str(sc["length"]))
                .replace("{LENGTH_WORDS}", str(int(sc["length"] / 1.7))).replace("{VARIATION}", str(sc["variation"]))
                .replace("{EXTRA}", ("- " + dom["gen_extra_fa"]) if dom.get("gen_extra_fa") else ""))
        try:
            # max_tokens 8192: the SY-A run used 5000 and ~3% of worked_problems docs got truncated mid-problem
            text, pin, pout, pth = await call(client, sc["teacher"], templates["gen_system"], user, 0.9,
                                              gsem if sc["teacher"] == FLASH else lsem, max_tokens=8192)
            ledger.add(sc["teacher"], pin, pout, pth)
            doc = parse_doc(text)
            if not doc:
                stats["bad"] += 1; print(f"  {sc['id']} unparseable/short ({len(text)} chars)", flush=True); return
            row = dict(sc); row.update({"anchor_id": anchor["id"], "title": doc[0], "text": doc[1],
                                        "usage": {"in": pin, "out": pout, "thoughts": pth}})
            async with lock:
                with open(fp, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["ok"] += 1
            if stats["ok"] % 50 == 0: print(f"  [gen] {stats} ${ledger.usd():.2f} {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            stats["bad"] += 1; print(f"  {sc['id']} {type(e).__name__}: {str(e)[:140]}", flush=True)
    await pool(todo, one, 12)
    print(f"[gen] done {stats}; ${ledger.usd():.2f}", flush=True)


# ---------------- stage: judge ----------------
async def stage_judge(templates, client, ledger, out, sem, cap):
    docs = read_jsonl(out / "docs.jsonl"); jp = out / "judged.jsonl"
    done = {r["id"] for r in read_jsonl(jp)}
    todo = [d for d in docs if d["id"] not in done]
    print(f"[judge] {len(docs)} docs, {len(todo)} to judge", flush=True)
    lock = asyncio.Lock()
    async def one(d):
        if ledger.usd() > cap: return
        prompt = (templates["judge"].replace("{DOMAIN}", d["domain"]).replace("{SUBDOMAIN}", d["subdomain"])
                  .replace("{DOC_TYPE}", d["doc_type"]).replace("{TOPIC}", d["topic"]).replace("{DOCUMENT}", d["text"]))
        try:
            text, pin, pout, pth = await call(client, FLASH, None, prompt, 0.0, sem, max_tokens=512, json_mime=True)
            ledger.add(FLASH, pin, pout, pth)
            m = re.search(r"\{.*\}", text, re.S); s = json.loads(m.group(0))
            scores = {k: int(s.get(k, 0)) for k in ("correctness", "natural_persian", "translationese_free", "informational_density", "overall")}
            scores["reason"] = str(s.get("reason", ""))[:300]
        except Exception as e:
            scores = {"overall": -1, "reason": f"judge_error {type(e).__name__}: {str(e)[:120]}"}
        async with lock:
            with open(jp, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": d["id"], "scores": scores}, ensure_ascii=False) + "\n")
    await pool(todo, one, 12)
    print(f"[judge] done; ${ledger.usd():.2f}", flush=True)


# ---------------- stage: render ----------------
def fmt_usd(x): return f"${x:,.0f}" if x >= 100 else f"${x:.2f}"


def projections(docs, judge_ledger, n_judged):
    """$ per 1B tokens of KEPT text: 3x oversample, judge scores every candidate, keep top third."""
    by_teacher = {}
    for t in (FLASH, LITE):
        ds = [d for d in docs if d["teacher"] == t]
        if not ds: continue
        mi = sum(d["usage"]["in"] for d in ds) / len(ds)
        mo_text = sum(d["usage"]["out"] for d in ds) / len(ds)
        mo_billed = sum(d["usage"]["out"] + d["usage"]["thoughts"] for d in ds) / len(ds)
        by_teacher[t] = (mi, mo_text, mo_billed)
    jv = judge_ledger.d.get(FLASH, {"in": 0, "out": 0, "thoughts": 0})
    judge_per_doc = ((jv["in"] * PRICES[FLASH][0] + (jv["out"] + jv["thoughts"]) * PRICES[FLASH][1]) / 1e6 / max(1, n_judged))
    rows = []
    for label, weights in [("all gemini-3.7-flash", {FLASH: 1.0}), ("all gemini-3.5-flash-lite", {LITE: 1.0}),
                           ("mix 60% flash / 40% lite", {FLASH: 0.6, LITE: 0.4})]:
        if any(t not in by_teacher for t in weights): continue
        gen_doc = sum(w * (by_teacher[t][0] * PRICES[t][0] + by_teacher[t][2] * PRICES[t][1]) / 1e6 for t, w in weights.items())
        mo_text = sum(w * by_teacher[t][1] for t, w in weights.items())
        docs_per_B = 1e9 / mo_text
        gen_only = 3 * docs_per_B * gen_doc
        total = gen_only + 3 * docs_per_B * judge_per_doc
        rows.append((label, gen_only, total, total * 0.5))   # batch mode -50% on both stages
    return rows, judge_per_doc


def stage_render(tax, out, args):
    docs = read_jsonl(out / "docs.jsonl")
    judged = {r["id"]: r["scores"] for r in read_jsonl(out / "judged.jsonl")}
    led = {k: Ledger(out / f"ledger_{k}.json") for k in ("topics", "gen", "judge")}
    md_root = out / "out_md"; md_root.mkdir(exist_ok=True)
    order = [d["name"] for d in tax["domains"]]
    for dom in order:
        ddir = md_root / dom; ddir.mkdir(exist_ok=True)
        for j, d in enumerate(sorted([x for x in docs if x["domain"] == dom], key=lambda x: x["id"]), 1):
            s = judged.get(d["id"], {})
            tag = "flash" if d["teacher"] == FLASH else "flashlite"
            head = (f"# {d['title']}\n\n"
                    f"- doc-type: {d['doc_type']}   |   domain: {d['domain']} / {d['subdomain']}\n"
                    f"- seed topic (EN): {d['topic']}\n"
                    f"- teacher: {d['teacher']}   |   billed tokens in/out: {d['usage']['in']}/{d['usage']['out']}\n"
                    f"- judge overall: {s.get('overall', '—')} "
                    f"(correctness {s.get('correctness', '—')}, natural {s.get('natural_persian', '—')}, "
                    f"no-translationese {s.get('translationese_free', '—')}, density {s.get('informational_density', '—')}) "
                    f"— {s.get('reason', '')}\n\n---\n\n")
            (ddir / f"{j:02d}_{tag}_{d['doc_type']}.md").write_text(head + d["text"] + "\n", encoding="utf-8")
    # ---- report ----
    R = ["# synth_v1 SY-A pilot report", "",
         f"date: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} | docs: {len(docs)} "
         f"(flash {sum(1 for d in docs if d['teacher']==FLASH)}, flash-lite {sum(1 for d in docs if d['teacher']==LITE)}) | judged: {len(judged)}",
         f"models (verified live on Vertex, location=global, 2026-08-31): {FLASH} (primary, thinking off), {LITE} (bulk comparison slice)", ""]
    rows, judge_per_doc = projections(docs, led["judge"], len(judged))
    R += ["## $/B-token projection (the number that matters)", "",
          "Basis: measured usage metadata below; kept tokens = generated tokens / 3 (3x oversample, judge keeps top third);",
          "judge (gemini-3.7-flash) scores every candidate. Prices: Vertex list 2026-08-31 "
          f"(flash {PRICES[FLASH][0]}/{PRICES[FLASH][1]}, lite {PRICES[LITE][0]}/{PRICES[LITE][1]} $/M in/out; flash intro price doubles 2027-01-01).", "",
          "| teacher mix | generation only | generation + judge | with batch mode (-50%) |", "|---|---:|---:|---:|"]
    for label, gen_only, total, batch in rows:
        R.append(f"| {label} | {fmt_usd(gen_only)}/B | {fmt_usd(total)}/B | {fmt_usd(batch)}/B |")
    if rows:
        best = min(r[3] for r in rows); worst = max(r[2] for r in rows)
        R += ["", f"=> 40-60B kept tokens: **{fmt_usd(best*40)} - {fmt_usd(worst*60)}** "
                  f"(mandate estimate was $8-15k; hard stop $20,000 buys ~{20000/best:.1f}B kept tokens at the cheapest mix).",
              f"   Judge cost per candidate doc: ${judge_per_doc:.4f}."]
    R += ["", "## Pilot spend and token counts (exact, from usage metadata)", "",
          "| stage | model | calls | tokens in | tokens out | thoughts | cost |", "|---|---|---:|---:|---:|---:|---:|"]
    total_usd = 0.0
    for stage, ledger in led.items():
        for m, v in ledger.d.items():
            usd = v["in"] / 1e6 * PRICES[m][0] + (v["out"] + v.get("thoughts", 0)) / 1e6 * PRICES[m][1]; total_usd += usd
            R.append(f"| {stage} | {m} | {v['calls']} | {v['in']:,} | {v['out']:,} | {v.get('thoughts',0):,} | ${usd:.2f} |")
    R.append(f"| **total** | | | | | | **${total_usd:.2f}** |")
    for t in (FLASH, LITE):
        ds = [d for d in docs if d["teacher"] == t]
        if not ds: continue
        mo = sum(d["usage"]["out"] for d in ds) / len(ds); mi = sum(d["usage"]["in"] for d in ds) / len(ds)
        ch = sum(len(d["text"]) for d in ds) / len(ds)
        cost = (mi * PRICES[t][0] + (mo + sum(d['usage']['thoughts'] for d in ds)/len(ds)) * PRICES[t][1]) / 1e6
        R.append(f"\n- {t}: {len(ds)} docs, mean {mo:.0f} out-tokens/doc ({ch:.0f} chars, {ch/mo:.2f} chars/token), mean in {mi:.0f}, **cost/doc ${cost:.4f}**")
    R += ["", "## Judge scores", "", "| domain | n | mean overall | >=8 | <=4 | mean by flash | mean by lite |", "|---|---:|---:|---:|---:|---:|---:|"]
    for dom in order:
        ids = [d for d in docs if d["domain"] == dom]
        sc = [judged[d["id"]]["overall"] for d in ids if d["id"] in judged and judged[d["id"]]["overall"] >= 0]
        fl = [judged[d["id"]]["overall"] for d in ids if d["id"] in judged and d["teacher"] == FLASH and judged[d["id"]]["overall"] >= 0]
        li = [judged[d["id"]]["overall"] for d in ids if d["id"] in judged and d["teacher"] == LITE and judged[d["id"]]["overall"] >= 0]
        if not sc: continue
        R.append(f"| {dom} | {len(sc)} | {sum(sc)/len(sc):.2f} | {sum(1 for x in sc if x>=8)} | {sum(1 for x in sc if x<=4)} | "
                 f"{sum(fl)/len(fl):.2f} | {(sum(li)/len(li)):.2f} |" if li else
                 f"| {dom} | {len(sc)} | {sum(sc)/len(sc):.2f} | {sum(1 for x in sc if x>=8)} | {sum(1 for x in sc if x<=4)} | "
                 f"{sum(fl)/len(fl):.2f} | — |")
    low = sorted([(judged[d["id"]]["overall"], d["id"], d["teacher"], judged[d["id"]]["reason"]) for d in docs
                  if d["id"] in judged and judged[d["id"]]["overall"] >= 0], key=lambda x: x[0])[:8]
    R += ["", "### Lowest-scored docs (judge reasons)", ""] + [f"- {s}/10 `{i}` ({t}): {r}" for s, i, t, r in low]
    R += ["", "## Quality concerns (human review)", "", "_filled in during review_", ""]
    (out / "out_md" / "_pilot_report.md").write_text("\n".join(R), encoding="utf-8")
    print(f"[render] {len(docs)} docs -> {md_root}; report written; pilot total ${total_usd:.2f}", flush=True)


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default=str(HERE / "taxonomy.yaml")); ap.add_argument("--prompts", default=str(HERE / "prompts"))
    ap.add_argument("--anchors", default="/data/synth_v1/anchors/anchors.jsonl"); ap.add_argument("--out", default="/data/synth_v1/pilot")
    ap.add_argument("--per-domain", type=int, default=50); ap.add_argument("--lite-per-domain", type=int, default=10)
    ap.add_argument("--topics-per-subdomain", type=int, default=25)
    ap.add_argument("--stage", default="all", choices=["all", "topics", "gen", "judge", "render"])
    ap.add_argument("--cap-usd", type=float, default=18.0); ap.add_argument("--seed", type=int, default=20260831)
    a = ap.parse_args()
    tax = yaml.safe_load(open(a.taxonomy, encoding="utf-8"))
    templates = {k: (Path(a.prompts) / f"{k}.md").read_text(encoding="utf-8") for k in ("gen_system", "gen_user", "topics_user", "judge")}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    led = {k: Ledger(out / f"ledger_{k}.json") for k in ("topics", "gen", "judge")}
    if a.stage != "render":
        from google import genai
        client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR-GCP-PROJECT"), location="global")
    gsem, lsem, jsem, tsem = asyncio.Semaphore(8), asyncio.Semaphore(4), asyncio.Semaphore(10), asyncio.Semaphore(6)
    if a.stage in ("all", "topics", "gen"):
        topics = await stage_topics(tax, templates["topics_user"], client, led["topics"], out, tsem, a.topics_per_subdomain, a.cap_usd)
    if a.stage in ("all", "gen"):
        anchors = read_jsonl(a.anchors)
        if len(anchors) < 20: raise SystemExit(f"only {len(anchors)} anchors at {a.anchors} — run sample_anchors.py first")
        scens = build_scenarios(tax, topics, a.per_domain, a.lite_per_domain, a.seed)
        await stage_gen(tax, templates, client, led["gen"], out, anchors, scens, gsem, lsem, a.cap_usd)
    if a.stage in ("all", "judge"):
        await stage_judge(templates, client, led["judge"], out, jsem, a.cap_usd)
    if a.stage in ("all", "render"):
        stage_render(tax, out, a)


if __name__ == "__main__":
    asyncio.run(amain())
