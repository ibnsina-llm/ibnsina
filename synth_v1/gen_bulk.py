#!/opt/pipe/bin/python3
"""synth_v1 bulk generation — one wave: topics (once) -> scenarios -> Vertex batch jobs per teacher
(chunked) -> poll -> parse to docs.jsonl. Resumable at every step via wave_state.json.
  gen_bulk.py --wave N --run [--docs-per-wave 120000] [--mode batch|online]
  gen_bulk.py --print-next-wave
Exit codes: 0 ok, 43 budget hard stop, 1 failure."""
import argparse, json, random, subprocess, sys, time
from pathlib import Path
import yaml
from bulk_common import (BULK, CODE, GCS, FLASH, LITE, HARD_STOP_USD, Ledger, append_jsonl, batch_line,
                         budget_events, build_gen_prompt, gcs, jdump, jload, list_output_jsonls,
                         load_templates, online_run, parse_batch_line, parse_doc, poll_batch, read_jsonl,
                         routing, submit_batch, TERMINAL)

CHUNK_LINES = 20000
EST_DOC_COST = {FLASH: 0.0060, LITE: 0.0034}   # batch-priced, from SY-A pilot (+ margin)


def topics_incomplete(tax, topics, per_sub=250):
    return [(d, s) for d in tax["domains"] for s in d["subdomains"]
            if len(topics.get(d["name"], {}).get(s["name"], [])) < per_sub * 0.6]


def ensure_topics(tax, templates, ledger, per_sub=250):
    # NOTE: runs asyncio.run(), which poisons google-genai's shared httpx client for later SYNC calls
    # (batches.create raised "client has been closed"). Only ever call this in a --topics-only subprocess.
    fp = BULK / "topics.json"
    topics = jload(fp, {})
    todo = topics_incomplete(tax, topics, per_sub)
    if not todo: return topics
    print(f"[topics] expanding {len(todo)} subdomains to ~{per_sub}", flush=True)
    import asyncio
    items = []
    for d, s in todo:
        for k in range(3):
            user = (templates["topics_user"].replace("{DOMAIN}", d["name"]).replace("{SUBDOMAIN}", s["name"])
                    .replace("{STRATEGY}", s["seed_topic_strategy"]).replace("{N}", "90")
                    + f"\n(variation seed {k}: produce a spread different from any other run)")
            items.append((f"{d['name']}|{s['name']}|{k}", None, user))
    results = {}
    def on_result(rid, text, usage):
        if text is None: print(f"  [topics] {rid} failed: {usage}", flush=True); return
        ledger.add(FLASH, usage["in"], usage["out"], usage["thoughts"])
        results[rid] = text
    asyncio.run(online_run(items, FLASH, 1.0, 8000, True, 6, on_result))
    for rid, text in results.items():
        dn, sn, _ = rid.split("|")
        try:
            i = text.find("[")
            arr, _junk = json.JSONDecoder().raw_decode(text[i:])
        except Exception:
            continue
        cur = topics.setdefault(dn, {}).setdefault(sn, [])
        seen = {t.strip().lower() for t in cur}
        for t in arr:
            t = str(t).strip()
            if t and t.lower() not in seen: cur.append(t); seen.add(t.lower())
    jdump(fp, topics)
    n = sum(len(v) for d in topics.values() for v in d.values())
    print(f"[topics] total {n}; ${ledger.usd():.2f}", flush=True)
    return topics


def build_scenarios(tax, topics, wave, docs_per_wave, seed):
    route = routing(tax)
    aspects = tax["generation"]["aspect_axis"]
    rows = []
    for dom in tax["domains"]:
        n = round(docs_per_wave * dom["weight"])
        rnd = random.Random(f"{seed}:{wave}:{dom['name']}")
        subs = [s for s in dom["subdomains"] if topics.get(dom["name"], {}).get(s["name"])]
        sw = [s["weight"] for s in subs]
        mix = dom["doc_type_mix"]; dts = list(mix); dtw = [mix[k] for k in dts]
        for i in range(n):
            sub = rnd.choices(subs, weights=sw)[0]
            tl = topics[dom["name"]][sub["name"]]
            rows.append({"id": f"w{wave:04d}-{dom['name']}-{i:06d}", "domain": dom["name"], "subdomain": sub["name"],
                         "topic": tl[rnd.randrange(len(tl))], "doc_type": rnd.choices(dts, weights=dtw)[0],
                         "length": rnd.choice(tax["targets"]["tokens_per_doc"]["sample_grid"]),
                         "aspect": rnd.choice(aspects), "teacher": route(dom["name"]), "variation": rnd.randrange(10 ** 6)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=0); ap.add_argument("--run", action="store_true")
    ap.add_argument("--print-next-wave", action="store_true")
    ap.add_argument("--docs-per-wave", type=int, default=120000)
    ap.add_argument("--mode", default="batch", choices=["batch", "online"])
    ap.add_argument("--topics-only", action="store_true")
    ap.add_argument("--seed", type=int, default=20260901)
    a = ap.parse_args()
    state = jload(BULK / "state.json", {"wave": 1, "kept_tokens": 0, "usd_notified": 0})
    if a.print_next_wave: print(state["wave"]); return
    if a.topics_only:
        tax = yaml.safe_load(open(CODE / "taxonomy.yaml", encoding="utf-8"))
        ensure_topics(tax, load_templates(), Ledger()); return
    wave = a.wave or state["wave"]
    wdir = BULK / f"waves/wave-{wave:04d}"; wdir.mkdir(parents=True, exist_ok=True)
    ws = jload(wdir / "wave_state.json", {"jobs": [], "parsed": [], "gen_done": False})
    if ws.get("gen_done"): print(f"[gen w{wave}] already done"); return
    tax = yaml.safe_load(open(CODE / "taxonomy.yaml", encoding="utf-8"))
    templates = load_templates()
    ledger = Ledger()
    anchors = read_jsonl("/data/synth_v1/anchors/anchors.jsonl")
    assert len(anchors) >= 100, "anchors missing"
    docs_n = min(a.docs_per_wave, 9000) if wave == 1 else a.docs_per_wave   # wave 1 = small end-to-end validation
    scen_fp = wdir / "scenarios.jsonl"
    if not scen_fp.exists():
        topics = jload(BULK / "topics.json", {})
        if topics_incomplete(tax, topics):    # expand in a subprocess: keeps this process's sync SDK client healthy
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--topics-only"], check=True)
            topics = jload(BULK / "topics.json", {})
        scens = build_scenarios(tax, topics, wave, docs_n, a.seed)
        append_jsonl(scen_fp, scens)
    scens = read_jsonl(scen_fp)
    by_id = {s["id"]: s for s in scens}
    est = sum(EST_DOC_COST[s["teacher"]] for s in scens)
    if ledger.usd() + est > HARD_STOP_USD:
        print(f"BUDGET HARD STOP: ${ledger.usd():.2f} + est ${est:.0f} > ${HARD_STOP_USD}"); sys.exit(43)
    done_ids = {r["id"] for r in read_jsonl(wdir / "docs.jsonl")}
    if a.mode == "online":
        run_online(tax, templates, ledger, wdir, anchors, [s for s in scens if s["id"] not in done_ids], state); finish(wdir, ws, scens); return
    # ---- batch: submit chunks per teacher ----
    if not ws["jobs"]:
        for teacher in (FLASH, LITE):
            mine = [s for s in scens if s["teacher"] == teacher]
            for ci in range(0, len(mine), CHUNK_LINES):
                chunk = mine[ci:ci + CHUNK_LINES]; pi = ci // CHUNK_LINES
                fp = wdir / f"gen_in_{teacher}_p{pi:02d}.jsonl"
                with open(fp, "w", encoding="utf-8") as f:
                    for s in chunk:
                        system, user = build_gen_prompt(tax, templates, anchors, s)
                        f.write(json.dumps(batch_line(s["id"], system, user, teacher, 0.9, 8192), ensure_ascii=False) + "\n")
                src = f"{GCS}/bulk/wave-{wave:04d}/{fp.name}"
                dest = f"{GCS}/bulk/wave-{wave:04d}/gen_out/{teacher}_p{pi:02d}/"
                gcs("cp", str(fp), src)
                info = submit_batch(teacher, src, dest)
                ws["jobs"].append(info); jdump(wdir / "wave_state.json", ws)
                print(f"[gen w{wave}] submitted {teacher} p{pi:02d} ({len(chunk)} reqs) -> {info['name']}", flush=True)
    # ---- poll + parse ----
    while True:
        states = []
        for info in ws["jobs"]:
            st, err = poll_batch(info); states.append(st)
            if st in TERMINAL and info["dest"] not in ws["parsed"]:
                if st in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
                    parse_outputs(info, by_id, wdir, ledger, done_ids)
                else:
                    print(f"[gen w{wave}] job {info['name']} {st} {err}", flush=True)
                ws["parsed"].append(info["dest"]); jdump(wdir / "wave_state.json", ws)
        print(f"[gen w{wave}] {time.strftime('%H:%M')} states={states} docs={sum(1 for _ in open(wdir/'docs.jsonl', encoding='utf-8')) if (wdir/'docs.jsonl').exists() else 0} ${ledger.usd():.2f}", flush=True)
        budget_events(ledger, state); jdump(BULK / "state.json", state)
        if all(s in TERMINAL for s in states): break
        time.sleep(180)
    finish(wdir, ws, scens)


def parse_outputs(info, by_id, wdir, ledger, done_ids):
    model_key = info["model"] + "@batch"
    rows, bad = [], 0
    for uri in list_output_jsonls(info["dest"]):
        out = gcs("cat", uri).stdout
        for line in out.splitlines():
            rid, text, usage, err = parse_batch_line(line)
            if err or not usage: bad += 1; continue
            ledger.add(model_key, usage["in"], usage["out"], usage["thoughts"])
            sc = by_id.get(rid)
            if not sc or rid in done_ids: continue
            doc = parse_doc(text)
            if not doc: bad += 1; continue
            row = dict(sc); row.update({"title": doc[0], "text": doc[1], "usage": usage})
            rows.append(row); done_ids.add(rid)
    append_jsonl(wdir / "docs.jsonl", rows)
    print(f"[gen] parsed {info['dest'].rsplit('/', 2)[-2]}: +{len(rows)} docs, {bad} bad", flush=True)


def run_online(tax, templates, ledger, wdir, anchors, todo, state):
    import asyncio
    for teacher in (FLASH, LITE):
        mine = [s for s in todo if s["teacher"] == teacher]
        items, id2sc = [], {s["id"]: s for s in mine}
        for s in mine:
            system, user = build_gen_prompt(tax, templates, anchors, s)
            items.append((s["id"], system, user))
        def on_result(rid, text, usage):
            if text is None: return
            ledger.add(teacher, usage["in"], usage["out"], usage["thoughts"])
            doc = parse_doc(text)
            if not doc: return
            row = dict(id2sc[rid]); row.update({"title": doc[0], "text": doc[1], "usage": usage})
            append_jsonl(wdir / "docs.jsonl", [row])
        asyncio.run(online_run(items, teacher, 0.9, 8192, False, 10, on_result,
                               cap_fn=lambda: ledger.usd() > HARD_STOP_USD))
        budget_events(ledger, state)


def finish(wdir, ws, scens):
    docs = read_jsonl(wdir / "docs.jsonl")
    ws["gen_done"] = True; jdump(wdir / "wave_state.json", ws)
    print(f"[gen] wave done: {len(docs)}/{len(scens)} docs parsed ({len(docs)/max(1,len(scens)):.1%})", flush=True)
    if len(docs) < 0.5 * len(scens): sys.exit(1)


if __name__ == "__main__":
    main()
