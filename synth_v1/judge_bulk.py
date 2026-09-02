#!/opt/pipe/bin/python3
"""synth_v1 bulk judge — judges EVERY candidate doc of a wave with gemini-3.7-flash via Vertex batch
(chunked), online fallback. Resumable. Exit codes: 0 ok, 43 budget hard stop, 1 failure.
  judge_bulk.py --wave N --run [--mode batch|online]"""
import argparse, json, re, sys, time
from pathlib import Path
from bulk_common import (BULK, CODE, GCS, FLASH, HARD_STOP_USD, Ledger, append_jsonl, batch_line,
                         budget_events, build_judge_prompt, gcs, jdump, jload, list_output_jsonls,
                         load_templates, online_run, parse_batch_line, poll_batch, read_jsonl, submit_batch, TERMINAL)

CHUNK_LINES = 12000
CRITERIA = ("correctness", "natural_persian", "translationese_free", "informational_density", "overall")


def parse_scores(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    s = json.loads(m.group(0))
    out = {k: int(s.get(k, 0)) for k in CRITERIA}
    out["reason"] = str(s.get("reason", ""))[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, required=True); ap.add_argument("--run", action="store_true")
    ap.add_argument("--mode", default="batch", choices=["batch", "online"])
    a = ap.parse_args()
    wdir = BULK / f"waves/wave-{a.wave:04d}"
    ws = jload(wdir / "wave_state.json", {})
    if ws.get("judge_done"): print(f"[judge w{a.wave}] already done"); return
    templates = load_templates(); ledger = Ledger()
    state = jload(BULK / "state.json", {"wave": 1, "kept_tokens": 0, "usd_notified": 0})
    docs = read_jsonl(wdir / "docs.jsonl"); by_id = {d["id"]: d for d in docs}
    done = {r["id"] for r in read_jsonl(wdir / "judged.jsonl")}
    todo = [d for d in docs if d["id"] not in done]
    print(f"[judge w{a.wave}] {len(docs)} docs, {len(todo)} to judge", flush=True)
    if ledger.usd() > HARD_STOP_USD: sys.exit(43)
    if a.mode == "online":
        import asyncio
        items = [(d["id"], None, build_judge_prompt(templates, d)) for d in todo]
        def on_result(rid, text, usage):
            if text is None: return
            ledger.add(FLASH, usage["in"], usage["out"], usage["thoughts"])
            try: scores = parse_scores(text)
            except Exception: scores = {"overall": -1, "reason": "judge_parse_error"}
            append_jsonl(wdir / "judged.jsonl", [{"id": rid, "scores": scores}])
        asyncio.run(online_run(items, FLASH, 0.0, 512, True, 12, on_result,
                               cap_fn=lambda: ledger.usd() > HARD_STOP_USD))
    else:
        jobs = ws.setdefault("judge_jobs", []); parsed = ws.setdefault("judge_parsed", [])
        if not jobs and todo:
            for ci in range(0, len(todo), CHUNK_LINES):
                chunk = todo[ci:ci + CHUNK_LINES]; pi = ci // CHUNK_LINES
                fp = wdir / f"judge_in_p{pi:02d}.jsonl"
                with open(fp, "w", encoding="utf-8") as f:
                    for d in chunk:
                        f.write(json.dumps(batch_line(d["id"], None, build_judge_prompt(templates, d), FLASH, 0.0, 512, json_mime=True), ensure_ascii=False) + "\n")
                src = f"{GCS}/bulk/wave-{a.wave:04d}/{fp.name}"
                dest = f"{GCS}/bulk/wave-{a.wave:04d}/judge_out/p{pi:02d}/"
                gcs("cp", str(fp), src)
                info = submit_batch(FLASH, src, dest)
                jobs.append(info); jdump(wdir / "wave_state.json", ws)
                print(f"[judge w{a.wave}] submitted p{pi:02d} ({len(chunk)} reqs) -> {info['name']}", flush=True)
        while jobs:
            states = []
            for info in jobs:
                st, err = poll_batch(info); states.append(st)
                if st in TERMINAL and info["dest"] not in parsed:
                    if st in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
                        rows, bad = [], 0
                        for uri in list_output_jsonls(info["dest"]):
                            for line in gcs("cat", uri).stdout.splitlines():
                                rid, text, usage, e = parse_batch_line(line)
                                if e or not usage: bad += 1; continue
                                ledger.add(FLASH + "@batch", usage["in"], usage["out"], usage["thoughts"])
                                if rid not in by_id: continue
                                try: scores = parse_scores(text)
                                except Exception: scores = {"overall": -1, "reason": "judge_parse_error"}; bad += 1
                                rows.append({"id": rid, "scores": scores})
                        append_jsonl(wdir / "judged.jsonl", rows)
                        print(f"[judge w{a.wave}] parsed {info['dest'].rsplit('/', 2)[-2]}: +{len(rows)}, {bad} bad", flush=True)
                    else:
                        print(f"[judge w{a.wave}] job {info['name']} {st} {err}", flush=True)
                    parsed.append(info["dest"]); jdump(wdir / "wave_state.json", ws)
            print(f"[judge w{a.wave}] {time.strftime('%H:%M')} states={states} ${ledger.usd():.2f}", flush=True)
            budget_events(ledger, state); jdump(BULK / "state.json", state)
            if all(s in TERMINAL for s in states): break
            time.sleep(180)
    judged = read_jsonl(wdir / "judged.jsonl")
    ws["judge_done"] = True; jdump(wdir / "wave_state.json", ws)
    print(f"[judge w{a.wave}] done: {len(judged)}/{len(docs)} judged", flush=True)
    if len(judged) < 0.8 * len(docs): sys.exit(1)


if __name__ == "__main__":
    main()
