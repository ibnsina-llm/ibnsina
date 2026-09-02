import os
#!/opt/pipe/bin/python3
"""sft_v2 judge — automatic checks + Gemini rubric scoring; keeps the top third per category; emits DPO sibling pairs and the STOP S-B report.
  judge.py --candidates /data/sft_v2/candidates --out /data/sft_v2 [--categories a,b] [--judge-model gemini-2.5-flash-lite] [--max N]
Outputs: judged/<cat>.jsonl (all candidates + scores), kept/<cat>.jsonl, dpo_pairs/<cat>.jsonl, report/<cat>.json + report/S-B.md"""
import argparse, asyncio, collections, json, random, re, sys, time
from pathlib import Path
import yaml
sys.path.insert(0, "/data/pipeline")
HERE = Path(__file__).resolve().parent
EXPECT_LANG = {"translation_fa_en": None, "language_discipline": None}   # None = mixed languages allowed; default: assistant turns must be Persian


def flat_text(content):
    return content if isinstance(content, str) else " ".join(p.get("text", "") for p in content if p.get("type") == "text")


def auto_checks(row, lid):
    flags = []; cat = row["category"]
    if cat == "toolcall": flags += verify_toolcalls(row)
    for i, m in enumerate(row["messages"]):
        t = flat_text(m["content"])
        if m["role"] == "assistant":
            if not (1 <= len(t) <= 6000) and not isinstance(m["content"], list): flags.append(f"len_assistant_{len(t)}")
            if cat not in EXPECT_LANG and len(t) >= 40:
                lang, p = lid.predict(t)
                if lang != "fa" and p > 0.8 and cat != "toolcall": flags.append(f"lang_{lang}")
            words = t.split()
            for n in (8, 12):
                grams = collections.Counter(tuple(words[j:j + n]) for j in range(max(0, len(words) - n)))
                if grams and grams.most_common(1)[0][1] >= 3: flags.append("repetition_loop"); break
            if re.search(r"(به عنوان یک مدل زبانی|as an ai language model|gemini|openai|chatgpt|claude|anthropic|moonshot|kimi)", t, re.I): flags.append("teacher_self_reference")
        else:
            if not (1 <= len(t) <= 4000): flags.append(f"len_user_{len(t)}")
    return flags


# ---- tool-call verification: allowlisted functions, real execution, outputs overwritten ----
import ast, re as _re
def gregorian_to_jalali(gy, gm, gd):
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]; gy2 = gy + 1 if gm > 2 else gy
    days = 355666 + 365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400 + gd + g_d_m[gm - 1]
    jy = -1595 + 33 * (days // 12053); days %= 12053; jy += 4 * (days // 1461); days %= 1461
    if days > 365: jy += (days - 1) // 365; days = (days - 1) % 365
    jm, jd = (1 + days // 31, 1 + days % 31) if days < 186 else (7 + (days - 186) // 30, 1 + (days - 186) % 30)
    return f"{jy:04d}-{jm:02d}-{jd:02d}"
def jalali_to_gregorian(jy, jm, jd):
    jy += 1595; days = -355668 + 365 * jy + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + jd + ((jm - 1) * 31 if jm < 7 else (jm - 7) * 30 + 186)
    gy = 400 * (days // 146097); days %= 146097
    if days > 36524:
        days -= 1; gy += 100 * (days // 36524); days %= 36524
        if days >= 365: days += 1
    gy += 4 * (days // 1461); days %= 1461
    if days > 365: gy += (days - 1) // 365; days = (days - 1) % 365
    gd = days + 1; leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    sal_a = [0, 31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]; gm = 0
    while gm < 13 and gd > sal_a[gm]: gd -= sal_a[gm]; gm += 1
    return f"{gy:04d}-{gm:02d}-{gd:02d}"
assert jalali_to_gregorian(1403, 1, 1) == "2024-03-20" and jalali_to_gregorian(1403, 7, 15) == "2024-10-06" and gregorian_to_jalali(2024, 3, 20) == "1403-01-01" and jalali_to_gregorian(1357, 11, 22) == "1979-02-11"
_ALLOWED_NAMES = {"round", "abs", "min", "max", "sum", "int", "float", "pow"}; _ALLOWED_FUNCS = {"jalali_to_gregorian": jalali_to_gregorian, "gregorian_to_jalali": gregorian_to_jalali}
def safe_eval(expr):
    tree = ast.parse(expr.strip(), mode="eval")
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in _ALLOWED_NAMES | set(_ALLOWED_FUNCS): raise ValueError("call")
        elif isinstance(n, ast.Name):
            if n.id not in _ALLOWED_NAMES | set(_ALLOWED_FUNCS): raise ValueError("name")
        elif not isinstance(n, (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop, ast.Load, ast.Tuple, ast.List)): raise ValueError(type(n).__name__)
    val = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {**{k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in _ALLOWED_NAMES}, **_ALLOWED_FUNCS})
    return str(val)
def verify_toolcalls(row):
    flags = []
    for m in row["messages"]:
        if m["role"] != "assistant" or not isinstance(m["content"], list): continue
        parts = m["content"]
        for i, p in enumerate(parts):
            if p["type"] != "python": continue
            nxt = parts[i + 1] if i + 1 < len(parts) else None
            if not nxt or nxt["type"] != "python_output": flags.append("toolcall_no_output"); continue
            expr = p["text"].strip()
            if _re.match(r"^search\((['\"]).*\1\)$", expr, _re.S): continue          # search outputs are synthetic by design
            try: real = safe_eval(expr)
            except Exception: flags.append("toolcall_invalid_python"); continue
            given = nxt["text"].strip()
            def num(s):
                try: return float(s)
                except Exception: return s
            if num(given) != num(real) and not (isinstance(num(given), float) and isinstance(num(real), float) and abs(num(given) - num(real)) < 1e-6 * max(1, abs(num(real)))): flags.append("toolcall_wrong_output")
            nxt["text"] = real
    return flags

def render_conv(msgs):
    out = []
    for m in msgs:
        c = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"], ensure_ascii=False)
        out.append(f"[{m['role']}]\n{c}")
    return "\n\n".join(out)


async def judge_one(client, model, template, cat, row, sem):
    from google.genai import types
    prompt = template.replace("{CATEGORY}", cat["name"]).replace("{RUBRIC}", cat["rubric"].strip()).replace("{CONVERSATION}", render_conv(row["messages"]))
    cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=512, response_mime_type="application/json", thinking_config=types.ThinkingConfig(thinking_budget=0))
    async with sem:
        r = await client.aio.models.generate_content(model=model, contents=prompt, config=cfg)
    m = re.search(r"\{.*\}", r.text, re.S); d = json.loads(m.group(0))
    return {k: int(d.get(k, 0)) for k in ("correctness", "natural_persian", "adherence", "register", "safety_respect", "overall")} | {"reason": str(d.get("reason", ""))[:300]}, (r.usage_metadata.prompt_token_count or 0), (r.usage_metadata.candidates_token_count or 0)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default=str(HERE / "sft_taxonomy.yaml")); ap.add_argument("--candidates", default="/data/sft_v2/candidates"); ap.add_argument("--out", default="/data/sft_v2")
    ap.add_argument("--categories", default=""); ap.add_argument("--judge-model", default=None); ap.add_argument("--concurrency", type=int, default=16); ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    tax = yaml.safe_load(open(a.taxonomy, encoding="utf-8")); J = tax["judge"]; model = a.judge_model or J["model"]
    template = (HERE / "prompts" / "judge.md").read_text(encoding="utf-8")
    from pipeline.common import LangID
    lid = LangID()
    from google import genai
    client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT") or exit("set GOOGLE_CLOUD_PROJECT (Vertex AI project)"), location="us-central1"); sem = asyncio.Semaphore(a.concurrency)
    out = Path(a.out); [ (out / d).mkdir(parents=True, exist_ok=True) for d in ("judged", "kept", "dpo_pairs", "report") ]
    usage = {"in": 0, "out": 0}; summary = []; lpath = out / f"ledger_judge_{__import__('hashlib').blake2b(a.categories.encode(), digest_size=4).hexdigest()}.json"
    for cat in tax["categories"]:
        if a.categories and cat["name"] not in a.categories.split(","): continue
        cp = Path(a.candidates) / f"{cat['name']}.jsonl"
        if not cp.exists(): continue
        rows = []
        for l in cp.read_text(encoding="utf-8").splitlines():
            if not l.strip(): continue
            try: rows.append(json.loads(l))
            except Exception: pass                          # torn line — ignore
        if a.max: rows = rows[:a.max]
        jp = out / "judged" / f"{cat['name']}.jsonl"; judged = {}
        if jp.exists():
            for l in jp.read_text(encoding="utf-8").splitlines():
                if not l.strip(): continue
                try: r = json.loads(l); judged[r["id"]] = r
                except Exception: pass
        todo = [r for r in rows if r["id"] not in judged]; t0 = time.time()
        print(f"[{cat['name']}] {len(rows)} candidates, {len(todo)} to judge with {model}", flush=True)

        async def one(r):
            flags = auto_checks(r, lid)
            try:
                s, pin, pout = await judge_one(client, model, template, cat, r, sem); usage["in"] += pin; usage["out"] += pout
            except Exception as e:
                s = {"overall": 0, "reason": f"judge_error {type(e).__name__}"}; flags = flags + ["judge_error"]
            rec = dict(r); rec["auto_flags"] = flags; rec["scores"] = s; rec["auto_ok"] = not flags
            judged[r["id"]] = rec
            with open(jp, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        await asyncio.gather(*(one(r) for r in todo))
        allj = [judged[r["id"]] for r in rows if r["id"] in judged]
        ok = [r for r in allj if r["auto_ok"] and r["scores"]["overall"] >= J["min_score"]]
        ok.sort(key=lambda r: (-r["scores"]["overall"], -r["scores"].get("correctness", 0), r["id"]))
        # selection: best-first up to the category target, never below the quality floor (overall >= keep_min, default 7).
        # At 3x oversample this is the "top third"; at 2x (general_instruction) it fills the target from the 8-10 band instead of stopping at 0.67x.
        keep_min = J.get("keep_min", 7); ok_high = [r for r in ok if r["scores"]["overall"] >= keep_min]
        n_keep = min(cat["target"], len(ok_high))
        kept = ok_high[:n_keep]; kept_ids = {r["id"] for r in kept}
        with open(out / "kept" / f"{cat['name']}.jsonl", "w", encoding="utf-8") as f:
            for r in kept: f.write(json.dumps({k: r[k] for k in ("id", "scenario_id", "category", "subtype", "persona", "register", "turns", "teacher", "messages", "scores")}, ensure_ascii=False) + "\n")
        # DPO pairs from siblings (same scenario), score gap >= 2, chosen must be auto-clean
        by_sc = collections.defaultdict(list)
        for r in allj: by_sc[r["scenario_id"]].append(r)
        pairs = []
        for sid, sib in by_sc.items():
            if len(sib) < 2: continue
            sib.sort(key=lambda r: -r["scores"]["overall"]); hi, lo = sib[0], sib[-1]
            if hi["auto_ok"] and hi["scores"]["overall"] - lo["scores"]["overall"] >= 2:
                pairs.append({"scenario_id": sid, "category": cat["name"], "prompt": hi["messages"][:1], "chosen": hi["messages"], "rejected": lo["messages"], "score_chosen": hi["scores"]["overall"], "score_rejected": lo["scores"]["overall"]})
        with open(out / "dpo_pairs" / f"{cat['name']}.jsonl", "w", encoding="utf-8") as f:
            for p in pairs: f.write(json.dumps(p, ensure_ascii=False) + "\n")
        hist = collections.Counter(r["scores"]["overall"] for r in allj); flagc = collections.Counter(fl.split("_")[0] for r in allj for fl in r["auto_flags"])
        rnd = random.Random(7); rejected = [r for r in allj if r["id"] not in kept_ids]
        rep = {"category": cat["name"], "candidates": len(allj), "auto_fail": sum(1 for r in allj if not r["auto_ok"]), "auto_flags": dict(flagc), "kept": len(kept), "keep_rate": round(len(kept) / max(1, len(allj)), 3),
               "target": cat["target"], "score_hist": {str(k): hist[k] for k in sorted(hist)}, "dpo_pairs": len(pairs), "judge_model": model,
               "samples_kept": rnd.sample(kept, min(20, len(kept))), "samples_rejected": rnd.sample(rejected, min(5, len(rejected)))}
        (out / "report" / f"{cat['name']}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        json.dump({model: {"in": usage["in"], "out": usage["out"], "calls": len(todo)}}, open(lpath, "w"))
        summary.append(rep); print(f"[{cat['name']}] kept {len(kept)}/{len(allj)} (rate {rep['keep_rate']}), auto_fail {rep['auto_fail']}, pairs {len(pairs)}, hist {rep['score_hist']} {time.time()-t0:.0f}s", flush=True)
        if rep["keep_rate"] < J["keep_rate_floor"]: print(f"!! [{cat['name']}] keep rate {rep['keep_rate']} < floor {J['keep_rate_floor']} — STOP and show Sina before spending more", flush=True)
    md = ["# STOP S-B — judge report", "", f"judge model: {model}; tokens in/out: {usage['in']:,}/{usage['out']:,}", "", "| category | candidates | auto-fail | kept | keep rate | target | DPO pairs | score histogram |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary: md.append(f"| {r['category']} | {r['candidates']} | {r['auto_fail']} | {r['kept']} | {r['keep_rate']} | {r['target']} | {r['dpo_pairs']} | {r['score_hist']} |")
    (out / "report" / "S-B.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    asyncio.run(main())
