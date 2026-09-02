#!/opt/pipe/bin/python3
"""synth_v1 judge calibration runner — scores the planted-flaw cases in cases.yaml with the real judge
prompt and checks each lands in its expected band (sft_v2 CALIBRATION.md approach).
  run_calibration.py [--model gemini-3.7-flash] [--out /data/synth_v1/calibration_results.jsonl]
Run on corpus-pipeline2 (Vertex, location=global). Compare models with two runs (--model gemini-3.5-flash-lite)."""
import argparse, asyncio, json, sys
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bulk_common import FLASH, build_judge_prompt, load_templates, online_run


def check(expect, scores):
    fails = []
    for k, v in expect.items():
        crit, bound = k.rsplit("_", 1)
        got = scores.get("overall" if crit == "overall" else crit, None)
        if got is None: fails.append(f"{crit}=missing"); continue
        if bound == "max" and got > v: fails.append(f"{crit} {got}>{v}")
        if bound == "min" and got < v: fails.append(f"{crit} {got}<{v}")
    return fails


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=FLASH)
    ap.add_argument("--out", default="/data/synth_v1/calibration_results.jsonl")
    a = ap.parse_args()
    cases = yaml.safe_load(open(HERE / "cases.yaml", encoding="utf-8"))["cases"]
    templates = load_templates(HERE.parent / "prompts")
    items = [(c["id"], None, build_judge_prompt(templates, {**c, "text": c["text"]})) for c in cases]
    results = {}
    def on_result(rid, text, usage):
        if text is None: results[rid] = {"error": str(usage)}; return
        try:
            import re
            m = re.search(r"\{.*\}", text, re.S); results[rid] = json.loads(m.group(0))
        except Exception:
            results[rid] = {"error": "parse", "raw": (text or "")[:200]}
    await online_run(items, a.model, 0.0, 512, True, 8, on_result)
    n_pass = 0
    out_rows = []
    print(f"\n== judge calibration on {a.model} ==")
    print(f"{'case':26s} {'overall':>7s} {'expect':28s} verdict")
    for c in cases:
        s = results.get(c["id"], {"error": "no_result"})
        fails = check(c["expect"], s) if "error" not in s else [s["error"]]
        ok = not fails
        n_pass += ok
        print(f"{c['id']:26s} {str(s.get('overall', '—')):>7s} {json.dumps(c['expect']):28s} {'PASS' if ok else 'FAIL: ' + '; '.join(fails)}")
        out_rows.append({"id": c["id"], "model": a.model, "scores": s, "expect": c["expect"], "pass": ok, "flaw": c["flaw"]})
    print(f"\n{n_pass}/{len(cases)} within expected bands")
    with open(a.out, "a", encoding="utf-8") as f:
        for r in out_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
