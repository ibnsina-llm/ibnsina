#!/usr/bin/env python3
"""Render the judge's per-category reports (report/<cat>.json) as one RTL HTML page for STOP S-B: keep rates, score histograms, 20 kept + 5 rejected samples per category.
usage: render_report.py /data/sft_v2/report out.html"""
import html, json, sys
from pathlib import Path
rep_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
def esc(s): return html.escape(str(s)).replace("\n", "<br>")
def content(c):
    if isinstance(c, str): return esc(c)
    return "<br>".join(esc(p.get("text", "")) if p.get("type") == "text" else f'<code dir="ltr">[{p.get("type")}] {esc(p.get("text",""))}</code>' for p in c)
def conv(r, cls):
    s = r.get("scores", {}); meta = f'{r.get("id","")} · {r.get("teacher","")} · overall {s.get("overall","?")} · {esc(s.get("reason",""))[:160]}'
    if r.get("auto_flags"): meta += " · flags: " + ", ".join(r["auto_flags"])
    body = "".join(f'<div class="msg {m["role"]}" dir="auto"><span class="role">{"کاربر" if m["role"]=="user" else "دستیار"}</span>{content(m["content"])}</div>' for m in r["messages"])
    return f'<article class="{cls}"><header>{meta}</header>{body}</article>'
secs, rows = [], []
for f in sorted(rep_dir.glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8")); c = r["category"]
    hist = " ".join(f'<span class="bar"><i style="height:{min(60, 2 + 58 * v / max(1, max(r["score_hist"].values())))}px"></i>{k}</span>' for k, v in sorted(r["score_hist"].items(), key=lambda kv: int(kv[0])))
    rows.append(f'<tr><td><a href="#{c}">{c}</a></td><td>{r["candidates"]}</td><td>{r["auto_fail"]}</td><td>{r["kept"]}</td><td class="{ "bad" if r["keep_rate"] < 0.15 else ""}">{r["keep_rate"]:.0%}</td><td>{r["target"]}</td><td>{r["dpo_pairs"]}</td><td class="hist">{hist}</td></tr>')
    secs.append(f'<section id="{c}"><h2>{c} <small>kept {r["kept"]}/{r["candidates"]} · rate {r["keep_rate"]:.0%} · target {r["target"]} · auto-fail {r["auto_fail"]} {esc(json.dumps(r.get("auto_flags", {}), ensure_ascii=False))}</small></h2>'
                + "<h3>نمونه‌های پذیرفته‌شده</h3>" + "".join(conv(x, "kept") for x in r["samples_kept"]) + "<h3>نمونه‌های ردشده</h3>" + "".join(conv(x, "rejected") for x in r["samples_rejected"]) + "</section>")
page = f"""<title>IbnSina sft_v2 — گزارش داور (S-B)</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&family=IBM+Plex+Mono&display=swap">
<style>:root{{--bg:#F6F8FA;--panel:#fff;--ink:#16202B;--muted:#5B6874;--line:#D8DFE6;--me:#DDF0F1;--accent:#0F8B93;--bad:#B5563A;--ok:#DDF0F1;--rej:#FBE9E4}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0F151B;--panel:#161E26;--ink:#E4EAF0;--muted:#98A6B3;--line:#2A3541;--me:#123A3E;--accent:#34B8C0;--bad:#D8836A;--ok:#123A3E;--rej:#3A1F18}}}}
:root[data-theme="dark"]{{--bg:#0F151B;--panel:#161E26;--ink:#E4EAF0;--muted:#98A6B3;--line:#2A3541;--me:#123A3E;--accent:#34B8C0;--bad:#D8836A;--ok:#123A3E;--rej:#3A1F18}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:Vazirmatn,system-ui,sans-serif;font-size:15.5px;line-height:1.8;direction:rtl}}main{{max-width:980px;margin:0 auto;padding:28px 18px 80px}}
h1{{font-size:26px;margin:0 0 14px}}table{{width:100%;border-collapse:collapse;background:var(--panel);font-size:13.5px;direction:ltr;font-family:"IBM Plex Mono",monospace}}th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}}th{{color:var(--muted);font-weight:500}}td.bad{{color:var(--bad);font-weight:700}}
.hist{{white-space:nowrap}}.bar{{display:inline-flex;flex-direction:column;align-items:center;width:16px;font-size:9px;color:var(--muted)}}.bar i{{display:block;width:10px;background:var(--accent);border-radius:2px 2px 0 0}}
h2{{font-size:20px;margin:34px 0 6px;border-bottom:1px solid var(--line)}}h2 small{{font-size:12px;color:var(--muted);font-family:"IBM Plex Mono",monospace;direction:ltr;display:inline-block}}h3{{font-size:15px;color:var(--muted);margin:14px 0 6px}}
article{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:10px 0}}article.kept{{border-right:4px solid var(--accent)}}article.rejected{{border-right:4px solid var(--bad);background:var(--rej)}}
article header{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted);direction:ltr;margin-bottom:6px}}.msg{{padding:6px 10px;border-radius:8px;margin:6px 0;white-space:pre-wrap}}.msg.user{{background:var(--me)}}.msg.assistant{{border:1px solid var(--line)}}
.role{{display:block;font-size:11px;color:var(--accent);font-weight:700}}code{{font-family:"IBM Plex Mono",monospace;font-size:12.5px;display:block;background:var(--bg);padding:3px 6px;border-radius:5px}}
</style><main><h1>IbnSina sft_v2 — گزارش داور (STOP S-B)</h1>
<table><tr><th>category</th><th>candidates</th><th>auto-fail</th><th>kept</th><th>keep rate</th><th>target</th><th>DPO pairs</th><th>score histogram (1–10)</th></tr>{''.join(rows)}</table>{''.join(secs)}</main>"""
out.write_text(page, encoding="utf-8"); print("wrote", out, f"{out.stat().st_size/1e3:.0f} KB")
