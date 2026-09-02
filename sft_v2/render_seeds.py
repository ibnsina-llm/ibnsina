#!/usr/bin/env python3
"""Render sft_v2/seeds/*.jsonl as one RTL HTML review page (STOP S-A). usage: render_seeds.py out.html [notes.json]"""
import html, json, sys
from pathlib import Path
import yaml
HERE = Path(__file__).resolve().parent; out = Path(sys.argv[1]); notes = json.load(open(sys.argv[2], encoding="utf-8")) if len(sys.argv) > 2 else {}
tax = yaml.safe_load(open(HERE / "sft_taxonomy.yaml", encoding="utf-8")); cats = [c["name"] for c in tax["categories"]]; info = {c["name"]: c for c in tax["categories"]}
def esc(s): return html.escape(s).replace("\n", "<br>")
def content(c):
    if isinstance(c, str): return esc(c)
    parts = []
    for p in c:
        if p["type"] == "text": parts.append(esc(p["text"]))
        elif p["type"] == "python": parts.append(f'<code class="tool" dir="ltr">&lt;|python_start|&gt;{esc(p["text"])}&lt;|python_end|&gt;</code>')
        else: parts.append(f'<code class="tool out" dir="ltr">&lt;|output_start|&gt;{esc(p["text"])}&lt;|output_end|&gt;</code>')
    return "<br>".join(parts)
secs, toc = [], []
for cat in cats:
    f = HERE / "seeds" / f"{cat}.jsonl"
    if not f.exists(): continue
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]; convs = [r for r in rows if "messages" in r]
    c = info[cat]; toc.append(f'<a href="#{cat}">{cat} ({len(convs)})</a>')
    body = [f'<section id="{cat}"><h2>{cat} <small>{c["group"]} · target {c["target"]:,} · {len(convs)} seeds</small></h2><p class="rubric">{esc(c["rubric"].strip())}</p>']
    if notes.get(cat): body.append(f'<p class="note"><b>یادداشت بازبینی:</b> {esc(notes[cat])}</p>')
    for r in convs:
        meta = " · ".join(str(r.get(k)) for k in ("subtype", "persona", "register") if r.get(k))
        body.append(f'<article><header><b>{esc(r["id"])}</b> <span>{esc(meta)}</span></header>')
        if r.get("notes"): body.append(f'<p class="why">{esc(r["notes"])}</p>')
        for m in r["messages"]: body.append(f'<div class="msg {m["role"]}" dir="auto"><span class="role">{"کاربر" if m["role"]=="user" else "دستیار"}</span>{content(m["content"])}</div>')
        body.append("</article>")
    secs.append("".join(body) + "</section>")
page = f"""<title>IbnSina sft_v2 — بازبینی بذرها (S-A)</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&family=IBM+Plex+Mono&display=swap">
<style>:root{{--bg:#F6F8FA;--panel:#fff;--ink:#16202B;--muted:#5B6874;--line:#D8DFE6;--me:#DDF0F1;--accent:#0F8B93;--warn:#FFF4E5}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0F151B;--panel:#161E26;--ink:#E4EAF0;--muted:#98A6B3;--line:#2A3541;--me:#123A3E;--accent:#34B8C0;--warn:#3A2E15}}}}
:root[data-theme="dark"]{{--bg:#0F151B;--panel:#161E26;--ink:#E4EAF0;--muted:#98A6B3;--line:#2A3541;--me:#123A3E;--accent:#34B8C0;--warn:#3A2E15}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:Vazirmatn,system-ui,sans-serif;font-size:16px;line-height:1.85;direction:rtl}}
main{{max-width:900px;margin:0 auto;padding:32px 20px 80px}}h1{{font-size:28px;margin:0 0 6px}}.lede{{color:var(--muted);margin:0 0 18px}}
nav{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 28px}}nav a{{font-size:13px;padding:4px 10px;border:1px solid var(--line);border-radius:999px;color:var(--ink);text-decoration:none;background:var(--panel)}}
h2{{font-size:22px;margin:36px 0 8px;border-bottom:1px solid var(--line);padding-bottom:6px}}h2 small{{font-size:13px;color:var(--muted);font-family:"IBM Plex Mono",monospace;margin-right:8px}}
.rubric{{white-space:pre-wrap;color:var(--muted);font-size:14px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px}}
.note{{background:var(--warn);border-radius:8px;padding:10px 14px;font-size:14.5px}}
article{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:14px 0}}article header{{display:flex;justify-content:space-between;gap:10px;font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted);direction:ltr}}
.why{{font-size:13.5px;color:var(--muted);margin:6px 0 10px}}.msg{{padding:8px 12px;border-radius:10px;margin:8px 0;white-space:pre-wrap}}.msg.user{{background:var(--me)}}.msg.assistant{{border:1px solid var(--line)}}
.role{{display:block;font-size:11.5px;color:var(--accent);font-weight:700;margin-bottom:2px}}code.tool{{display:block;font-family:"IBM Plex Mono",monospace;font-size:13px;background:var(--bg);padding:4px 8px;border-radius:6px;margin:4px 0}}code.out{{opacity:.8}}
</style><main><h1>IbnSina sft_v2 — بذرهای طلایی</h1><p class="lede">STOP S-A · {sum(len([1 for l in (HERE/'seeds'/f'{c}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip() and '"messages"' in l]) for c in cats if (HERE/'seeds'/f'{c}.jsonl').exists())} گفتگو در {len(secs)} دسته · لحن تولید انبوه از همین‌ها گرفته می‌شود</p><nav>{' '.join(toc)}</nav>{''.join(secs)}</main>"""
out.write_text(page, encoding="utf-8"); print("wrote", out, f"{out.stat().st_size/1e3:.0f} KB")
