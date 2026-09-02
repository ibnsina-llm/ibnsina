"""Generate the two-band PersianMedQA comparison chart (EN + FA SVG) for the READMEs / card /
report. Reads docs/eval/frontier_persianmedqa_2026-08-30.json + small_band_2026-08-31.json.
Self-contained SVG: explicit background (readable on GitHub dark mode), system fonts only
(GitHub blocks external fonts inside <img>), tabular layout, IbnSina highlighted.
Usage:  python3 docs/eval/gen_chart.py
"""
import json, html, pathlib

D = pathlib.Path(__file__).parent
frontier = json.load(open(D / "frontier_persianmedqa_2026-08-30.json"))["models"]
small = json.load(open(D / "small_band_2026-08-31.json"))

SKIP = {"z-ai__glm-5", "anthropic__claude-fable-5-local"}
front_rows = sorted(
    ([v["display"], v["acc_all"], v["unparsed"], False] for k, v in frontier.items() if k not in SKIP),
    key=lambda r: -r[1])

small_rows = []
for band in ("hosted_small", "local"):
    for k, v in small[band].items():
        if v.get("acc_all") is None:
            continue
        small_rows.append([v["display"], v["acc_all"], v["unparsed"], bool(v.get("highlight"))])
small_rows.sort(key=lambda r: -r[1])

INK, MUTED, GRID, BG = "#24292f", "#6a737d", "#e8e8e4", "#fdfdfb"
BAR, HI = "#87a7b3", "#0f8b93"

def render(lang):
    fa = lang == "fa"
    W, ROW, PAD_L, PAD_R, TOP = 920, 26, 250, 150, 92
    bands = [
        ("Frontier / large — hosted APIs" if not fa else "پیشتازان / مدل‌های بزرگ — از راه سرویس ابری", front_rows),
        ("Small / laptop-class — identical harness" if not fa else "مدل‌های کوچک / لپ‌تاپی — با همان پروتکل", small_rows),
    ]
    n_rows = sum(len(b[1]) for b in bands)
    H = TOP + n_rows * ROW + len(bands) * 34 + 56
    x0, x1 = PAD_L, W - PAD_R
    sc = lambda v: x0 + (x1 - x0) * v / 100.0
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system, Segoe UI, Vazirmatn, Tahoma, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>']
    title = "PersianMedQA — accuracy on all 5,235 questions" if not fa else "PersianMedQA — دقت روی هر ۵٬۲۳۵ پرسش"
    sub = ("Same protocol for every row: zero-shot, temperature 0, option-number answer. Unparsed replies count as wrong."
           if not fa else "پروتکل برای همهٔ ردیف‌ها یکسان است: بدون مثال، دمای صفر، پاسخ = شمارهٔ گزینه؛ پاسخ بدون گزینه، غلط حساب می‌شود.")
    # FA headline texts: centered — immune to mixed-direction (bidi) width quirks across renderers.
    anchor = f'middle" x="{W // 2}' if fa else 'start" x="20'
    s.append(f'<text text-anchor="{anchor}" y="34" font-size="19" font-weight="700" fill="{INK}">{html.escape(title)}</text>')
    s.append(f'<text text-anchor="{anchor}" y="56" font-size="12.5" fill="{MUTED}">{html.escape(sub)}</text>')
    for gv in (0, 25, 50, 75, 100):
        gx = sc(gv)
        s.append(f'<line x1="{gx:.1f}" y1="{TOP - 10}" x2="{gx:.1f}" y2="{H - 46}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{gx:.1f}" y="{H - 28}" font-size="11" fill="{MUTED}" text-anchor="middle">{gv}</text>')
    y = TOP
    for btitle, rows in bands:
        s.append(f'<text x="{20 if not fa else W // 2}" y="{y}" font-size="13" font-weight="700" fill="{INK}" text-anchor="{"start" if not fa else "middle"}">{html.escape(btitle)}</text>')
        y += 14
        for disp, acc, unp, hi in rows:
            bw = sc(acc) - x0
            col = HI if hi else BAR
            s.append(f'<rect x="{x0}" y="{y + 4}" width="{bw:.1f}" height="{ROW - 10}" rx="3" fill="{col}"/>')
            s.append(f'<text x="{x0 - 8}" y="{y + ROW - 9}" font-size="12" fill="{INK}" text-anchor="end" font-weight="{700 if hi else 400}">{html.escape(disp)}</text>')
            label = f"{acc:.1f}"
            if unp and unp / 5235 > 0.02 and acc < 70:  # long labels on high bars overflow the canvas; counts live in Table B
                label += (f" · {unp} unparsed" if not fa else f" · {unp} بدون گزینه")
            s.append(f'<text x="{sc(acc) + 6:.1f}" y="{y + ROW - 9}" font-size="11.5" fill="{MUTED}" font-weight="{700 if hi else 400}">{html.escape(label)}</text>')
            y += ROW
        y += 20
    foot = ("Dorna2-8B: could not run the identical protocol (gated weights, access pending). Per-model unparsed counts are in the report table."
            if not fa else "Dorna2-8B با همین پروتکل قابل اجرا نبود (وزن‌ها دسترسی‌بسته‌اند). شمار پاسخ‌های بدون گزینهٔ هر مدل در جدول گزارش آمده است.")
    s.append(f'<text text-anchor="{anchor}" y="{H - 8}" font-size="10.5" fill="{MUTED}">{html.escape(foot)}</text>')
    s.append("</svg>")
    return "\n".join(s)

for lang in ("en", "fa"):
    out = D / f"persianmedqa_chart_{lang}.svg"
    out.write_text(render(lang), encoding="utf-8")
    print("wrote", out)
