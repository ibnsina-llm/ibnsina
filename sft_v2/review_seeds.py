#!/usr/bin/env python3
"""Seed QA for STOP S-A: per-category counts + automatic lint (role order, Arabic ي/ك, non-ASCII digits, translationese phrases, id/fields) + a sample."""
import collections, json, re, sys
from pathlib import Path
SEEDS = Path(__file__).resolve().parent / "seeds"
BAD_PHRASES = ["این یک سوال عالی", "قطعاً!", "به عنوان یک مدل زبانی", "امیدوارم کمک کرده باشد", "امیدوارم این کمک", "خوشحالم که", "البته! "]
ARABIC = re.compile(r"[يك]"); NONASCII_DIGITS = re.compile(r"[۰-۹٠-٩]")
def text_of(c): return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c)
total = 0; problems = 0
for f in sorted(SEEDS.glob("*.jsonl")):
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    convs = [r for r in rows if "messages" in r]; total += len(convs)
    regs = collections.Counter(r.get("register") for r in convs); turns = collections.Counter(len(r["messages"]) // 2 for r in convs); subs = collections.Counter(r.get("subtype") for r in convs)
    issues = []
    for r in convs:
        for i, m in enumerate(r["messages"]):
            if m["role"] != ("user" if i % 2 == 0 else "assistant"): issues.append(f"{r.get('id')}: role order")
            t = text_of(m["content"])
            if ARABIC.search(t): issues.append(f"{r.get('id')}: arabic ي/ك")
            if NONASCII_DIGITS.search(t): issues.append(f"{r.get('id')}: non-ascii digits")
            if m["role"] == "assistant":
                for b in BAD_PHRASES:
                    if b in t: issues.append(f"{r.get('id')}: phrase «{b.strip()}»")
        for k in ("id", "category", "persona", "register", "notes"):
            if not r.get(k): issues.append(f"{r.get('id')}: missing {k}")
    problems += len(issues)
    print(f"== {f.stem}: {len(convs)} seeds | turns {dict(sorted(turns.items()))} | register {dict(regs)} | subtypes {len([s for s in subs if s])} | issues {len(issues)}")
    for i in sorted(set(issues))[:8]: print("   !", i)
    if "--sample" in sys.argv and convs:
        r = convs[0]; print("   sample:", r["messages"][0]["content"][:120].replace("\n", " | "), "→", text_of(r["messages"][1]["content"])[:160].replace("\n", " | "))
print(f"TOTAL {total} seeds, {problems} lint issues")
