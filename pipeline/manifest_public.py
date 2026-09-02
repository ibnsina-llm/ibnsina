"""Derive the PUBLIC, category-level mix manifest from a build's full mix_manifest.json.

The full manifest enumerates every source by name (including private-overlay sources); the public rule for this project is
"category level only" (same as licenses.json). This keeps per-slice shares, token/doc totals, source COUNTS and licence
classes, and drops every source name and any section that lists names. It refuses to write if a private name survives.

  python pipeline/manifest_public.py --in /path/mix_manifest.json --out docs/data/mix_manifest_public_v1_1.json \\
      --private pipeline/sources_private.json pipeline/licenses_private.json
"""
import argparse, json, re, sys

PUBLIC_CLASS = {"proprietary": "proprietary/curated"}   # public wording: curated material without an open licence, never a single source

STRUCTURAL = {"class", "epochs", "note", "notes", "sources", "license", "licence", "license_class", "layer", "rule", "share", "url", "domain", "slice", "tokens", "docs", "chars", "tpc", "path", "prefix", "name"}

def private_names(paths, public_slices=()):
    """Identifiers that must never appear publicly: source ids (keys that look like ids: contain '_' or '.') and domain-like
    values found anywhere in the private overlay files, minus structural words and public slice names."""
    names = set()
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                k = str(k)
                if ("_" in k or "." in k) and k not in STRUCTURAL and k not in public_slices: names.add(k)
                walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
        elif isinstance(x, str) and re.fullmatch(r"[a-z0-9.\-]+\.[a-z]{2,}", x): names.add(x)
    for p in paths:
        try: walk(json.load(open(p, encoding="utf-8")))
        except FileNotFoundError: continue
    return names

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--private", nargs="*", default=[]); a = ap.parse_args()
    m = json.load(open(a.inp, encoding="utf-8")); src = m.get("sources", {})
    out = {k: m[k] for k in ("name", "target_tokens", "total_tokens_est", "seed", "val_permille", "tokenizer_proxy", "built_at") if k in m}
    out["note"] = "Category-level public manifest: per-slice shares and totals only; individual sources are listed by count and licence class, never by name. The corpus itself is not redistributed."
    slices = {}
    for sl, info in (m.get("by_slice") or {}).items():
        members = [s for s, d in src.items() if d.get("slice") == sl]
        classes = sorted({PUBLIC_CLASS.get(str(src[s].get("license_class", "unknown")), str(src[s].get("license_class", "unknown"))) for s in members})
        slices[sl] = {"share": round(float(info.get("share", 0)), 4), "tokens_est": int(info.get("tokens_est", 0)), "docs": int(info.get("docs", 0)),
                      "n_sources": len(members), "licence_classes": classes}
        cfg = (m.get("slices_config") or {}).get(sl, {})
        for k in ("layer", "rule", "target_share"):
            if k in cfg: slices[sl][k] = cfg[k]
        ep = cfg.get("epochs")
        if isinstance(ep, dict) and ep: slices[sl]["epochs"] = {"min": min(float(v) for v in ep.values()), "max": max(float(v) for v in ep.values())}   # per-source dict keyed by name -> range only
        elif ep is not None: slices[sl]["epochs"] = ep
    out["slices"] = slices
    out["n_sources_total"] = len(src)
    out["shards"] = {"count": len(m.get("shards", [])) if isinstance(m.get("shards"), list) else m.get("shards")}
    blob = json.dumps(out, ensure_ascii=False, indent=1)
    priv = private_names(a.private, public_slices=set(slices)); leaked = sorted(n for n in priv if n in blob)
    if leaked: print("REFUSING to write: private identifiers present:", leaked, file=sys.stderr); sys.exit(2)
    for n in src:   # no source names at all in the public file
        if re.search(rf"\b{re.escape(n)}\b", blob): print("REFUSING to write: source name present:", n, file=sys.stderr); sys.exit(2)
    open(a.out, "w", encoding="utf-8").write(blob + "\n"); print("wrote", a.out, "| slices:", list(slices), "| sources:", len(src))

if __name__ == "__main__":
    main()
