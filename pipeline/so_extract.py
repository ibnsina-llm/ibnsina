#!/opt/pipe/bin/python3
"""Stack Overflow Posts.7z -> clean/stackoverflow/part-*.jsonl.gz  (Q + accepted/top answers, HTML stripped, code fenced).

pass 1: stream `7z x -so Posts.7z` XML, bucket questions by Id % NB and answers by ParentId % NB (disk).
pass 2: per bucket (parallel): join, filter, render docs.
Keeps a question if score >= 0 and it has >= 1 answer that is accepted or has score >= 1; renders accepted + up to 2 best others.
"""
from __future__ import annotations
import argparse, html, json, multiprocessing as mp, os, re, shutil, subprocess, sys, time
from pathlib import Path
from xml.etree import ElementTree as ET

import orjson

sys.path.insert(0, str(Path(__file__).parent))
from common import BUCKET, DATA, ShardWriter, Stats, gcs_download, gcs_list, gcs_upload_dir, gsutil, log, normalize_generic

NAME = "stackoverflow"
SRC = f"{BUCKET}/raw/code/d6bcdb31bd/stackoverflow.com-Posts.7z"
NB = 128

# ------------------------------------------------------------------ html -> text with fenced code
_PRE = re.compile(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", re.S | re.I)
_CODE = re.compile(r"<code[^>]*>(.*?)</code>", re.S | re.I)
_BR = re.compile(r"<br\s*/?>|</p>|</div>|</li>|</h[1-6]>|</tr>", re.I)
_LI = re.compile(r"<li[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_NL = re.compile(r"\n{3,}")


def html_to_text(s: str) -> str:
    s = _PRE.sub(lambda m: "\n```\n" + html.unescape(m.group(1)).strip("\n") + "\n```\n", s)
    s = _CODE.sub(lambda m: "`" + html.unescape(m.group(1)) + "`", s)
    s = _BR.sub("\n", s); s = _LI.sub("\n- ", s)
    s = _TAG.sub("", s)
    s = html.unescape(s)
    return _NL.sub("\n\n", s).strip()


# ------------------------------------------------------------------ pass 1
def pass1(sevenz: Path, bdir: Path):
    if (bdir / "_DONE").exists():
        return
    shutil.rmtree(bdir, ignore_errors=True); (bdir / "q").mkdir(parents=True); (bdir / "a").mkdir()
    qf = [open(bdir / "q" / f"{i}.jsonl", "wb") for i in range(NB)]
    af = [open(bdir / "a" / f"{i}.jsonl", "wb") for i in range(NB)]
    p = subprocess.Popen(["7z", "x", "-so", str(sevenz)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 20)
    n = 0; t0 = time.time()
    for _, el in ET.iterparse(p.stdout, events=("end",)):
        if el.tag != "row":
            continue
        a = el.attrib; t = a.get("PostTypeId")
        if t == "1":
            i = int(a["Id"])
            qf[i % NB].write(orjson.dumps({"id": i, "t": a.get("Title", ""), "b": a.get("Body", ""), "s": int(a.get("Score", 0)),
                                          "tags": a.get("Tags", ""), "acc": int(a.get("AcceptedAnswerId", 0) or 0), "d": a.get("CreationDate", "")[:10]}) + b"\n")
        elif t == "2":
            pid = int(a["ParentId"])
            af[pid % NB].write(orjson.dumps({"id": int(a["Id"]), "p": pid, "b": a.get("Body", ""), "s": int(a.get("Score", 0))}) + b"\n")
        el.clear(); n += 1
        if n % 2_000_000 == 0:
            log(f"[so] pass1 {n/1e6:.0f}M rows, {n/(time.time()-t0):,.0f} rows/s")
    for f in qf + af:
        f.close()
    p.wait(); (bdir / "_DONE").touch()
    log(f"[so] pass1 done: {n:,} rows in {time.time()-t0:,.0f}s")


# ------------------------------------------------------------------ pass 2
def pass2(i, bdir: Path, out_dir: Path):
    ans = {}
    with open(bdir / "a" / f"{i}.jsonl", "rb") as f:
        for line in f:
            o = orjson.loads(line); ans.setdefault(o["p"], []).append(o)
    w = ShardWriter(out_dir, "part", i); st = Stats()
    with open(bdir / "q" / f"{i}.jsonl", "rb") as f:
        for line in f:
            q = orjson.loads(line); st.inc("docs_in")
            al = ans.get(q["id"])
            if q["s"] < 0 or not al:
                st.inc("reject:no_answer" if not al else "reject:negative_score"); continue
            acc = [x for x in al if x["id"] == q["acc"]]
            good = [x for x in al if x["s"] >= 1 and x["id"] != q["acc"]]
            if not acc and not good:
                st.inc("reject:no_good_answer"); continue
            good.sort(key=lambda x: -x["s"])
            chosen = acc + good[:2] if acc else good[:3]
            tags = [t for t in q["tags"].replace("><", "|").strip("<>").split("|") if t]
            parts = [f"Title: {html.unescape(q['t'])}", "Question:\n" + html_to_text(q["b"])]
            for k, x in enumerate(chosen):
                label = "Answer (accepted)" if acc and k == 0 else "Answer"
                parts.append(f"{label} [score {x['s']}]:\n" + html_to_text(x["b"]))
            text = normalize_generic("\n\n".join(parts))
            if len(text) < 100:
                st.inc("reject:too_short"); continue
            w.write({"id": f"{NAME}:{q['id']}", "text": text, "source": NAME, "url": f"https://stackoverflow.com/questions/{q['id']}",
                     "lang": "en", "meta": {"category": "code", "score": q["s"], "tags": tags[:6], "answers": len(chosen), "date": q["d"]}})
            st.inc("docs_out"); st.inc("chars_out", len(text))
    w.close(); st.inc("bytes_out", w.bytes)
    return st


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    remote = f"{BUCKET}/clean/{NAME}"
    if not a.force and gcs_list(remote, "_DONE.json"):
        log(f"[so] already done"); return
    t0 = time.time()
    in_dir = DATA / "in" / NAME; bdir = in_dir / "buckets"; out_dir = DATA / "clean" / NAME
    sevenz = gcs_download(SRC, in_dir / "Posts.7z")
    log(f"[so] have {sevenz} ({sevenz.stat().st_size/1e9:.1f} GB)")
    pass1(sevenz, bdir)
    shutil.rmtree(out_dir, ignore_errors=True); out_dir.mkdir(parents=True)
    total = Stats()
    with mp.Pool(a.workers) as pool:
        for st in pool.starmap(pass2, [(i, bdir, out_dir) for i in range(NB)], chunksize=1):
            total.merge(st)
    stats = total.to_dict(); stats.update({"dataset": NAME, "lang": "en", "category": "code", "seconds": round(time.time() - t0),
                                           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": "Q + accepted/top-scored answers; HTML stripped; code fenced"})
    (out_dir / "_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"[so] docs_in={stats['docs_in']:,} docs_out={stats['docs_out']:,} chars={stats['chars_out']:,} rejects={stats['rejects']}")
    gsutil("rm", "-r", "-q", remote, check=False); gcs_upload_dir(out_dir, remote)
    (out_dir / "_DONE.json").write_text(json.dumps({"dataset": NAME, "docs_out": stats["docs_out"], "finished_at": stats["finished_at"]}))
    gsutil("cp", "-q", str(out_dir / "_DONE.json"), f"{remote}/_DONE.json")
    shutil.rmtree(out_dir, ignore_errors=True); shutil.rmtree(in_dir, ignore_errors=True)
    log(f"[so] uploaded to {remote}")


if __name__ == "__main__":
    main()
