"""Readers: each takes a local path and yields dicts {text, url, meta}. No normalization here."""
from __future__ import annotations
import bz2, glob, gzip, io, json, os, re, shutil, subprocess, tarfile, tempfile, zipfile
from pathlib import Path

import orjson

CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".kt", ".swift",
            ".dart", ".php", ".rb", ".scala", ".sh", ".sql", ".r", ".jl", ".lua", ".m", ".md", ".rst", ".txt", ".yaml", ".yml", ".toml",
            ".json", ".html", ".css", ".ipynb", ".proto", ".gradle", ".cmake", ".mk", "Makefile", "Dockerfile"}


def _open_text(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    if path.endswith(".zst"):
        import zstandard
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8", errors="ignore")
    if path.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="ignore")
    if path.endswith(".xz"):
        import lzma
        return io.TextIOWrapper(lzma.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


# --------------------------------------------------------------------------- tabular / jsonl
def read_parquet(path, text_col="text", url_col="url", meta_cols=(), row_groups=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    cols = [c for c in dict.fromkeys([text_col, url_col, *meta_cols]) if c in names]
    for batch in pf.iter_batches(batch_size=1024, columns=cols, row_groups=row_groups):
        d = batch.to_pydict()
        n = len(d[text_col])
        urls = d.get(url_col) or [None] * n
        for i in range(n):
            meta = {c: d[c][i] for c in meta_cols if c in d}
            yield {"text": d[text_col][i] or "", "url": urls[i], "meta": meta}


def read_jsonl(path, text_key="text", url_key="url", meta_keys=()):
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                o = orjson.loads(line)
            except Exception:
                continue
            yield {"text": o.get(text_key) or "", "url": o.get(url_key), "meta": {k: o[k] for k in meta_keys if k in o}}


def read_cc100_xz(path, min_lines=1):
    """CC-100: one paragraph per line, documents separated by blank lines."""
    buf = []
    with _open_text(path) as f:
        for line in f:
            if line.strip():
                buf.append(line.rstrip("\n"))
            elif buf:
                yield {"text": "\n".join(buf), "url": None, "meta": {}}; buf = []
    if buf:
        yield {"text": "\n".join(buf), "url": None, "meta": {}}


# --------------------------------------------------------------------------- wikipedia (wikiextractor --json output)
def read_wikiextractor(path):
    with _open_text(path) as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            text = (o.get("text") or "").strip()
            if not text:
                continue
            title = o.get("title") or ""
            yield {"text": f"{title}\n\n{text}" if title and not text.startswith(title) else text,
                   "url": o.get("url"), "meta": {"title": title, "wiki_id": o.get("id")}}


# --------------------------------------------------------------------------- plain text
_HEADER = re.compile(r"^(?:[\w.-]+\.txt|number of beyts: *\d+)\s*$", re.I)


def read_text_file(path, chunk_lines=0, meta=None):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    name = Path(path).name
    head = text.split("\n", 3)
    while head and _HEADER.match(head[0]):  # corpus files start with "<file>.txt\nnumber of beyts: N"
        head.pop(0)
    text = "\n".join(head)
    if not chunk_lines:
        yield {"text": text, "url": None, "meta": {"file": name, **(meta or {})}}
        return
    lines = text.split("\n"); buf = []; idx = 0
    for i, ln in enumerate(lines):
        buf.append(ln)
        if len(buf) >= chunk_lines and (not ln.strip() or len(buf) >= chunk_lines * 2):
            yield {"text": "\n".join(buf), "url": None, "meta": {"file": name, "chunk": idx, **(meta or {})}}; buf = []; idx += 1
    if any(l.strip() for l in buf):
        yield {"text": "\n".join(buf), "url": None, "meta": {"file": name, "chunk": idx, **(meta or {})}}


def read_html_file(path):
    import trafilatura
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True) or ""
    m = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
    yield {"text": text, "url": None, "meta": {"file": Path(path).name, "title": (m.group(1).strip() if m else "")[:200]}}


# --------------------------------------------------------------------------- parallel corpora
def _pair_docs(pairs, group, corpus):
    buf = []
    for fa, en in pairs:
        fa, en = fa.strip(), en.strip()
        if not fa or not en:
            continue
        buf.append(f"fa: {fa}\nen: {en}")
        if len(buf) >= group:
            yield {"text": "\n\n".join(buf), "url": None, "meta": {"corpus": corpus, "pairs": len(buf)}}; buf = []
    if buf:
        yield {"text": "\n\n".join(buf), "url": None, "meta": {"corpus": corpus, "pairs": len(buf)}}


def read_opus_zip(path, group=32, corpus=""):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        en = [n for n in names if n.endswith(".en")]; fa = [n for n in names if n.endswith(".fa")]
        if not en or not fa:
            return
        with z.open(en[0]) as fe, z.open(fa[0]) as ff:
            te = io.TextIOWrapper(fe, encoding="utf-8", errors="ignore"); tf = io.TextIOWrapper(ff, encoding="utf-8", errors="ignore")
            yield from _pair_docs(zip(tf, te), group, corpus or Path(path).parent.name)


def read_opus100_parquet(path, group=32, corpus="opus-100"):
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    col = "translation" if "translation" in t.column_names else t.column_names[0]
    rows = t.column(col).to_pylist()
    yield from _pair_docs(((r.get("fa", ""), r.get("en", "")) for r in rows if isinstance(r, dict)), group, corpus)


# --------------------------------------------------------------------------- PDFs
_PRESENTATION = re.compile("[ﭐ-﷿ﹰ-﻿]+")


def _fix_presentation_forms(text: str) -> str:
    import unicodedata
    return _PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)


_FA_LET = re.compile("[\u0600-\u06ff]"); _LA_LET = re.compile("[A-Za-z]")


def _pdftotext_pages(path):
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", path, "-"], capture_output=True)
    out = r.stdout.decode("utf-8", errors="replace")
    if not out.strip():
        raise RuntimeError((r.stderr.decode(errors="ignore") or "empty pdftotext output")[:200])
    return [p for p in out.split("\f") if p.strip()]


def _pdf_quality(pages, lang):
    text = "".join(pages); n = max(1, len(text))
    bad = text.count("\ufffd") / n
    fa, la = len(_FA_LET.findall(text)), len(_LA_LET.findall(text))
    share = (fa if lang.startswith("fas") else la) / max(1, fa + la)
    return round(share, 3), round(bad, 4)


def read_pdf(path, ocr_lang="fas+eng", min_chars_per_page=80, force_ocr=False, meta=None):
    """pdftotext (poppler: correct bidi reading order) with a quality gate; falls back to tesseract OCR."""
    name = Path(path).name
    why = "forced"
    if not force_ocr:
        try:
            pages = _pdftotext_pages(path)
            avg = sum(len(p.strip()) for p in pages) / max(1, len(pages))
            share, bad = _pdf_quality(pages, ocr_lang)
            if avg >= min_chars_per_page and bad < 0.005 and share >= 0.6:
                text = "\n\n".join(_fix_presentation_forms(p.strip()) for p in pages)
                yield {"text": text, "url": None, "meta": {"file": name, "pages": len(pages), "extraction": "pdftotext",
                                                            "script_share": share, "bad_char_ratio": bad, **(meta or {})}}
                return
            why = f"avg_chars={avg:.0f} share={share} bad={bad}"
        except Exception as e:
            why = f"pdftotext failed: {type(e).__name__}"
    for d in read_pdf_ocr(path, ocr_lang, meta=meta):
        d["meta"]["ocr_reason"] = why; yield d


def read_pdf_ocr(path, lang="fas+eng", dpi=200, low_conf=60.0, meta=None):
    """pdftoppm -> tesseract (tsv) per page. Keeps every page; low-confidence pages are listed in meta."""
    name = Path(path).name
    tmp = tempfile.mkdtemp(prefix="ocr_")
    try:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", path, f"{tmp}/p"], check=True, capture_output=True)
        pages, low, confs_all = [], [], []
        imgs = sorted(glob.glob(f"{tmp}/p-*.png"))
        for pi, img in enumerate(imgs, 1):
            r = subprocess.run(["tesseract", img, "stdout", "-l", lang, "--psm", "6", "tsv"], capture_output=True, text=True)
            lines, cur_key, cur, confs = [], None, [], []
            for row in r.stdout.splitlines()[1:]:
                p = row.split("\t")
                if len(p) < 12 or not p[11].strip():
                    continue
                key = (p[2], p[3], p[4])
                if key != cur_key and cur:
                    lines.append(" ".join(cur)); cur = []
                cur_key = key; cur.append(p[11]); confs.append(float(p[10]))
            if cur:
                lines.append(" ".join(cur))
            mean = sum(confs) / len(confs) if confs else 0.0
            confs_all.append(mean)
            if mean < low_conf:
                low.append(pi)
            pages.append("\n".join(lines))
        text = "\n\n".join(p for p in pages if p.strip())
        yield {"text": text, "url": None, "meta": {"file": name, "pages": len(imgs), "extraction": "tesseract", "ocr_lang": lang,
                                                    "ocr_mean_conf": round(sum(confs_all) / max(1, len(confs_all)), 1),
                                                    "low_conf_pages": low, **(meta or {})}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def read_zip_of_pdfs(path, **kw):
    tmp = tempfile.mkdtemp(prefix="zip_")
    try:
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        for p in sorted(Path(tmp).rglob("*.pdf")):
            for d in read_pdf(str(p), **kw):
                d["meta"]["archive"] = Path(path).name; yield d
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- code repos (tar.zst, .git stripped)
def read_tar_zst_code(path, max_bytes=1_000_000):
    import zstandard
    repo = Path(path).name.replace(".tar.zst", "")
    with open(path, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as r, tarfile.open(fileobj=r, mode="r|") as tar:
        for m in tar:
            if not m.isfile() or m.size > max_bytes or m.size == 0:
                continue
            ext = Path(m.name).suffix.lower() or Path(m.name).name
            if ext not in CODE_EXT:
                continue
            data = tar.extractfile(m).read()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            lines = text.split("\n")
            if lines and sum(len(l) for l in lines) / max(1, len(lines)) > 400:  # minified / generated
                continue
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
            yield {"text": text, "url": None, "meta": {"repo": repo, "path": rel, "ext": ext}}
