"""Shared pieces for the Persian-corpus pipeline: normalization, language id, shard writers, stats, GCS helpers."""
from __future__ import annotations
import gzip, io, json, os, re, subprocess, sys, time, unicodedata
from collections import Counter
from pathlib import Path

import orjson
import xxhash

BUCKET = os.environ.get("CORPUS_BUCKET", "gs://YOUR-BUCKET")
DATA = Path(os.environ.get("PIPE_DATA", "/data"))
LID_MODEL = os.environ.get("LID_MODEL", "/models/lid.176.bin")

# ----------------------------------------------------------------------------- normalization
_FA_CHAR_MAP = str.maketrans({
    "\u0643": "\u06a9",  # Arabic kaf -> Persian keheh
    "\u064a": "\u06cc",  # Arabic yeh -> Farsi yeh
    "\u0649": "\u06cc",  # alef maksura -> yeh
    "\u06d2": "\u06cc",  # yeh barree -> yeh
    "\u0629": "\u0647",  # teh marbuta -> heh
    "\u0640": None,       # tatweel
    "\u200b": None, "\u200d": None, "\u200e": None, "\u200f": None, "\u2060": None, "\ufeff": None,  # zero-width / bidi marks
    "\u202a": None, "\u202b": None, "\u202c": None, "\u202d": None, "\u202e": None, "\u2066": None, "\u2067": None, "\u2068": None, "\u2069": None,  # bidi embeddings/isolates
    "\u00a0": " ", "\u2007": " ", "\u2008": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ", "\u3000": " ",  # odd spaces
    "\u066a": "%",
})
_DIGITS = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGITS.update({ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})
_ZWNJ_RUN = re.compile("‌{2,}")
_ZWNJ_EDGE = re.compile(r"(?:(?<=\s)‌+)|(?:‌+(?=\s))|^‌+|‌+$", re.M)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")
_SPACES = re.compile(r"[ \t　]+")
_MULTI_NL = re.compile(r"\n{3,}")
_TRAIL = re.compile(r"[ \t]+\n")


def normalize_fa(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = text.translate(_FA_CHAR_MAP).translate(_DIGITS)
    text = _ZWNJ_RUN.sub("‌", text)
    text = _ZWNJ_EDGE.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _TRAIL.sub("\n", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def normalize_generic(text: str) -> str:
    """Light touch for English/code: line endings, NBSP, control chars, trailing spaces. Never reflows code."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ").replace("﻿", "")
    text = _CTRL.sub("", text)
    text = _TRAIL.sub("\n", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def normalize(text: str, lang: str) -> str:
    return normalize_fa(text) if lang == "fa" else normalize_generic(text)


_FA_LETTERS = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")


def script_ratio(text: str) -> tuple[float, float]:
    """(persian/arabic-script share, latin share) over letters — cheap sanity check next to fastText."""
    fa = len(_FA_LETTERS.findall(text)); la = len(_LATIN.findall(text)); tot = fa + la
    return (fa / tot, la / tot) if tot else (0.0, 0.0)


# ----------------------------------------------------------------------------- language id
class LangID:
    def __init__(self, path: str = LID_MODEL):
        import fasttext
        fasttext.FastText.eprint = lambda *a, **k: None
        self.m = fasttext.load_model(path)

    def predict(self, text: str, head: int = 3000) -> tuple[str, float]:
        s = text[:head].replace("\n", " ")
        labels, probs = self.m.predict(s, k=1)
        return labels[0].replace("__label__", ""), float(probs[0])


# ----------------------------------------------------------------------------- boilerplate lines
SHORT_LINE = 25


def short_line_keys(text: str, domain: str):
    """xxh64 keys of (domain, line) for lines shorter than SHORT_LINE chars."""
    pre = domain.encode() + b"\x1f"
    for line in text.split("\n"):
        line = line.strip()
        if 0 < len(line) < SHORT_LINE:
            yield xxhash.xxh64_intdigest(pre + line.encode())


def drop_boilerplate_lines(text: str, domain: str, banned: set) -> tuple[str, int]:
    if not banned:
        return text, 0
    pre = domain.encode() + b"\x1f"
    out, dropped = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if 0 < len(s) < SHORT_LINE and xxhash.xxh64_intdigest(pre + s.encode()) in banned:
            dropped += 1; continue
        out.append(line)
    return "\n".join(out), dropped


def domain_of(url: str | None, fallback: str) -> str:
    if not url:
        return fallback
    m = re.match(r"^(?:https?://)?([^/:?#]+)", url)
    return m.group(1).lower() if m else fallback


# ----------------------------------------------------------------------------- writers
class ShardWriter:
    """Gzip JSONL shards of ~max_bytes uncompressed. Files: {prefix}-{worker:03d}-{n:04d}.jsonl.gz"""

    def __init__(self, out_dir: Path, prefix: str, worker: int, max_bytes: int = 256 << 20):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.worker, self.max_bytes = prefix, worker, max_bytes
        self.n = -1; self.f = None; self.written = 0; self.files = []
        self.docs = 0; self.chars = 0; self.bytes = 0

    def _open(self):
        if self.f: self.f.close()
        self.n += 1; self.written = 0
        p = self.dir / f"{self.prefix}-{self.worker:03d}-{self.n:04d}.jsonl.gz"
        self.files.append(p)
        self.f = gzip.open(p, "wb", compresslevel=4)

    def write(self, doc: dict):
        if self.f is None or self.written >= self.max_bytes:
            self._open()
        line = orjson.dumps(doc) + b"\n"
        self.f.write(line); self.written += len(line)
        self.docs += 1; self.chars += len(doc.get("text", "")); self.bytes += len(line)

    def close(self):
        if self.f: self.f.close(); self.f = None


# ----------------------------------------------------------------------------- stats
class Stats:
    def __init__(self):
        self.c = Counter()

    def inc(self, k, n=1):
        self.c[k] += n

    def merge(self, other: "Stats"):
        self.c.update(other.c)

    def to_dict(self):
        d = dict(self.c)
        rej = {k[7:]: v for k, v in d.items() if k.startswith("reject:")}
        return {"docs_in": d.get("docs_in", 0), "docs_out": d.get("docs_out", 0), "chars_out": d.get("chars_out", 0),
                "bytes_out_jsonl": d.get("bytes_out", 0), "lines_dropped_boilerplate": d.get("bp_lines", 0),
                "rejects": dict(sorted(rej.items(), key=lambda kv: -kv[1])), "raw": d}


# ----------------------------------------------------------------------------- gcs
def gsutil(*args, check=True, capture=False, tries=3):
    """gcloud storage wrapper with retries for transient failures (rsync/cp against a busy bucket)."""
    cmd = ["gcloud", "--no-user-output-enabled", "storage", *args]
    for attempt in range(tries):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 or not check:
            return r
        log(f"gcloud storage {args[0]} failed (try {attempt + 1}/{tries}): {r.stderr.strip()[-300:]}")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"gcloud storage failed after {tries} tries: {' '.join(str(a) for a in args[:3])} ...\n{r.stderr.strip()[-300:]}")


def gcs_list(prefix: str, pattern: str = "**") -> list[tuple[str, int]]:
    r = subprocess.run(["gcloud", "storage", "ls", "-l", f"{prefix.rstrip('/')}/{pattern}"], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2].startswith("gs://") and not parts[2].endswith("/"):
            out.append((parts[2], int(parts[0])))
    return out


def gcs_download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    gsutil("cp", "-q", url, str(tmp)); tmp.rename(dest)
    return dest


def gcs_upload_dir(local: Path, remote: str):
    gsutil("rsync", "-r", "-q", str(local), remote)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def doc_id(dataset: str, shard: str, i: int) -> str:
    return f"{dataset}:{Path(shard).name}:{i}"
