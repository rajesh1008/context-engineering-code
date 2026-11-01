"""contextlab.parsing: turn PDF pages into blocks with a page and a kind (Chapter 10, v1.8).

Five ways to parse a page, from cheapest to most expensive:
  text_layer  the PDF's own text, in content-stream order (pypdf)
  layout      words with positions: columns, furniture, tables (pdfplumber)
  ocr         render the page and read the pixels (RapidOCR)
  vision      a deterministic stand-in for a vision-language parser
  routed      deterministic first; OCR or the vision stand-in only where needed
Every block keeps the page it came from, so every chunk can cite one.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Block:
    kind: str                      # heading, text, table, furniture
    text: str
    page: int
    level: int = 0
    rows: list[list[str]] | None = None
    continued: bool = False        # a table that carries on from the previous page


@dataclass
class ParsedDoc:
    doc_id: str
    method: str
    blocks: list[Block] = field(default_factory=list)
    seconds: float = 0.0
    pages: int = 0
    usd: float = 0.0               # illustrative, vision stand-in only


def furniture_key(line: str) -> str:
    """Normalise a line so the same header or footer matches on every page."""
    return re.sub(r"\d+", "#", line.strip().lower())


def repeated_lines(pages: list[list[str]], share: float = 0.5) -> set[str]:
    """Lines (normalised) that recur on at least `share` of a document's pages."""
    if len(pages) < 2:
        return set()
    seen = Counter(k for p in pages for k in {furniture_key(l) for l in p if l.strip()})
    return {k for k, n in seen.items() if n >= max(2, share * len(pages))}


HEADING_RE = re.compile(r"^(\d{1,3}\.\s+[A-Z][A-Za-z ,'-]{2,60}|Section [IVX]+ [A-Z][A-Za-z ]+|[A-Z][A-Za-z ]{2,40})$")


def guess_heading(line: str) -> bool:
    """Text-only heading heuristic for parsers that lose font information."""
    s = line.strip()
    return bool(HEADING_RE.match(s)) and not s.endswith(".") and len(s.split()) <= 9


# ---------------------------------------------------------------- 1. text layer
def parse_text_layer(path: str, doc_id: str) -> ParsedDoc:
    from pypdf import PdfReader
    t0, out = time.perf_counter(), ParsedDoc(doc_id, "text_layer")
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages, 1):
        for line in (page.extract_text() or "").splitlines():
            if line.strip():
                out.blocks.append(Block("heading" if guess_heading(line) else "text", line.strip(), i))
    out.pages, out.seconds = len(reader.pages), time.perf_counter() - t0
    return out


# ---------------------------------------------------------------- 2. layout
def _lines_from_words(words, tol=3.0):
    rows: dict[int, list] = {}
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        key = next((k for k in rows if abs(k - w["top"]) <= tol), None)
        rows.setdefault(round(w["top"]) if key is None else key, []).append(w)
    return [sorted(ws, key=lambda w: w["x0"]) for _, ws in sorted(rows.items())]


def parse_layout(path: str, doc_id: str, strip_furniture: bool = True,
                 tables: bool = True, columns: bool = True, band_only: bool = True) -> ParsedDoc:
    import pdfplumber
    t0, out = time.perf_counter(), ParsedDoc(doc_id, "layout")
    staged = []
    with pdfplumber.open(path) as pdf:
        out.pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            found = page.find_tables() if tables else []
            boxes = [t.bbox for t in found]
            words = [w for w in page.extract_words(extra_attrs=["fontname", "size"])
                     if not any(b[0] <= w["x0"] <= b[2] and b[1] <= w["top"] <= b[3] for b in boxes)]
            mid = page.width / 2
            crossing = sum(w["x0"] < mid < w["x1"] for w in words)
            two = columns and crossing < 0.01 * max(1, len(words)) and \
                sum(w["x0"] > mid for w in words) > 0.2 * max(1, len(words))
            band = [w for w in words if w["top"] < 60 or w["top"] > page.height - 60]
            body = [w for w in words if w not in band]
            cols = ([[w for w in body if w["x0"] < mid], [w for w in body if w["x0"] >= mid]]
                    if two else [body])
            items = []
            groups = [(-1, [w for w in band if w["top"] < 60])] + list(enumerate(cols)) + \
                     [(99, [w for w in band if w["top"] >= 60])]
            for ci, ws in groups:
                for ln in _lines_from_words(ws):
                    text = " ".join(w["text"] for w in ln)
                    bold = all("Bold" in w["fontname"] for w in ln) and ln[0]["size"] >= 10
                    items.append(("heading" if bold else "text", text, ln[0]["top"], ci))
            for t in found:
                rows = [[(c or "").strip() for c in r] for r in t.extract()]
                items.append(("table", rows, t.bbox[1], 0))
            items.sort(key=lambda x: (x[3], x[2]))
            staged.append(items)
    # furniture is a line that repeats in the same margin band; a body line that
    # happens to repeat (template wording in every clause) is never furniture
    furn = repeated_lines([[x[1] for x in it if x[0] != "table" and (x[3] in (-1, 99) or not band_only)] for it in staged]) \
        if strip_furniture else set()
    last_cols = None
    for i, items in enumerate(staged, 1):
        body_seen = False
        for kind, payload, _top, ci in items:
            if kind == "table":
                cont = last_cols == len(payload[0]) and not body_seen
                out.blocks.append(Block("table", " | ".join(" | ".join(r) for r in payload), i,
                                        rows=payload, continued=cont))
                last_cols, body_seen = len(payload[0]), True
                continue
            if (ci in (-1, 99) or not band_only) and furniture_key(payload) in furn:
                continue
            if ci not in (-1, 99):
                body_seen = True
            out.blocks.append(Block(kind, payload, i))
        if not any(k == "table" for k, *_ in items):
            last_cols = None
    out.seconds = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------- 3. OCR
_OCR = None


def ocr_page_lines(path: str, index: int, dpi: int = 150) -> list[tuple]:
    """Render one page and return (x0, y, text, x1) per detected line, in points."""
    global _OCR
    import pypdfium2 as pdfium
    from rapidocr_onnxruntime import RapidOCR
    import numpy as np
    _OCR = _OCR or RapidOCR()
    pdf = pdfium.PdfDocument(path)
    img = pdf[index].render(scale=dpi / 72).to_pil().convert("RGB")
    res, _ = _OCR(np.array(img))
    s = 72 / dpi
    return [(b[0][0] * s, b[0][1] * s, txt, b[1][0] * s) for b, txt, _conf in (res or [])]


def parse_ocr(path: str, doc_id: str, cached: dict | None = None) -> ParsedDoc:
    """OCR every page. `cached` maps page index -> lines, so the lab can reuse a run."""
    import pypdfium2 as pdfium
    t0, out = time.perf_counter(), ParsedDoc(doc_id, "ocr")
    n = len(pdfium.PdfDocument(path))
    pages = []
    for i in range(n):
        lines = [tuple(l) for l in (cached[i] if cached is not None else ocr_page_lines(path, i))]
        width = 612
        crossing = sum(l[0] < width / 2 < l[3] for l in lines)
        two = crossing < 0.05 * max(1, len(lines)) and sum(l[0] > width / 2 for l in lines) > 0.2 * max(1, len(lines))
        band = [l for l in lines if l[1] < 60 or l[1] > 792 - 60]
        body = [l for l in lines if l not in band]
        cols = [[l for l in body if l[0] < width / 2], [l for l in body if l[0] >= width / 2]] if two else [body]
        ordered = []
        for c in [[l for l in band if l[1] < 60]] + cols + [[l for l in band if l[1] >= 60]]:
            rows: dict[int, list] = {}
            for x, y, t, _x1 in sorted(c, key=lambda l: (l[1], l[0])):
                k = next((k for k in rows if abs(k - y) <= 4), None)
                rows.setdefault(round(y) if k is None else k, []).append((x, t))
            ordered += [" ".join(t for _, t in sorted(v)) for _, v in sorted(rows.items())]
        pages.append(ordered)
    furn = repeated_lines(pages)
    for i, lines in enumerate(pages, 1):
        for ln in lines:
            if furniture_key(ln) not in furn:
                out.blocks.append(Block("heading" if guess_heading(ln) else "text", ln, i))
    out.pages = n
    out.seconds = time.perf_counter() - t0 if cached is None else 0.0
    return out


# ---------------------------------------------------------------- 4. vision stand-in
VISION_USD_PER_PAGE = 0.012    # illustrative, US dollars
VISION_SECONDS_PER_PAGE = 3.5  # modelled latency, not measured


def parse_vision_standin(truth_pages: list[dict], doc_id: str, seed: int = 10) -> ParsedDoc:
    """A deterministic stand-in for a vision-language parser. It reads the
    ground truth and injects the error kinds such parsers are reported to make:
    a substituted word, a misread table digit, an omitted paragraph on a
    dense page. The rates are set, not measured."""
    import random
    rng = random.Random(seed + sum(map(ord, doc_id)))
    out = ParsedDoc(doc_id, "vision")
    for tp in truth_pages:
        dense = len(tp["lines"]) > 90
        for ln in tp["lines"]:
            if ln["kind"] == "table":
                rows = [[re.sub(r"\d", lambda m: str((int(m.group()) + 1) % 10), c)
                         if rng.random() < 0.02 else c for c in r] for r in ln["rows"]]
                out.blocks.append(Block("table", " | ".join(" | ".join(r) for r in rows), tp["page"], rows=rows))
                continue
            if dense and rng.random() < 0.01:
                continue                                  # omission
            words = ln["text"].split()
            if words and rng.random() < 0.02:
                words[rng.randrange(len(words))] = rng.choice(["the", "a", "of", "loss"])
            kind = "heading" if ln["kind"].startswith("h") else "text"
            out.blocks.append(Block(kind, " ".join(words), tp["page"]))
    out.pages = len(truth_pages)
    out.seconds = VISION_SECONDS_PER_PAGE * out.pages
    out.usd = VISION_USD_PER_PAGE * out.pages
    return out


# ---------------------------------------------------------------- 5. routed
def route_page(has_text_layer: bool, has_ruled_table: bool) -> str:
    """Deterministic first: the model earns its seat only on scans with tables."""
    if has_text_layer:
        return "layout"
    return "vision" if has_ruled_table else "ocr"
