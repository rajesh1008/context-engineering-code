"""contextlab.chunker: cut parsed documents into citable units (Chapter 10, v1.8).

Strategies: fixed, recursive, structural, semantic. Late chunking is not a
way of cutting; it is a way of embedding the cuts, so it lives in
contextlab.chunkeval as a retriever option that reads Chunk.context.
Every chunk carries document, edition, section path and pages: a chunk
that cannot be cited is not admitted.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from contextlab.parsing import Block, ParsedDoc

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
SENT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def count_tokens(text: str) -> int:
    """The lab's tokenizer: words and punctuation, about 1.2 per English word."""
    return len(TOKEN_RE.findall(text))


@dataclass
class Element:
    kind: str                       # heading, para, table
    text: str
    pages: list[int]
    level: int = 0
    rows: list[list[str]] | None = None
    row_pages: list[int] | None = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    label: str
    edition: str
    kind: str
    text: str
    pages: list[int]
    path: tuple[str, ...] = ()
    method: str = ""
    prefix: str = ""
    context: str = ""               # neighbouring text, for late-chunking retrieval
    parser: str = ""                # which parser produced the text (lineage)
    source_sha: str = ""            # checksum of the source file (lineage)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)

    def indexed_text(self) -> str:
        return f"{self.prefix}\n{self.text}" if self.prefix else self.text

    def citation(self) -> str:
        sec = " > ".join(self.path) if self.path else "no section"
        return f"{self.label} ({self.edition}), {sec}, page {self.pages[0]}"

    def digest(self) -> str:
        return hashlib.sha1(self.indexed_text().encode()).hexdigest()[:12]


def heading_level(text: str) -> int:
    return 2 if re.match(r"^(\d{1,3}\.|[A-Z]\.\d)", text) else 1


def elements(parsed: ParsedDoc, join_tables: bool = True) -> list[Element]:
    """Merge lines into paragraphs; join a table that continues across a page."""
    out: list[Element] = []
    widths = [len(b.text) for b in parsed.blocks if b.kind == "text"] or [80]
    full = sorted(widths)[int(0.9 * (len(widths) - 1))]
    open_para = False
    for b in parsed.blocks:
        if b.kind == "heading":
            prev = out[-1] if out else None
            # a heading that wraps onto a second line is one heading, not two
            if (prev is not None and prev.kind == "heading" and prev.pages[-1] == b.page
                    and heading_level(b.text) == 1 and not b.text.startswith("Section")
                    and (prev.level == 2 or b.text[:1].islower() or b.text.split()[0] in ("Or", "And", "Of"))):
                prev.text += " " + b.text
                continue
            out.append(Element("heading", b.text, [b.page], heading_level(b.text)))
            open_para = False
        elif b.kind == "table":
            if join_tables and b.continued and out and out[-1].kind == "table":
                out[-1].rows += b.rows
                out[-1].row_pages += [b.page] * len(b.rows)
                out[-1].pages = sorted(set(out[-1].pages) | {b.page})
            else:
                out.append(Element("table", b.text, [b.page], rows=[list(r) for r in b.rows],
                                   row_pages=[b.page] * len(b.rows)))
            open_para = False
        elif b.kind == "text":
            if open_para and out[-1].kind == "para":
                out[-1].text += " " + b.text
                if b.page not in out[-1].pages:
                    out[-1].pages.append(b.page)
            else:
                out.append(Element("para", b.text, [b.page]))
            # a short line ending a sentence closes the paragraph
            open_para = not (b.text.rstrip().endswith((".", ":")) and len(b.text) < 0.8 * full)
    for e in out:
        if e.kind == "table":
            e.text = table_text(e.rows)
    return out


def table_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(r) for r in rows)


class Chunker:
    """Cut a ParsedDoc into Chunks.

    strategy  fixed | recursive | structural | semantic
    size      target tokens per chunk (a maximum for structural)
    overlap   tokens shared by neighbouring fixed windows
    prefix    prepend document, edition and section path to the indexed text
    """

    def __init__(self, strategy: str = "structural", size: int = 512, overlap: int = 0,
                 prefix: bool = True, min_tokens: int = 24, rows_per_chunk: int = 1,
                 semantic_pct: float = 0.15, context_tokens: int = 400):
        self.strategy, self.size, self.overlap, self.prefix = strategy, size, overlap, prefix
        self.min_tokens, self.rows_per_chunk = min_tokens, rows_per_chunk
        self.semantic_pct, self.context_tokens = semantic_pct, context_tokens

    @property
    def name(self) -> str:
        if self.strategy == "fixed":
            return f"fixed-{self.size}" + (f"/o{self.overlap}" if self.overlap else "")
        if self.strategy == "structural":
            return "structural" + ("+path" if self.prefix else "")
        return f"{self.strategy}-{self.size}"

    # ------------------------------------------------------------ public
    def chunk(self, parsed: ParsedDoc, meta: dict) -> list[Chunk]:
        els = elements(parsed, join_tables=self.strategy == "structural")
        pieces = getattr(self, "_" + self.strategy)(els)
        chunks = []
        for i, (text, pages, path) in enumerate(pieces):
            if not text.strip():
                continue
            c = Chunk(f"{meta['doc_id']}:{self.name}:{i:04d}", meta["doc_id"], meta["label"],
                      meta["edition"], meta["kind"], text.strip(), sorted(set(pages)) or [1],
                      tuple(path), self.name, parser=parsed.method,
                      source_sha=meta.get("sha256", "")[:12])
            if self.prefix and self.strategy == "structural":
                c.prefix = f"{c.label} ({c.edition}) > " + " > ".join(path)
            chunks.append(c)
        self._contexts(chunks)
        return chunks

    # ------------------------------------------------------------ strategies
    def _words(self, els):
        for e in els:
            for w in e.text.split():
                yield w, e.pages[0] if len(e.pages) == 1 else e.pages[min(len(e.pages) - 1, 0)]

    def _fixed(self, els):
        words = []
        for e in els:            # spread a multi-page paragraph's words over its pages
            ws = e.text.split()
            for k, w in enumerate(ws):
                words.append((w, e.pages[min(len(e.pages) - 1, k * len(e.pages) // max(1, len(ws)))]))
        out, i = [], 0
        while i < len(words):
            j, tok = i, 0
            while j < len(words) and tok < self.size:
                tok += count_tokens(words[j][0]); j += 1
            out.append((" ".join(w for w, _ in words[i:j]), [p for _, p in words[i:j]], ()))
            if j >= len(words):
                break
            back, k = 0, j
            while k > i + 1 and back < self.overlap:
                k -= 1; back += count_tokens(words[k][0])
            i = k if self.overlap else j
        return out

    def _split_long(self, text: str, limit: int) -> list[str]:
        if count_tokens(text) <= limit:
            return [text]
        sents, parts, cur = SENT_RE.split(text), [], ""
        for s in sents:
            if cur and count_tokens(cur + " " + s) > limit:
                parts.append(cur); cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            parts.append(cur)
        out = []
        for p in parts:                      # a single sentence longer than the limit
            ws = p.split()
            while count_tokens(" ".join(ws)) > limit:
                k = max(1, int(len(ws) * limit / count_tokens(" ".join(ws))))
                out.append(" ".join(ws[:k])); ws = ws[k:]
            if ws:
                out.append(" ".join(ws))
        return out

    def _recursive(self, els):
        units = []
        for e in els:
            for part in self._split_long(e.text, self.size):
                units.append((part, e.pages))
        out, cur, pages = [], "", []
        for text, pg in units:
            if cur and count_tokens(cur + "\n\n" + text) > self.size:
                out.append((cur, pages, ())); cur, pages = "", []
            cur = (cur + "\n\n" + text).strip(); pages = pages + pg
        if cur:
            out.append((cur, pages, ()))
        return out

    def _structural(self, els):
        out, path = [], ["(untitled)"]
        buf: list[Element] = []

        def flush():
            if not buf:
                return
            paras = [e for e in buf]
            cur, pages = "", []
            for e in paras:
                for part in self._split_long(e.text, self.size):
                    if cur and count_tokens(cur + "\n\n" + part) > self.size:
                        out.append((cur, pages, tuple(path))); cur, pages = "", []
                    cur = (cur + "\n\n" + part).strip(); pages = pages + e.pages
            if cur:
                out.append((cur, pages, tuple(path)))
            buf.clear()

        for e in els:
            if e.kind == "heading":
                flush()
                path = path[: e.level - 1] if e.level > 1 and len(path) >= e.level - 1 else []
                path = (path + [e.text])[-3:]
            elif e.kind == "table":
                flush()
                head, body, rp = e.rows[0], e.rows[1:], e.row_pages[1:]
                for k in range(0, max(1, len(body)), self.rows_per_chunk):
                    out.append((table_text([head] + body[k:k + self.rows_per_chunk]),
                                rp[k:k + self.rows_per_chunk] or e.pages, tuple(path)))
            else:
                buf.append(e)
        flush()
        merged = []                         # fold a tiny chunk into its neighbour in the same section
        for text, pages, p in out:
            if merged and merged[-1][2] == p and count_tokens(text) < self.min_tokens \
                    and count_tokens(merged[-1][0]) + count_tokens(text) <= self.size and "|" not in text:
                merged[-1] = (merged[-1][0] + "\n\n" + text, merged[-1][1] + pages, p)
            else:
                merged.append((text, pages, p))
        return merged

    def _semantic(self, els):
        sents = []
        for e in els:
            for s in SENT_RE.split(e.text):
                if s.strip():
                    sents.append((s.strip(), e.pages))
        if not sents:
            return []
        vecs = [Counter(w.lower() for w in re.findall(r"[A-Za-z]{3,}", s)) for s, _ in sents]
        df = Counter(w for v in vecs for w in v)
        n = len(vecs)
        idf = {w: math.log((n + 1) / (d + 0.5)) for w, d in df.items()}

        def cos(a, b):
            num = sum(a[w] * b[w] * idf[w] ** 2 for w in a if w in b)
            na = math.sqrt(sum((a[w] * idf[w]) ** 2 for w in a)) or 1
            nb = math.sqrt(sum((b[w] * idf[w]) ** 2 for w in b)) or 1
            return num / (na * nb)

        sims = [cos(vecs[i], vecs[i + 1]) for i in range(n - 1)]
        cut = sorted(sims)[int(self.semantic_pct * (len(sims) - 1))] if sims else 0
        out, cur, pages = [], "", []
        for i, (s, pg) in enumerate(sents):
            cur = (cur + " " + s).strip(); pages = pages + pg
            boundary = i < n - 1 and sims[i] <= cut
            if boundary or count_tokens(cur) >= self.size or i == n - 1:
                out.append((cur, pages, ())); cur, pages = "", []
        return out

    def _contexts(self, chunks):
        """Neighbouring text within the same document, for late-chunking retrieval."""
        for i, c in enumerate(chunks):
            left, right, j, k = [], [], i - 1, i + 1
            budget = self.context_tokens
            while budget > 0 and (j >= 0 or k < len(chunks)):
                if j >= 0:
                    left.insert(0, chunks[j].text); budget -= chunks[j].tokens; j -= 1
                if k < len(chunks) and budget > 0:
                    right.append(chunks[k].text); budget -= chunks[k].tokens; k += 1
            c.context = " ".join(left + [c.text] + right)
