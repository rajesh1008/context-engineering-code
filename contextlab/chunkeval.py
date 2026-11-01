"""contextlab.chunkeval: rank chunks and score them as evidence (Chapter 10, v1.8).

BM25 is the keyword ranker the pipeline has used since Chapter 2. VectorStandIn
is NOT an embedding model: it is a deterministic stand-in that scores word and
character-four-gram overlap, so it forgives "froze" for "frozen" the way a dense
model forgives paraphrase, without any learned meaning. late=True mixes each
chunk's vector with its neighbours' text, which imitates the effect of late
chunking (chunk vectors that carry document context), not its mechanism.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from contextlab.chunker import Chunk

WORD = re.compile(r"[a-z0-9$%]+")
STOP = set("a an the of to and or in on for by is are be it this that with as at from we you your our "
           "not do does any if when which will under only but".split())


def words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(words(c.indexed_text())) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(w for t in self.tf for w in t)
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q = words(query)
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


def grams(text: str) -> Counter:
    c = Counter()
    for w in words(text):
        c["w:" + w] += 1
        p = f"#{w}#"
        for i in range(max(1, len(p) - 3)):
            c["g:" + p[i:i + 4]] += 0.25
    return c


class VectorStandIn:
    def __init__(self, chunks: list[Chunk], late: bool = False, mix: float = 0.35):
        self.chunks = chunks
        raw = [grams(c.indexed_text()) for c in chunks]
        df = Counter(k for v in raw for k in v)
        n = len(raw)
        self.idf = {k: math.log((n + 1) / (d + 0.5)) for k, d in df.items()}
        self.vecs = [self._norm(v) for v in raw]
        if late:
            ctx = [self._norm(grams(c.context)) for c in chunks]
            self.vecs = [self._norm(self._mix(v, cv, mix)) for v, cv in zip(self.vecs, ctx)]

    def _norm(self, v: Counter) -> dict:
        w = {k: x * self.idf.get(k, 1.0) for k, x in v.items()}
        n = math.sqrt(sum(x * x for x in w.values())) or 1.0
        return {k: x / n for k, x in w.items()}

    @staticmethod
    def _mix(a: dict, b: dict, m: float) -> Counter:
        out = Counter({k: (1 - m) * x for k, x in a.items()})
        for k, x in b.items():
            out[k] += m * x
        return out

    def scores(self, query: str) -> list[float]:
        q = self._norm(grams(query))
        return [sum(x * v.get(k, 0.0) for k, x in q.items()) for v in self.vecs]


def ranked(index, query: str, allowed=None) -> list[Chunk]:
    sc = index.scores(query)
    order = sorted(range(len(sc)), key=lambda i: (-sc[i], index.chunks[i].chunk_id))
    return [index.chunks[i] for i in order if allowed is None or allowed(index.chunks[i])]


def coverage(chunk_text: str, gold_text: str) -> float:
    """Share of the gold clause's content words that the chunk contains, in order-free terms."""
    g = Counter(words(gold_text))
    c = Counter(words(chunk_text))
    tot = sum(g.values())
    return sum(min(n, c[w]) for w, n in g.items()) / tot if tot else 0.0


def citable(chunk: Chunk, gold: dict, min_cov: float = 0.8) -> bool:
    """Evidence, not just text: right document and edition, the clause's words
    present, and a cited page on which the clause actually appears."""
    return (chunk.label == gold["label"] and chunk.edition == gold["edition"]
            and chunk.pages[0] in gold["pages"] and coverage(chunk.text, gold["text"]) >= min_cov)


def row_intact(text: str, row: list[str]) -> bool:
    """A table row is evidence only if its cells still sit together, in order."""
    flat = re.sub(r"\s+", " ", text)
    pat = r"\W{0,6}".join(re.escape(c) for c in row)
    return re.search(pat, flat) is not None


def evidence_hit(chunk: Chunk, gold: dict) -> bool:
    if "row" in gold:
        return (chunk.label == gold["label"] and chunk.edition == gold["edition"]
                and chunk.pages[0] in gold["pages"] and row_intact(chunk.text, gold["row"])
                and all(h.lower() in chunk.text.lower() for h in gold["table"]["head"][:3]))
    return citable(chunk, gold)


def names_clause(chunk: Chunk, gold: dict) -> bool:
    """A citation names the clause when the chunk's section path ends at the
    clause's heading, or when most of the chunk is the clause itself."""
    gp = gold.get("path") or []
    if gp and chunk.path and chunk.path[-1].strip().lower() == str(gp[-1]).strip().lower():
        return True
    words = re.findall(r"[a-z0-9]+", chunk.text.lower())
    gw = set(re.findall(r"[a-z0-9]+", gold["text"].lower()))
    return bool(words) and sum(w in gw for w in words) / len(words) >= 0.5


def cited(chunk: Chunk, gold: dict) -> bool:
    """Evidence the reader can cite as the clause: present, on the right page, and named."""
    if "row" in gold:
        return evidence_hit(chunk, gold)
    return evidence_hit(chunk, gold) and names_clause(chunk, gold)
