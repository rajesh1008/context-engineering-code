"""contextlab.rankers: keyword and stand-in vector rankers (moved from chunkeval)."""
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
