"""Chapter 20: the Lakeshore retrieval stack as one configurable build.

Every stage Part III added is a switch, so the same script can run the
baseline the workshop audits, the rebuilt stack, and every ablation between.
The corpus and the 1,000 queries are Chapter 12's (synthetic, seeded).
"""
import hashlib
from dataclasses import dataclass, replace

import numpy as np

from contextlab.chunker import count_tokens
from contextlab.chunkeval import coverage
from contextlab.retrieval.corpus import PRECEDENCE, Span, admitted, build_corpus
from contextlab.retrieval.dense import DIM, SimEncoder
from contextlab.retrieval.fusion import ranked, weighted
from contextlab.retrieval.index import InvertedIndex
from contextlab.retrieval.rerank import SimCrossEncoder


@dataclass(frozen=True)
class Config:
    chunking: str = "structural"     # "structural" spans or "fixed" windows
    window: int = 256                # tokens per fixed window
    retriever: str = "hybrid"        # keyword, dense or hybrid
    rerank: bool = True
    authority: bool = True           # Chapter 15 admission; False = inventory only
    precedence: bool = True          # authority order within the top topic
    graph: bool = True               # follow modifies/governs links
    index_model: str = "sim-embed@1"
    query_model: str = "sim-embed@1"
    reindexed: float = 0.0           # share of the index re-embedded by query_model


BASELINE = Config(chunking="fixed", retriever="keyword", rerank=False,
                  authority=False, precedence=False, graph=False)
REBUILT = Config()


class SimEncoderV2(SimEncoder):
    """The 'upgraded' embedder: same interface, its own geometry."""
    name = "sim-embed@2"

    def encode(self, text):
        if text in self._cache:
            return self._cache[text]
        v = super().encode(text)
        seed = int.from_bytes(hashlib.sha256(b"sim-embed@2").digest()[:8], "big")
        rot = np.random.default_rng(seed).permutation(DIM)
        sign = np.where(np.random.default_rng(seed + 1).random(DIM) < 0.5, -1.0, 1.0)
        out = v[rot] * sign
        self._cache[text] = out
        return out


ENCODERS = {"sim-embed@1": SimEncoder, "sim-embed@2": SimEncoderV2}


def inventory_admits(span, q) -> bool:
    """Chapter 9's inventory: licensed sources, right family, right form edition."""
    if span.doc_type == "underwriting":
        return False
    if span.family not in (q.family, "ALL"):
        return False
    return not (span.doc_type == "form" and span.family == "HO-3"
                and span.edition != q.edition)


def windows(corpus: dict, size: int) -> dict:
    """Fixed windows over each document's text, as before Chapter 10."""
    docs: dict[str, list] = {}
    for s in corpus.values():
        docs.setdefault(s.doc, []).append(s)
    out = {}
    for doc, spans in docs.items():
        words = " ".join(s.full for s in spans).split()
        step = max(1, int(size / 1.2))           # about 1.2 tokens per word
        for n, i in enumerate(range(0, len(words), step)):
            first = spans[0]
            out[f"W:{doc}:{n}"] = replace(first, span_id=f"W:{doc}:{n}", section=doc,
                                          heading=f"window {n + 1}", text=" ".join(words[i:i + step]),
                                          clause="", links=[], superseded_by="")
    return out


class Stack:
    """One build: units (spans or windows), a lexical index, a dense index."""

    def __init__(self, cfg: Config, corpus: dict | None = None):
        self.cfg, self.spans = cfg, corpus or build_corpus()
        self.units = (self.spans if cfg.chunking == "structural"
                      else windows(self.spans, cfg.window))
        self.texts = {i: u.full for i, u in self.units.items()}
        self.delivers = {i: self._delivered(u) for i, u in self.units.items()}
        self.index = InvertedIndex()
        for i, t in self.texts.items():
            self.index.add(i, t)
        old, new = ENCODERS[cfg.index_model](), ENCODERS[cfg.query_model]()
        self.ids = sorted(self.texts)              # the first `cut` are re-embedded
        cut = int(round(cfg.reindexed * len(self.ids))) if old.name != new.name else 0
        self.matrix = np.vstack([(new if n < cut else old).encode(self.texts[i])
                                 for n, i in enumerate(self.ids)])
        self.qenc = new
        self.cross = SimCrossEncoder(self.index, ENCODERS[cfg.query_model]())
        self.build_id = hashlib.sha256(repr(cfg).encode()).hexdigest()[:8]

    def _delivered(self, u) -> list[str]:
        if self.cfg.chunking == "structural":
            return [u.span_id]
        return [s.span_id for s in self.spans.values()
                if s.doc == u.doc and coverage(u.text, s.full) >= 0.8]

    def admit(self, uid, q) -> bool:
        u = self.units[uid]
        return admitted(u, q) if self.cfg.authority else inventory_admits(u, q)

    # ------------------------------------------------------------ retrieval
    def dense(self, text: str) -> dict[str, float]:
        sims = self.matrix @ self.qenc.encode(text)
        return dict(zip(self.ids, map(float, sims)))

    def candidates(self, q, depth: int = 100) -> list[str]:
        c = self.cfg
        lex = self.index.bm25(q.text) if c.retriever != "dense" else {}
        den = self.dense(q.text) if c.retriever != "keyword" else {}
        scores = lex if not den else den if not lex else weighted(lex, den, 0.5)
        return [i for i in ranked(scores)[:depth] if self.admit(i, q)]

    def retrieve(self, q, k: int = 5) -> list[str]:
        out = self.candidates(q)
        if self.cfg.rerank:
            out = self.cross.rerank(q.text, out[:30], self.texts)[0]
        if self.cfg.graph and self.cfg.chunking == "structural":
            out = self._expand(out, q)
        if self.cfg.precedence and self.cfg.chunking == "structural":
            out = self._authority_order(out, k)
        return out

    def _expand(self, out, q, seeds: int = 3):
        out = list(out)
        for parent in list(out[:seeds]):
            for nb in self.units[parent].links:
                if nb in out[:5] or not self.admit(nb, q):
                    continue
                if nb in out:
                    out.remove(nb)
                out.insert(out.index(parent) + 1, nb)
        return out

    def _authority_order(self, out, k):
        """Within the top topic, the span with authority leads; superseded drop."""
        top, rest = out[:k], out[k:]
        if not top:
            return out
        live = [i for i in top if not self.units[i].superseded_by] or top
        topic = self.units[live[0]].clause
        lead = max((i for i in live if self.units[i].clause == topic),
                   key=lambda i: (PRECEDENCE[self.units[i].doc_type], -live.index(i)))
        return [lead] + [i for i in live if i != lead] + rest

    # ------------------------------------------------------------- measures
    def packet(self, q, k: int = 5) -> dict:
        units = self.retrieve(q, k)[:k]
        got = [s for u in units for s in self.delivers[u]]
        first = self.delivers[units[0]] if units else []
        return {"units": units, "spans": got,
                "hit": any(g in got for g in q.gold),
                "gov": q.governing in got,
                "gov1": q.governing in first,
                "leaks": sum(not admitted(self.spans[s], q) for s in got),
                "leaky": any(not admitted(self.spans[s], q) for s in got),
                "tokens": sum(count_tokens(self.texts[u]) for u in units)}


# ---------------------------------------------------------------- the guard

class Pinned:
    """Adapter so Chapter 14's EmbeddingRegistry can fingerprint sim-embed."""

    def __init__(self, model_id: str, encoder=None):
        self.model_id, self.dim, self.max_input_tokens = model_id, DIM, 512
        self.enc = encoder or ENCODERS[model_id]()

    def encode(self, texts) -> np.ndarray:
        return np.vstack([self.enc.encode(t) for t in texts])


PROBES = ["water backed up through the floor drain", "frozen pipe in a vacant house",
          "hail damage to the roof shingles", "rideshare delivery collision",
          "LM-HO 2311 limited fungi", "notice of claim within 60 days"]
