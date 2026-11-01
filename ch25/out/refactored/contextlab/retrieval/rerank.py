"""A deterministic cross-encoder stand-in ("sim-rerank").

A cross-encoder reads the query and one candidate together and returns one
score. This stand-in does the same with stated features, each of which needs
both texts at once: how much of the query's weighted vocabulary the span
covers, whether identifiers and numbers in the query appear verbatim, and
whether the span's polarity (grants cover or takes it away) matches what the
question asks. It is slow on purpose: the cost model charges per pair.
"""
import re

from .dense import LEXICON, SimEncoder
from .text import analyse, stem, tokenize

MS_PER_PAIR = 6.0          # cost model for a small cross-encoder on a CPU
EXCLUDING = re.compile(r"\b((do|does|will) not (insure|cover|pay|apply|include)|"
                       r"not covered|excluded|no coverage)\b")
ASKS_EXCLUSION = re.compile(r"\b(not (covered|insured|pay|paid|apply)|"
                            r"isn't covered|doesn't cover|excluded|excludes|"
                            r"exclusion|no coverage)\b")


def concepts(text: str) -> set[str]:
    return {LEXICON.get(t) or LEXICON.get(stem(t)) for t in tokenize(text)} - {None}


def identifiers(text: str) -> set[str]:
    return {t for t in tokenize(text) if any(c.isdigit() for c in t) and len(t) > 1}


class SimCrossEncoder:
    name = "sim-rerank"

    def __init__(self, index, encoder: SimEncoder | None = None):
        self.index, self.encoder = index, encoder or SimEncoder()
        self.pairs_scored = 0

    def score(self, query: str, text: str) -> float:
        self.pairs_scored += 1
        q_terms, s_terms = set(analyse(query)), set(analyse(text))
        idf = {t: self.index.idf(t) for t in q_terms}
        lexical = sum(idf[t] for t in q_terms & s_terms) / (sum(idf.values()) or 1)
        semantic = float(self.encoder.encode(query) @ self.encoder.encode(text))
        qc = concepts(query)
        coverage = len(qc & concepts(text)) / len(qc) if qc else 0.0
        qi = identifiers(query)
        ident = len(qi & identifiers(text)) / len(qi) if qi else 0.0
        polarity = 0.0
        if ASKS_EXCLUSION.search(query.lower()):
            polarity = 0.15 if EXCLUDING.search(text.lower()) else -0.15
        return (0.35 * lexical + 0.30 * semantic + 0.25 * coverage
                + 0.15 * ident + polarity)

    def rerank(self, query: str, candidates: list[str], texts: dict[str, str]):
        scores = {d: self.score(query, texts[d]) for d in candidates}
        return sorted(candidates, key=lambda d: (-scores[d], d)), scores
