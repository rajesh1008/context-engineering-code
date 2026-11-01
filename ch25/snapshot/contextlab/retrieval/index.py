"""An inverted index with TF-IDF and BM25 scoring, from scratch."""
import math
from collections import Counter, defaultdict

from .text import analyse


class InvertedIndex:
    def __init__(self, analyser=analyse):
        self.analyser = analyser
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.length: dict[str, int] = {}

    def add(self, doc_id: str, text: str) -> None:
        terms = self.analyser(text)
        self.length[doc_id] = len(terms)
        for term, tf in Counter(terms).items():
            self.postings[term][doc_id] = tf

    @property
    def n(self) -> int:
        return len(self.length)

    @property
    def avgdl(self) -> float:
        return sum(self.length.values()) / self.n

    def df(self, term: str) -> int:
        return len(self.postings.get(term, {}))

    def idf(self, term: str) -> float:
        """BM25's idf, floored at zero, as Lucene's variant does."""
        df = self.df(term)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def bm25(self, query: str, k1: float = 1.2, b: float = 0.75,
             allowed=None) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        avgdl = self.avgdl
        for term in set(self.analyser(query)):
            idf = self.idf(term)
            for doc, tf in self.postings.get(term, {}).items():
                if allowed is not None and doc not in allowed:
                    continue
                norm = k1 * (1 - b + b * self.length[doc] / avgdl)
                scores[doc] += idf * tf * (k1 + 1) / (tf + norm)
        return dict(scores)

    def tfidf(self, query: str, allowed=None) -> dict[str, float]:
        """Raw tf times log idf, no length normalisation: the textbook trap."""
        scores: dict[str, float] = defaultdict(float)
        for term in set(self.analyser(query)):
            df = self.df(term)
            if not df:
                continue
            idf = math.log(self.n / df)
            for doc, tf in self.postings[term].items():
                if allowed is None or doc in allowed:
                    scores[doc] += tf * idf
        return dict(scores)


def top(scores: dict[str, float], k: int) -> list[str]:
    """Ranked ids, ties broken by id so every run is identical."""
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]
