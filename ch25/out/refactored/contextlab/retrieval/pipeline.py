"""The retrieval ladder: one function per rung, same inputs, same outputs."""
from .corpus import admitted, build_corpus
from .dense import DenseIndex, SimEncoder
from .fusion import ranked, reciprocal_rank_fusion, weighted
from .index import InvertedIndex
from .rerank import SimCrossEncoder

RUNGS = ["keyword", "dense", "hybrid", "reranked", "graph"]


class Ladder:
    def __init__(self, corpus=None, depth: int = 100, rerank_depth: int = 30,
                 k1: float = 1.2, b: float = 0.75, alpha: float = 0.5):
        self.corpus = corpus or build_corpus()
        self.texts = {i: s.full for i, s in self.corpus.items()}
        self.index, self.dense = InvertedIndex(), DenseIndex(SimEncoder())
        for i, t in self.texts.items():
            self.index.add(i, t)
            self.dense.add(i, t)
        self.cross = SimCrossEncoder(self.index, self.dense.encoder)
        self.depth, self.rerank_depth, self.k1, self.b = depth, rerank_depth, k1, b
        self.alpha = alpha

    def _admit(self, ids, q):
        return [i for i in ids if admitted(self.corpus[i], q)]

    def keyword(self, q):
        return self._admit(ranked(self.index.bm25(q.text, self.k1, self.b))[:self.depth], q)

    def dense_rung(self, q):
        return self._admit(ranked(self.dense.search(q.text))[:self.depth], q)

    def hybrid(self, q, how="weighted", alpha=None):
        lex, den = self.index.bm25(q.text, self.k1, self.b), self.dense.search(q.text)
        a = self.alpha if alpha is None else alpha
        fused = reciprocal_rank_fusion(lex, den) if how == "rrf" else weighted(lex, den, a)
        return self._admit(ranked(fused)[:self.depth], q)

    def reranked(self, q, depth=None):
        cands = self.hybrid(q)[:depth or self.rerank_depth]
        return self.cross.rerank(q.text, cands, self.texts)[0]

    def graph(self, q, seeds: int = 3):
        """Follow 'modifies' and 'governs' links from the top spans."""
        out = list(self.reranked(q))
        for parent in list(out[:seeds]):
            for nb in self.corpus[parent].links:
                if nb in out[:5] or not admitted(self.corpus[nb], q):
                    continue
                if nb in out:
                    out.remove(nb)
                out.insert(out.index(parent) + 1, nb)
        return out

    def run(self, rung, q):
        return {"keyword": self.keyword, "dense": self.dense_rung, "hybrid": self.hybrid,
                "reranked": self.reranked, "graph": self.graph}[rung](q)
