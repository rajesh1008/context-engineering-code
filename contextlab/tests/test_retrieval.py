"""Tests for contextlab.retrieval (Chapter 12)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from contextlab.retrieval import (HNSW, InvertedIndex, SimEncoder, analyse,  # noqa: E402
                                  hit_at_k, mrr, ndcg_at_k, recall_at_k, rrf, weighted)
from contextlab.retrieval.pipeline import Ladder  # noqa: E402
from contextlab.retrieval.queries import TWINS, build_queries  # noqa: E402


def small_index():
    ix = InvertedIndex()
    ix.add("a", "frozen pipe frozen pipe heat")
    ix.add("b", "wind damage to the roof")
    ix.add("c", "pipe " + "word " * 40)
    return ix


def test_stop_list_keeps_negation():
    assert "not" in analyse("this is not covered")


def test_idf_formula():
    ix = small_index()
    assert math.isclose(ix.idf("pipe"), math.log(1 + (3 - 2 + 0.5) / (2 + 0.5)))


def test_bm25_saturates_and_normalises_length():
    ix = small_index()
    s = ix.bm25("pipe")
    assert s["a"] > s["c"]                         # shorter span wins at similar tf
    assert ix.bm25("pipe", b=0.0)["c"] > 0


def test_rrf_uses_ranks_only():
    f = rrf({"x": 100.0, "y": 1.0}, {"y": 0.9, "x": 0.1})
    assert math.isclose(f["x"], f["y"])


def test_weighted_alpha_extremes():
    lex, den = {"x": 5.0, "y": 1.0}, {"x": 0.1, "y": 0.9}
    assert max(weighted(lex, den, 0.0), key=weighted(lex, den, 0.0).get) == "x"
    assert max(weighted(lex, den, 1.0), key=weighted(lex, den, 1.0).get) == "y"


def test_metrics():
    gold = {"g": 2, "s": 1}
    r = ["x", "g", "s"]
    assert hit_at_k(r, gold, 1) == 0 and hit_at_k(r, gold, 2) == 1
    assert recall_at_k(r, gold, 3) == 1 and mrr(r, gold) == 0.5
    assert 0 < ndcg_at_k(r, gold, 3) < 1


def test_encoder_nearly_ignores_negation():
    e = SimEncoder()
    (neg, _), (pos, _) = TWINS["neg"], TWINS["pos"]
    assert float(e.encode(neg) @ e.encode(pos)) > 0.98


def test_hnsw_recall():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((800, 16))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    h = HNSW(16, M=12)
    for v in X:
        h.add(v)
    q = X[5] + 0.01
    q /= np.linalg.norm(q)
    assert 5 in h.search(q, 5, ef=40)


def test_ladder_climbs_on_recall_at_5():
    L = Ladder()
    Q = build_queries(L.corpus)
    rec = {r: sum(recall_at_k(L.run(r, q), q.gold, 5) for q in Q) for r in
           ("keyword", "hybrid", "graph")}
    assert rec["keyword"] < rec["hybrid"] <= rec["graph"]


def test_admission_never_leaks_other_states():
    L = Ladder()
    for q in build_queries(L.corpus)[:100]:
        for d in L.graph(q)[:10]:
            assert L.corpus[d].state in ("", q.state)
