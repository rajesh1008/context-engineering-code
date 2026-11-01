from dataclasses import replace

import ledger20 as L
import rebuild as R
from contextlab.embeddings import EmbeddingRegistry
from contextlab.retrieval.corpus import build_corpus
from contextlab.retrieval.queries import build_queries

CORPUS = build_corpus()
QUERIES = build_queries(CORPUS)


def score(cfg, key):
    st = R.Stack(cfg, CORPUS)
    return sum(st.packet(q)[key] for q in QUERIES)


def test_rebuild_beats_baseline_on_both_measures():
    assert score(R.BASELINE, "gov") == 850 and score(R.REBUILT, "gov") == 967
    assert score(R.REBUILT, "gov1") == 880


def test_fixed_windows_lose_fifteen_spans():
    st = R.Stack(R.BASELINE, CORPUS)
    held = {s for v in st.delivers.values() for s in v}
    assert len(set(CORPUS) - held) == 15


def test_authority_filter_removes_every_leak():
    assert score(R.REBUILT, "leaky") == 0
    assert score(replace(R.REBUILT, authority=False), "leaky") > 500


def test_precedence_without_authority_hurts():
    assert score(replace(R.REBUILT, authority=False), "gov1") < score(R.BASELINE, "gov1")


def test_upgrade_without_reindex_collapses_dense_and_guard_refuses():
    dense = replace(R.REBUILT, retriever="dense", rerank=False, query_model="sim-embed@2")
    assert score(dense, "gov") < 150
    reg = EmbeddingRegistry(R.PROBES)
    reg.register(R.Pinned("sim-embed@1"), "self-hosted")
    reg.bind_index("spans", "sim-embed@1")
    assert reg.may_query("spans", R.Pinned("sim-embed@1"))
    assert not reg.may_query("spans", R.Pinned("sim-embed@1", R.SimEncoderV2()))


def test_ledger_rebuild_fixes_only_the_index_cause():
    left = [r for r in L.RESIDUAL if not L.fixed_by_rebuild(r, L.GRAPH_BUILD_REBUILT)]
    assert len(left) == 7 and {r.layer for r in left} == {"record", "document", "feed"}
    assert not any(L.fixed_by_rebuild(r, L.GRAPH_BUILD_BEFORE) for r in L.RESIDUAL)
