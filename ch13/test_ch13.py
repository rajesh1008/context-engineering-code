"""Contract tests for the Chapter 13 store interface.

Run from 04-code:  .venv-ch13/bin/python -m pytest ch13/test_ch13.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contextlab.vectorstore import (FlatStore, Hit, IncompleteResult, Result, Row,  # noqa: E402
                                    Scope, TenantLeak, check)

DIM = 32


def rows(n_big=1900, n_small=100, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for t, n in (("big", n_big), ("small", n_small)):
        for i in range(n):
            v = rng.normal(size=DIM).astype(np.float32)
            out.append(Row(f"{t}-{i}", t, "form", f"{t} span {i}", v / np.linalg.norm(v)))
    return out


def query(seed=1):
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def test_scope_rejects_unsafe_tenant_ids():
    with pytest.raises(ValueError):
        Scope(("big' OR '1'='1",))
    with pytest.raises(ValueError):
        Scope(())
    assert Scope(("big", "public")).where() == "tenant IN ('big', 'public')"


def test_check_catches_a_leak_and_an_incomplete_result():
    s = Scope(("small",))
    leaked = Result([Hit("big-1", 0.9, "big")], k=10, eligible=100, engine="x")
    with pytest.raises(TenantLeak):
        check(leaked, s)
    short = Result([Hit("small-1", 0.9, "small")], k=10, eligible=100, engine="x")
    assert not short.complete
    with pytest.raises(IncompleteResult):
        check(short, s, strict=True)
    # fewer hits than k is fine when fewer rows were eligible
    assert Result([Hit("small-1", 0.9, "small")], k=10, eligible=1, engine="x").complete


def test_flat_store_filters_first_and_is_complete():
    st = FlatStore()
    st.upsert(rows())
    r = check(st.search(query(), 10, Scope(("small",))), Scope(("small",)), strict=True)
    assert len(r.hits) == 10 and r.eligible == 100


@pytest.fixture(scope="module")
def lance(tmp_path_factory):
    pytest.importorskip("lancedb")
    from contextlab.vectorstore_lance import LanceStore
    path = str(tmp_path_factory.mktemp("lance"))
    st = LanceStore(path, "spans-test")
    st.upsert(rows())
    st.build_index("ivf_pq", partitions=16, sub_vectors=8)
    return path


def test_prefilter_is_complete_and_never_leaks(lance):
    from contextlab.vectorstore_lance import LanceStore
    st, s = LanceStore(lance, "spans-test", nprobes=16, refine=10), Scope(("small",))
    for seed in range(20):
        check(st.search(query(seed), 10, s), s, strict=True)


def test_postfilter_drops_rows_for_a_small_tenant(lance):
    from contextlab.vectorstore_lance import LanceStore
    st, s = LanceStore(lance, "spans-test", prefilter=False), Scope(("small",))
    short = [not st.search(query(seed), 10, s).complete for seed in range(20)]
    assert sum(short) >= 15          # 5% selectivity: the top 10 is mostly 'big'


def test_exact_routing_matches_the_flat_store(lance):
    from contextlab.vectorstore_lance import LanceStore
    flat = FlatStore()
    flat.upsert(rows())
    st, s = LanceStore(lance, "spans-test", exact_below=500), Scope(("small",))
    for seed in range(10):
        assert st.search(query(seed), 10, s).ids == flat.search(query(seed), 10, s).ids


def test_upsert_replaces_by_span_id(lance):
    from contextlab.vectorstore_lance import LanceStore
    st = LanceStore(lance, "spans-test")
    before = st.t.count_rows()
    r = rows()[:1]
    r[0].text = "edited"
    r[0].content_hash = ""
    r[0].__post_init__()
    st.upsert(r)
    assert st.t.count_rows() == before
    got = st.t.search().where("span_id = 'big-0'").to_list()[0]
    assert got["text"] == "edited" and got["content_hash"] == r[0].content_hash


def test_hybrid_search_respects_the_scope(lance):
    from lancedb.index import FTS
    from contextlab.vectorstore_lance import LanceStore
    st = LanceStore(lance, "spans-test")
    st.t.create_index("text", config=FTS(), replace=True)
    s = Scope(("small",))
    got = (st.t.search(query_type="hybrid").vector(query()).text("span")
           .where(s.where(), prefilter=True).limit(10).to_list())
    assert got and all(r["tenant"] == "small" for r in got)


def test_a_version_filter_never_mixes_embedding_versions():
    st = FlatStore()
    rs = rows()
    for r in rs[::2]:
        r.embed_version = "2"
    st.upsert(rs)
    s = Scope(("big", "small"))
    assert st.search(query(), 20, s, version="1").versions == ("1",)
    assert st.search(query(), 20, s).versions == ("1", "2")
