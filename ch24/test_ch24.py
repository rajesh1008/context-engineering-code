"""Tests for contextlab.freshness and the Chapter 24 lab."""
from datetime import datetime, timedelta

import lab
import lakeshore24 as L
from contextlab import freshness as F

T = datetime(2026, 9, 15, 10, 0)


def test_applier_is_idempotent_and_ordered():
    ap = F.Applier()
    a, b = F.Change("x", "k", 1, T, "v1"), F.Change("x", "k", 2, T, "v2")
    assert ap.apply(b) and not ap.apply(a) and not ap.apply(b)
    assert ap.state[("x", "k")].version == "v2"
    assert (ap.duplicates, ap.out_of_order) == (1, 1)


def test_next_nightly():
    assert F.next_nightly(T) == datetime(2026, 9, 16, 3, 40)
    assert F.next_nightly(datetime(2026, 9, 15, 1, 0)) == datetime(2026, 9, 15, 3, 40)


def test_tag_invalidation_misses_new_documents_epoch_does_not():
    for mode, expect_hit in (("tags", True), ("epoch", False)):
        c = F.ScopedCache(timedelta(hours=24), mode)
        c.put("HL", ("MN", "HL"), ["old"], T, {"BUL-MN-HL-1@1"})
        c.on_change(F.Change("docstore", "BUL-MN-HL-2@1", 1, T, "1", (("MN", "HL"),)))
        assert (c.get("HL", ("MN", "HL"), T) is not None) == expect_hit


def test_staleness_incident():
    req = [("B1", T - timedelta(hours=1)), ("B2", T + timedelta(hours=1))]
    assert F.staleness_incident(T, req, {"B1": T + timedelta(minutes=5)}) == ["B1"]
    assert F.staleness_incident(T, req, {"B1": T - timedelta(minutes=5)}) == []


def test_lab_headlines():
    bs = L.bulletins()
    v = lab.visibility(bs)
    _, res = lab.exposure(bs, v)
    assert res["A nightly + graph full builds"]["iowa"] > 0
    assert res["D C + watermark routing"]["graph"] == 0
    assert lab.cache_day("tags")[:2] == (0, 11) and lab.cache_day("epoch")[:2] == (0, 0)
    assert lab.cdc_check()[1]["keyed"][:2] == (0, 0)


def test_feed_is_reconciled_against_the_log():
    n, bad, _, case = lab.feed_check()
    assert n == 160 and case["wrong_before"] is True
    assert all(a != b for a, b in bad)
