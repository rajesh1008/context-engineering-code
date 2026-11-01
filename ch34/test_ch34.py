import json
import os
from contextlab.caching import (Segment, PrefixCache, Prices, cache_layout,
                                canonical_json, lint_packet, ResultCache, hit_rate)
from lakeshore34 import Builder, run_day, traffic

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def segs(*names):
    return [Segment(n, n, 600) for n in names]


def test_exact_prefix_hit_and_divergence():
    c = PrefixCache(min_tokens=1024)
    c.call(0, segs("a", "b", "c"), 10)
    r = c.call(1, segs("a", "b", "x"), 10)
    assert r.cached == 1200 and r.hit_boundary == "b"
    assert c.call(2, segs("z", "b", "c"), 10).cached == 0


def test_minimum_length_and_ttl():
    c = PrefixCache(min_tokens=1024, ttl_s=300)
    c.call(0, segs("a"), 10)
    assert c.call(1, segs("a"), 10).cached == 0           # 600 < 1,024
    c.call(0, segs("a", "b"), 10)
    assert c.call(299, segs("a", "b"), 10).cached == 1200
    assert c.call(900, segs("a", "b"), 10).cached == 0    # expired


def test_write_premium_billing():
    p = Prices(input=1e6 / 1e6, output=0, read=0.1, write=1.25)
    c = PrefixCache(prices=p, explicit=True)
    s = [Segment("a", "a", 2000, breakpoint=True), Segment("q", "q", 100, "call")]
    first, second = c.call(0, s, 0), c.call(10, s, 0)
    assert first.written == 2000 and abs(first.cost - (100 + 2500) / 1e6) < 1e-12
    assert second.cached == 2000 and abs(second.cost - (100 + 200) / 1e6) < 1e-12


def test_layout_and_serialisation():
    mixed = [Segment("q", "q", 1, "call"), Segment("s", "s", 1, "route"),
             Segment("i", "i", 1)]
    assert [s.name for s in cache_layout(mixed)] == ["i", "s", "q"]
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_lint_flags_timestamp_and_memo():
    s = [Segment("instructions", "rules now 10:04:31", 2000)]
    rules = {r for r, _ in lint_packet(s, memo_keys={"m": "state"})}
    assert {"C2-volatile-value-in-prefix", "C5-memo-without-watermark"} <= rules


def test_result_cache_key_moves_with_watermark():
    k1 = ResultCache.key("C1", "Determine", "s", "wm-1", "fp", "m")
    assert k1 == ResultCache.key("C1", " determine ", "s", "wm-1", "fp", "m")
    assert k1 != ResultCache.key("C1", "Determine", "s", "wm-2", "fp", "m")


def test_day_replay_is_deterministic_and_fix_helps():
    a = run_day(Builder(()), n=400)
    b = run_day(Builder(()), n=400)
    fixed = run_day(Builder(("stamp_end", "canonical", "reorder")), n=400)
    assert a["cost_per_det"] == b["cost_per_det"]
    assert hit_rate(a["results"]) < 0.25 < 0.6 < fixed["hit_rate"]
    assert fixed["cost_per_det"] < a["cost_per_det"]


def test_lab_outputs_match_chapter():
    f = json.load(open(os.path.join(OUT, "fixes.json")))
    assert abs(f["steps"][0]["hit_rate"] - 0.126) < 0.001
    assert 0.44 < f["fall"] < 0.47
    fr = json.load(open(os.path.join(OUT, "freshness.json")))
    assert fr["v7.5 as first built"]["alert_fired"]
    assert not fr["memo keyed on watermark"]["alert_fired"]
    assert fr["memo keyed on watermark"]["stale_wi_determinations"] == 0
