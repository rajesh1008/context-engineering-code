"""Chapter 35 tests: the claims the chapter rests on."""
import pytest

import lakeshore35 as L
import sim35 as S
from contextlab.costs import Call, CacheLayer, Router, Route, cascade, pareto
import lab

DAY = lab.DAY


def test_thinking_is_billed_as_output_and_never_cached():
    c = Call("draft", "large", uncached=1000, cached=1000, output=100, thinking=1000)
    d = c.dollars()
    assert d["thinking"] == pytest.approx(1000 * 12.00 / 1e6)
    assert d["cached"] == pytest.approx(d["uncached"] * 0.10)


def test_baseline_and_chosen_hold_the_wrong_clause_rate():
    assert lab.base["wrong"] == 4 and lab.s9["wrong"] == 4


def test_cost_falls_about_55_percent_while_tokens_barely_move():
    assert 0.54 <= lab.fall <= 0.57
    assert abs(lab.s9["tokens"] / lab.base["tokens"] - 1) < 0.10


def test_quality_rises_with_the_cut():
    assert lab.s9["clean"] > lab.base["clean"] and lab.s9["flagged"] < lab.base["flagged"]


def test_no_true_hard_determination_reaches_the_small_tier():
    outs, _, _ = S.run(lab.CHOSEN, DAY)
    assert not [o for o in outs if o.d.band == 2 and o.route.startswith("small")]


def test_strict_router_refuses_a_renamed_feature():
    with pytest.raises(KeyError):
        lab.ROUTER.route({k: v for k, v in DAY[0].feats().items() if k != "ambiguity"})


def test_blind_router_is_cheaper_wrong_and_seen_by_the_shadow_audit():
    assert lab.s10["cost"] < lab.s9["cost"] and lab.s10["wrong"] > 50
    assert sum(o.shadow_disagree for o in lab.o10) > 0


def test_retrieval_cache_key_must_carry_the_bulletin_version():
    assert sum(o.stale for o in lab.o7u) > 0
    outs, _, _ = S.run(lab.CHOSEN, DAY)
    assert not any(o.stale for o in outs)


def test_cascade_stops_at_first_pass():
    seen = []
    steps = [(n, (lambda n=n: seen.append(n) or n)) for n in ("small", "mid", "large")]
    d, name, attempts = cascade(steps, lambda x: x == "mid")
    assert (d, name, attempts, seen) == ("mid", "mid", ["small", "mid"], ["small", "mid"])


def test_cache_layer_counts_hits():
    c = CacheLayer("t", lambda x: x)
    for k in (1, 1, 2, 1):
        c.get_or(k, lambda: 0)
    assert c.stats()["hits"] == 2 and c.stats()["misses"] == 2


def test_pareto_drops_dominated_points():
    pts = [("a", 1.0, 5), ("b", 2.0, 4), ("c", 3.0, 9), ("d", 0.5, 1)]
    assert [p[0] for p in pareto(pts)] == ["d", "a", "c"]


def test_output_matches_lab():
    text = (lab.HERE / "output.txt").read_text()
    assert f"falls {lab.fall:.1%}" in text
