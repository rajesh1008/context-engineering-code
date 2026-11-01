"""Tests for Chapter 45: contextlab.operations v9.8 and the simulated year."""
from datetime import date

import lakeshore45 as L
from contextlab.operations import (VERSION, Artefact, Calendar, Instruction, Inventory, Job,
                                   TunedParam, headroom, mix_adjusted, prune, recalibrate_floor,
                                   replay_compare, stale_params)


def test_version():
    assert VERSION == "9.8"


def test_loop_gaps():
    j = Job("x", "daily", "a", stop="s", state="st", verifier="a", budget="b", human="h")
    assert j.gaps() == ["verifier is the assembler"]
    assert set(Job("y", "weekly", "a").gaps()) == {"stop", "state", "verifier", "budget", "human"}
    assert Calendar([j]).runs_per_year() == 360


def test_stale_params():
    p = TunedParam("K", 40, {"corpus_pages": 40000, "embed_model": "m1"})
    assert stale_params([p], {"corpus_pages": 49000, "embed_model": "m1"}) == []
    assert stale_params([p], {"corpus_pages": 51000})[0][0] == "K"
    assert "m1 -> m2" in stale_params([p], {"embed_model": "m2"})[0][1]


def test_headroom():
    assert headroom([1, 2, 31, 40, 41], 40) == 0.5
    assert headroom([], 40) == 0.0


def test_inventory_plan_and_left_behind():
    inv = Inventory([Artefact("idx", "index", "m", "re-embed", order=1),
                     Artefact("floor", "threshold", "m", "recalibrate", order=6),
                     Artefact("other", "index", "n", "x")])
    assert [s.split(" ")[0] for s in inv.plan("m", "m2")] == ["idx", "floor"]
    assert inv.left_behind("m", {"idx"}) == ["floor"]


def test_replay_compare_blocks_and_passes():
    old = [{"id": i, "s": "a", "w": False} for i in range(10)]
    new = [{"id": i, "s": "a", "w": i < 2} for i in range(10)]
    assert not replay_compare(old, new, "s", {"w": 0.1})["passed"]
    assert replay_compare(old, old, "s", {"w": 0.0})["passed"]


def test_recalibrate_floor_keeps_share():
    old = [i / 100 for i in range(100)]
    new = [x - 0.1 for x in old]
    f = recalibrate_floor(old, new, 0.30)
    assert abs(f - 0.20) < 0.011


def test_mix_adjusted():
    assert mix_adjusted({"a": 0.1, "b": 0.3}, {"a": 1, "b": 1}) == 0.2


def test_prune_keeps_undocumented():
    lines = [Instruction("1", "t", "Nov", "why", "T1"), Instruction("2", "t", "Nov"),
             Instruction("3", "t", "Nov", "why", "T3")]
    kept, removed, undoc = prune(lines, lambda ln: ln.id == "3")
    assert removed == ["1"] and undoc == 1 and [k.id for k in kept] == ["2", "3"]


def test_year_is_deterministic_and_doubles():
    assert L.pages(date(2027, 10, 15)) / L.pages(date(2026, 11, 15)) > 2.0
    a = [L.decide(c, L.AS_RUN) for c in L.claims(8, 500)]
    b = [L.decide(c, L.AS_RUN) for c in L.claims(8, 500)]
    assert a == b


def test_operated_beats_run_in_month_nine():
    cl = L.claims(8, 5000, "t")
    w = {r.name: sum(L.decide(c, r)["outcome"] == "wrong" for c in cl) for r in (L.AS_RUN, L.AS_OPERATED)}
    assert w["as operated"] < w["as run"] / 2


def test_new_floor_and_memory_fix():
    assert 0.5 < L.NEW_FLOOR < 0.6
    assert L.AS_OPERATED.memory_ok(date(2027, 7, 1)) and not L.AS_RUN.memory_ok(date(2027, 7, 1))


def test_cohort_jitter_removes_backlog():
    assert max(L.revalidation_backlog(True)[0].values()) == 0
    assert max(L.revalidation_backlog(False)[0].values()) > 10000


def test_pending_flag_zero_superseded():
    rs = [L.decide(c, L.AS_OPERATED, L.windows(L.AS_OPERATED)) for c in L.claims(4, 3000)]
    assert not any(r["superseded"] and r["outcome"] == "wrong" for r in rs)
