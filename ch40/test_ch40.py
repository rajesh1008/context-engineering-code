"""Chapter 40 tests: the claims the chapter rests on."""
import pathlib
import subprocess
import sys

import lakeshore40 as L
from contextlab.pyramid import Check, Suite, ladder, lint
import lab

HERE = pathlib.Path(__file__).resolve().parent


def test_clean_system_passes_every_check():
    assert all(r.passed for r in L.suite.run(L.System()))


def test_layer_one_is_deterministic_free_and_fast():
    assert lint(L.suite.checks) == []
    assert lab.LAYERCOST[1]["dollars"] == 0 and lab.LAYERCOST[1]["seconds"] < 120


def test_misplaced_judge_is_flagged_by_lint():
    rules = {r for n, r in lab.LINT_BEFORE if n == lab.SMOKE.name}
    assert rules == {"L1 must be deterministic", "L1 must not spend on models"}


def test_judge_smoke_flakes_and_retries_hide_the_regression():
    once, retried = lab.flake["clean"]
    assert 0.30 <= once <= 0.38 and retried < 0.06
    assert lab.flake["F4"] == lab.flake["clean"]          # blind to the bug it was written for
    assert lab.flake["F9"][1] < 0.35


def test_schedule_test_catches_f4_at_layer_one():
    assert lab.FIX_F4 == (True, False)


def test_every_fault_is_caught_somewhere_and_f12_only_in_production():
    for f, row in lab.MATRIX.items():
        assert max(row.values()) >= 0.75, f
    assert [n for n, v in lab.MATRIX["F12"].items() if v] == [5]
    assert lab.MATRIX["F1"][3] == 0.0                     # a faithful quote of the wrong edition


def test_end_to_end_misses_packet_faults():
    assert all(lab.MATRIX[f][4] == 0 for f in ("F2", "F3", "F4", "F6", "F9"))


def test_2025_pair_goes_red_at_layer_one_for_nothing():
    a = lab.FIRST["F1"][1]
    assert a["rung"] == "every commit" and a["dollars"] == 0
    assert (1, "endorsement on the declarations survives dedupe") in lab.PAIR_FAILS


def test_ladder_stops_f9_at_the_candidate_rung():
    assert lab.F9RUN["rung"] == 2 and lab.CLEAN["rung"] is None


def test_release_costs_about_the_same_but_spends_it_differently():
    assert abs(sum(lab.REL_USD.values()) - sum(lab.BEFORE_REL.values())) < 5
    assert lab.REL_USD["every commit"] == 0 and lab.BEFORE_REL["every commit"] > 150


def test_ladder_stops_at_first_failing_rung():
    s = Suite()
    s.add("a", 1, "evidence", "x")(lambda sys_, seed: True)
    s.add("b", 2, "evidence", "x", sample=1, dollars=2.0)(lambda sys_, seed: False)
    s.add("c", 3, "evidence", "x", sample=1, dollars=9.0)(lambda sys_, seed: False)
    r = ladder(s, None, [(1,), (2,), (3,)])
    assert r["rung"] == 1 and r["failed"] == ["b"] and r["dollars"] == 2.0


def test_output_matches_lab():
    out = subprocess.run([sys.executable, str(HERE / "lab.py")], capture_output=True, text=True,
                         cwd=HERE.parent).stdout
    assert out == (HERE / "output.txt").read_text()
