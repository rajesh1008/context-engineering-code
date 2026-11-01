"""Tests for Chapter 51. Run: python3 -m pytest -q (from 04-code/ch51)."""
import lakeshore51 as L
from contextlab.finish import prioritise, run_checklist, drill, compare, targets, EdgeCase


def test_priority_differs_from_frequency():
    p = [c.key for c in prioritise(L.BACKLOG, "priority")]
    f = [c.key for c in prioritise(L.BACKLOG, "frequency")]
    assert p[0] == "E07" and f[0] == "E13"
    assert len(set(p[:5]) & set(f[:5])) == 1


def test_fully_caught_case_has_no_priority():
    assert EdgeCase("X", "x", 50, 5, 1.0, 1).priority() == 0


def test_checklist_before_and_after():
    _, before = run_checklist(L.CHECKS, L.CONFIG_BEFORE)
    _, after = run_checklist(L.CHECKS, L.CONFIG_AFTER)
    assert [c.key for c in before] == ["H12", "H13", "H16", "H17", "H18", "H24"]
    assert after == []


def test_failure_injection_missing_evidence_fails_closed():
    cfg = dict(L.CONFIG_AFTER)
    del cfg["restore_tested"]
    _, failed = run_checklist(L.CHECKS, cfg)
    assert [c.key for c in failed] == ["H18"]


def test_runbook_beats_improvising():
    for rb in L.RUNBOOKS:
        d = L.DISCOVERY[rb.key]
        assert drill(rb, d, True) < drill(rb, d, False)


def test_exam_drill_finds_one_gap():
    s = L.exam_sample()
    assert sum(x["verifies"] for x in s) == 30
    assert sum(not x["reason_code"] for x in s) == 1


def test_gold_misses_precedence_change_but_probe_catches_it():
    gold = L.gold_items()
    prod, cand = L.run_model(gold, "production"), L.run_model(gold, "candidate")
    rows = compare(L.diagnostics(adapter=True), prod, cand)
    assert not any(r["alarm"] for r in rows)
    probes = L.probe_items()
    assert sum(r["wrong"] for r in L.run_model(probes, "candidate")) == 11
    assert sum(r["wrong"] for r in L.run_model(probes, "production")) == 0


def test_format_change_is_caught():
    gold = L.gold_items()
    rows = compare(L.diagnostics(adapter=False), L.run_model(gold, "production"),
                   L.run_model(gold, "candidate"))
    alarms = {r["diagnostic"] for r in rows if r["alarm"]}
    assert "citation parse rate, %" in alarms


def test_targets():
    t = targets({"lookup_minutes": 1.29, "wrong_clause_pct": 0.3, "traceable_pct": 96.67}, L.GOALS)
    assert t["wrong_clause_pct"]["met"] and not t["traceable_pct"]["met"]
