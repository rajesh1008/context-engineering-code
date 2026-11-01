"""Chapter 48 tests: the claims the chapter rests on."""
import copy
import pathlib
import subprocess
import sys

import lakeshore48 as L
from contextlab.configlint import CATALOGUE, lint, overlap, resolve

HERE = pathlib.Path(__file__).resolve().parent


def rules(findings):
    return {f.rule for f in findings}


def test_catalogue_has_twelve_entries():
    assert len(CATALOGUE) == 12


def test_clean_determination_passes_lint_and_resolve():
    assert lint(L.DETERMINATION) == []
    assert resolve(L.DETERMINATION, L.world(0)) == []


def test_every_injected_antipattern_is_caught_by_its_own_rule():
    for rule in CATALOGUE:
        assert rule in rules(lint(L.with_antipattern(L.DETERMINATION, rule))), rule


def test_audit_warnings_are_the_two_the_chapter_reports():
    assert [f.rule for f in lint(L.ASSISTANT)] == ["AP09"]
    assert [f.rule for f in lint(L.SUBRO)] == ["AP12"]


def test_lookalike_descriptions_score_above_threshold():
    t = {x["name"]: x["description"] for x in L.SUBRO["tools"]}
    assert overlap(t["claim_notes"], t["claim_diary"]) > 0.6
    assert overlap(t["policy_record"], t["claim_facts"]) < 0.2


def test_lint_stays_green_while_the_world_moves():
    for m in range(6):
        assert not [f for c in L.CONFIGS for f in lint(c) if f.severity == "error"]


def test_resolve_finds_each_change_in_the_month_it_lands():
    assert rules(resolve(L.SUBRO, L.world(2))) == {"AP05"}
    assert rules(resolve(L.DETERMINATION, L.world(3))) == {"AP10"}
    assert rules(resolve(L.DETERMINATION, L.world(4))) == {"AP02", "AP10"}


def test_rot_moves_outcomes_without_a_fix():
    assert L.month_outcomes(L.DETERMINATION, L.SUBRO, 1) == (3, 0)
    wrong, broken = L.month_outcomes(L.DETERMINATION, L.SUBRO, 5)
    assert wrong > 30 and broken > 200


def test_fixing_on_resolve_keeps_the_scoreboard():
    det, sub = copy.deepcopy(L.DETERMINATION), copy.deepcopy(L.SUBRO)
    for m in range(6):
        if resolve(det, L.world(m)) + resolve(sub, L.world(m)):
            det, sub = L.fixed(det, m), L.fixed(sub, m)
        assert L.month_outcomes(det, sub, m) == (3, 0)


def test_pin_on_renamed_rule_pins_nothing():
    sub = copy.deepcopy(L.SUBRO)
    sub["compaction"]["pins"] = ["SUBRO-HOLD-01"]
    assert lint(sub) == [] or rules(lint(sub)) == {"AP12"}
    assert rules(resolve(sub, L.world(2))) == {"AP05"}


def test_lab_output_matches_committed_output():
    out = subprocess.run([sys.executable, str(HERE / "lab.py")], capture_output=True,
                         text=True, check=True).stdout
    assert out == (HERE / "output.txt").read_text()
