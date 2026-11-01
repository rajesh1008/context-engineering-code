"""Chapter 8: the numbers the chapter prints, and the mechanisms behind them."""
from lab import R09, R10, R11_BOOK, R11_FIXED, R8_NO_START, evaluate
from standin import CORPUS
from contextlab.rulebook import Rulebook, check, probes
from gate import gate


def test_eval_coverage():
    ev = CORPUS.eval
    assert len(ev) == 200
    assert sum(d.topic in R10.topics() for d in ev) == 119


def test_review_finds_r11_and_only_r11():
    res = check(R09, CORPUS)
    assert sum(r.checked for r in res) == 296
    bad = [r for r in res if r.conflicts]
    assert [r.rule.rule_id for r in bad] == ["R11"]
    assert [c.det.determination_id for c in bad[0].conflicts] == [
        "DET-0030", "DET-0252", "DET-0326", "DET-0393", "DET-0415"]
    assert not any(r.conflicts for r in check(R10, CORPUS))


def test_probes_and_rates():
    assert len(probes(R10, CORPUS)) == 107
    want = {"sim-large": (7, 5, 105), "sim-mid": (9, 5, 103), "sim-small": (9, 6, 102)}
    for m, (none, r10, adh) in want.items():
        assert evaluate(m, None)["wrong"] == none
        assert evaluate(m, R10)["wrong"] == r10
        assert evaluate(m, None)["probes_ok"] == adh


def test_r11_yaml_is_the_fixed_rule():
    assert R11_FIXED.rule_id == "R11" and len(R11_FIXED.rows) == 1
    assert R11_FIXED.rows[0].governs == "WI-OCI-2024-11"


def test_dates_matter_for_r8():
    assert check(Rulebook("x", (R8_NO_START,)), CORPUS)[0].agree == 18


def test_gate():
    assert gate(R09, R10)[0] is True
    assert gate(R10, R11_BOOK)[0] is False


def test_fingerprint_ignores_version():
    assert R10.fingerprint() == R10.with_rule(R10.rules[0], "r9.9", "x").fingerprint()
    assert R09.fingerprint() != R10.fingerprint()
