"""Chapter 29 tests. Run from 04-code: python3 -m pytest -q ch29/test_ch29.py"""
from collections import Counter
from contextlab.diagnostics import Diagnostics, NOT_CONTEXT
from standin29 import run, Case
from tests29 import TESTS, GOLD
import ledger29 as L

diag = lambda c: Diagnostics.diagnose(c, run, TESTS)


def test_each_probe_family_is_found_and_fixed():
    for cls, cases in L.probes().items():
        test = next(t for t in TESTS if t.cls == cls)
        for c in cases:
            assert run(test.fix(c))
            if not run(c):
                assert diag(c).primary == cls


def test_error_analysis_of_the_82():
    rows, dets = L.ledger()
    assert len(dets) == 1000 and len(rows) == 82
    found = Counter(diag(r[4]).primary for r in rows)
    assert found == Counter({"confusion": 48, "missing_evidence": 14, "fragmentation": 13,
                             "poisoning": 5, "drift": 1, NOT_CONTEXT: 1})
    assert sum(diag(r[4]).joint for r in rows) == 6


def test_injection_fits_two_classes_and_admission_wins():
    c = L.injection()[0]
    d = diag(c)
    assert (d.primary, d.secondary) == ("confusion", ["lost_in_middle"])
    conf = next(t for t in TESTS if t.cls == "confusion")
    assert all(run(conf.fix(x)) for x in L.injection())


def test_counterfactuals_beat_first_signal_on_mixed_packets():
    cases = L.mixed()
    first = sum([t.cls for t in TESTS if t.signal(c)][0] == k for k, c in cases)
    assert all(diag(c).primary == k for k, c in cases) and first < 70


def test_chatbot_incident_needs_all_three_fixes():
    d = diag(dict((i[0], i[4]) for i in L.incidents())["I01"])
    assert d.joint and d.primary == "missing_evidence"
    assert set(d.secondary) == {"confusion", "clash"}


def test_sound_packet_is_not_a_context_failure_and_ch04_api_kept():
    assert diag(Case("ok", (GOLD,), slip=True)).primary == NOT_CONTEXT
    for name in ("rot_curve", "knee", "clash", "confusion", "distraction", "rot", "summary"):
        assert hasattr(Diagnostics, name)
