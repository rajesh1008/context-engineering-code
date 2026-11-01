"""Pins for Chapter 30's numbers (stand-in reader; see lakeshore30.py)."""
import pytest
import lab


@pytest.fixture(scope="module")
def R():
    for f in (lab.s1_control, lab.s2_scripts, lab.s3_signals, lab.s4_side_by_side,
              lab.s5_dose, lab.s6_clash, lab.s7_fragmentation, lab.s8_flips,
              lab.s9_regression):
        f()
    return lab.RES


def test_control_and_scripts(R):
    assert R["control"] == {"naive": 1, "current": 1}
    s = R["scripts"]
    assert {k: v["naive"] for k, v in s.items()} == {
        "overflow": 1000, "missing_evidence": 1000, "drift": 1000, "poisoning": 631,
        "confusion": 501, "clash": 457, "fragmentation": 1000, "distraction": 252,
        "lost_in_middle": 13, "rot": 86}
    assert all(v["current"] <= 1 for k, v in s.items() if k != "distraction")
    assert s["distraction"]["current"] == 252 and s["missing_evidence"]["handoff"] == 1000
    assert s["lost_in_middle"]["shape"] == {"every seed": 0, "some seeds": 11, "no seed": 39}


def test_diagnosis_recovers_the_script(R):
    d = R["diagnosis"]
    for cls in ("poisoning", "confusion", "lost_in_middle"):
        assert set(d[cls]) == {cls}
    assert d["distraction"] == {"distraction": 153, "not_context": 98, "unresolved": 1}
    assert R["signals"]["distraction"]["off-topic share > 50%"] == 0


def test_side_by_side_dose_clash_fragmentation(R):
    assert R["side_by_side"]["confusion"]["labels"] == 1
    assert R["side_by_side"]["distraction"]["labels"] == 252
    assert R["side_by_side"]["distraction"]["dedup (keyed)"] == 126
    on = R["dose"]["on"]
    assert all(on[k] < on[k + 1] for k in range(8)) and max(R["dose"]["off"].values()) == 1
    assert R["clash"]["skill text newer"]["naive"] == 630
    assert R["fragmentation"]["second half not retrieved"]["before"] == 1000
    assert R["fragmentation"]["second half not retrieved"]["handoff"] == 1000


def test_flips_and_suite(R):
    assert R["flips"]["re-embedded, naive, all scripts"] == (5940, 5927, 227, 240)
    su = R["suite"]
    assert all(v[2] and v[3] for v in su["current"].values())
    bad = [k for k, v in su["current + dedup"].items() if not v[3]]
    assert bad == ["confusion"]
    assert all(v[3] for v in su["keyed dedup, guard tightened"].values())
