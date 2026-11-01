"""Pins for Chapter 41's numbers (stand-in world; see lakeshore41.py)."""
import pytest
import lab
from contextlab.erroranalysis import cohen_kappa, saturation, tvd, reweight, wilson_upper


@pytest.fixture(scope="module")
def R():
    for f in lab.SECTIONS:
        f()
    return lab.RES


def test_module_basics():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert saturation(["a", "a", "b"] + ["a"] * 5, run=3) == ([1, 1, 2, 2, 2, 2, 2, 2], 6)
    assert tvd({"x": 1.0}, {"x": .5, "y": .5}) == .5
    assert reweight({"x": .9}, {"x": .5, "y": .5}) == (.9, .5)
    assert round(wilson_upper(0, 100), 4) == 0.037


def test_score_up_production_down(R):
    assert R["gold"] == {"before": 94.33, "released": 96.35}
    assert R["override_aug"] == 7.97 and R["override_oct"] == 8.67
    d = R["drift"]
    assert (d["d_raw"], d["d_rw"], d["d_prod"]) == (2.02, 2.09, -1.11)
    assert d["seg_prod"]["released"]["multi_peril"] == 45.71
    assert d["months"]["Oct"] == {"tvd": .128, "unsupported": .09}


def test_coding(R):
    assert R["open"]["pool"] == 832 and R["open"]["acceptable"] == 23
    assert R["axial"]["v1"]["None"] == 58 and R["axial"]["v2"]["second_peril"] == 55
    assert R["axial"]["moves"]["missing_evidence->second_peril"] == 20
    assert R["pareto"][0] == ["second_peril", 55, 165]
    a = R["agreement"]
    assert (a["before"]["kappa"], a["after"]["kappa"]) == (0.46, 0.89)


def test_saturation_and_fix(R):
    s = R["saturation"]
    assert s["gold-set failures"]["second_peril_at"] is None and s["gold-set failures"]["final"] == 6
    assert s["July overrides"]["second_peril_at"] == 6 and s["October overrides"]["stop"] == 51
    f = R["fix"]
    assert f["released"]["multi_wrong"] == 475 and f["per_peril"]["multi_wrong"] == 61
    assert f["per_peril"]["multi_upper"] == 8.85 and f["worse"] == []
    assert f["extended"] == {"before": 93.18, "released": 92.13, "per_peril": 96.24}
    assert R["ship"]["exit_pass"] and R["ship"]["alarm_month"] == "Jun"


def test_questions_per_peril():
    from questions41 import questions_for, complete
    c = dict(id="C1", perils=["WIND", "WATER"], form="HO-3", state="WI")
    qs = questions_for(c)
    assert [q["peril"] for q in qs] == ["WIND", "WATER"] and qs[0]["budget"] == 800
    assert complete(c, qs[:1]) == (False, ["WATER"]) and complete(c, qs) == (True, [])


def test_stop_rule_misses_rare_known_classes(R):
    assert R["saturation"]["October overrides"]["after_stop"] == [[52, "superseded_edition"],
                                                                 [118, "precedence_missed"]] or \
        R["saturation"]["October overrides"]["after_stop"] == [(52, "superseded_edition"), (118, "precedence_missed")]
