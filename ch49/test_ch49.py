"""Pins for Chapter 49's numbers (stand-in world; see lakeshore49.py)."""
import pytest
import lab
from contextlab.hiring import Review, cohen_kappa, score_anchored, score_vocab, spearman, top_k_overlap
from lakeshore49 import DEFECTS


@pytest.fixture(scope="module")
def R():
    for f in lab.SECTIONS:
        f()
    return lab.RES


def test_module_basics():
    assert spearman([1, 2, 3], [10, 20, 30]) == 1.0
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert top_k_overlap([3, 2, 1], [1, 2, 3], k=1) == 0
    r = Review("x", tool_names=10, claimed=2)
    assert score_vocab(r) == 22 and score_anchored(r, DEFECTS) == 0


def test_exercise(R):
    assert R["exercise"] == {"spans": 11, "defects": 8, "no_page": ["S5"]}
    assert R["found"]["D8"] == 12 and R["found"]["D6"] == 28


def test_injection_and_fix(R):
    assert R["vocab"] == {"rho_drill": .501, "rho_fluency": .923, "top5": 0}
    assert R["anch"] == {"rho_drill": .789, "rho_fluency": .337, "top5": 3}
    assert R["vocab_top"][0] == "C33" and R["anch_top"][0] == "C09"
    assert R["kappa"] == {"impression": .49, "anchored": .77}
