import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import lakeshore39 as L  # noqa: E402
from contextlab.evaluator import (Evaluator, GoldSet, attribute, cohen_kappa,  # noqa: E402
                                  corrected_rate, mrr, ndcg_at_k, recall_at_k, wilson)


def test_metrics_by_hand():
    ranked = ["a", "g", "s", "b", "c"]
    grades = {"g": 3, "s": 2, "x": 2, "a": 1, "b": 1}
    assert abs(recall_at_k(ranked, {"g", "s", "x"}, 5) - 2 / 3) < 1e-9
    assert mrr(ranked, {"g"}) == 0.5
    assert 0.6 < ndcg_at_k(ranked, grades, 5) < 0.8


def test_kappa_known_value():
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert abs(cohen_kappa(a, b) - 0.4) < 1e-9


def test_correction_inverts_a_known_judge():
    true, tpr, tnr = 0.88, 0.93, 0.90
    observed = true * tpr + (1 - true) * (1 - tnr)
    assert abs(corrected_rate(observed, tpr, tnr) - true) < 1e-9


def test_attribution_orders_stages():
    base = {"governing": "g", "rank": 3, "in_packet": True, "cited": "g", "supported": True}
    assert attribute(base) == "ok"
    assert attribute({**base, "rank": 14}) == "retrieval"
    assert attribute({**base, "in_packet": False}) == "assembly"
    assert attribute({**base, "cited": "look"}) == "grounding: wrong citation"
    assert attribute({**base, "supported": False}) == "grounding: unsupported"
    assert attribute({**base, "governing": None, "cited": None}) == "correct referral"


def test_baseline_matches_scoreboard():
    claims = L.make_claims()
    _, recs = L.pipeline(claims, "rerank")
    e = Evaluator(GoldSet("t", (), ())).run(recs)
    assert e["wrong_clause"] == 4 and abs(e["faithfulness"] - 0.880) < 1e-9


def test_dedupe_hides_behind_wrong_clause():
    claims = L.make_claims()
    _, base = L.pipeline(claims, "rerank")
    _, dd = L.pipeline(claims, "rerank", dedupe_by_form=True)
    b, d = Evaluator(GoldSet("t", (), ())).run(base), Evaluator(GoldSet("t", (), ())).run(dd)
    assert d["wrong_clause"] <= b["wrong_clause"]
    assert d["packet_sufficiency"] < b["packet_sufficiency"] - 0.03


def test_wilson_interval_for_rare_events():
    lo, hi = wilson(4, 1000)
    assert lo < 0.003 and hi > 0.009


def test_lab_reproduces_reference_output():
    out = subprocess.run([sys.executable, os.path.join(HERE, "lab.py")], capture_output=True,
                         text=True, check=True, cwd=os.path.dirname(HERE)).stdout
    with open(os.path.join(HERE, "output.txt")) as f:
        assert out == f.read()
