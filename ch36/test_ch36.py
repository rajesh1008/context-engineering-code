import json
import os
import subprocess
import sys

import pytest

from contextlab.optimise import Gate, Playbook, Prompt, evaluate, paired_ci, transitions
from lakeshore36 import HAND_WRITTEN, faithfulness, make, program, rule_text, wrong_clause

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = Prompt(*HAND_WRITTEN)


def test_deterministic_baseline():
    held = make("H", 400, 3)
    assert evaluate(program, BASE, held, faithfulness).mean == pytest.approx(0.8675)
    assert wrong_clause(BASE, make("F", 1000, 4, misses=4)) == 4


def test_pinned_edit_raises_wrong_clause():
    full = make("F", 1000, 4, misses=4)
    assert wrong_clause(Prompt(("P1r", "P2"), BASE.rules), full) > 4


def test_transitions_and_ci():
    held = make("H", 400, 3)
    a = evaluate(program, BASE, held, faithfulness)
    b = evaluate(program, BASE.with_rule("G5"), held, faithfulness)
    t = transitions(a, b)
    assert t["net"] == t["recoveries"] - t["regressions"]
    mean, lo, hi = paired_ci(a, b)
    assert lo <= mean <= hi


def test_gate_blocks_pinned_and_narrow():
    gate = Gate(BASE.pinned_hash)
    ok, why = gate.check(Prompt(("P1r", "P2")), (0.01, 0.0, 0.02), 4, 9, ["K:x"])
    assert not ok and any("pinned" in w for w in why) and any("wrong-clause" in w for w in why)


def test_rewrite_collapses_to_generic():
    pb = Playbook({r: [1, 0] for r in ("G1", "G2", "G4", "G5")})
    pb.rewrite(240, "N4", lambda r: len(rule_text(r)))
    assert "N4" in pb.bullets and "G1" not in pb.bullets


def test_lab_reproduces_reference_output():
    root = os.path.dirname(HERE)
    out = subprocess.run([sys.executable, os.path.join(HERE, "lab.py")], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    with open(os.path.join(HERE, "output.txt")) as fh:
        assert out == fh.read()
    rec = json.load(open(os.path.join(HERE, "out", "manifest.json")))
    assert rec["pinned_hash"] == BASE.pinned_hash


def test_dspy_signature_and_metric_contract():
    dspy = pytest.importorskip("dspy")
    from dspy_module36 import Determine, metric
    lm = dspy.utils.DummyLM([{"outcome": "covered", "cited_span": "S3",
                              "rationale": "The endorsement adds water back-up cover."}])
    dspy.configure(lm=lm)
    pred = dspy.Predict(Determine)(claim="Sump pump failure, WI", packet="S3 HO 04 95")
    ex = dspy.Example(claim="", packet="", gold_span="S3", support={"S3": ["water back-up"]})
    res = metric(ex, pred)
    assert res.score == 1.0 and "S3" in res.feedback
