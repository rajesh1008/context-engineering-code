"""The chapter's numbers. Run with: python3 -m pytest -q"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from standin import CORPUS, PipelineConfig, faithfulness, lint, run_set, unsupported, wrong  # noqa: E402
from contextlab.instructions import V10, V11, V12, V20, V21, V22, V23, V23B  # noqa: E402

import lab  # noqa: E402

EVAL = CORPUS.eval


def runs(ins, model, dets=EVAL):
    return run_set(PipelineConfig(model=model, instructions=ins), dets)


def test_versions_large():
    assert [wrong(runs(v, "sim-large")) for v in (V10, V11, V12, V20, V22)] == [12, 8, 8, 7, 7]


def test_versions_small():
    assert [wrong(runs(v, "sim-small")) for v in (V10, V11, V12)] == [16, 8, 8]


def test_quote_only():
    assert unsupported(runs(V11, "sim-large")) == 6 and unsupported(runs(V12, "sim-large")) == 1
    assert unsupported(runs(V11, "sim-small")) == 18 and unsupported(runs(V12, "sim-small")) == 5
    assert round(faithfulness(runs(V12, "sim-large")), 3) == 0.998


def test_v22_fixes_six_breaks_det0061():
    a = {t.det.determination_id for t in runs(V10, "sim-large") if t.wrong}
    b = {t.det.determination_id for t in runs(V22, "sim-large") if t.wrong}
    assert len(a - b) == 6 and b - a == {"DET-0061"}


def test_injection_and_lint():
    assert len(lint(V21.examples)) == 1 and not lint(V22.examples)


def test_fingerprints():
    assert V10.fingerprint() == "ca0d1f284a19"
    assert V22.fingerprint() == V20.fingerprint() == "6bc1f97b940b"
    assert V23.fingerprint() == "91a37cb95c66"


def test_gate():
    assert lab.gate(V22, V23, "sim-large")["gate"] == "BLOCK"
    assert lab.gate(V22, V23B, "sim-large")["gate"] == "pass"
