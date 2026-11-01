"""Chapter 2 tests: the lab's mechanisms, the smoke checks, the routing rule, and (when the
manuscript is beside the code) that every printed output block is reproduced line for line."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

import lab as L
from contextlab import scoreboard as sb
from contextlab.pipeline import Assembler
import flag
import smoke

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parent.parent / "02-manuscript" / "ch02.md"
SCRIPTS = ["lab.py", "stage_variants.py", "ablation.py", "eval_probes.py", "smoke.py", "smoke.py --stale",
           "trace_det8.py", "flag.py"]


@pytest.fixture(scope="module")
def dets():
    return L.corpus.eval_set(200)


@pytest.fixture(scope="module")
def runs(dets):
    return {"base": L.run_all(L.base, dets), "stale": L.run_all(L.base, dets, L.stale_view),
            "fixed": L.run_all(L.fixed, dets, L.fix_view)}


def test_corpus_census():
    spans = L.corpus.spans
    assert len(spans) == 378
    assert sum(s.tokens for s in spans) == 22_326
    assert len(L.corpus.eval_set(200)) == 200 and len(L.corpus.holdout()) == 300


def test_statistics():
    lo, hi = sb.wilson(17, 200)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (5.4, 13.2)
    assert sb.sign_test(0, 0) == 1.0
    assert sb.sign_test(10, 0) == pytest.approx(2 / 2 ** 10)


def test_headline_rows(runs):
    s = {k: sb.score(v) for k, v in runs.items()}
    assert s["base"].wrong == 17 and round(s["base"].hit1, 2) == 0.59
    assert s["fixed"].wrong < s["base"].wrong < s["stale"].wrong


def test_deterministic(dets):
    a = [t.answer.text for t in L.run_all(L.base, dets[:20])]
    b = [t.answer.text for t in L.run_all(L.base, dets[:20])]
    assert a == b


def test_packet_drives_the_answer(dets):
    """The reader answers from the packet: drop the cited span and the answer changes."""
    d = dets[0]
    t = L.base.run(d)
    assert t.answer.cited_span_id in t.packet.span_ids()
    without = L.base.with_config(k=10, padding=()).run(d)
    assert without.answer.cited_span_id == t.answer.cited_span_id
    stale = L.base.run(L.stale_view(d))
    assert stale.rendered.text != t.rendered.text


def test_reconcile_repairs_the_stale_pointer(dets):
    stale = [L.stale_view(d) for d in dets]
    fixed = [L.reconcile(d) for d in stale]
    assert any(note for _, note in fixed)
    assert all(d.form == L.corpus.policies[d.policy_number]["form"] for d, _ in fixed)


def test_filter_keeps_only_applicable_spans(dets):
    for d in dets[:40]:
        t = L.fixed.run(L.fix_view(d))
        assert all(L.fixed.policy.applies(s, L.fix_view(d)).ok for s in t.packet.spans)


@pytest.mark.parametrize("fmt", Assembler.FORMATS)
def test_formats_render(dets, fmt):
    t = L.base.with_config(fmt=fmt).run(dets[0])
    assert t.rendered.fmt == fmt and t.rendered.tokens > 0
    assert ("[S1]" in t.rendered.text) == (fmt == "markdown")


def test_route_fails_closed():
    d = L.corpus.eval_set(200)[0]
    rec = L.corpus.policies[d.policy_number]
    assert flag.route(d, None) == (False, "no policy record")
    assert flag.route(d, {**rec, "endorsements": None}) == (False, "endorsements unknown")


def test_smoke_checks():
    assert smoke.main() == 0
    assert smoke.main(stale=True) != 0


def _outputs() -> set[str]:
    text = ""
    for s in SCRIPTS:
        text += subprocess.run([sys.executable, *s.split()], cwd=HERE, capture_output=True, text=True).stdout + "\n"
    return {line.rstrip() for line in text.splitlines()}


@pytest.mark.skipif(not MANUSCRIPT.exists(), reason="manuscript not beside the code")
def test_every_printed_block_reproduces():
    md = re.sub(r"(?m)^> ?", "", MANUSCRIPT.read_text())
    out = _outputs()
    missing = []
    for m in re.finditer(r"```(\w*)\n(.*?)```", md, re.S):
        if m.group(1) in ("python", "py", "bash", "sh", "json", "yaml", "toml", "markdown", "md"):
            continue
        missing += [line for line in m.group(2).splitlines() if line.strip() and line.rstrip() not in out]
    assert not missing, missing[:10]


@pytest.mark.skipif(not MANUSCRIPT.exists(), reason="manuscript not beside the code")
def test_retrieval_modes_table_reproduces():
    out = subprocess.run([sys.executable, "retrieval_modes.py"], cwd=HERE, capture_output=True, text=True).stdout
    rows = [line for line in out.splitlines() if line.startswith("| ")]
    md = MANUSCRIPT.read_text()
    assert len(rows) == 6
    assert "\n".join(rows) in md
