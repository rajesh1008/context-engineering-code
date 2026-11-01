"""Tests for the Chapter 4 lab: the chapter's printed numbers, the diagnostics
API (Listings 4.5-4.7) and the knee edge case Section 4.8 says deserves a test.
Run: python -m pytest 04-code/ch04 -q"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import pytest  # noqa: E402
import lab, monitor_ch04, standin as S  # noqa: E402
from contextlab.diagnostics import Diagnostics, knee  # noqa: E402


def test_corpus_and_sets():
    assert len(S.CORPUS) == 378 and sum(s.tokens for s in S.CORPUS) == 22_326
    assert len(lab.dets) == 100 and len(lab.all_dets) == 200
    d = lab.all_dets[3]
    assert (d.determination_id, d.holder, d.state, d.form, d.edition) == ("DET-0004", "Grace Murphy", "IA", "BOP", "2025-01")


@pytest.fixture(scope="module")
def output():
    lines = []
    lab.main(lines.append)
    return "\n".join(lines)


def test_rot_curves_and_knees(output):
    assert "   64,000     71,497  27.0%      71,497  78.0%      64,000 100.0%*" in output
    assert "knee  32,000 padded tokens (prompt 36,336 tokens" in output
    assert "knee  12,000 padded tokens (prompt 14,447 tokens; 6.0% -> 11.0% at 16,000)  knee prompt/window 45%" in output


def test_distraction_and_split(output):
    assert "sim-mid      16,000    9.0%      23.0%" in output
    assert "an inapplicable look-alike 14, an applicable span, wrong clause 9" in output
    assert "rejects 100 of 100 calls" in output


def test_task_types(output):
    assert "endorsement or bulletin   47  21.3%  21.3%  21.3%  29.8%  36.2%  44.7%  46.8%  61.7%    8,000" in output
    assert "family BOP                17   0.0%   0.0%   0.0%   0.0%   0.0%  11.8%  23.5%  52.9%   16,000" in output


def test_injection_and_example(output):
    assert "sim-mid    injected + cap 3,000           8.0%" in output
    assert "sim-mid    clean + cap 3,000              7.0%" in output
    assert "sim-small  injected (look-alikes)        54.0%" in output
    assert "example DET-0004" in output and "endorsement not on policy" in output
    assert "Section I Exclusions B.2.h, page 3" in output


def test_scoreboard_unchanged_by_cap():
    pipe = lab.current.with_config(model="sim-large")
    for p in (pipe, pipe.with_config(budget=lab.CAP)):
        w, tok, cost, minutes = lab.row_stats(lab.run(p, lab.all_dets), lab.all_dets)
        assert (round(w * 200), round(tok), round(cost, 5), round(minutes, 1)) == (12, 1_103, 0.00399, 3.5)


def test_overflow_raises_before_the_model():
    p = lab.current.with_config(model="sim-small", pad_fn=lambda d: S.neutral(d, 40_000))
    with pytest.raises(ValueError):
        p.run(lab.dets[0])


def test_two_point_check():
    assert monitor_ch04.two_point_check(lab.current.with_config(model="sim-large"), lab.dets, 32_000)[-1]
    assert not monitor_ch04.two_point_check(lab.current.with_config(model="sim-mid"), lab.dets, 32_000)[-1]


def test_knee_edge_cases():
    assert knee([]) is None
    assert knee([(0, 1, 0.04), (1, 2, 0.06), (2, 3, 0.07)]) == 1
    # Section 4.8's documented mismatch: second point already out of band returns 0, not None
    assert knee([(0, 1, 0.04), (1, 2, 0.20)]) == 0


def test_rot_curve_pad_pool_and_args():
    pool = [s.span_id for s in S.CORPUS if s.kind == "bulletin"][:20]
    rows = Diagnostics.rot_curve(lab.current.with_config(model="sim-mid"), lab.dets[:10], [0, 4_000], pad_pool=pool)
    assert rows[0][0] == 0 and rows[1][1] > rows[0][1]
    with pytest.raises(ValueError):
        Diagnostics.rot_curve(lab.current, lab.dets[:2], [0])
