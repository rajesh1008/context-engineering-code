"""Tests for the Chapter 5 lab: the chapter's printed numbers, the stuffing modes
(Listing 5.1), the cache replay and the benchmark scorer (Listing 5.16).
Run: python -m pytest 04-code/ch05 -q"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import pytest  # noqa: E402
import benchmark, explore, families, lab, standin as S, why_wrong  # noqa: E402
from contextlab.longctx import stuffed_spans  # noqa: E402


def collect(main):
    lines = []
    main(lines.append)
    return "\n".join(lines)


@pytest.fixture(scope="module")
def output():
    return collect(lab.main)


def test_corpus_and_sets():
    assert len(S.CORPUS.eval) == 200 and len(S.CORPUS.holdout) == 300
    d = S.CORPUS.det("DET-0001")
    assert (d.holder, d.state, d.form) == ("Emily Carter", "WI", "HO-3 (2024-06)")


def test_stuffing_modes():
    d = S.CORPUS.det("DET-0001")
    every = stuffed_spans(S.CORPUS, d, "all")
    fam = stuffed_spans(S.CORPUS, d, "family")
    docs = stuffed_spans(S.CORPUS, d, "documents")
    assert len(every) == 351 and len(fam) == 183 and len(docs) == 47
    assert set(docs) <= set(fam) <= set(every)
    with pytest.raises(ValueError):
        stuffed_spans(S.CORPUS, d, "everything")


def test_quality_and_cost(output):
    assert "sim-large  stuff all      27.5%        31.4%         14.9%         0         34,054" in output
    assert "sim-small  stuff all                   all calls rejected: prompt over the window" in output
    assert "sim-large  hybrid          1.5%   0.01186   0.00681       51%" in output
    assert "  sim-large, stuff all: 55 wrong, 48 of them cite a span that does not apply to the policy" in output


def test_failure_injection(output):
    assert "  gold clause cut away for 18 of 113 HO-3 determinations (31 are governed by an endorsement or bulletin)" in output
    assert "sim-large  stuff family, cut 6,000         23.0%         64.5%       9,820" in output


def test_explore_and_why_wrong():
    ex = collect(explore.main)
    assert "  hybrid fixes 9, breaks 0, wrong in both 3" in ex
    assert "retrieve holdout                              12/300   4.0% [ 2.3%,  6.9%]  tok mean   1,053 p95   1,184 max   1,247" in ex
    ww = collect(why_wrong.main)
    assert "   12  applies (wrong clause)" in ww


def test_families():
    out = collect(families.main)
    assert "  family   wrong  2" in out and "DET-0006, DET-0049" in out


def test_cache_replay_rejected_calls_cost_nothing():
    pipe = S.pipeline("stuff all", model="sim-small")
    traces = S.run_all(pipe, S.CORPUS.eval[:5])
    assert all(t.answer.rejected for t in traces)
    assert S.replay_cache(traces) == [0] * 5


def test_benchmark_scorer_matches_scoreboard(tmp_path):
    lines = []
    benchmark.main(lines.append, tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == [
        "hybrid.jsonl", "retrieve.jsonl", "stuff_all.jsonl", "stuff_family.jsonl"]
    text = "\n".join(lines)
    assert "stuff_all     200 prompts  mean   34,054 tokens   30.7 MB" in text
    for name, n in (("retrieve", 12), ("stuff_all", 55), ("stuff_family", 33), ("hybrid", 3)):
        assert f"{name:<13} simulated reader, scored from its reply text: wrong {n}/200" in text
    assert all(p.stat().st_size < 50e6 for p in tmp_path.glob("*.jsonl"))


def test_scoreboard_rows():
    import scoreboard_ch05
    out = collect(scoreboard_ch05.main)
    assert "Ch5 shipped: BOAT whole (eval)     wrong  6.0% (12/200)  lookup 3.5 min  faith 0.985  tokens 1,213  $0.00432" in out


def test_calibration_is_reproducible(tmp_path):
    import json, calibrate
    assert calibrate.build() == json.loads((HERE / "calibration.json").read_text())
