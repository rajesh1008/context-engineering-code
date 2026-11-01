"""The chapter's numbers, checked against the stand-in (python3 -m pytest -q)."""
from __future__ import annotations

import math

import pytest

import standin as S
from contextlab.cache import PromptCache
from contextlab.llm import PROFILES, position_weight
from contextlab.tokens import cost_usd
from experiments import padded_packet, render


@pytest.fixture(scope="module")
def dets():
    return S.load_dets()


def test_corpus():
    spans = S.CORPUS.spans
    assert len(spans) == 378
    assert sum(s.tokens for s in spans) == 22_326
    assert sum(s.provenance.effective_from > S.CUT for s in spans) == 99


def test_governing_and_filter(dets):
    d = dets[0]
    assert (d.determination_id, d.holder) == ("DET-0001", "Emily Carter")
    assert d.gold_span_id == "HO 04 95 (2024-06)|Coverage"
    assert all(S.CORPUS.governing(x).span_id == x.gold_span_id for x in dets)
    pk = S.Pipeline(S.PipelineConfig(policy_filter=True)).build_packet(d)[0]
    assert all(S.applies(S.CORPUS.by_id[e.span_id], d) for e in pk.evidence)


def test_packet_sizes(dets):
    def mean(cfg, ds=dets):
        p = S.Pipeline(S.PipelineConfig(**cfg))
        return round(sum(p.render(d).tokens for d in ds) / len(ds))
    assert mean(dict(policy_filter=True)) == 1_046
    assert mean(dict(policy_filter=True), dets[:60]) == 1_046
    assert mean({}) == 1_031 and mean({}, dets[:60]) == 1_018
    assert mean(dict(stuff="family")) == 14_463
    assert mean(dict(stuff="all")) == 34_054
    assert S.Pipeline(S.PipelineConfig(policy_filter=True)).render(dets[0]).tokens == 1_182


def test_cost():
    assert f"{cost_usd('sim-large', 1_046, 57):.5f}" == "0.00399"
    assert f"{3_200 * cost_usd('sim-large', 60_000, 57):.2f}" == "578.74"


def test_position_weight():
    p = PROFILES["sim-large"]
    assert position_weight(0.0, 60_000, p) == pytest.approx(1.0)
    assert position_weight(0.5, 60_000, p) == pytest.approx(1 - 0.30 * (1 - math.exp(-4)))
    assert position_weight(0.5, 60_000, p) < position_weight(0.5, 2_000, p)


def test_padded_packet_puts_the_clause_at_depth(dets):
    d = dets[1]
    for depth in (0.0, 0.5, 1.0):
        r = render(padded_packet(d, 60_000, depth), d)
        assert r.tokens >= 60_000
        ids = [s.span_id for s in r.evidence]
        assert d.gold_span_id in ids


def test_failure_injection(dets):
    model = S.SimulatedModel("sim-large")
    ds = dets[:60]
    hit = sum(model.generate(render(padded_packet(d, 60_000, 0.5), d), key=d.determination_id).cited_span_id
              == d.gold_span_id for d in ds)
    assert hit == 9


def test_window_is_enforced(dets):
    d = dets[0]
    with pytest.raises(ValueError, match="exceeds sim-small window 32000"):
        S.SimulatedModel("sim-small").generate(render(padded_packet(d, 60_000, 0.5), d), key=d.determination_id)


def test_fix_and_strategies(dets):
    def wrong(cfg):
        return sum(t.wrong for t in S.Pipeline(S.PipelineConfig(**cfg)).run_many(dets))
    assert wrong(dict(policy_filter=True)) == 12
    assert wrong({}) == 17
    assert wrong(dict(policy_filter=True, order="edges")) == 13
    assert wrong(dict(stuff="family")) == 37


def test_prompt_cache_prefix_rule():
    c = PromptCache()
    a = [("x" * 10, 2_048), ("y", 100)]
    assert c.lookup(a) == 0 and c.lookup(a) == 2_048
    assert c.lookup([("z", 10)] + a) == 0
