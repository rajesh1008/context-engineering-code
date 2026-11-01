"""Chapter 7 checks: the printed numbers, the fingerprints, the gates, and that seed 2026
reproduces Chapter 6. Run: python3 -m pytest -q"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from common import HERE, MODELS, ROOF, WATER, adh, corpus, pipe, traces
from contextlab.instructions import (HISTORY, HISTORY_7, V22, V30, V30_WATER, V31, V31_WATER,
                                     V32, adherence_probes, clashes, lint)
from standin import cost, faithfulness, mean_tokens, rate, unsupported, wrong
import explore

PROBES = adherence_probes(corpus, "precedence")
ROOF_P = [d for d in PROBES if d.topic in ROOF]


def test_fingerprints():
    fp = {i.version: i.fingerprint() for i in HISTORY_7 + (V30_WATER, V31_WATER)}
    assert fp == {"v1.0": "ca0d1f284a19", "v1.1": "6aa36fe51271", "v1.2": "560d9d9973f0",
                  "v2.0": "6bc1f97b940b", "v2.1": "334aee811204", "v2.2": "6bc1f97b940b",
                  "v2.3": "91a37cb95c66", "v3.0": "a84d0f106ca6", "v3.1": "4ae5de6cdc10",
                  "v3.2": "40a0ad22ae1b", "v3.0-water": "8ea887fc31b0", "v3.1-water": "929c44666623"}
    assert explore.hot_edit(V32).fingerprint() == "3848805dd4bb"


def test_clash_check():
    assert clashes(V22) == [] and clashes(V32) == []
    assert len(clashes(V30)) == len(clashes(V31)) == len(clashes(V30_WATER)) == 1
    assert all(lint(i, corpus) == [] for i in (V22, V30, V31, V32))


def test_probe_sets():
    assert (len(PROBES), len(ROOF_P)) == (115, 15)
    app = adherence_probes(corpus, "applicability")
    assert (len(app), sum(d.topic in ROOF for d in app), sum(d.topic in WATER for d in app)) == (208, 36, 40)


@pytest.mark.parametrize("ins,expect", [(V22, (15, 15, 15)), (V30, (12, 11, 12)),
                                        (V31, (15, 15, 14)), (V32, (15, 15, 15))])
def test_roof_probes(ins, expect):
    assert tuple(adh(pipe(ins, m), ROOF_P) for m in MODELS) == expect


def test_history_large():
    got = [adh(pipe(i, "sim-large"), PROBES) for i in HISTORY_7]
    assert got == [98, 111, 111, 113, 108, 113, 99, 110, 113, 113]


def test_leak_at_scale_small():
    w = [d for d in corpus.determinations if d.topic in WATER]
    n = [sum(unsupported(traces(i, "sim-small", w, s)) for s in range(2026, 2036))
         for i in (V22, V30_WATER, V31_WATER)]
    assert (len(w), n) == (77, [13, 39, 15])


def test_gate_v2():
    verdicts = {i.version: explore.release_gate_v2(i, V22)[0]
                for i in (V30, V31, V32, V30_WATER, V31_WATER)}
    assert verdicts == {"v3.0": False, "v3.1": False, "v3.2": True,
                        "v3.0-water": False, "v3.1-water": False}
    assert not any(explore.release_gate_v1(i, V22)[0] for i in (V22, V30, V31, V32))


def test_scoreboard():
    ev = traces(V32, "sim-large", corpus.eval)
    assert (round(mean_tokens(ev)), f"{cost(ev):.5f}", rate(ev)) == (1550, "0.00551", 3.5)


def _standin6():
    spec = importlib.util.spec_from_file_location("standin6", HERE.parent / "ch06" / "standin.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["standin6"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_seed_2026_reproduces_chapter_6():
    """Chapter 6's printed measurements hold on this chapter's world at seed 2026."""
    s6 = _standin6()
    for ins in HISTORY:
        for m in ("sim-large", "sim-small"):
            a = traces(ins, m, corpus.eval)
            b = s6.run_set(s6.PipelineConfig(model=m, instructions=ins), s6.CORPUS.eval)
            assert (rate(a), f"{faithfulness(a):.3f}", unsupported(a)) == \
                   (s6.rate(b), f"{s6.faithfulness(b):.3f}", s6.unsupported(b)), (ins.version, m)
            assert f"{mean_tokens(a):,.0f}" == f"{s6.mean_tokens(b):,.0f}", (ins.version, m)
    # Chapter 6's gate prints the hold-out (66 claims) for v2.2 and v2.3 on two models
    from contextlab.instructions import V23
    for ins in (V22, V23):
        for m in ("sim-large", "sim-small"):
            h7 = traces(ins, m, corpus.hard())
            h6 = s6.run_set(s6.PipelineConfig(model=m, instructions=ins), s6.CORPUS.hard())
            assert (wrong(h7), len(h7)) == (s6.wrong(h6), len(h6)), (ins.version, m)
