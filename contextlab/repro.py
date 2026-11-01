"""contextlab.repro: failure scripts, reproduction rates and regression cases (Chapter 30).

contextlab v6.6. A failure script takes a sound case and injects exactly one defect
from the Failure Taxonomy (`contextlab.diagnostics.CLASSES`). Running it many times,
on many cases, under fixed seeds, turns "it went wrong once" into a rate with a
control. A regression case pins two numbers per script: the rate at which it must
still reproduce on the reference pipeline (the canary, which shows the script still
works) and the rate it must stay under on the pipeline being shipped (the guard).

Duck-typed: `run(case, seed)` returns "right", "wrong" or "handoff"; cases are any
objects the script's `inject(case, rng)` accepts.
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Callable

from contextlab.diagnostics import CLASSES


def rng_for(*parts) -> random.Random:
    """A seeded generator from any labels, so a script draws the same defect every time."""
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()))


@dataclass(frozen=True)
class Script:
    cls: str                      # one of CLASSES
    inject: Callable              # (case, rng) -> case with one defect
    what: str = ""

    def __post_init__(self):
        if self.cls not in CLASSES:
            raise ValueError(f"unknown failure class {self.cls!r}")


@dataclass
class Repro:
    cls: str
    runs: int
    wrong: int
    handoff: int
    control_wrong: int
    per_case: dict = field(default_factory=dict)     # case id -> wrong count over seeds

    @property
    def rate(self) -> float:
        return self.wrong / self.runs

    @property
    def control_rate(self) -> float:
        return self.control_wrong / self.runs

    def shape(self, seeds: int) -> dict[str, int]:
        """Cases that fail on every seed, on some seeds, or on none."""
        out = {"every seed": 0, "some seeds": 0, "no seed": 0}
        for n in self.per_case.values():
            out["every seed" if n == seeds else "no seed" if n == 0 else "some seeds"] += 1
        return out


def reproduce(script: Script, cases, run: Callable, seeds: int = 20) -> Repro:
    """Wrong and hand-off counts for the injected cases, beside the same cases untouched."""
    r = Repro(script.cls, 0, 0, 0, 0)
    for c in cases:
        bad = script.inject(c, rng_for(script.cls, c.cid))
        n = 0
        for s in range(seeds):
            out, ctl = run(bad, s), run(c, s)
            r.runs += 1
            n += out == "wrong"
            r.handoff += out == "handoff"
            r.control_wrong += ctl == "wrong"
        r.wrong += n
        r.per_case[c.cid] = n
    return r


def flips(cases, perturb: Callable, run: Callable, seeds: int = 20) -> tuple[int, int, int, int]:
    """(wrong before, wrong after, right->wrong, wrong->right) under common seeds.

    The same seed is used before and after, so a flip is caused by the perturbation,
    not by sampling. Aggregate counts can match while both flip counts are large."""
    before = after = rw = wr = 0
    for c in cases:
        p = perturb(c)
        for s in range(seeds):
            a, b = run(c, s) == "wrong", run(p, s) == "wrong"
            before += a; after += b
            rw += (not a) and b; wr += a and not b
    return before, after, rw, wr


def seeds_needed(p: float, confidence: float = 0.95) -> int:
    """Reruns needed to see a failure at least once, if it happens with probability p."""
    if p <= 0:
        return math.inf
    if p >= 1:
        return 1
    return math.ceil(math.log(1 - confidence) / math.log(1 - p))


@dataclass(frozen=True)
class Regression:
    """One script turned into a test: it must still fail on `reference`, and not on `candidate`."""
    script: Script
    canary: float                 # minimum wrong rate on the reference pipeline
    guard: float                  # maximum wrong rate on the candidate pipeline
    seeds: int = 10


def run_suite(suite, cases, reference: Callable, candidate: Callable) -> list[dict]:
    """Every regression case on both pipelines; a broken canary is a broken test."""
    rows = []
    for t in suite:
        ref = reproduce(t.script, cases, reference, t.seeds)
        cand = reproduce(t.script, cases, candidate, t.seeds)
        rows.append({"cls": t.script.cls, "reference": ref.rate, "candidate": cand.rate,
                     "handoff": cand.handoff / cand.runs,
                     "canary_ok": ref.rate >= t.canary, "guard_ok": cand.rate <= t.guard})
    return rows
