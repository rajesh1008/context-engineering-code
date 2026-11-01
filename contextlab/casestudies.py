"""Case-study harness: layer cards, outcome scoring, transplants and gates (contextlab v11.1).

Chapter 50 compares four context layers with Lakeshore's. A layer is described
by a card with the same twelve rows for every case, scored on its own questions
with the same outcome rules, and probed by transplanting another layer's rules.
Self-contained: standard library only.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

ROWS = ("unit", "sources", "required", "as_of", "precedence", "admission",
        "budget", "gate", "eval", "exit", "handback", "trace")
DEFER = ("HANDBACK", "ASK", "REFUSE")


@dataclass
class LayerCard:
    name: str
    rows: dict = field(default_factory=dict)

    def missing(self) -> list:
        return [r for r in ROWS if not self.rows.get(r)]


def compare(cards: list) -> dict:
    """Row by row: which layers declare it (True) and which leave it implicit."""
    return {r: {c.name: bool(c.rows.get(r)) for c in cards} for r in ROWS}


@dataclass(frozen=True)
class Q:
    qid: str
    intent: str
    truth: str
    kind: str = ""
    ctx: tuple = ()


@dataclass
class Tally:
    n: int = 0
    right: int = 0
    wrong: int = 0
    deferred: int = 0
    by: Counter = field(default_factory=Counter)
    wrong_by: Counter = field(default_factory=Counter)

    def rate(self) -> float:
        return self.wrong / self.n if self.n else 0.0


def outcome(answer: str, truth: str) -> str:
    """Right, wrong (cited the wrong evidence) or deferred (asked, refused, handed back)."""
    if answer == truth:
        return "right"
    return "deferred" if answer in DEFER else "wrong"


def score(questions, answer) -> Tally:
    t = Tally()
    for q in questions:
        o = outcome(answer(q), q.truth)
        t.n += 1
        setattr(t, o, getattr(t, o) + 1)
        t.by[(q.intent, o)] += 1
        if o == "wrong":
            t.wrong_by[q.kind or q.intent] += 1
    return t


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


def transplant(native: dict, donor: dict, keys) -> dict:
    """The native configuration with the donor's rule for each named key."""
    cfg = dict(native)
    for k in keys:
        cfg[k] = donor[k]
    return cfg


def ablate(base: dict, target: dict, run) -> dict:
    """Score the base with one rule at a time swapped for the target's."""
    out = {}
    for k in sorted(base):
        if base[k] != target[k]:
            out[k] = run(transplant(base, target, [k]))
    return out


def route(q: float, hi: float, lo: float) -> str:
    """Document quality gate: proceed, review or escalate by confidence band."""
    if q >= hi:
        return "auto"
    return "review" if q >= lo else "escalate"


def gate_curve(docs, lo: float, his) -> list:
    """For each upper band edge: review load and expected escaped errors."""
    rows = []
    for hi in his:
        load = sum(1 for q, _ in docs if route(q, hi, lo) == "review")
        esc = sum(e for q, e in docs if route(q, hi, lo) == "auto")
        rows.append((hi, load, round(esc, 1)))
    return rows


def fit_band(docs, lo: float, his, capacity: int) -> tuple:
    """Highest upper edge whose review load fits the reviewers' capacity."""
    fits = [r for r in gate_curve(docs, lo, his) if r[1] <= capacity]
    return max(fits, key=lambda r: r[0]) if fits else None


def rare_event_exit(replayed: list, live_reviewed: int, need: int) -> tuple:
    """Exit rule written in the unit failures happen in: rule changes, not questions."""
    if not replayed or not all(replayed):
        return ("hold", f"{sum(replayed)}/{len(replayed)} past changes replay clean")
    if live_reviewed < need:
        return ("human-in-loop", f"{live_reviewed}/{need} new changes reviewed")
    return ("exit", f"{len(replayed)} replayed, {live_reviewed} reviewed")
