"""Chapter 29 stand-in: packets as span lists and a deterministic reader.

Note: no language model runs here. The reader is a deterministic stand-in: it
scores each span by a fixed rule (similarity, position, length, completeness,
labels) and cites the highest. The rule encodes the mechanisms Chapters 3, 4,
10, 15 and 24 measured, so that a counterfactual fix can be re-run and checked.
"""
from __future__ import annotations
from dataclasses import dataclass, replace

INSTR = 450                 # instruction tokens, always first in the packet
ONSET = 14_447              # sim-mid knee from Chapter 4: reading degrades past it
EDGE = 4_000                # below this, position barely matters (Chapter 3)


@dataclass(frozen=True)
class Span:
    sid: str
    kind: str               # gold, lookalike, rival, offq, false, pad
    tokens: int = 300
    sim: float = 0.70
    applies: bool = True    # does it govern this claim (state, edition, authority)?
    true: bool = True       # does it say what its source says now?
    lag: bool = False       # copy behind its source (watermark or fingerprint)
    origin: str = "source"  # source, generated, adversarial, extraction, feed, summary
    complete: bool = True   # whole governing unit in one span
    labelled: bool = False  # applicability marked in the packet
    ranked: bool = False    # precedence marked against the gold span


@dataclass(frozen=True)
class Case:
    cid: str
    spans: tuple
    window: int = 32_000
    slip: bool = False      # a reading failure on a sound packet

    @property
    def tokens(self) -> int:
        return INSTR + sum(s.tokens for s in self.spans)


def score(case: Case, i: int) -> float:
    s, n, t = case.spans[i], len(case.spans), case.tokens
    v = s.sim
    if s.kind == "gold":
        pos = (i + 0.5) / n
        if t > EDGE and 0.2 < pos < 0.8:
            v -= 0.10                          # lost in the middle
        if t > ONSET:
            v -= 0.30 * (t - ONSET) / ONSET    # rot
        if not s.complete:
            v -= 0.25                          # fragment: half the rule
    if s.labelled and not s.applies:
        v -= 0.50
    if s.kind == "rival" and s.ranked:
        v -= 0.50
    return v


def run(case: Case) -> bool:
    """True when the reader cites the gold span and the span is sound."""
    if case.tokens > case.window or case.slip:
        return False                           # truncated from the front, or a slip
    if not case.spans:
        return False
    i = max(range(len(case.spans)), key=lambda k: score(case, k))
    s = case.spans[i]
    return s.kind == "gold" and s.true and s.complete


def with_spans(case: Case, spans) -> Case:
    return replace(case, spans=tuple(spans))
