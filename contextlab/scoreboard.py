"""Chapter 2: the scoreboard. Wrong-clause rate with a Wilson interval, retrieval measures on the
packet, faithfulness, tokens, cost, and the lookup-time proxy (Listing 2.1)."""
from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from typing import Any

from contextlab.tokens import cost_usd

Trace = Any   # contextlab.pipeline.Trace; kept loose so the proxy can score any trace-shaped record

# Minutes, illustrative constants for the proxy (not measurements).
VERIFY_MEDIAN = 1.2     # open the cited page and confirm the clause
VERSION_CHECK = 2.2     # look up whether that version was in force on the date of loss
LOCATE_MEDIAN = 4.5     # citation has no version or page: find the clause first
MANUAL_MEDIAN = 11.0    # wrong citation or abstention: Lakeshore's manual lookup

def lookup_minutes(t: Trace, exact: bool, governing: bool) -> float:
    rng = random.Random(int(hashlib.sha256(t.det.determination_id.encode()).hexdigest()[:12], 16))
    jitter = rng.lognormvariate(0, 0.35)
    if t.answer.abstained or not governing:
        return MANUAL_MEDIAN * jitter + 0.5
    if not exact:
        return LOCATE_MEDIAN * jitter + VERSION_CHECK
    return VERIFY_MEDIAN * jitter + (0.0 if t.answer.version_checked else VERSION_CHECK)


def correct(t: Trace) -> bool:
    return (not t.answer.abstained) and t.answer.cited_span_id == t.det.gold_span_id


def exact(t: Trace) -> bool:
    """The citation names the edition and the page of the span it cites."""
    c = t.answer.citation
    return bool(c) and t.answer.cited_span_id is not None and "page" in c and "(" in c


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def sign_test(a: int, b: int) -> float:
    """Exact two-sided sign test on a and b discordant pairs."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def gold_rank(t: Trace) -> int | None:
    ids = t.packet.span_ids()
    return ids.index(t.det.gold_span_id) + 1 if t.det.gold_span_id in ids else None


def faithfulness(t: Trace) -> float:
    """Share of the answer's claims that the cited span supports (the stand-in's claims are checkable)."""
    claims = t.answer.claims
    if not claims or t.answer.cited_span_id is None:
        return 1.0
    text = t.packet_span_text() if hasattr(t, "packet_span_text") else None
    src = next((s.text for s in t.packet.spans if s.span_id == t.answer.cited_span_id), text or "")
    return sum(c in src for c in claims) / len(claims)


@dataclass
class Score:
    n: int
    wrong: int
    abstained: int
    hit1: float
    hit5: float
    mrr: float
    faith: float
    tokens: float
    cost: float
    lookup: float
    lo: float
    hi: float

    @property
    def rate(self) -> float:
        return self.wrong / self.n


def score(traces: list[Trace]) -> Score:
    n = len(traces)
    wrong = sum(not correct(t) for t in traces)
    ranks = [gold_rank(t) for t in traces]
    lo, hi = wilson(wrong, n)
    return Score(
        n=n, wrong=wrong, abstained=sum(t.answer.abstained for t in traces),
        hit1=sum(r == 1 for r in ranks) / n, hit5=sum(r is not None and r <= 5 for r in ranks) / n,
        mrr=sum(1 / r for r in ranks if r) / n,
        faith=statistics.mean(faithfulness(t) for t in traces),
        tokens=statistics.mean(t.rendered.tokens for t in traces),
        cost=statistics.mean(cost_usd(t.config.model, t.rendered.tokens, t.answer.tokens_out) for t in traces),
        lookup=statistics.median(lookup_minutes(t, exact(t), correct(t)) for t in traces), lo=lo, hi=hi)


def row(name: str, s: Score) -> str:
    """One line in the format the chapter prints for a configuration."""
    return (f"{name:<42}{s.wrong}/{s.n}{100 * s.rate:6.1f}% [{100 * s.lo:4.1f}%,{100 * s.hi:4.1f}%] "
            f"abst {100 * s.abstained / s.n:.1f}% hit1 {s.hit1:.2f} hit5 {s.hit5:.2f} mrr {s.mrr:.3f} "
            f"faith {s.faith:.3f} tok {s.tokens:,.0f} ${s.cost:.5f} lookup {s.lookup:.2f}")


def run_row(name: str, s: Score) -> str:
    return (f"{name:<40}{s.wrong:>2}/{s.n}{100 * s.rate:7.1f}%  [{100 * s.lo:4.1f}%, {100 * s.hi:4.1f}%]  "
            f"hit@1 {s.hit1:.2f}  hit@5 {s.hit5:.2f}{s.tokens:>8,.0f} tok")
