"""Chapter 30 stand-in: Lakeshore packets, a sampling reader and two pipelines.

Note: no language model runs here. The reader is a deterministic stand-in. It
scores each span by a fixed rule (the Chapter 29 rule: similarity, position,
length, completeness, labels, precedence, plus a small preference for the newer
source) and then samples one span to cite from a softmax at a low temperature,
using a seed. The same case and seed always give the same answer; different
seeds give the run-to-run variation a real model shows at non-zero temperature.
`naive` is a packet builder with none of the defences of Chapters 9 to 28;
`current` has them, except any defence against on-topic distraction.
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, replace

INSTR = 450            # instruction tokens, always first
ONSET = 14_447         # Chapter 4's knee: reading degrades past it
EDGE = 4_000           # below this, position barely matters (Chapter 3)
WINDOW = 32_000
BUDGET = 8_000         # current pipeline's packet budget
TEMP = 0.04            # reader temperature (stand-in)
STATES = ("WI", "MN", "IA", "IL", "MI")
TOPICS = ("water_backup", "roof", "theft", "wind_hail", "liability")


@dataclass(frozen=True)
class Span:
    sid: str
    kind: str                # gold, support, lookalike, rival, false, distractor, pad
    tokens: int = 300
    sim: float = 0.70
    topic: str = "general"
    applies: bool = True     # governs this claim (state, edition, authority)
    true: bool = True        # says what its source says now
    lag: bool = False        # copy behind its source
    origin: str = "source"   # source, summary, instruction, claim_file
    complete: bool = True    # whole governing rule in one span
    labelled: bool = False   # applicability marked in the packet
    ranked: bool = False     # precedence stated against the governing span
    newer: bool = False      # dated later than the other side of a conflict
    dup: str = ""            # near-duplicate text group
    edition: str = ""


@dataclass(frozen=True)
class Case:
    cid: str
    topic: str
    spans: tuple
    window: int = WINDOW
    handoff: bool = False

    @property
    def tokens(self) -> int:
        return INSTR + sum(s.tokens for s in self.spans)


def with_spans(case: Case, spans) -> Case:
    return replace(case, spans=tuple(spans))


def _r(*parts) -> random.Random:
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()))


def base_cases(n: int = 50) -> list[Case]:
    """Sound packets of about 1,500 tokens: the governing span first, four supports."""
    out = []
    for k in range(n):
        r = _r("base", k)
        st, tp = STATES[k % 5], TOPICS[(k // 5) % 5]
        cid = f"S{k + 1:02d}-{st}-{tp}"
        gold = Span(f"{cid}:gov", "gold", r.randint(250, 420), round(r.uniform(0.82, 0.88), 3),
                    tp, dup=f"{tp}:{st}:clause", edition="2024-01")
        sup = [Span(f"{cid}:sup{j}", "support", r.randint(150, 300),
                    round(r.uniform(0.40, 0.56), 3), "general") for j in range(4)]
        out.append(Case(cid, tp, (gold, *sup)))
    return out


# ------------------------------------------------------------------ the reader
def score(case: Case, i: int) -> float:
    s, n, t = case.spans[i], len(case.spans), case.tokens
    v = s.sim
    if s.kind == "gold":
        pos = (i + 0.5) / n
        if t > EDGE and 0.2 < pos < 0.8:
            v -= 0.10                              # lost in the middle
        if t > ONSET:
            v -= 0.30 * (t - ONSET) / ONSET        # rot
    if s.labelled and not s.applies:
        v -= 0.50
    if s.kind == "rival" and s.ranked:
        v -= 0.50
    if s.newer:
        v += 0.03                                  # recency preference
    return v


def probs(case: Case) -> list[float]:
    z = [math.exp(score(case, i) / TEMP) for i in range(len(case.spans))]
    tot = sum(z)
    return [x / tot for x in z]


def _truncate(case: Case) -> Case:
    """What an unbudgeted builder does at the window: drop from the front."""
    spans = list(case.spans)
    while spans and INSTR + sum(s.tokens for s in spans) > case.window:
        spans.pop(0)
    return with_spans(case, spans)


def run(case: Case, seed: int) -> str:
    """'right', 'wrong' or 'handoff' for one read of one packet under one seed."""
    if case.handoff:
        return "handoff"
    if case.tokens > case.window:
        case = _truncate(case)
    if not case.spans:
        return "wrong"
    u, acc = _r("read", case.cid, seed).random(), 0.0
    p = probs(case)
    for s, q in zip(case.spans, p):
        acc += q
        if u < acc:
            break
    return "right" if s.kind == "gold" and s.true and s.complete else "wrong"


# ------------------------------------------------------------------ pipelines
def naive(case: Case) -> Case:
    """Chapter 8's builder: whatever retrieval returned, in retrieval order."""
    return case


def _budget(c: Case) -> Case:
    spans = list(c.spans)
    for s in sorted([s for s in spans if s.kind == "pad"], key=lambda s: s.sim):
        if INSTR + sum(x.tokens for x in spans) <= BUDGET:
            break
        spans.remove(s)
    return with_spans(c, spans)


def _merge(c: Case) -> Case:
    """Structural spans: the two halves of a cut rule are read as one."""
    parts = [s for s in c.spans if s.kind == "gold" and not s.complete]
    if len(parts) < 2:
        return c
    whole = replace(parts[0], tokens=sum(p.tokens for p in parts), complete=True,
                    sim=max(p.sim for p in parts) + 0.04, sid=parts[0].sid.split("#")[0])
    rest = [s for s in c.spans if s not in parts]
    return with_spans(c, [whole] + rest)


def defend(c: Case, strict: bool = True) -> Case:
    """The defences of Chapters 9 to 28, in pipeline order (none for distraction).
    `strict` adds this chapter's change: a rule still incomplete after merging is
    treated as missing, so the claim is handed off instead of read."""
    c = _budget(c)                                               # Ch 3-4 budget
    if not any(s.kind == "gold" for s in c.spans):
        return replace(c, handoff=True)                          # Ch 9 sufficiency
    spans = [replace(s, lag=False, true=True) if s.lag else s for s in c.spans]  # Ch 24
    spans = [s for s in spans if s.true or s.origin not in ("summary", "generated")]  # Ch 9, 22
    spans = [replace(s, labelled=True) if not s.applies else s for s in spans]   # Ch 15
    spans = [replace(s, ranked=True) if s.kind == "rival" else s for s in spans]  # Ch 6-7
    c = _merge(with_spans(c, spans))                             # Ch 10
    if strict and any(s.kind == "gold" and not s.complete for s in c.spans):
        return replace(c, handoff=True)                          # Ch 30
    return with_spans(c, sorted(c.spans, key=lambda s: -s.sim))  # Ch 3 edge placement


def dedup(c: Case, keyed: bool = False) -> Case:
    """Drop near-duplicates, keeping the first of each group in packet order.
    Unkeyed, two editions of one clause count as duplicates; keyed adds the edition."""
    seen, out = set(), []
    for s in c.spans:
        key = (s.dup, s.edition) if keyed else s.dup
        if s.dup and key in seen:
            continue
        seen.add(key)
        out.append(s)
    return with_spans(c, out)


def current(case: Case) -> Case:
    return defend(case)


def current_dedup(case: Case) -> Case:
    """The candidate change: the dedup step from the platform team, after the checks."""
    return dedup(defend(case))


def current_dedup_keyed(case: Case) -> Case:
    return dedup(defend(case), keyed=True)


def runner(pipeline):
    """`run(case, seed)` for a pipeline, the shape contextlab.repro expects."""
    return lambda case, seed: run(pipeline(case), seed)
