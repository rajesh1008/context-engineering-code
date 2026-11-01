"""contextlab.diagnostics: length and packet diagnostics (Chapter 4).

`rot_curve` and `knee` are the functions printed as Listings 4.5 to 4.7. They are
duck-typed: any pipeline with `with_config(pad_fn=...)` and `run(det)` works, where
`run` returns a trace with `.answer.cited_span_id`, `.det.gold_span_id` and
`.rendered.tokens`, and raises ValueError when the prompt exceeds the window.
The four packet checks (clash, confusion, distraction, rot) count findings over
traces; they are deliberately simple so their blind spots can be shown.
"""
from __future__ import annotations

import logging
from itertools import cycle
from typing import Callable

log = logging.getLogger("contextlab.diagnostics")
Pipeline = Determination = Span = object      # duck types; see the module docstring


def rot_curve(pipeline: Pipeline, dets: list[Determination], lengths: list[int],
              pad_pool: list[str] | None = None,
              pad_fn: Callable[[Determination, int], list[Span]] | None = None,
              ) -> list[tuple[int, float, float]]:
    """Wrong-clause rate as padding grows: [(target_tokens, mean_tokens, wrong_rate)].

    Padding comes from `pad_pool` (span ids, cycled, the same for every claim)
    or from `pad_fn(det, target)` (per-claim spans). It is appended after the
    retrieved evidence, so the only change between points is what was added.
    A prompt larger than the model's window counts as wrong (overflow).
    """
    if (pad_pool is None) == (pad_fn is None):
        raise ValueError("give exactly one of pad_pool or pad_fn")
    rows = []
    for target in lengths:
        if pad_fn is not None:
            pad = lambda d, t=target: pad_fn(d, t)
        else:
            pad = lambda d, t=target: _from_pool(pipeline, pad_pool, t)
        pipe = pipeline.with_config(pad_fn=pad)
        wrong = tokens = 0.0
        for d in dets:
            try:
                t = pipe.run(d)
            except ValueError:            # overflow: the call is rejected
                wrong += 1
                tokens += target
                continue
            wrong += t.answer.cited_span_id != t.det.gold_span_id
            tokens += t.rendered.tokens
        log.info("rot_curve target=%d tokens=%.0f wrong=%.3f", target, tokens / len(dets),
                 wrong / len(dets))
        rows.append((target, tokens / len(dets), wrong / len(dets)))
    return rows


def _from_pool(pipeline, pool_ids, target):
    by_id = {s.span_id: s for s in pipeline.corpus}
    out, n = [], 0
    if target <= 0:
        return out
    for sid in cycle(pool_ids):
        out.append(by_id[sid])
        n += by_id[sid].tokens
        if n >= target:
            return out


def knee(rows: list[tuple[int, float, float]], tolerance: float = 0.02) -> int | None:
    """Largest padded length whose wrong-clause rate stays within `tolerance`
    (absolute) of the shortest packet's rate, provided every shorter length
    does too. None if even the second point is past tolerance."""
    if not rows:
        return None
    base = rows[0][2]
    last = None
    for target, _, wrong in rows:
        if wrong - base > tolerance + 1e-9:   # rates are ratios; ignore float dust
            break
        last = target
    return last


# ------------------------------------------------------------------ packet checks
def _binding(spans):
    return [s for s in spans if getattr(s, "binding", True)]


def clash(trace) -> int:
    """Findings: a grant and an exclusion on the same topic from the same form edition."""
    by = {}
    for s in _binding(trace.rendered.spans):
        if s.effect in ("grant", "exclusion"):
            by.setdefault((s.form, s.edition, s.topic), set()).add(s.effect)
    return sum(1 for v in by.values() if len(v) == 2)


def confusion(trace, applies: Callable) -> int:
    """Findings: a clause present in several editions, plus every on-topic binding
    span that does not apply to this policy (wrong edition, state, date, not bought)."""
    d = trace.det
    eds = {}
    for s in _binding(trace.rendered.spans):
        eds.setdefault((s.form, s.section), set()).add(s.edition)
    versions = sum(1 for v in eds.values() if len(v) > 1)
    inapplicable = sum(1 for s in _binding(trace.rendered.spans)
                       if s.topic in (d.topic, "general") and not applies(s, d))
    return versions + inapplicable


def distraction(trace, threshold: float = 0.5) -> int:
    """Finding when more than `threshold` of the packet's span tokens are off the claim's
    topic. A similarity-style check: on-topic look-alikes pass it, claim-file pages fail it."""
    spans = trace.rendered.spans
    total = sum(s.tokens for s in spans) or 1
    off = sum(s.tokens for s in spans if s.topic not in (trace.det.topic, "general"))
    return int(off / total > threshold)


def rot(trace) -> int:
    """Finding when the prompt is past the model's configured onset."""
    return int(trace.rendered.tokens > trace.model.onset)


class Diagnostics:
    """The suite, so the curve reads as one of the checks."""
    rot_curve = staticmethod(rot_curve)
    knee = staticmethod(knee)
    clash = staticmethod(clash)
    confusion = staticmethod(confusion)
    distraction = staticmethod(distraction)
    rot = staticmethod(rot)

    @staticmethod
    def summary(traces, applies) -> dict[str, int]:
        """Traces flagged by clash, distraction and rot; total confusion findings."""
        return {"clash": sum(clash(t) > 0 for t in traces),
                "confusion": sum(confusion(t, applies) for t in traces),
                "distraction": sum(distraction(t) for t in traces),
                "rot": sum(rot(t) for t in traces)}


# ------------------------------------------------------- Chapter 29: the flowchart
# contextlab v6.5. Ten classes in pipeline order: the earliest is the cheapest fix.
from dataclasses import dataclass, field
from functools import reduce

CLASSES = ("overflow", "missing_evidence", "drift", "poisoning", "confusion",
           "clash", "fragmentation", "distraction", "lost_in_middle", "rot")
NOT_CONTEXT = "not_context"          # packet sound: reading or reasoning (Chapter 28)


@dataclass(frozen=True)
class Test:
    """One box of Signature 7: a cheap signal on the trace, then a counterfactual.

    `signal(case)` says whether the class could apply; `fix(case)` returns the case
    with only that defect repaired, so a rerun shows whether the defect was causal."""
    cls: str
    signal: Callable
    fix: Callable
    remedy: str = ""


@dataclass
class Diagnosis:
    case_id: str
    primary: str
    secondary: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)   # fired but not causal alone
    joint: bool = False                                  # only the combined fix worked


def diagnose(case, run: Callable, tests: list[Test]) -> Diagnosis:
    """Run every test; classes whose fix alone turns the answer right are causal.

    The primary class is the first causal one in pipeline order. If no single fix
    works but all signalled fixes together do, the first signal is primary (joint).
    No signal and no flip means the packet was sound: NOT_CONTEXT."""
    order = {c: i for i, c in enumerate(CLASSES)}
    tests = sorted(tests, key=lambda t: order[t.cls])
    fired = [t for t in tests if t.signal(case)]
    causal = [t.cls for t in fired if run(t.fix(case))]
    cid = getattr(case, "cid", "?")
    if causal:
        return Diagnosis(cid, causal[0], causal[1:],
                         [t.cls for t in fired if t.cls not in causal])
    if fired:
        joint = reduce(lambda c, t: t.fix(c), fired, case)
        if run(joint):
            return Diagnosis(cid, fired[0].cls, [t.cls for t in fired[1:]], [], True)
        return Diagnosis(cid, "unresolved", [], [t.cls for t in fired])
    return Diagnosis(cid, NOT_CONTEXT)


def blast_radius(traces, touches: Callable, wrong: Callable) -> tuple[int, int]:
    """(exposed, wrong): determinations whose packet read the defective item, and
    how many of those were wrong. Exposure, not errors, sizes the replay."""
    exposed = [t for t in traces if touches(t)]
    return len(exposed), sum(1 for t in exposed if wrong(t))


Diagnostics.diagnose = staticmethod(diagnose)
Diagnostics.blast_radius = staticmethod(blast_radius)
Diagnostics.CLASSES = CLASSES
