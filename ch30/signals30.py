"""Signals read off a packet, and Signature 7's tests for the stand-in (Chapter 29).

A signal needs no rerun. A test pairs a signal with a fix that repairs one defect,
so contextlab.diagnostics.diagnose can check that a script's failures are the
class the script claims to inject.
"""
from __future__ import annotations
from dataclasses import replace
from contextlab.diagnostics import Test
from lakeshore30 import Case, EDGE, ONSET, with_spans, _budget, _merge


def _gold_i(c):
    return next((i for i, s in enumerate(c.spans) if s.kind == "gold"), None)


def off_topic_share(c: Case) -> float:
    tot = sum(s.tokens for s in c.spans) or 1
    return sum(s.tokens for s in c.spans if s.topic not in (c.topic, "general")) / tot


def crowding(c: Case) -> int:
    """Applicable, on-topic spans other than the governing one, within 0.10 of its similarity."""
    i = _gold_i(c)
    if i is None:
        return 0
    g = c.spans[i]
    return sum(1 for s in c.spans if s is not g and s.applies and s.topic == c.topic
               and s.kind != "gold" and s.sim >= g.sim - 0.10)


SIGNALS = {
    "over window": lambda c: c.tokens > c.window,
    "governing span absent": lambda c: _gold_i(c) is None,
    "copy behind source": lambda c: any(s.lag for s in c.spans),
    "derived, unverified span": lambda c: any(not s.true and not s.lag for s in c.spans),
    "inapplicable, unlabelled": lambda c: any(not s.applies and not s.labelled for s in c.spans),
    "rival, no precedence": lambda c: any(s.kind == "rival" and not s.ranked for s in c.spans),
    "rule incomplete": lambda c: any(not s.complete for s in c.spans),
    "off-topic share > 50%": lambda c: off_topic_share(c) > 0.5,
    "crowding >= 2": lambda c: crowding(c) >= 2,
    "rule mid-packet, > 4,000": lambda c: (_gold_i(c) is not None and c.tokens > EDGE
                                           and 0.2 < (_gold_i(c) + 0.5) / len(c.spans) < 0.8),
    "past onset": lambda c: c.tokens > ONSET,
}


def _edge(c):
    i = _gold_i(c)
    if i is None:
        return c
    s = list(c.spans)
    return with_spans(c, [s.pop(i)] + s)


def _restore(c):
    from lakeshore30 import base_cases
    src = {b.cid: b for b in base_cases()}[c.cid]
    g = next(s for s in src.spans if s.kind == "gold")
    return with_spans(c, [g] + list(c.spans))


S = SIGNALS
TESTS = [
    Test("overflow", S["over window"], _budget),
    Test("missing_evidence", S["governing span absent"], _restore),
    Test("drift", S["copy behind source"],
         lambda c: with_spans(c, [replace(s, lag=False, true=True) if s.lag else s
                                  for s in c.spans])),
    Test("poisoning", S["derived, unverified span"],
         lambda c: with_spans(c, [s for s in c.spans if s.true or s.lag])),
    Test("confusion", S["inapplicable, unlabelled"],
         lambda c: with_spans(c, [replace(s, labelled=True) if not s.applies else s
                                  for s in c.spans])),
    Test("clash", S["rival, no precedence"],
         lambda c: with_spans(c, [replace(s, ranked=True) if s.kind == "rival" else s
                                  for s in c.spans])),
    Test("fragmentation", S["rule incomplete"], _merge),
    Test("distraction", S["crowding >= 2"],
         lambda c: with_spans(c, [s for s in c.spans if s.kind != "distractor"])),
    Test("lost_in_middle", S["rule mid-packet, > 4,000"], _edge),
    Test("rot", S["past onset"],
         lambda c: with_spans(c, [s for s in c.spans if s.kind != "pad"])),
]
