"""The ten tests of Signature 7 on stand-in cases: a signal, then a fix to re-run."""
from dataclasses import replace
from contextlab.diagnostics import Test
from standin29 import ONSET, EDGE, Span, with_spans

GOLD = Span("gold", "gold", 320, 0.85)


def _gold_at(c):
    return next((i for i, s in enumerate(c.spans) if s.kind == "gold"), None)


def _fit(c):
    keep, spans = c.tokens - c.window, list(c.spans)
    for s in [s for s in spans if s.kind in ("pad", "offq")]:
        if keep <= 0:
            break
        spans.remove(s)
        keep -= s.tokens
    return with_spans(c, spans)


def _to_edge(c):
    i = _gold_at(c)
    if i is None:
        return c
    spans = list(c.spans)
    g = spans.pop(i)
    return with_spans(c, spans + [g])


def _middle(c):
    i = _gold_at(c)
    return i is not None and c.tokens > EDGE and 0.2 < (i + 0.5) / len(c.spans) < 0.8


TESTS = [
    Test("overflow", lambda c: c.tokens > c.window, _fit,
         "budget the packet; refuse rather than truncate"),
    Test("missing_evidence", lambda c: _gold_at(c) is None,
         lambda c: with_spans(c, list(c.spans) + [GOLD]),
         "find why retrieval, filter or corpus lost it"),
    Test("drift", lambda c: any(s.lag for s in c.spans),
         lambda c: with_spans(c, [replace(s, lag=False, true=True) if s.lag else s
                                  for s in c.spans]),
         "re-sync the copy; watermark it"),
    Test("poisoning", lambda c: any(not s.true and not s.lag for s in c.spans),
         lambda c: with_spans(c, [replace(s, true=True) if s.kind == "gold" else s
                                  for s in c.spans
                                  if s.kind == "gold" or s.true or s.lag]),
         "trace lineage; quarantine; re-extract"),
    Test("confusion", lambda c: any(not s.applies and not s.labelled for s in c.spans),
         lambda c: with_spans(c, [replace(s, labelled=True) if not s.applies else s
                                  for s in c.spans]),
         "admit by applicability; label edition, state, authority"),
    Test("clash", lambda c: any(s.kind == "rival" and not s.ranked for s in c.spans),
         lambda c: with_spans(c, [replace(s, ranked=True) if s.kind == "rival" else s
                                  for s in c.spans]),
         "state precedence in the packet"),
    Test("fragmentation", lambda c: any(not s.complete for s in c.spans),
         lambda c: with_spans(c, [replace(s, complete=True) for s in c.spans]),
         "span by structure; fetch the continuation"),
    Test("distraction", lambda c: any(s.kind == "offq" and s.sim >= 0.8 for s in c.spans),
         lambda c: with_spans(c, [s for s in c.spans if s.kind != "offq"]),
         "drop or fence off-question material"),
    Test("lost_in_middle", _middle, _to_edge, "put governing spans at the edges"),
    Test("rot", lambda c: c.tokens > ONSET,
         lambda c: with_spans(c, [s for s in c.spans if s.kind != "pad"]),
         "shorten below the knee"),
]
