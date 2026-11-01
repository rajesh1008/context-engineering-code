"""The ten failure scripts of Chapter 30: one defect each, injected into a sound case.

Each script names the incident or probe it was cut from (Figure 29.15 and the
Chapter 29 probe families). An injection changes one thing about the packet and
leaves the claim, the reader and the seed alone.
"""
from __future__ import annotations
from dataclasses import replace
from contextlab.repro import Script
from lakeshore30 import Span, Case, EDGE, ONSET, WINDOW, INSTR, with_spans


def _gold(c: Case) -> Span:
    return next(s for s in c.spans if s.kind == "gold")


def _pads(c: Case, rng, total: int, tag: str) -> list[Span]:
    out, n = [], 0
    while n < total:
        t = rng.randint(250, 350)
        out.append(Span(f"{c.cid}:{tag}{len(out)}", "pad", t, round(rng.uniform(0.25, 0.40), 3),
                        "other"))
        n += t
    return out


def overflow(c, rng):           # claim file pasted whole; no budget
    return with_spans(c, list(c.spans) + _pads(c, rng, WINDOW - c.tokens + 2_000, "file"))


def missing(c, rng):            # Ch 14: the governing span never indexed
    return with_spans(c, [s for s in c.spans if s.kind != "gold"])


def drift(c, rng):              # Ch 24: the copy is behind its source
    g = _gold(c)
    return with_spans(c, [replace(s, lag=True, true=False) if s is g else s for s in c.spans])


def poisoning(c, rng):          # Ch 22: a derived summary states the wrong limit
    g = _gold(c)
    bad = Span(f"{c.cid}:summary", "false", 180, round(g.sim + rng.uniform(0.0, 0.05), 3),
               c.topic, true=False, origin="summary")
    return with_spans(c, [bad] + list(c.spans))


def confusion(c, rng):          # Ch 1: the superseded edition, unlabelled
    g = _gold(c)
    look = replace(g, sid=f"{c.cid}:ed2019", kind="lookalike", applies=False, edition="2019-03",
                   sim=round(g.sim + rng.uniform(-0.04, 0.04), 3))
    return with_spans(c, [look] + list(c.spans))


def clash(c, rng):              # Ch 7: the skill and the bulletin disagree, no precedence
    g = _gold(c)
    newer = rng.random() < 0.5
    rival = Span(f"{c.cid}:rival", "rival", 220, round(g.sim - 0.01, 3), c.topic,
                 origin="instruction", newer=newer)
    return with_spans(c, [replace(s, newer=not newer) if s is g else s for s in c.spans]
                      + [rival])


def fragmentation(c, rng):      # Ch 10: the rule cut at a page break
    g, k = _gold(c), rng.uniform(0.4, 0.7)
    a = replace(g, sid=g.sid + "#1", tokens=int(g.tokens * k), complete=False,
                sim=round(g.sim - 0.04, 3))
    b = replace(g, sid=g.sid + "#2", tokens=g.tokens - a.tokens, complete=False,
                sim=round(g.sim - 0.22, 3))
    return with_spans(c, [a, b] + [s for s in c.spans if s is not g])


def distraction(c, rng, k: int = 4, on_topic: bool = True):   # earlier claims' notes
    lo, hi = (0.70, 0.78) if on_topic else (0.35, 0.45)
    ds = [Span(f"{c.cid}:note{j}", "distractor", rng.randint(150, 250),
               round(rng.uniform(lo, hi), 3), c.topic if on_topic else "other",
               origin="claim_file", dup=f"{c.cid}:claimfile") for j in range(k)]
    return with_spans(c, list(c.spans) + ds)


def lost_middle(c, rng):        # Ch 3 probe: the rule in the middle of a long packet
    g = _gold(c)
    rest = [s for s in c.spans if s is not g] + _pads(c, rng, 7_000, "mid")
    h = len(rest) // 2
    return with_spans(c, rest[:h] + [g] + rest[h:])


def rot(c, rng):                # Ch 4 probe: long packet, rule at the front
    return with_spans(c, list(c.spans) + _pads(c, rng, 21_000, "rot"))


SCRIPTS = [
    Script("overflow", overflow, "claim file pasted whole past the window"),
    Script("missing_evidence", missing, "governing span removed from the index"),
    Script("drift", drift, "governing copy one revision behind"),
    Script("poisoning", poisoning, "derived summary with the wrong limit"),
    Script("confusion", confusion, "superseded edition, unlabelled"),
    Script("clash", clash, "skill text against the bulletin, no precedence"),
    Script("fragmentation", fragmentation, "rule cut across two spans"),
    Script("distraction", distraction, "four earlier-claim notes on the same topic"),
    Script("lost_in_middle", lost_middle, "rule in the middle of 8,500 tokens"),
    Script("rot", rot, "rule at the front of 22,500 tokens"),
]
BY_CLASS = {s.cls: s for s in SCRIPTS}
