"""Packet builders shared by the Chapter 3 labs (and by calibrate.py, which must build the
same packets the labs build)."""
from __future__ import annotations

import standin as S

SIZES = (2_000, 15_000, 30_000, 60_000)
DEPTHS = tuple(round(0.1 * i, 1) for i in range(11))


def dets() -> list[S.Determination]:
    return S.load_dets()


def sweep_set(ds=None) -> list[S.Determination]:
    """The lab's 60: the first 60 of the 200, in ID order."""
    return (ds or dets())[:60]


def rivals(d: S.Determination) -> list[S.CSpan]:
    """The nine strongest retrieved spans other than the governing one (no filter)."""
    by, gold = S.CORPUS.by_id, S.CORPUS.governing(d).span_id
    return [by[sid] for sid, _ in S.CAL["retrieval"][d.determination_id] if sid != gold][:9]


def padded_packet(d: S.Determination, size: int, depth: float) -> S.Packet:
    """Two strongest rivals on top, then the body (seven rivals and claim-file pages, in a
    fixed shuffled order) with the governing clause inserted at `depth` of the body."""
    rv = [S.ev_of(s) for s in rivals(d)]
    gold = S.ev_of(S.CORPUS.governing(d))
    base = S.Packet(determination_id=d.determination_id, instructions=S.instructions_for(d),
                    evidence=rv + [gold], question=d.question)
    fixed = S.Assembler().render(base, S.claim_facts(d)).tokens
    pages, n, i = [], fixed, 0
    while n < size:
        p = S.claim_page(1 + (S.rng("pg", d.determination_id).randrange(5000) + i) % 5000)
        pages.append(p)
        n += S.count_tokens(S.Assembler().span_text(1, p).split("\n", 1)[0]) + S.count_tokens(p.text)
        i += 1
    body = rv[2:] + pages
    S.rng("body", d.determination_id).shuffle(body)
    at = round(depth * len(body))
    ev = rv[:2] + body[:at] + [gold] + body[at:]
    return base.model_copy(update={"evidence": ev})


def render(pk: S.Packet, d: S.Determination, order: str = "rank") -> S.Rendered:
    return S.Assembler("markdown", order).render(pk, S.claim_facts(d))


def retrieved(d: S.Determination, policy_filter: bool) -> S.Packet:
    return S.Pipeline(S.PipelineConfig(policy_filter=policy_filter)).build_packet(d)[0]
