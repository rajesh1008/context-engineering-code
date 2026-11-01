"""contextlab.longctx (v0.8, Chapter 5): building a packet without retrieval.

Three ways to stuff the window (the whole corpus, the policy's product family, or
the documents the applicability check admits, each read whole) and the cut that a
"stay under the knee" rule performs. The applicability check is Chapter 2's
policy, restated here so that the stuffing logic and the filter share one source.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Kind(str, Enum):
    FORM = "form"
    ENDORSEMENT = "endorsement"
    BULLETIN = "bulletin"
    INSTRUCTION = "instruction"
    CLAIM = "claim facts"


DOC_KINDS = (Kind.FORM, Kind.ENDORSEMENT, Kind.BULLETIN)
STUFF_MODES = ("all", "family", "documents")
KIND_RANK = {Kind.FORM: 0, Kind.ENDORSEMENT: 1, Kind.BULLETIN: 2}


def family_of(span) -> str:
    """The product family a document belongs to; bulletins belong to every family."""
    return span.meta.get("family", "")


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str


class Policy:
    """Chapter 2's applicability check: does this span apply to this policy on this date?"""

    def applies(self, span, det) -> Verdict:
        p = span.provenance
        if span.kind == Kind.BULLETIN:
            if p.jurisdiction != det.state:
                return Verdict(False, f"wrong state ({p.jurisdiction})")
            if not p.in_force(det.loss):
                return Verdict(False, "bulletin not in force on date of loss")
            return Verdict(True, "ok")
        if span.kind == Kind.FORM:
            if p.label == det.form:
                return Verdict(True, "ok")
            doc, _, ver = det.form.partition(" (")
            if p.doc_id == doc and p.version < ver.rstrip(")"):
                return Verdict(False, "superseded")
            return Verdict(False, "not on policy")
        if p.jurisdiction != "ALL":                       # a state amendatory endorsement
            if p.jurisdiction != det.state:
                return Verdict(False, f"wrong state ({p.jurisdiction})")
            if family_of(span) != det.family:
                return Verdict(False, "not on policy")
            return Verdict(True, "ok")
        if p.label in det.endorsements:
            return Verdict(True, "ok")
        on_policy = {e.partition(" (")[0]: e.partition(" (")[2].rstrip(")") for e in det.endorsements}
        if p.doc_id in on_policy and p.version < on_policy[p.doc_id]:
            return Verdict(False, "superseded")
        return Verdict(False, "endorsement not on policy")


def stuffed_spans(corpus: Corpus, det: Determination, how: str,
                  policy: Policy | None = None) -> list[Span]:
    """Spans for one stuffing strategy, in corpus (document) order."""
    if how not in STUFF_MODES:
        raise ValueError(f"unknown stuffing mode {how!r}; use one of {STUFF_MODES}")
    docs = [s for s in corpus.spans if s.kind in DOC_KINDS]
    if how == "all":
        return docs
    if how == "family":
        return [s for s in docs if s.kind == Kind.BULLETIN or family_of(s) == det.family]
    policy = policy or Policy()
    keep = {(s.provenance.doc_id, s.provenance.version) for s in docs
            if policy.applies(s, det).ok}
    return [s for s in docs if (s.provenance.doc_id, s.provenance.version) in keep]


def corpus_order(spans: list) -> list:
    """The corpus filing order: forms, then endorsements, then bulletins; documents by ID."""
    return sorted(spans, key=lambda s: (KIND_RANK[s.kind], s.provenance.doc_id,
                                        s.provenance.version, s.meta.get("seq", 0)))


def cache_order(spans: list) -> list:
    """The cache-friendly prompt order: documents sorted by ID, bulletins last, sections in order."""
    return sorted(spans, key=lambda s: (s.kind == Kind.BULLETIN, s.provenance.doc_id,
                                        s.provenance.version, s.meta.get("seq", 0)))


def truncate(spans: list, cap: int | None) -> tuple[list, list]:
    """Keep whole spans in order while the running total of text tokens stays within cap.

    This is the cut a "stay under the knee" rule performs. It returns (kept, dropped)
    so that the trace can record every dropped span.
    """
    if cap is None:
        return list(spans), []
    kept, dropped, total = [], [], 0
    for s in spans:
        if not dropped and total + s.tokens <= cap:
            kept.append(s)
            total += s.tokens
        else:
            dropped.append(s)
    return kept, dropped
