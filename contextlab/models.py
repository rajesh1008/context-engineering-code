"""contextlab.models (Chapter 2): the typed objects a context pipeline passes around.

Provenance, Span, Source and ContextPacket are Listings 2.2, 2.3 and 2.5. Determination is
one row of the evaluation set: the claim facts the claims system holds, the adjuster's
question and the gold label.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from functools import cached_property
from typing import Any

from pydantic import BaseModel, Field, computed_field

from contextlab.chunker import count_tokens


class Kind(str, Enum):
    FORM = "form"
    ENDORSEMENT = "endorsement"
    BULLETIN = "bulletin"
    GUIDANCE = "guidance"
    UW_NOTE = "uw_note"


class Provenance(BaseModel):
    """Where a span came from and what authority it carries."""

    doc_id: str                      # e.g. "HO-3"
    version: str | None              # e.g. "2024-06"; None fails closed
    effective_from: date | None = None
    effective_to: date | None = None  # exclusive; None means still in force
    superseded_by: str | None = None
    jurisdiction: str = "ALL"        # two-letter state code or "ALL"
    entitlement: frozenset[str] = frozenset({"claims", "underwriting"})
    precedence: int = 0
    page: int | None = None
    source_system: str = "docstore"
    trust: float = 1.0               # 0..1; user turns and memory start lower

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        return f"{self.doc_id} ({self.version})" if self.version else self.doc_id

    def in_force(self, on: date) -> bool:
        if self.effective_from and on < self.effective_from:
            return False
        return not (self.effective_to and on >= self.effective_to)

class Span(BaseModel):
    """A unit of evidence or instruction that can be placed in the window."""

    span_id: str
    kind: Kind
    text: str
    provenance: Provenance
    topic: str | None = None
    section: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def tokens(self) -> int:
        return count_tokens(self.text)

    def citation(self) -> str:
        p = self.provenance
        page = f", page {p.page}" if p.page is not None else ""
        sec = f", {self.section}" if self.section else ""
        return f"{p.label}{sec}{page}"


class Source(BaseModel):
    """A registered knowledge source: one row of the knowledge inventory."""

    name: str
    kind: Kind
    owner: str
    system: str
    refresh: str = "on change"
    requires_version: bool = True
    entitlement: frozenset[str] = frozenset({"claims", "underwriting"})

    def admit(self, span: Span) -> tuple[bool, str]:
        """Fail closed: a span with no version cannot become evidence."""
        if span.kind != self.kind:
            return False, f"kind {span.kind} does not belong to source {self.name}"
        if self.requires_version and not span.provenance.version:
            return False, f"{span.span_id}: no version, rejected (fail closed)"
        return True, "ok"


class ContextPacket(BaseModel):
    """The assembled input for one determination, before rendering."""

    determination_id: str
    question: str
    instructions: list[Span] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)
    budget_tokens: int | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    dropped: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def tokens(self) -> int:
        return sum(s.tokens for s in self.instructions + self.spans)

    def span_ids(self) -> list[str]:
        return [s.span_id for s in self.spans]

    def rules(self) -> set[str]:
        """Rule tags carried by the instruction spans (e.g. 'precedence')."""
        return {r for s in self.instructions for r in s.meta.get("rules", [])}


class Determination(BaseModel):
    """One coverage question on one claim, with the claim facts and the gold label."""

    determination_id: str
    claim_id: str
    claimant: str
    policy_number: str
    state: str
    line: str
    form: str                        # form edition label, e.g. "HO-3 (2024-06)"
    endorsements: list[str] = Field(default_factory=list)
    date_of_loss: date
    adjuster: str
    question: str
    topic: str
    gold_span_id: str
    gold_label: str
    gold_kind: str
    verdict: str
    difficulty: str                  # simple (the form governs) | complex


class Applicability(BaseModel):
    ok: bool
    reason: str = "applies"
