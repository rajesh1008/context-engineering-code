"""contextlab.pipeline (Chapter 2): the six-stage baseline pipeline (Listings 2.6 and 2.7).

discover (query + claim facts) -> retrieve -> select (filter, cut, budget) -> assemble -> generate -> record.
The pipeline is generic: the corpus it is given supplies the retriever, the applicability policy
and the model, so the same code runs on any chapter's synthetic world.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from contextlab.chunker import count_tokens
from contextlab.models import ContextPacket, Determination, Kind, Provenance, Span

INSTRUCTION = ("You are the claims assistant for Lakeshore Mutual Insurance. Answer the adjuster's coverage "
               "question from the documents provided and name the governing provision.")
RULE_TEXT = {
    "precedence": "A state bulletin in force on the date of loss outranks an endorsement, and an endorsement outranks the form.",
    "applicability": "Use only documents that apply to this policy, its state and its date of loss.",
    "exact_citation": "Cite the document, edition, section and page of the provision you rely on.",
}


def instruction(rules: tuple[str, ...] | set[str] = ()) -> Span:
    """The instruction span: the task sentence plus any rule sentences, tagged with the rules."""
    rules = [r for r in RULE_TEXT if r in set(rules)]
    text = " ".join([INSTRUCTION] + [RULE_TEXT[r] for r in rules])
    return Span(span_id="INSTR", kind=Kind.GUIDANCE, text=text,
                provenance=Provenance(doc_id="instructions", version="v1", source_system="pipeline"),
                meta={"rules": rules})


def claim_facts(det: Determination) -> str:
    """The claim header: what the claims system already knows about the case."""
    ends = ", ".join(det.endorsements) if det.endorsements else "none"
    return (f"Claim facts: Claim {det.claim_id}, claimant {det.claimant}, policy {det.policy_number}. "
            f"Policy form: {det.form}. Endorsements: {ends}. State: {det.state}. "
            f"Date of loss: {det.date_of_loss.isoformat()}. Adjuster: {det.adjuster}.")


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = "bm25"                   # retrieval mode
    k: int = 10                          # spans retrieved into the packet
    query_facts: bool = True             # append form label and state to the query
    policy_filter: bool = False          # drop inapplicable spans before ranking cut
    rules: tuple[str, ...] = ()          # instruction rules
    fmt: str = "markdown"
    order: str = "rank"
    budget: int | None = None            # evidence token budget
    packer: str = "greedy"               # greedy | knapsack | elastic
    model: str = "sim-large"
    include_history: bool = False
    padding: tuple[str, ...] = ()        # span ids appended (failure injection)
    pad_fn: Callable[[Determination], list[Span]] | None = None  # per-claim spans appended
    seed: int = 2026


@dataclass
class Entry:
    """One evidence span as placed in the rendered text."""
    tag: str                             # "S1", "S2", ... (empty in plain text)
    span: Span
    rank: int                            # position in the packet before ordering
    labelled: bool                       # does the reader see its provenance header?


@dataclass
class Rendered:
    text: str
    tokens: int
    fmt: str
    entries: list[Entry]
    facts: str
    question: str


@dataclass
class Answer:
    text: str
    cited_span_id: str | None
    cited_label: str | None             # the tag the answer cites, e.g. "S2"
    citation: str | None                # the citation string in the answer
    tokens_out: int
    version_checked: bool = False
    abstained: bool = False
    claims: list[str] = field(default_factory=list)   # the sentences that assert something


@dataclass
class Trace:
    det: Determination
    config: PipelineConfig
    retrieved: list[str]                 # ranked span ids before filtering
    packet: ContextPacket
    rendered: Rendered
    answer: Answer
    dropped: dict[str, str] = field(default_factory=dict)
    cached_tokens: int = 0
    seconds: float = 0.0


def _header_fields(s: Span) -> dict[str, Any]:
    p = s.provenance
    return {"label": p.label, "section": s.section or "no section",
            "page": p.page if p.page is not None else "none",
            "effective": p.effective_from.isoformat() if p.effective_from else "unknown",
            "jurisdiction": p.jurisdiction, "authority": s.kind.value}


class Assembler:
    """Render a packet as text: instruction, claim header, evidence spans, question."""

    FORMATS = ("markdown", "xml", "json", "plain")
    ORDERS = ("rank", "edges", "precedence")

    def __init__(self, fmt: str = "markdown", order: str = "rank"):
        assert fmt in self.FORMATS and order in self.ORDERS, (fmt, order)
        self.fmt, self.order = fmt, order

    def arrange(self, spans: list[Span]) -> list[tuple[int, Span]]:
        ranked = list(enumerate(spans))
        if self.order == "edges":            # strongest at both ends, weakest in the middle
            front, back = ranked[0::2], ranked[1::2]
            return front + back[::-1]
        if self.order == "precedence":       # bulletin, endorsement, form, guidance, note; rank breaks ties
            return sorted(ranked, key=lambda x: (-x[1].provenance.precedence, x[0]))
        return ranked

    def block(self, tag: str, s: Span) -> str:
        h = _header_fields(s)
        if self.fmt == "markdown":
            return (f"### [{tag}] {h['label']} | {h['section']} | page {h['page']} | effective {h['effective']} "
                    f"| jurisdiction {h['jurisdiction']} | authority {h['authority']}\n{s.text}")
        if self.fmt == "xml":
            return (f'<span id="{tag}">\n{h["label"]} | {h["section"]} | page {h["page"]} | effective {h["effective"]} '
                    f'| jurisdiction {h["jurisdiction"]} | authority {h["authority"]}\n{s.text}\n</span>')
        if self.fmt == "json":
            return json.dumps({"id": tag, "label": h["label"], "page": h["page"], "authority": h["authority"],
                               "text": s.text})
        return f"- {s.text}"                    # plain: a bare list of clauses, no provenance

    def render(self, packet: ContextPacket, facts: str) -> Rendered:
        entries, blocks = [], []
        for n, (rank, s) in enumerate(self.arrange(packet.spans), 1):
            tag = f"S{n}" if self.fmt != "plain" else ""
            entries.append(Entry(tag=tag, span=s, rank=rank, labelled=self.fmt != "plain"))
            blocks.append(self.block(tag, s))
        instr = "\n".join(s.text for s in packet.instructions)
        if self.fmt == "xml":
            blocks = ["<documents>", *blocks, "</documents>"]
        parts = [instr, facts] + blocks + [f"Question: {packet.question}"]
        text = "\n\n".join(parts)
        return Rendered(text=text, tokens=count_tokens(text), fmt=self.fmt, entries=entries, facts=facts,
                        question=packet.question)


class Pipeline:
    """The baseline pipeline. The corpus supplies the retriever, the policy and the model."""

    def __init__(self, corpus: Any, config: PipelineConfig | None = None, _shared: dict | None = None):
        self.corpus = corpus
        self.config = config or PipelineConfig()
        shared = _shared if _shared is not None else {}
        self._shared = shared
        if "retriever" not in shared:
            shared["retriever"] = corpus.retriever()
            shared["policy"] = corpus.policy()
            shared["models"] = {}
        self.retriever = shared["retriever"]
        self.policy = shared["policy"]
        if self.config.model not in shared["models"]:
            shared["models"][self.config.model] = corpus.model(self.config.model)
        self.model = shared["models"][self.config.model]

    def with_config(self, **changes: Any) -> "Pipeline":
        return Pipeline(self.corpus, replace(self.config, **changes), self._shared)

    def query(self, det: Determination) -> str:
        if not self.config.query_facts:
            return det.question
        return f"{det.question} {det.form} {' '.join(det.endorsements)} {det.state}"

    def build_packet(self, det: Determination) -> tuple[ContextPacket, list[str], dict[str, str]]:
        c = self.config
        flt = (lambda s: self.policy.applies(s, det).ok) if c.policy_filter else None
        hits = self.retriever.search(self.query(det), k=max(c.k, 30), mode=c.mode)
        retrieved = [s.span_id for s, _ in hits]
        dropped = {}
        if flt:
            kept_hits = [(s, x) for s, x in hits if flt(s)]
            dropped = {s.span_id: self.policy.applies(s, det).reason for s, _ in hits if not flt(s)}
        else:
            kept_hits = hits
        kept_hits = kept_hits[:c.k]
        spans = [s for s, _ in kept_hits]
        if c.include_history:
            spans += self.corpus.history_for(det)
        spans += [self.corpus.span(x) for x in c.padding]
        if c.pad_fn is not None:
            spans += c.pad_fn(det)
        packet = ContextPacket(determination_id=det.determination_id, question=det.question,
                               instructions=[instruction(c.rules)], spans=spans,
                               budget_tokens=c.budget,
                               scores={s.span_id: x for s, x in kept_hits})
        if c.budget is not None:
            packet = self._pack(packet)
        return packet, retrieved, dropped

    def _pack(self, packet: ContextPacket) -> ContextPacket:
        """Greedy packing in rank order under the evidence budget (Chapter 32 adds the others)."""
        kept, used = [], 0
        for s in packet.spans:
            if used + s.tokens <= (packet.budget_tokens or 0):
                kept.append(s)
                used += s.tokens
        dropped = [s.span_id for s in packet.spans if s not in kept]
        return packet.model_copy(update={"spans": kept, "dropped": dropped})

    def run(self, det: Determination) -> Trace:
        t0 = time.perf_counter()
        packet, retrieved, dropped = self.build_packet(det)
        rendered = Assembler(self.config.fmt, self.config.order).render(packet, claim_facts(det))
        answer = self.model.generate(rendered, key=det.determination_id, topic=det.topic)
        return Trace(det=det, config=self.config, retrieved=retrieved, packet=packet, rendered=rendered,
                     answer=answer, dropped=dropped, seconds=time.perf_counter() - t0)
