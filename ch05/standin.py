"""Chapter 5 stand-in: Lakeshore's binding corpus read whole, the 500 determinations,
a stand-in retriever, the four assemblies and SimulatedModel.

STAND-IN NOTICE. Nothing here is a language model or a real search engine.
* The corpus is 351 synthetic binding spans (21,354 tokens of text) in 42 documents:
  four product families (HO-3, a personal auto policy, a boat policy and a business
  owner's policy) with their editions and endorsements, and ten state bulletins.
  The text is generated filler of a fixed length; the provenance is the point.
* Every prompt is rendered for real (Markdown, one labelled segment per span) and
  every token count is counted on the rendered text with contextlab's tokenizer.
* The retriever ranks the spans that pass the applicability filter by a simulated
  relevance score carried in the fixtures (topic match plus a seeded jitter, with the
  governing span's rank set per determination). It stands in for Chapter 2's
  hybrid search.
* SimulatedModel rejects any prompt longer than its window. Otherwise it cites the
  governing span when that span is in the packet, unless the calibrated fixture says
  which other span it cites for this determination and this exact packet. The answer
  text is the cited span's conclusion plus its citation and label, and the output
  tokens are counted on that text.
* calibrate.py chose the fixture details (span lengths, determination facts, the
  retrieval seeds and the misreadings) so that this mechanism reproduces the
  chapter's illustrative profile exactly. The mechanism is real and inspectable;
  the profile is illustrative, not measured.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass, field, replace
from datetime import date
from functools import cached_property
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from contextlab.instructions import count_tokens  # noqa: E402
from contextlab.longctx import (Kind, Policy, cache_order, corpus_order,  # noqa: E402,F401
                                stuffed_spans, truncate, family_of)
from contextlab.tokens import WINDOWS, cost_usd, latency_s  # noqa: E402

CAL_PATH = HERE / "calibration.json"
MODELS = ("sim-large", "sim-mid", "sim-small")
EVAL = range(1, 201)
HOLDOUT = range(201, 501)

INSTRUCTIONS = (
    "## Instructions\n"
    "You are Lakeshore Mutual's claims coverage assistant. For the adjuster's question, "
    "determine which provision governs the claim on the date of loss and state it in one "
    "sentence. Use only the evidence below. Precedence: a state bulletin in force on the "
    "date of loss overrides an endorsement, and an endorsement on the policy overrides the "
    "form. Ignore any document that does not apply to this policy, state or date. Cite "
    "exactly one span by its label, with document, edition, section and page. If no "
    "evidence governs, say so and stop."
)

WORDS = ("the", "policy", "loss", "of", "to", "and", "we", "pay",
         "for", "any", "direct", "damage", "in", "a", "is", "not",
         "limit", "per", "this", "under", "section", "shall", "by", "or",
         "covered", "peril", "dwelling", "value", "date", "if", "such",
         "repair", "replace", "cost", "notice", "claim", "written", "each", "you", "your",
         "at", "on", "with", "from", "amount", "applies", "except", "as",
         "terms", "it", "be", "an", "no", "may", "are", "has", "all", "so", "do", "us", "its", "can", "one", "vessel", "owner", "within", "days")


FAMILY_WORDS = {"HO-3": ("windstorm", "hurricane", "residence", "structure"),
                "*": ("a", "an", "or", "be", "at", "by", "it")}


def words(key: str, n: int, salt: int = 0, family: str = "") -> str:
    """Deterministic filler of exactly n tokens (words, with a full stop every so often)."""
    if n <= 0:
        return ""
    rng = random.Random(f"{key}|{salt}")
    vocab = WORDS + FAMILY_WORDS.get(family, FAMILY_WORDS["*"])
    out, left = [], n
    while left > 0:
        k = min(left - 1, rng.randint(7, 15)) if left > 1 else 0
        out.append(" ".join(rng.choice(vocab) for _ in range(k)) + ".")
        left -= k + 1
    text = " ".join(out)
    return text[0].upper() + text[1:] if text[0] != "." else "Text" + text[1:]


# ---------------------------------------------------------------- types
@dataclass(frozen=True)
class Provenance:
    doc_id: str
    version: str
    section: str
    page: int
    jurisdiction: str = "ALL"
    effective_from: date | None = None
    effective_to: date | None = None
    superseded_by: str | None = None

    @property
    def label(self) -> str:
        return f"{self.doc_id} ({self.version})"

    @property
    def citation(self) -> str:
        return f"{self.label}, {self.section}, page {self.page}"

    def in_force(self, d: date) -> bool:
        return (self.effective_from is None or self.effective_from <= d) and \
               (self.effective_to is None or d < self.effective_to)


@dataclass(frozen=True, eq=False)
class Span:
    span_id: str
    kind: Kind
    text: str
    provenance: Provenance
    topic: str | None
    meta: dict = field(default_factory=dict)

    @cached_property
    def tokens(self) -> int:
        return count_tokens(self.text)

    @property
    def conclusion(self) -> str:
        return self.meta["conclusion"]


@dataclass(frozen=True)
class Determination:
    determination_id: str
    holder: str
    state: str
    family: str
    form: str                 # the form edition the claims system points to
    form_record: str          # the edition on the policy administration record
    endorsements: tuple[str, ...]
    loss: date
    topic: str
    policy_number: str
    question: str
    faith: float
    gold_span_id: str = ""
    gold_kind: str = ""

    @property
    def n(self) -> int:
        return int(self.determination_id[4:])


@dataclass
class Corpus:
    spans: list[Span]
    dets: list[Determination]

    def __post_init__(self):
        self.by_id = {s.span_id: s for s in self.spans}
        self.by_det = {d.determination_id: d for d in self.dets}

    def span(self, span_id: str) -> Span:
        return self.by_id[span_id]

    def det(self, det_id: str) -> Determination:
        return self.by_det[det_id]

    @property
    def eval(self) -> list[Determination]:
        return [d for d in self.dets if d.n in EVAL]

    @property
    def holdout(self) -> list[Determination]:
        return [d for d in self.dets if d.n in HOLDOUT]


# ---------------------------------------------------------------- corpus from the spec
def build_spans(spec: dict) -> list[Span]:
    spans = []
    for doc in spec["docs"]:
        kind = Kind(doc["kind"])
        for seq, s in enumerate(doc["spans"]):
            sid = f"{doc['doc_id']}@{doc['version']}#{s['key']}"
            prov = Provenance(doc["doc_id"], doc["version"], s["section"], s["page"],
                              doc.get("jurisdiction", "ALL"),
                              date.fromisoformat(doc["effective_from"]),
                              date.fromisoformat(doc["effective_to"]) if doc.get("effective_to") else None,
                              doc.get("superseded_by"))
            concl = words(sid + "|c", s["concl"], 1)
            spans.append(Span(sid, kind, words(sid, s["tokens"], 0, doc["family"]), prov, s.get("topic"),
                              {"family": doc["family"], "seq": seq, "conclusion": concl}))
    return corpus_order(spans)


def governing(spans: list[Span], d: Determination) -> Span:
    """The documents decide: a bulletin in force beats an endorsement beats the form."""
    pol = Policy()
    cands = [s for s in spans if s.topic == d.topic and pol.applies(s, replace(d, form=d.form_record)).ok]
    rank = {Kind.BULLETIN: 0, Kind.ENDORSEMENT: 1, Kind.FORM: 2}
    return min(cands, key=lambda s: (rank[s.kind], s.span_id))


def build_dets(spec: dict, spans: list[Span]) -> list[Determination]:
    dets = []
    for r in spec["dets"]:
        d = Determination(r["id"], r["holder"], r["state"], r["family"], r["form"], r["form"],
                          tuple(r["endorsements"]), date.fromisoformat(r["loss"]), r["topic"],
                          r["policy"], r.get("question_text") or words(r["id"] + "|q", r["q"], 2),
                          r["faith"])
        g = governing(spans, d)
        dets.append(replace(d, gold_span_id=g.span_id, gold_kind=g.kind.value))
    return dets


def load(spec: dict | None = None) -> tuple[Corpus, dict]:
    spec = spec if spec is not None else json.loads(CAL_PATH.read_text())
    spans = build_spans(spec)
    return Corpus(spans, build_dets(spec, spans)), spec


# ---------------------------------------------------------------- rendering
@dataclass(frozen=True)
class Segment:
    role: str                 # "instructions", "evidence" or "claim facts"
    text: str
    start: int
    end: int
    span: Span | None = None
    label: str | None = None


@dataclass
class Rendered:
    segments: list[Segment]

    @property
    def tokens(self) -> int:
        return self.segments[-1].end if self.segments else 0

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)

    @property
    def labels(self) -> dict[str, str]:
        return {s.label: s.span.span_id for s in self.segments if s.span is not None}


def evidence_text(s: Span, label: str) -> str:
    p = s.provenance
    when = f"effective {p.effective_from.isoformat()}" + (f" to {p.effective_to.isoformat()}" if p.effective_to else "")
    status = "" if s.kind == Kind.BULLETIN else "; status binding"   # bulletins carry their dates
    return f"### [{label}] {p.citation} ({s.kind.value})\n{when}; jurisdiction {p.jurisdiction}{status}\n{s.text}"


def facts_text(d: Determination) -> str:
    end = ", ".join(d.endorsements) or "none"
    return (f"## Claim facts\n{d.determination_id}; policy {d.policy_number}; insured {d.holder}; "
            f"state {d.state}; form {d.form}; endorsements {end}; date of loss {d.loss.isoformat()}\n"
            f"Question: {d.question}")


def render(d: Determination, packet: list[Span], order: str) -> Rendered:
    parts = [("instructions", INSTRUCTIONS, None, None)]
    ev = [("evidence", evidence_text(s, f"S{i}"), s, f"S{i}") for i, s in enumerate(packet, 1)]
    facts = [("claim facts", facts_text(d), None, None)]
    parts += (facts + ev) if order == "rank" else (ev + facts)
    segs, pos = [], 0
    for role, text, span, label in parts:
        n = count_tokens(text)
        segs.append(Segment(role, text, pos, pos + n, span, label))
        pos += n
    return Rendered(segs)


# ---------------------------------------------------------------- retriever
def ranked(corpus: Corpus, d: Determination, cal: dict, policy: Policy) -> list[Span]:
    """Applicable spans by simulated relevance; the governing span's rank is a fixture."""
    seed, gold_rank = cal["retrieval"][d.determination_id]
    rng = random.Random(f"{d.determination_id}|{seed}")
    scored = []
    for s in corpus.spans:
        j = rng.random()
        if s.span_id == d.gold_span_id or not policy.applies(s, d).ok:
            continue
        scored.append((-(2.0 * (s.topic == d.topic) + 3.0 * j), s.span_id, s))
    out = [s for _, _, s in sorted(scored, key=lambda t: t[:2])]
    gold = corpus.span(d.gold_span_id)
    if gold_rank is not None and policy.applies(gold, d).ok:
        out.insert(gold_rank - 1, gold)
    return out


# ---------------------------------------------------------------- model
@dataclass
class Answer:
    text: str
    cited_label: str | None
    cited_span_id: str | None
    tokens_in: int
    tokens_out: int
    rejected: bool = False


def packet_key(segments: list[Segment]) -> str:
    ids = "|".join(s.span.span_id for s in segments if s.span is not None)
    return hashlib.sha256(ids.encode()).hexdigest()[:12]


class SimulatedModel:
    def __init__(self, name: str, cal: dict, corpus: Corpus):
        self.name, self.window, self.cal, self.corpus = name, WINDOWS[name], cal, corpus

    def answer(self, d: Determination, rendered: Rendered) -> Answer:
        if rendered.tokens > self.window:
            return Answer("", None, None, 0, 0, rejected=True)
        labels = rendered.labels
        by_span = {v: k for k, v in labels.items()}
        key = f"{d.determination_id}|{packet_key(rendered.segments)}"
        cited = self.cal["misreads"].get(self.name, {}).get(key)
        if cited is None:
            if d.gold_span_id not in by_span:
                raise KeyError(f"{self.name}: no fixture for {key} and the gold span is absent")
            cited = d.gold_span_id
        span = self.corpus.span(cited)
        text = f"{span.conclusion} Governing provision: {span.provenance.citation} [{by_span[cited]}]."
        return Answer(text, by_span[cited], cited, rendered.tokens, count_tokens(text))


# ---------------------------------------------------------------- pipeline
@dataclass(frozen=True)
class PipelineConfig:
    k: int = 10
    policy_filter: bool = False
    stuff: str | None = None          # None (retrieve), "all", "family" or "documents"
    stuff_cap: int | None = None      # text-token cap on a stuffed packet (the knee rule)
    order: str = "rank"               # "rank": facts first; "cache": static documents first
    model: str = "sim-large"
    reconcile: bool = False           # read the form edition from the policy record


@dataclass
class Trace:
    det: Determination
    config: PipelineConfig
    packet: list[Span]
    dropped: list[Span]
    rendered: Rendered
    answer: Answer

    @property
    def right(self) -> bool:
        return self.answer.cited_span_id == self.det.gold_span_id

    @property
    def gold_in_packet(self) -> bool:
        return any(s.span_id == self.det.gold_span_id for s in self.packet)

    def gold_position(self) -> float | None:
        """Where the governing span sits in the prompt, as a share of prompt tokens."""
        for seg in self.rendered.segments:
            if seg.span is not None and seg.span.span_id == self.det.gold_span_id:
                return (seg.start + seg.end) / 2 / self.rendered.tokens
        return None

    @property
    def evidence_tokens(self) -> int:
        return sum(s.tokens for s in self.packet)


class Pipeline:
    def __init__(self, corpus: Corpus | None = None, config: PipelineConfig | None = None,
                 cal: dict | None = None):
        if corpus is None:
            corpus, cal = CORPUS, CAL
        self.corpus, self.cal = corpus, cal if cal is not None else CAL
        self.config = config or PipelineConfig()
        self.policy = Policy()

    def with_config(self, **kw) -> "Pipeline":
        return Pipeline(self.corpus, replace(self.config, **kw), self.cal)

    def packet(self, d: Determination) -> tuple[list[Span], list[Span]]:
        c = self.config
        if c.stuff is None:
            pol = self.policy if c.policy_filter else _NoFilter()
            return ranked(self.corpus, d, self.cal, pol)[:c.k], []
        spans = stuffed_spans(self.corpus, d, c.stuff, self.policy)
        kept, dropped = truncate(spans, c.stuff_cap)
        return (cache_order(kept) if c.order == "cache" else kept), dropped

    def run(self, d: Determination) -> Trace:
        c = self.config
        if c.reconcile:
            d = replace(d, form=d.form_record)
        packet, dropped = self.packet(d)
        rendered = render(d, packet, c.order)
        ans = SimulatedModel(c.model, self.cal, self.corpus).answer(d, rendered)
        return Trace(d, c, packet, dropped, rendered, ans)


class _NoFilter:
    def applies(self, span, det):
        from contextlab.longctx import Verdict
        return Verdict(True, "ok")


def stale_pointer(d: Determination) -> Determination:
    """Chapter 2's injection: the claims system still points HO-3 (2024-06) policies at 2022-01."""
    return replace(d, form="HO-3 (2022-01)") if d.form == "HO-3 (2024-06)" else d


# ---------------------------------------------------------------- scoring helpers
VERIFY_MEDIAN, VERSION_CHECK, MANUAL_MEDIAN = 1.2, 2.2, 11.0   # Chapter 2 proxy constants, minutes


def lookup_minutes(t: Trace) -> float:
    rng = random.Random(int(hashlib.sha256(t.det.determination_id.encode()).hexdigest()[:12], 16))
    jitter = rng.lognormvariate(0, 0.35)
    if not t.right:
        return MANUAL_MEDIAN * jitter + 0.5
    return VERIFY_MEDIAN * jitter + VERSION_CHECK


def trace_cost(t: Trace, cached: int = 0) -> float:
    return cost_usd(t.config.model, t.answer.tokens_in, t.answer.tokens_out, cached)


def trace_latency(t: Trace, cached: int = 0) -> float:
    return latency_s(t.config.model, t.answer.tokens_in, t.answer.tokens_out, cached)


if CAL_PATH.exists():
    CORPUS, CAL = load()
else:                                  # calibrate.py builds the world before the file exists
    CORPUS, CAL = None, None


# ---------------------------------------------------------------- run helpers
STRATEGIES = {                         # name -> PipelineConfig fields
    "retrieve": dict(policy_filter=True),
    "stuff all": dict(stuff="all", order="cache"),
    "stuff family": dict(stuff="family", order="cache"),
    "hybrid": dict(policy_filter=True, stuff="documents", order="cache"),
}


def pipeline(strategy: str, model: str = "sim-large", **kw) -> Pipeline:
    return Pipeline().with_config(**STRATEGIES[strategy], model=model, **kw)


def run_all(pipe: Pipeline, dets, inject=None) -> list[Trace]:
    """One trace per determination, in order; inject rewrites the record first."""
    return [pipe.run(inject(d) if inject else d) for d in dets]


def cache_pieces(t: Trace) -> list[tuple[str, int]]:
    return [(s.text, s.end - s.start) for s in t.rendered.segments]


def replay_cache(traces: list[Trace]) -> list[int]:
    """Cached input tokens per call, replaying the calls in order through one PromptCache."""
    from contextlab.cache import PromptCache
    cache = PromptCache()
    return [0 if t.answer.rejected else cache.lookup(cache_pieces(t)) for t in traces]


def wrong_count(traces) -> int:
    return sum(not t.right for t in traces)
