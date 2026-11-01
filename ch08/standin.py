"""Chapter 8 stand-in: Lakeshore's reviewed determinations, the documents that govern
them, the v2.2 pipeline with an optional Rulebook span, and SimulatedModel.

The mechanism is real: the governing document is computed from the documents (a
bulletin in force for the state and topic beats an endorsement on the policy, which
beats the form edition); Rulebook tables are resolved and checked against that; the
packet is assembled from the instruction spans, the topic's rule span and the
applicable documents; and the model answers from the packet. What each simulated model
gets wrong without help (its misses and the claims where an example pulls it) is a
calibrated fixture (calibration.json, built by calibrate.py), illustrative rather than
measured.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from contextlab.instructions import V22, Instructions, Span, count_tokens  # noqa: E402
from contextlab.rulebook import Rulebook  # noqa: E402

CAL = json.loads((HERE / "calibration.json").read_text())

# ---------------------------------------------------------------- documents
FORM_SECTIONS = {
    "HO-3": {
        "water_backup": ("Section I Exclusions A.3", 7), "flood": ("Section I Exclusions A.3", 7),
        "food_spoilage": ("Section I Exclusions A.4", 8), "earth_movement": ("Section I Exclusions A.2", 5),
        "mold": ("Section I Exclusions A.6", 8), "personal_property_rc": ("Section I Conditions D.2", 4),
        "roof_settlement": ("Section I Conditions D.3", 4), "roof_matching": ("Section I Conditions D.5", 4),
        "frozen_pipes": ("Section I Perils Insured Against 12", 2),
        "tree_debris": ("Section I Additional Coverages 1.b", 9),
        "hail_cosmetic": ("Section I Exclusions A.11", 3), "wind_hail": ("Section I Perils Insured Against 2", 1),
        "ice_dam": ("Section I Perils Insured Against 13", 2)},
    "PAP": {
        "deer_swerve": ("Part D Definitions", 2), "deer_collision": ("Part D Coverage for Damage to Your Auto", 2),
        "rental_reimbursement": ("Part D Transportation Expenses", 3), "rideshare": ("Part A Exclusions A.5", 1)},
    "BOP": {
        "spoilage": ("Section I Exclusions B.2.k", 12), "employee_dishonesty": ("Section I Exclusions B.2.c", 10),
        "equipment_breakdown": ("Section I Additional Coverages 5.k", 6)},
    "BOAT": {"navigation_limits": ("Declarations Navigation Warranty", 3), "hull_damage": ("Coverage A Hull", 2)},
}
ENDORSEMENT_TOPIC = {
    "HO 04 95": "water_backup", "HO 04 98": "food_spoilage", "HO 04 54": "earth_movement",
    "HO 04 90": "personal_property_rc", "LM HO 17": "roof_settlement", "LM PP 21": "rental_reimbursement",
    "LM PP 34": "rideshare", "LM BP 12": "spoilage", "BP 04 50": "employee_dishonesty", "BT 05": "navigation_limits",
}
BULLETINS = [  # doc_id, state, topic, effective (the day it takes effect)
    ("WI-OCI-2023-07", "WI", "roof_matching", date(2023, 7, 1)),
    ("WI-OCI-2025-03", "WI", "roof_matching", date(2025, 3, 15)),
    ("WI-OCI-2024-11", "WI", "deer_swerve", date(2024, 11, 1)),
    ("IA-ID-2024-08", "IA", "tree_debris", date(2024, 8, 15)),
    ("IL-DOI-2025-05", "IL", "frozen_pipes", date(2025, 5, 1)),
    ("MN-COMM-2026-01", "MN", "hail_cosmetic", date(2026, 1, 1)),
]
# A bulletin's version label is its year and month of issue.
BULLETIN_VERSION = {b[0]: b[0][-7:] for b in BULLETINS}


@dataclass(frozen=True)
class Provision:
    doc_id: str
    version: str
    kind: str          # form | endorsement | bulletin
    section: str
    page: int

    @property
    def citation(self) -> str:
        return f"{self.doc_id} ({self.version}), {self.section}, page {self.page}"

    def __str__(self) -> str:
        return f"{self.citation} ({self.kind})"

    @property
    def doc_label(self) -> str:
        return f"{self.doc_id} ({self.version})"


def provision(doc_id: str, version: str, topic: str, form: str | None = None) -> Provision:
    if doc_id in BULLETIN_VERSION:
        return Provision(doc_id, BULLETIN_VERSION[doc_id], "bulletin", "Requirements", 1)
    if doc_id in ENDORSEMENT_TOPIC:
        return Provision(doc_id, version, "endorsement", "Coverage", 1)
    sec, page = FORM_SECTIONS[doc_id][topic]
    return Provision(doc_id, version, "form", sec, page)


# ---------------------------------------------------------------- determinations
@dataclass
class Determination:
    determination_id: str
    name: str
    state: str
    form_id: str
    form_version: str
    endorsements: list[tuple[str, str]]
    topic: str
    loss: date

    @property
    def n(self) -> int:
        return int(self.determination_id[4:])

    @property
    def form(self) -> str:
        return f"{self.form_id} ({self.form_version})"

    def carries(self, endorsement: str) -> bool:
        """'HO 04 95' matches any edition; 'HO 04 95 (2024-06)' only that edition."""
        return any(endorsement in (e, f"{e} ({v})") for e, v in self.endorsements)


class Corpus:
    def __init__(self, cal: dict = CAL):
        self.determinations = [
            Determination(d["id"], d["name"], d["state"], d["form"], d["form_version"],
                          [tuple(e) for e in d["endorsements"]], d["topic"], date.fromisoformat(d["loss"]))
            for d in cal["determinations"]]
        self.by_id = {d.determination_id: d for d in self.determinations}

    @property
    def eval(self) -> list[Determination]:
        return [d for d in self.determinations if d.n <= 200]

    def bulletin_in_force(self, d: Determination) -> str | None:
        live = [b for b in BULLETINS if b[1] == d.state and b[2] == d.topic and b[3] <= d.loss]
        return max(live, key=lambda b: b[3])[0] if live else None

    def applicable(self, d: Determination) -> list[Provision]:
        """The documents that apply to this policy on this date, for this topic."""
        out = [provision(d.form_id, d.form_version, d.topic)]
        out += [provision(e, v, d.topic) for e, v in d.endorsements if ENDORSEMENT_TOPIC.get(e) == d.topic]
        out += [provision(b[0], "", d.topic) for b in BULLETINS
                if b[1] == d.state and b[2] == d.topic and b[3] <= d.loss]
        return out

    def governing(self, d: Determination) -> Provision:
        """Precedence as the documents state it: bulletin in force > endorsement > form."""
        b = self.bulletin_in_force(d)
        if b:
            return provision(b, "", d.topic)
        for e, v in d.endorsements:
            if ENDORSEMENT_TOPIC.get(e) == d.topic:
                return provision(e, v, d.topic)
        return provision(d.form_id, d.form_version, d.topic)

    def resolve_doc(self, d: Determination, doc_id: str) -> Provision:
        """The provision a rule's doc_id points to for this claim (its edition or version)."""
        if doc_id in BULLETIN_VERSION:
            return provision(doc_id, "", d.topic)
        for e, v in d.endorsements:
            if e == doc_id:
                return provision(e, v, d.topic)
        if doc_id == d.form_id:
            return provision(d.form_id, d.form_version, d.topic)
        ver = {"HO-3": "2024-06", "PAP": "2023-01", "BOP": "2025-01", "BOAT": "2025-03"}.get(doc_id, "")
        if doc_id in ENDORSEMENT_TOPIC:
            return Provision(doc_id, ver or "2024-06", "endorsement", "Coverage", 1)
        return provision(doc_id, ver, d.topic)


CORPUS = Corpus()


# ---------------------------------------------------------------- packet, model, pipeline
@dataclass
class Segment:
    span: Span | None
    text: str


@dataclass
class Rendered:
    segments: list[Segment]
    overhead: int = 0                    # model chat-template tokens

    @property
    def tokens(self) -> int:
        return sum(count_tokens(s.text) for s in self.segments) + self.overhead


@dataclass
class Answer:
    provision: Provision

    @property
    def citation(self) -> str:
        return self.provision.citation


@dataclass(frozen=True)
class PipelineConfig:
    policy_filter: bool = True
    model: str = "sim-large"
    instructions: Instructions = V22
    rulebook: Rulebook | None = None


@dataclass
class Trace:
    config: PipelineConfig
    det: Determination
    rendered: Rendered
    answer: Answer
    gold: Provision

    @property
    def wrong(self) -> bool:
        return self.answer.provision != self.gold


class SimulatedModel:
    """Deterministic stand-in reader.

    Reads the packet: the topic's rule span if there is one, the examples in the
    instructions, and the applicable documents. Behaviour, in order:
      1. a rule whose row fires is followed (the rule names an exception), unless the
         document it names is superseded by a later one of the same kind in the packet;
      2. a rule with no rows states its default as the answer for every claim on its
         topic, and the model follows it (fixture: the claims where it resists);
      3. otherwise the model reads the documents. Its own misses come from the fixture:
         claims where it cites the form over the governing document, claims where an
         example with the same state and topic pulls it to the example's answer (only
         while that example is in the packet), and its other wrong-clause errors.
    """
    OVERHEAD = {"sim-large": 0, "sim-mid": 1, "sim-small": 1}

    def __init__(self, name: str, corpus: Corpus = CORPUS):
        self.name, self.corpus = name, corpus
        p = CAL["models"][name]
        self.miss = set(p["miss"])               # cites the form over the governing document
        self.pull = set(p["pull"])               # follows an example's answer
        self.other = p["other"]                  # id -> [doc, version, section, page]
        self.resist = set(p["resist"])           # ignores a no-row rule

    def own(self, d: Determination, examples) -> Provision:
        gov = self.corpus.governing(d)
        if d.determination_id in self.other:
            doc, ver, sec, page = self.other[d.determination_id]
            return Provision(doc, ver, "form", sec, page)
        if d.determination_id in self.pull:
            ex = [e for e in examples if e["state"] == d.state and e["topic"] == d.topic]
            if ex:
                return self.corpus.resolve_doc(d, ex[0]["cites"])
        if d.determination_id in self.miss:
            return provision(d.form_id, d.form_version, d.topic)
        return gov

    def answer(self, d: Determination, packet: Rendered) -> Answer:
        examples = [s.span.meta["example"] for s in packet.segments if s.span and "example" in s.span.meta]
        rule_spans = [s.span for s in packet.segments if s.span and "rule" in s.span.meta]
        own = self.own(d, examples)
        if rule_spans:
            meta = rule_spans[0].meta["rule"]
            from contextlab.rulebook import Rule, Row
            rule = Rule(meta["rule_id"], meta["topic"], meta["statement"],
                        tuple(Row(**{k: (date.fromisoformat(v) if k in ("on_or_after", "before") else v)
                                     for k, v in r.items()}) for r in meta["rows"]),
                        meta["default"], meta["expert"])
            named = rule.resolve(d)
            if rule.exception(d):
                later = self.corpus.bulletin_in_force(d)
                if named in BULLETIN_VERSION and later and later != named:
                    return Answer(own)                       # superseded in the packet
                return Answer(self.corpus.resolve_doc(d, named))
            if not rule.rows and d.determination_id not in self.resist:
                return Answer(self.corpus.resolve_doc(d, named))
        return Answer(own)


class Pipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig(), corpus: Corpus = CORPUS):
        self.config, self.corpus = config, corpus
        self.model = SimulatedModel(config.model, corpus)

    def packet(self, d: Determination) -> Rendered:
        segs = [Segment(s, s.text) for s in self.config.instructions.spans(d)]
        if self.config.rulebook is not None:
            rs = self.config.rulebook.span(d)
            if rs is not None:
                segs.append(Segment(rs, rs.text))
        docs = self.corpus.applicable(d) if self.config.policy_filter else []
        for i, p in enumerate(docs):
            sp = Span(f"S{i + 1}", f"[{p.doc_label}] {p.section}, page {p.page}", {"kind": "evidence"})
            segs.append(Segment(sp, sp.text))
        # the retrieved passages' text, whose length is a per-claim fixture
        segs.append(Segment(None, "x " * CAL["evidence_tokens"][d.determination_id]))
        return Rendered(segs, SimulatedModel.OVERHEAD[self.config.model])

    def run(self, d: Determination) -> Trace:
        r = self.packet(d)
        return Trace(self.config, d, r, self.model.answer(d, r), self.corpus.governing(d))


def run_set(cfg: PipelineConfig, dets) -> list[Trace]:
    p = Pipeline(cfg)
    return [p.run(d) for d in dets]
