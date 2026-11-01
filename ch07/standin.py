"""Chapter 7 stand-in: Chapter 6's world (04-code/ch06/standin.py) extended with seeds,
instruction clashes and adherence measurements. The determinations, documents and model
fixtures start from Chapter 6's calibration; ch07/calibrate.py records the adjustments.

Added for this chapter:
  - a seed per run. Seed 2026 is Chapter 6's run: Chapter 6's printed results hold (test_ch07.py).
  - clashes. A skill annotated `negates=(rule,)` competes with that rule whenever it is
    loaded. The stand-in settles the competition with a recorded draw u per model, seed,
    claim and skill: without the hierarchy rule the skill (the later instruction) wins when
    u < ModelProfile.recency (0.75); with it, the skill wins only when
    u >= ModelProfile.hierarchy (0.96). When the
    skill wins, the rule is dropped for that answer. A skill that only leans towards the
    form (`prefers="form"`, no negation) moves a model with `lean` > 0 when u >= 1 - lean.
  - where a dropped precedence rule matters: the claims the model gets wrong without the
    rule ('noprec', Chapter 6) and the claims a skill preferring the form tips over
    ('skill_pull', this chapter's fixture).
  - unsupported additions per seed (Chapter 6's are seed 2026's), and the gold span's rank
    in the retriever's list, from which hit@5 is computed.

What is real: the governing document is computed from the documents (a bulletin in force
for the state and topic on the date of loss beats an endorsement on the policy, which beats
the form edition); the packet is assembled from the instruction spans (rules, the skills
loaded for the claim's topic, the examples) and the applicable evidence; tokens are counted;
the fingerprint is the sha256 of the rendered instructions; lint resolves each example's
source determination against the documents.

What is calibrated (calibration.json, built by calibrate.py): which claims each stand-in
model gets wrong without a given rule, how strongly each claim resists an example's pull,
which answers carry an unsupported addition, and the length of the retrieved passages.
These fixtures are illustrative, chosen so that the lab reproduces the chapter; they are not
measurements of any production model.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from contextlab.instructions import V22, Example, Instructions, Span, count_tokens  # noqa: E402

CAL = json.loads((HERE / "calibration.json").read_text())
SEED = 2026                 # Chapter 6's run

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
        "ice_dam": ("Section I Perils Insured Against 13", 2),
        "dog_bite": ("Section II Coverage E", 17), "wear_tear": ("Section I Exclusions A.9", 8),
        "snow_load": ("Section I Perils Insured Against 8", 2)},
    "PAP": {
        "deer_swerve": ("Part D Definitions", 2), "deer_collision": ("Part D Coverage for Damage to Your Auto", 2),
        "rental_reimbursement": ("Part D Transportation Expenses", 3), "rideshare": ("Part A Exclusions A.5", 1)},
    "BOP": {
        "spoilage": ("Section I Exclusions B.2.k", 12), "employee_dishonesty": ("Section I Exclusions B.2.c", 10),
        "equipment_breakdown": ("Section I Additional Coverages 5.k", 6)},
    "BOAT": {"navigation_limits": ("Declarations Navigation Warranty", 3), "hull_damage": ("Coverage A Hull", 2)},
}
EDITIONS = {"HO-3": ["2019-03", "2022-01", "2024-06"], "PAP": ["2018-09", "2023-01"],
            "BOAT": ["2021-05", "2025-03"], "BOP": ["2020-04", "2025-01"]}
ENDORSEMENT_TOPIC = {
    "HO 04 95": "water_backup", "HO 04 98": "food_spoilage", "HO 04 54": "earth_movement",
    "HO 04 90": "personal_property_rc", "LM HO 17": "roof_settlement", "LM PP 21": "rental_reimbursement",
    "LM PP 34": "rideshare", "LM BP 12": "spoilage", "BP 04 50": "employee_dishonesty", "BT 05": "navigation_limits",
}
BULLETINS = [  # doc_id, state, topic, effective
    ("WI-OCI-2023-07", "WI", "roof_matching", date(2023, 7, 1)),
    ("WI-OCI-2025-03", "WI", "roof_matching", date(2025, 3, 15)),
    ("WI-OCI-2024-11", "WI", "deer_swerve", date(2024, 11, 1)),
    ("IA-ID-2024-08", "IA", "tree_debris", date(2024, 8, 15)),
    ("IL-DOI-2025-05", "IL", "frozen_pipes", date(2025, 5, 1)),
    ("MN-COMM-2026-01", "MN", "hail_cosmetic", date(2026, 1, 1)),
]
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

    @property
    def doc_label(self) -> str:
        return f"{self.doc_id} ({self.version})"

    def __str__(self) -> str:
        return f"{self.citation} ({self.kind})"


def provision(doc_id: str, version: str, topic: str) -> Provision:
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

    @property
    def facts(self) -> tuple:
        """The facts that decide which document governs, apart from the date."""
        return (self.topic, self.state, self.form, tuple(sorted(self.endorsements)))


class Corpus:
    def __init__(self, cal: dict = CAL):
        self.determinations = [
            Determination(d["id"], d["name"], d["state"], d["form"], d["form_version"],
                          [tuple(e) for e in d["endorsements"]], d["topic"], date.fromisoformat(d["loss"]))
            for d in cal["determinations"]]
        self.by_id = {d.determination_id: d for d in self.determinations}
        self.editions = EDITIONS

    @property
    def eval(self) -> list[Determination]:
        return [d for d in self.determinations if d.n <= 200]

    @property
    def holdout(self) -> list[Determination]:
        return [d for d in self.determinations if d.n > 200]

    def hard(self) -> list[Determination]:
        """The gate's hold-out: claims 201-500 governed by a bulletin or an endorsement."""
        return [d for d in self.holdout if self.governing(d).kind != "form"]

    def bulletin_in_force(self, d: Determination) -> str | None:
        live = [b for b in BULLETINS if b[1] == d.state and b[2] == d.topic and b[3] <= d.loss]
        return max(live, key=lambda b: b[3])[0] if live else None

    def applicable(self, d: Determination) -> list[Provision]:
        out = [provision(d.form_id, d.form_version, d.topic)]
        out += [provision(e, v, d.topic) for e, v in d.endorsements if ENDORSEMENT_TOPIC.get(e) == d.topic]
        out += [provision(b[0], "", d.topic) for b in BULLETINS
                if b[1] == d.state and b[2] == d.topic and b[3] <= d.loss]
        return out

    def inapplicable(self, d: Determination) -> list[Provision]:
        """What the policy filter removes: other editions of the form, endorsements for the
        topic that the policy does not carry, bulletins for other states or not yet in force."""
        out = [provision(d.form_id, v, d.topic) for v in EDITIONS[d.form_id] if v != d.form_version]
        carried = {e for e, _ in d.endorsements}
        out += [Provision(e, "2024-06", "endorsement", "Coverage", 1) for e, t in ENDORSEMENT_TOPIC.items()
                if t == d.topic and e not in carried]
        out += [provision(b[0], "", d.topic) for b in BULLETINS
                if b[2] == d.topic and (b[1] != d.state or b[3] > d.loss)]
        return out

    def governing(self, d: Determination) -> Provision:
        b = self.bulletin_in_force(d)
        if b:
            return provision(b, "", d.topic)
        for e, v in d.endorsements:
            if ENDORSEMENT_TOPIC.get(e) == d.topic:
                return provision(e, v, d.topic)
        return provision(d.form_id, d.form_version, d.topic)

    def resolve_doc(self, d: Determination, doc_id: str) -> Provision:
        """The provision a cited doc_id points to for this claim."""
        if doc_id in BULLETIN_VERSION:
            return provision(doc_id, "", d.topic)
        for e, v in d.endorsements:
            if e == doc_id:
                return provision(e, v, d.topic)
        if doc_id == d.form_id:
            return provision(d.form_id, d.form_version, d.topic)
        if doc_id in ENDORSEMENT_TOPIC:
            return Provision(doc_id, "2024-06", "endorsement", "Coverage", 1)
        return provision(doc_id, EDITIONS[doc_id][-1], d.topic)


CORPUS = Corpus()


def lint(examples, corpus: Corpus = CORPUS) -> list[str]:
    """Does each example's cited provision govern its own source determination?"""
    out = []
    for e in examples:
        src = corpus.by_id[e.source]
        cited = e.text.rsplit("Governing provision: ", 1)[1].rstrip(".")
        gov = corpus.governing(src)
        if cited != gov.citation:
            out.append(f"{e.example_id}: cites {cited}, but {gov} governs a {src.state} claim on {src.loss}")
    return out


# ---------------------------------------------------------------- answers
BODY = {
    "deer_swerve": ("If your covered auto leaves the road or strikes another object while you avoid an animal, "
                    "and there is no contact with the animal, the loss is collision and the collision "
                    "deductible applies."),
    "deer_collision": "Loss caused by contact with a bird or animal is not considered collision.",
}
GENERIC = {"form": "The policy form itself answers this question for the loss as described in the claim file.",
           "endorsement": ("The endorsement on this policy changes the form's answer for this loss, "
                           "so the endorsement governs here."),
           "bulletin": ("The state bulletin in force on the date of loss sets the answer for this loss, "
                        "ahead of the policy form.")}
PLAIN_OPENING = "From the documents, "   # with no rules at all, the stand-in opens with a stock phrase


@dataclass
class Answer:
    provision: Provision
    span_index: int                      # which evidence span it cites, [S1], [S2] ...
    addition: str | None = None          # a sentence the cited provision does not support
    version_check: bool = False          # cite_exact: the answer states the version was checked
    plain: bool = False                  # no rules in the instructions
    dropped: tuple = ()                  # rules a clashing skill displaced for this answer

    @property
    def citation(self) -> str:
        return self.provision.citation

    def text(self, topic: str) -> str:
        body = BODY.get(topic)
        if body is None:
            body = GENERIC[self.provision.kind]
            if self.plain:
                body = PLAIN_OPENING + body[0].lower() + body[1:]
        parts = [body]
        if self.addition:
            parts.append(self.addition)
        parts.append(f"Governing provision: {self.citation} [S{self.span_index}].")
        if self.version_check:
            parts.append(f"Version {self.provision.doc_label} checked against the date of loss.")
        return " ".join(parts)


# ---------------------------------------------------------------- packet, models, pipeline
@dataclass(frozen=True)
class PipelineConfig:
    model: str = "sim-large"
    instructions: Instructions = V22
    policy_filter: bool = True
    examples: tuple | None = None        # per-claim example override: None = the instructions' own
    seed: int = SEED


@dataclass
class Packet:
    spans: list[Span]
    evidence: list[Provision]
    overhead: int

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.spans) + self.overhead

    @property
    def instruction_tokens(self) -> int:
        return sum(s.tokens for s in self.spans if s.meta.get("kind") == "instruction")

    def examples(self) -> list[dict]:
        return [s.meta["example"] for s in self.spans if "example" in s.meta]

    def skills(self) -> list[str]:
        return [s.meta["skill"] for s in self.spans if "skill" in s.meta]


@dataclass(frozen=True)
class ModelProfile:
    name: str
    imitation: float = 0.45     # pull of an on-topic example; a quarter of it for the same kind of document
    quote_only: float = 0.25    # an unsupported addition survives quote_only if its strength > 1 - this
    overhead: int = 0           # chat-template tokens
    price_in: float = 3.00      # US$ per million tokens (illustrative)
    price_out: float = 15.00
    recency: float = 0.75       # Chapter 7: without a hierarchy rule, the later (clashing) instruction wins this often
    hierarchy: float = 0.96     # Chapter 7: with the hierarchy rule, the higher tier wins this often
    lean: float = 0.0           # pull of a skill that prefers the form without countermanding a rule


PROFILES = {
    "sim-large": ModelProfile("sim-large", overhead=0, price_in=3.00, price_out=15.00),
    "sim-mid": ModelProfile("sim-mid", overhead=1, price_in=1.00, price_out=5.00),
    "sim-small": ModelProfile("sim-small", overhead=1, price_in=0.25, price_out=1.25, lean=0.01),
}


class SimulatedModel:
    """Deterministic stand-in reader. For each claim, in order:
      1. it can only cite what is in the packet (a retrieval miss leaves the governing span out);
      2. an example on the claim's topic pulls it to the example's document when
         imitation x 1.0 exceeds the claim's resistance; without the precedence rule, any
         example pulls at a quarter of that towards the same kind of document (the form);
      3. without the precedence rule it cites the amended form on its 'noprec' claims, and
         with the rule on its 'prec' claims; the small model also falls back to the form on
         its 'noskill' claims when no skill span is loaded;
      4. with the policy filter off, it is misled by an inapplicable document on its 'offapp'
         claims, fewer of them when the applicability rule is present;
      5. otherwise it cites the governing provision.
    Chapter 7 adds, before step 3: a loaded skill that countermands a rule may displace it
    (see the module docstring), and a dropped precedence rule sends the 'noprec' and
    'skill_pull' claims to the form.
    """

    def __init__(self, name: str, corpus: Corpus = CORPUS, seed: int = SEED):
        self.name, self.corpus, self.profile, self.seed = name, corpus, PROFILES[name], seed
        p = CAL["models"][name]
        self.skill_pull = set(p.get("skill_pull", []))
        self.noprec, self.prec = set(p["noprec"]), set(p["prec"])
        self.noskill = set(p.get("noskill", []))
        self.offapp_rule, self.offapp_norule = set(p["offapp_rule"]), set(p["offapp_norule"])
        self.resistance = p["resistance"]          # default 1.0: no example moves it
        self.additions = dict(p["additions"])      # id -> [sentence, strength]
        if seed != SEED:                           # other seeds: the water claims' draws
            water = CAL["water"]
            self.additions = {i: a for i, a in self.additions.items() if i not in water}
            self.additions.update(CAL["seed_additions"][name].get(str(seed), {}))

    def draw(self, d: Determination, skill: str) -> float:
        """The recorded draw for this model, seed, claim and skill (a hash where none is recorded)."""
        rec = CAL["draws"].get(self.name, {}).get(skill, {}).get(str(self.seed), {})
        if d.determination_id in rec:
            return rec[d.determination_id]
        h = hashlib.sha256(f"{self.name}|{skill}|{self.seed}|{d.determination_id}".encode()).hexdigest()
        return int(h[:8], 16) / 2 ** 32

    def settle(self, d: Determination, rules: tuple, skills) -> tuple[tuple, tuple, bool]:
        """Rules in force after any clash, the rules dropped, and whether a leaning skill moved it."""
        dropped, leaned = [], False
        for sk in skills:
            u = self.draw(d, sk.name)
            for r in getattr(sk, "negates", ()):
                if r in rules and r not in dropped:
                    wins = u >= self.profile.hierarchy if "hierarchy" in rules else u < self.profile.recency
                    if wins:
                        dropped.append(r)
            if getattr(sk, "prefers", "") == "form" and not getattr(sk, "negates", ()) and \
                    self.profile.lean and u >= 1 - self.profile.lean:
                leaned = True
        return tuple(r for r in rules if r not in dropped), tuple(dropped), leaned

    def answer(self, d: Determination, packet: Packet, rules: tuple, skills=()) -> Answer:
        rules, dropped, leaned = self.settle(d, rules, skills)
        ev = packet.evidence
        gov = self.corpus.governing(d)
        form = provision(d.form_id, d.form_version, d.topic)
        choice = gov
        if gov not in ev:                                    # 1. retrieval miss
            choice = [p for p in ev if p.doc_id == d.form_id][0]
        else:
            res = self.resistance.get(d.determination_id, 1.0)
            on_topic = [e for e in packet.examples() if e["topic"] == d.topic]
            pulled = None
            if on_topic and self.profile.imitation > res:   # 2. on-topic pull
                pulled = self.corpus.resolve_doc(d, on_topic[0]["cites"])
            elif packet.examples() and "precedence" not in rules and self.profile.imitation / 4 > res:
                pulled = form                                # weak pull towards the form
            if pulled is not None:
                choice = pulled
            elif "precedence" in dropped and d.determination_id in self.skill_pull | self.noprec:
                choice = form                                # 3a. the clashing skill won
            elif leaned and d.determination_id in self.skill_pull | self.noprec:
                choice = form                                # 3b. a leaning skill moved it
            elif d.determination_id in (self.prec if "precedence" in rules else self.noprec):
                choice = form                                # 3. the amended document
            elif d.determination_id in self.noskill and not packet.skills():
                choice = form
            else:                                            # 4. filter off
                stray = [p for p in ev if p in self.corpus.inapplicable(d)]
                misled = self.offapp_rule if "applicability" in rules else self.offapp_norule
                if stray and d.determination_id in misled:
                    choice = stray[0]
        add = self.additions.get(d.determination_id)
        addition = None
        if add and ("quote_only" not in rules or add[1] > 1 - self.profile.quote_only):
            addition = add[0]
        idx = ev.index(choice) + 1 if choice in ev else 1
        return Answer(choice, idx, addition, "cite_exact" in rules, not rules, dropped)


@dataclass
class Trace:
    config: PipelineConfig
    det: Determination
    packet: Packet
    answer: Answer
    gold: Provision
    fingerprint: str

    @property
    def wrong(self) -> bool:
        return self.answer.provision != self.gold

    @property
    def gold_in_packet(self) -> bool:
        return self.gold in self.packet.evidence

    @property
    def answer_text(self) -> str:
        return self.answer.text(self.det.topic)

    @property
    def output_tokens(self) -> int:
        return count_tokens(self.answer_text)

    @property
    def lookup_minutes(self) -> float:
        """Lookup proxy (illustrative constants): 1.3 minutes to open the cited page, plus
        2.2 minutes to check the version against the date of loss unless the answer did."""
        return 1.3 + (0.0 if self.answer.version_check else 2.2)


TOP_K = 10   # spans of evidence per packet


class Pipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig(), corpus: Corpus = CORPUS):
        self.config, self.corpus = config, corpus
        self.model = SimulatedModel(config.model, corpus, config.seed)

    def evidence(self, d: Determination) -> list[Provision]:
        miss = CAL["retrieval_miss"].get(d.determination_id)
        ev = self.corpus.applicable(d)
        if miss:  # the governing section was not retrieved; a neighbouring section was
            ev = [Provision(d.form_id, d.form_version, "form", miss[0], miss[1]) if p.kind == "form" else p
                  for p in ev]
        if not self.config.policy_filter:
            ev = ev + self.corpus.inapplicable(d)
        # the rest of the top ten: neighbouring sections of the same form, about other topics
        for topic, (sec, page) in sorted(FORM_SECTIONS[d.form_id].items()):
            if len(ev) >= TOP_K:
                break
            p = Provision(d.form_id, d.form_version, "form", sec, page)
            if topic != d.topic and p not in ev:
                ev.append(p)
        return ev

    def packet(self, d: Determination) -> Packet:
        ins = self.config.instructions
        if self.config.examples is not None:
            ins = ins.bump(ins.version, examples=tuple(self.config.examples.get(d.determination_id, ())))
        spans = list(ins.spans(d))
        if ins.rules and {"precedence", "applicability"} & set(ins.rules):
            spans.append(Span("FACTS", f"Claim facts: state {d.state}; date of loss {d.loss}.", {"kind": "facts"}))
        ev = self.evidence(d)
        for i, p in enumerate(ev):
            spans.append(Span(f"S{i + 1}", f"[{p.doc_label}] {p.section}, page {p.page}", {"kind": "evidence"}))
        spans.append(Span("PASSAGES", "x " * CAL["evidence_tokens"][d.determination_id], {"kind": "passages"}))
        return Packet(spans, ev, PROFILES[self.config.model].overhead)

    def run(self, d: Determination) -> Trace:
        pk = self.packet(d)
        rules = self.config.instructions.rules
        skills = self.config.instructions.loaded_skills(d)
        return Trace(self.config, d, pk, self.model.answer(d, pk, rules, skills), self.corpus.governing(d),
                     self.config.instructions.fingerprint())


def run_set(cfg: PipelineConfig, dets) -> list[Trace]:
    p = Pipeline(cfg)
    return [p.run(d) for d in dets]


# ---------------------------------------------------------------- metrics
def wrong(traces) -> int:
    return sum(t.wrong for t in traces)


def rate(traces) -> float:
    return 100.0 * wrong(traces) / len(traces) if traces else 0.0


def unsupported(traces) -> int:
    return sum(t.answer.addition is not None for t in traces)


def faithfulness(traces) -> float:
    """Share of checked statements supported by the cited span. Each answer has two: its
    statement of what the provision says (fails if it adds a sentence) and its citation."""
    return 1 - unsupported(traces) / (2 * len(traces))


def mean_tokens(traces) -> float:
    return statistics.fmean(t.packet.tokens for t in traces)


def instruction_tokens(traces) -> float:
    return statistics.fmean(t.packet.instruction_tokens for t in traces)


def cost(traces) -> float:
    p = PROFILES[traces[0].config.model]
    return statistics.fmean(t.packet.tokens * p.price_in / 1e6 + t.output_tokens * p.price_out / 1e6
                            for t in traces)


def hit_at_5(traces) -> float:
    """Share of claims whose governing span is in the retriever's top five (rank recorded per
    claim in calibration.json; Chapter 5's retriever, before the policy filter)."""
    return sum(CAL["gold_rank"][t.det.determination_id] <= 5 for t in traces) / len(traces)


def lookup(traces) -> float:
    return statistics.median(t.lookup_minutes for t in traces)


# ---------------------------------------------------------------- examples from the reviewed pool
QUESTION = "Question: Is this {t} loss covered, and which policy document governs?"


def example_from(d: Determination, example_id: str, cited: Provision | None = None,
                 corpus: Corpus = CORPUS) -> Example:
    """A worked example built from a reviewed determination: its facts and its governing provision
    (or, to inject a failure, another provision)."""
    cited = cited or corpus.governing(d)
    ends = "; ".join(f"{e} ({v})" for e, v in d.endorsements) or "none"
    text = (f"Example {example_id}. {QUESTION.format(t=d.topic.replace('_', ' '))} Facts: form {d.form}; "
            f"endorsements {ends}; state {d.state}; date of loss {d.loss}. "
            f"Answer: Governing provision: {cited.citation}.")
    return Example(example_id, text, d.determination_id, d.state, d.topic, str(d.loss), cited.doc_id)


def pool(corpus: Corpus = CORPUS) -> list[Example]:
    """The 34 reviewed hold-out determinations available as examples, each citing what governs it."""
    return [example_from(corpus.by_id[i], i) for i in CAL["pool"]]


def _nearest(d: Determination, exs: list[Example]) -> Example:
    return min(exs, key=lambda e: (abs((date.fromisoformat(e.loss) - d.loss).days), e.source))


def select_same_topic(d: Determination, exs: list[Example]) -> tuple:
    same = [e for e in exs if e.topic == d.topic]
    return (_nearest(d, same),) if same else ()


def select_facts(d: Determination, exs: list[Example], corpus: Corpus = CORPUS) -> tuple:
    """Same topic, state, form edition and endorsement set as the claim; no example if none matches."""
    same = [e for e in exs if corpus.by_id[e.source].facts == d.facts]
    return (_nearest(d, same),) if same else ()
