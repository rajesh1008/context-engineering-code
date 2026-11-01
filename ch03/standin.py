"""Chapter 3 stand-in: Lakeshore's synthetic corpus (378 spans), 200 reviewed
determinations, the Chapter 2 pipeline (retrieve, applicability filter, assemble), the
four render formats, and SimulatedModel.

STAND-IN NOTICE. Nothing here is a language model or a real search engine.

What is real: the corpus text and its token counts (contextlab's counter, and tiktoken
when installed); the governing provision, computed from the documents (a bulletin in force
for the state and topic on the date of loss beats an endorsement on the policy, which
beats the form edition); the applicability filter; packet assembly, padding, ordering and
rendering; the position weight (contextlab.llm); the prompt cache (contextlab.cache); the
prices and latency model (contextlab.tokens); the cache identity hashes.

What is calibrated (calibration.json, built by calibrate.py): the retriever's candidate
lists and passage lengths, and which packets each stand-in model misreads, keyed by the
determination and the exact sequence of spans it was shown. When the model misreads, it
cites the rival with the highest relevance times position weight. These fixtures are
illustrative, chosen so that the lab reproduces the chapter; they are not measurements of
any production model.
"""
from __future__ import annotations

import functools
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from contextlab.cache import PromptCache  # noqa: E402,F401
from contextlab.instructions import V22, count_tokens  # noqa: E402
from contextlab.llm import PROFILES, position_weight  # noqa: E402
from contextlab.tokens import cost_usd, latency_s  # noqa: E402

_CALF = HERE / "calibration.json"
CAL = json.loads(_CALF.read_text()) if _CALF.exists() else {}
CUT = date(2024, 5, 31)
OUT_TOKENS = 57            # answer with citation (calibrated answer length)
OUT_EXTRA_WRONG = 1        # a wrong citation adds a token of hedging (calibrated)
RECENCY = 0.75             # Chapter 7: with no hierarchy rule, the later (clashing) instruction wins this often
HIERARCHY = 0.96           # Chapter 7: with the hierarchy rule, the higher tier wins this often
CLIENT_S = 0.03            # assembly and network overhead per call, seconds (illustrative)


def rng(*key) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, ("ch03",) + key)).encode()).hexdigest()[:12], 16))


# ---------------------------------------------------------------- documents
STATES = ["WI", "IL", "MI", "IA", "MN", "IN"]
FORM_SECTIONS = {
    "HO-3": {
        "water_backup": ("Section I Exclusions A.3", 7), "flood": ("Section I Exclusions A.3", 7),
        "food_spoilage": ("Section I Exclusions A.4", 8), "earth_movement": ("Section I Exclusions A.2", 5),
        "mold": ("Section I Exclusions A.6", 8), "personal_property_rc": ("Section I Conditions D.2", 4),
        "roof_settlement": ("Section I Conditions D.3", 4), "roof_matching": ("Section I Conditions D.5", 4),
        "frozen_pipes": ("Section I Perils Insured Against 12", 2),
        "tree_debris": ("Section I Additional Coverages 1.b", 9),
        "hail_cosmetic": ("Section I Exclusions A.11", 3), "wind_hail": ("Section I Perils Insured Against 2", 1),
        "ice_dam": ("Section I Perils Insured Against 13", 2), "dog_bite": ("Section II Coverage E", 17),
        "wear_tear": ("Section I Exclusions A.9", 8), "snow_load": ("Section I Perils Insured Against 8", 2)},
    "PAP": {
        "deer_swerve": ("Part D Definitions", 2), "deer_collision": ("Part D Coverage for Damage to Your Auto", 2),
        "rental_reimbursement": ("Part D Transportation Expenses", 3), "rideshare": ("Part A Exclusions A.5", 1)},
    "BOP": {
        "spoilage": ("Section I Exclusions B.2.k", 12), "employee_dishonesty": ("Section I Exclusions B.2.c", 10),
        "equipment_breakdown": ("Section I Additional Coverages 5.k", 6)},
    "BOAT": {"navigation_limits": ("Declarations Navigation Warranty", 3), "hull_damage": ("Coverage A Hull", 2)},
}
TOPIC_FAMILY = {t: f for f, ts in FORM_SECTIONS.items() for t in ts}
EDITIONS = {"HO-3": ["2019-03", "2022-01", "2024-06"], "PAP": ["2018-09", "2023-01"],
            "BOAT": ["2021-05", "2025-03"], "BOP": ["2020-04", "2025-01"]}
ENDORSEMENTS = [  # code, family, topic, editions
    ("HO 04 95", "HO-3", "water_backup", ["2019-03", "2024-06"]), ("HO 04 98", "HO-3", "food_spoilage", ["2024-06"]),
    ("HO 04 54", "HO-3", "earth_movement", ["2022-01"]), ("HO 04 90", "HO-3", "personal_property_rc", ["2022-01"]),
    ("LM HO 17", "HO-3", "roof_settlement", ["2024-06"]), ("LM PP 21", "PAP", "rental_reimbursement", ["2023-01"]),
    ("LM PP 34", "PAP", "rideshare", ["2023-01"]), ("LM BP 12", "BOP", "spoilage", ["2025-01"]),
    ("BP 04 50", "BOP", "employee_dishonesty", ["2020-04"]), ("BT 05", "BOAT", "navigation_limits", ["2025-03"]),
]
END_SECTIONS = ["Coverage", "Schedule", "Limit of Liability", "Conditions", "Definitions"]
BULLETINS = [  # doc_id, state, topic, family, effective, ends
    ("WI-OCI-2023-07", "WI", "roof_matching", "HO-3", date(2023, 7, 1), date(2025, 3, 14)),
    ("WI-OCI-2025-03", "WI", "roof_matching", "HO-3", date(2025, 3, 15), None),
    ("WI-OCI-2024-11", "WI", "deer_swerve", "PAP", date(2024, 11, 1), None),
    ("IA-ID-2024-08", "IA", "tree_debris", "HO-3", date(2024, 8, 15), None),
    ("IN-DOI-2025-02", "IN", "wind_hail", "HO-3", date(2025, 2, 1), None),
    ("IL-DOI-2025-05", "IL", "frozen_pipes", "HO-3", date(2025, 5, 1), None),
    ("MI-DIFS-2023-09", "MI", "roof_settlement", "HO-3", date(2023, 9, 1), date(2025, 7, 31)),
    ("MI-DIFS-2025-08", "MI", "roof_settlement", "HO-3", date(2025, 8, 1), None),
    ("MN-COMM-2026-01", "MN", "hail_cosmetic", "HO-3", date(2026, 1, 1), None),
]
BULLETIN_SECTIONS = ["Requirements", "Scope", "Effective Date"]
GENERAL = ["Definitions", "Conditions", "Declarations", "Loss Settlement", "Duties After Loss", "Other Insurance",
           "Policy Period", "Concealment or Fraud", "Appraisal", "Mortgage Clause", "Cancellation", "Nonrenewal",
           "Subrogation", "Suit Against Us", "Liberalization", "Waiver or Change", "Assignment", "Death",
           "Loss Payment", "Abandonment", "Recovered Property", "Our Option", "Mortgagee Interests",
           "Insurable Interest", "Pair or Set", "Glass Replacement", "Volcanic Eruption", "Fraud",
           "No Benefit", "Nuclear Hazard Clause", "Concealment", "Inspections and Surveys",
           "Examination", "Premiums"]
N_GENERAL = {("HO-3", "2019-03"): 34, ("HO-3", "2022-01"): 34, ("HO-3", "2024-06"): 15, ("PAP", "2018-09"): 33,
             ("PAP", "2023-01"): 33, ("BOAT", "2021-05"): 33, ("BOAT", "2025-03"): 8, ("BOP", "2020-04"): 33,
             ("BOP", "2025-01"): 10}

HAND_TEXT = {  # the four spans Listing 3.4 embeds
    "HO 04 95 (2019-03)|Coverage": (
        "We cover direct physical loss to property covered under Section I caused by water or waterborne "
        "material which backs up through sewers or drains, or overflows or is discharged from a sump, sump "
        "pump or related equipment. The most we pay for this coverage is $5,000 in any one loss."),
    "HO 04 95 (2024-06)|Coverage": (
        "We cover direct physical loss to property covered under Section I caused by water or waterborne "
        "material which backs up through sewers or drains, or overflows or is discharged from a sump, sump "
        "pump or related equipment, even if caused by mechanical breakdown of the sump pump. The most we pay "
        "for this coverage is $10,000 in any one loss, after a $1,000 deductible. Coverage does not apply if "
        "the loss results from your failure to maintain the sump pump."),
    "HO 04 95 (2019-03)|Schedule": (
        "Schedule. Water Back-Up and Sump Discharge or Overflow. Limit of insurance $5,000. Deductible as "
        "shown in the Declarations. Premium charged at inception."),
    "HO 04 95 (2024-06)|Schedule": (
        "Schedule. Water Back-Up and Sump Discharge or Overflow. Limit of insurance $10,000. Deductible "
        "$1,000. Premium charged at inception."),
}
VOCAB = ("the insured policy coverage loss property dwelling premises damage claim we you any one occurrence "
         "limit deductible amount payment written notice condition endorsement section applies except caused by "
         "direct physical including resulting from shall must within days after date of value replacement cost "
         "actual cash repair vehicle vessel business personal structure other premises described declarations "
         "agreement provision exclusion schedule state law requires").split()


def filler(key: str, n: int) -> str:
    """Deterministic text of exactly n contextlab tokens (words and full stops)."""
    r, out, used, sent = rng("text", key), [], 0, 0
    while used < n:
        if n - used >= 2 and sent >= r.randint(7, 14):
            out[-1] += "."
            used, sent = used + 1, 0
            continue
        out.append(r.choice(VOCAB))
        used, sent = used + 1, sent + 1
    txt = " ".join(out)
    txt = txt[0].upper() + txt[1:]
    assert count_tokens(txt) == n, (key, n)
    return txt


@dataclass(frozen=True)
class Provenance:
    doc_id: str
    label: str                     # e.g. "HO 04 95 (2024-06)"
    version: str
    kind: str                      # form | endorsement | bulletin
    family: str
    jurisdiction: str = "ALL"
    effective_from: date | None = None
    effective_to: date | None = None

    def in_force(self, d: date) -> bool:
        return (self.effective_from is None or self.effective_from <= d) and (
            self.effective_to is None or d <= self.effective_to)


@dataclass(frozen=True, eq=False)
class CSpan:
    span_id: str
    provenance: Provenance
    section: str
    page: int
    topic: str
    text: str = ""

    @property
    def kind(self) -> str:
        return self.provenance.kind

    @property
    def label(self) -> str:
        return f"{self.provenance.label}, {self.section}, page {self.page}"

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


def _ed_date(ed: str) -> date:
    y, m = ed.split("-")
    return date(int(y), int(m), 1)


def corpus_skeleton() -> list[tuple]:
    """(span_id, Provenance, section, page, topic) for all 378 spans."""
    out = []
    for fam, eds in EDITIONS.items():
        for i, ed in enumerate(eds):
            lab = f"{fam} ({ed})"
            nxt = _ed_date(eds[i + 1]) if i + 1 < len(eds) else None
            prov = Provenance(fam, lab, ed, "form", fam, effective_from=_ed_date(ed),
                              effective_to=date.fromordinal(nxt.toordinal() - 1) if nxt else None)
            for t, (sec, pg) in FORM_SECTIONS[fam].items():
                if t == "flood":
                    continue
                out.append((f"{lab}|{t}", prov, sec, pg, t))
            for g in GENERAL[:N_GENERAL[fam, ed]]:
                out.append((f"{lab}|{g}", prov, g, 20 + GENERAL.index(g), "general"))
    for code, fam, t, eds in ENDORSEMENTS:
        for i, ed in enumerate(eds):
            lab = f"{code} ({ed})"
            nxt = _ed_date(eds[i + 1]) if i + 1 < len(eds) else None
            prov = Provenance(code, lab, ed, "endorsement", fam, effective_from=_ed_date(ed),
                              effective_to=date.fromordinal(nxt.toordinal() - 1) if nxt else None)
            for n, sec in enumerate(END_SECTIONS):
                out.append((f"{lab}|{sec}", prov, sec, 1 + n // 2, t if sec in ("Coverage", "Schedule") else "general"))
    for doc, st, t, fam, vf, vt in BULLETINS:
        prov = Provenance(doc, f"{doc} ({doc[-7:]})", doc[-7:], "bulletin", fam, st, vf, vt)
        for n, sec in enumerate(BULLETIN_SECTIONS):
            out.append((f"{doc}|{sec}", prov, sec, 1 + n, t if n == 0 else "general"))
    return out


class Corpus:
    def __init__(self, sizes: dict | None = None):
        sizes = sizes or CAL.get("sizes", {})
        self.spans = []
        for sid, prov, sec, pg, t in corpus_skeleton():
            text = HAND_TEXT.get(sid) or filler(sid, sizes.get(sid, 50))
            self.spans.append(CSpan(sid, prov, sec, pg, t, text))
        self.tenant = "lakeshore"

    @property
    def by_id(self) -> dict:
        return {s.span_id: s for s in self.spans}

    def version(self) -> str:
        h = hashlib.sha256()
        for s in sorted(self.spans, key=lambda s: s.span_id):
            h.update(f"{s.span_id}|{s.provenance.version}|{s.text}\n".encode())
        return h.hexdigest()[:12]

    def governing(self, d) -> CSpan:
        by = self.by_id
        for doc, st, t, fam, vf, vt in BULLETINS:
            s = by.get(f"{doc}|Requirements")
            if s and st == d.state and t == d.topic and s.provenance.in_force(d.date_of_loss):
                return s
        for e in d.endorsements:
            s = by.get(f"{e}|Coverage")
            if s and s.topic == d.topic:
                return s
        return by[f"{d.form}|{d.topic if d.topic != 'flood' else 'water_backup'}"]


# ---------------------------------------------------------------- determinations
class Determination(BaseModel):
    model_config = ConfigDict(frozen=True)
    determination_id: str
    holder: str
    state: str
    family: str
    form: str                      # e.g. "HO-3 (2024-06)"
    endorsements: list[str]
    policy_number: str
    date_of_loss: date
    topic: str
    question: str
    gold_span_id: str


def load_dets(key: str = "determinations") -> list[Determination]:
    return [Determination(**d) for d in CAL.get(key, [])]


def claim_facts(d: Determination) -> str:
    ends = ", ".join(d.endorsements) or "none"
    return (f"Claim facts: policyholder {d.holder}; policy {d.policy_number}; form {d.form}; "
            f"endorsements {ends}; state {d.state}; date of loss {d.date_of_loss.isoformat()}.")


def applies(s: CSpan, d: Determination) -> bool:
    """Chapter 2's applicability filter: right edition, on the policy, right state, in force."""
    p = s.provenance
    if p.kind == "form":
        return p.label == d.form
    if p.kind == "endorsement":
        return p.label in d.endorsements
    return p.jurisdiction == d.state and p.in_force(d.date_of_loss)


def relevance(s, d: Determination) -> float:
    """The retriever's (and the stand-in reader's) relevance of a span to a question."""
    if getattr(s, "kind", "") == "page":
        return 0.05
    r = 1.0 if s.topic == d.topic else 0.25
    if s.provenance.family == d.family:
        r += 0.2
    return r + rng("rel", d.determination_id, s.span_id).random() * 0.1


# ---------------------------------------------------------------- packets
class ISpan(BaseModel):
    span_id: str
    text: str
    volatile: str = ""        # the topic skill inside text; the cache order moves it after the shared documents


class Ev(BaseModel):
    span_id: str
    label: str
    text: str
    kind: str
    effective: str = ""
    scope: str = ""


class Packet(BaseModel):
    determination_id: str
    instructions: list[ISpan]
    evidence: list[Ev]
    question: str


def ev_of(s: CSpan, text: str | None = None) -> Ev:
    eff = s.provenance.effective_from.isoformat() if s.provenance.effective_from else ""
    return Ev(span_id=s.span_id, label=s.label, text=s.text if text is None else text, kind=s.kind, effective=eff,
              scope=s.provenance.jurisdiction)


ADJ = ["Mileage for the field visit recorded at {n} miles round trip.",
       "Called the insured at {h}:{m:02d}; left a voicemail asking for photographs of the damage.",
       "Received {n} photographs by email and attached them to the file.",
       "Contractor estimate number {n} received and logged for review.",
       "Insured confirmed the mailing address and the preferred phone number.",
       "Diary set for a follow-up call in {k} business days.",
       "Mortgagee details checked against the declarations page.",
       "Supervisor review requested on reserve change of {n} dollars."]


@functools.lru_cache(maxsize=None)
def claim_page(i: int) -> Ev:
    r, out = rng("page", i), []
    while count_tokens(" ".join(out)) < 285:
        out.append(r.choice(ADJ).format(n=r.randint(12, 480), h=r.randint(8, 17), m=r.randint(0, 59),
                                        k=r.randint(2, 10)))
    return Ev(span_id=f"PAGE-{i:04d}", label=f"Claim file, adjuster log, page {i}", text=" ".join(out),
              kind="page")


def passage(s: CSpan, n: int) -> str:
    """The retrieved excerpt of a span: its first n tokens' worth of words."""
    if n >= s.tokens:
        return s.text
    words, out = s.text.split(), []
    for w in words:
        if count_tokens(" ".join(out + [w])) > n:
            break
        out.append(w)
    return " ".join(out)


# ---------------------------------------------------------------- rendering
@dataclass
class Segment:
    text: str
    start: int
    end: int
    kind: str
    span_id: str = ""


@dataclass
class Rendered:
    segments: list[Segment]
    labels: dict = field(default_factory=dict)     # "S1" -> span_id

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)

    @property
    def tokens(self) -> int:
        return self.segments[-1].end if self.segments else 0

    @property
    def evidence(self) -> list[Segment]:
        return [s for s in self.segments if s.kind == "evidence"]


ORDERS = ("rank", "reversed", "edges", "precedence", "cache")
KIND_ORDER = {"bulletin": 0, "endorsement": 1, "form": 2, "page": 3}


def arrange(ev: list[Ev], order: str) -> list[Ev]:
    if order in ("rank", "cache_rank"):
        return list(ev)
    if order == "reversed":
        return list(reversed(ev))
    if order == "edges":
        return ev[0::2] + list(reversed(ev[1::2]))
    if order == "precedence":
        return sorted(ev, key=lambda e: KIND_ORDER[e.kind])
    if order == "cache":
        return sorted(ev, key=lambda e: e.span_id)
    raise ValueError(order)


class Assembler:
    def __init__(self, fmt: str = "markdown", order: str = "rank"):
        self.fmt, self.order = fmt, order

    def span_text(self, i: int, e: Ev) -> str:
        if self.fmt == "plain":
            return e.text
        if self.fmt == "markdown":
            eff = f" (effective {e.effective}; {e.kind}; in {e.scope})" if e.effective else ""
            return f"[S{i}] {e.label}{eff}\n{e.text}"
        if self.fmt == "xml":
            return f'<span id="S{i}" source="{e.label}" effective="{e.effective}">{e.text}</span>'
        if self.fmt == "json":
            return json.dumps({"id": f"S{i}", "source": e.label, "effective": e.effective, "text": e.text})
        raise ValueError(self.fmt)

    def render(self, pk: Packet, facts: str) -> Rendered:
        ev = arrange(pk.evidence, self.order)
        parts = [(i.text, "instructions", i.span_id) for i in pk.instructions]
        evp = [(self.span_text(n, e), "evidence", e.span_id) for n, e in enumerate(ev, 1)]
        if self.order == "cache":           # stable rules first, the per-topic skill after the shared documents
            parts = [(i.text.replace("\n\n" + i.volatile, "", 1) if i.volatile else i.text, "instructions", i.span_id)
                     for i in pk.instructions]
            parts += evp + [(i.volatile, "instructions", i.span_id + "-skill") for i in pk.instructions if i.volatile]
            parts += [(facts, "facts", "")]
        else:
            parts += [(facts, "facts", "")] + evp
        parts.append((f"Question: {pk.question}", "question", ""))
        segs, pos = [], 0
        for text, kind, sid in parts:
            n = count_tokens(text)
            segs.append(Segment(text, pos, pos + n, kind, sid))
            pos += n
        return Rendered(segs, {f"S{n}": e.span_id for n, e in enumerate(ev, 1)})


# ---------------------------------------------------------------- the stand-in reader
@dataclass
class Answer:
    cited_span_id: str
    cited_label: str
    tokens_out: int


def signature(model: str, r: Rendered) -> str:
    ids = "|".join(s.span_id for s in r.evidence)
    return hashlib.sha256(f"{model}|{ids}".encode()).hexdigest()[:12]


class SimulatedModel:
    """Scores every evidence span by relevance x position weight and cites the best; when
    the fixture says it misreads this packet, it cites the best span other than the one
    that governs."""

    def __init__(self, name: str = "sim-large", dets: dict | None = None, corpus: Corpus | None = None):
        self.name, self.profile = name, PROFILES[name]
        self.misreads = CAL.get("misreads", {}).get(name, {})
        self.dets = dets if dets is not None else {d.determination_id: d for d in load_dets() + load_dets("probes")}
        self.corpus = corpus or CORPUS

    def scores(self, r: Rendered, d: Determination) -> dict:
        by, total, out = self.corpus.by_id, r.tokens, {}
        for s in r.evidence:
            src = by.get(s.span_id)
            rel = relevance(src, d) if src else 0.05
            out[s.span_id] = rel * position_weight((s.start + s.end) / 2 / total, total, self.profile)
        return out

    def generate(self, r: Rendered, key: str, topic: str = "") -> Answer:
        if r.tokens > self.profile.window:
            raise ValueError(f"prompt of {r.tokens} tokens exceeds {self.name} window {self.profile.window}")
        d = self.dets[key]
        try:
            gold = self.corpus.governing(d).span_id
        except KeyError:                          # a snapshot without the form edition in force
            gold = None
        sc = self.scores(r, d)
        fx = self.misreads.get(key, {}).get(signature(self.name, r))
        if fx in sc:                              # a calibrated misread: the fixture names the rival
            cited = fx
        elif gold in sc and fx is None:
            cited = gold
        else:                                     # misread, or the governing span is not in the packet
            cited = max((s for s in sc if s != gold), key=lambda s: sc[s])
        lab = next(s.text.split("\n")[0] for s in r.evidence if s.span_id == cited)
        return Answer(cited, lab, OUT_TOKENS + (OUT_EXTRA_WRONG if cited != gold else 0))


# ---------------------------------------------------------------- the pipeline
class PipelineConfig(BaseModel):
    model: str = "sim-large"
    policy_filter: bool = False
    k: int = 10
    order: str = "rank"
    stuff: str | None = None          # None (retrieve) | "family" | "all"
    fmt: str = "markdown"


@dataclass
class Trace:
    determination_id: str
    cited: str
    gold: str
    tokens_in: int
    tokens_out: int
    cached: int
    cost: float
    latency: float
    gold_in_packet: bool

    @property
    def wrong(self) -> bool:
        return self.cited != self.gold


def instructions_for(d: Determination, ins=V22) -> list[ISpan]:
    """The instruction set's rules, the loaded skill and the examples, as one system block."""
    spans = ins.spans(d)
    skill = "\n\n".join(s.text for s in spans if s.meta.get("skill"))
    return [ISpan(span_id=f"INSTR-{ins.version}", text="\n\n".join(s.text for s in spans), volatile=skill)]


class Pipeline:
    def __init__(self, cfg: PipelineConfig | None = None, corpus: Corpus | None = None, cache=None):
        self.cfg, self.corpus, self.cache = cfg or PipelineConfig(), corpus or CORPUS, cache
        self.model = SimulatedModel(self.cfg.model, corpus=self.corpus)

    def evidence(self, d: Determination) -> list[Ev]:
        by, cfg = self.corpus.by_id, self.cfg
        if cfg.stuff:
            pool = [s for s in self.corpus.spans if cfg.stuff == "all" or s.provenance.family == d.family
                    or s.provenance.kind == "bulletin"]       # the family: its forms and endorsements, every bulletin
            return [ev_of(s) for s in sorted(pool, key=lambda s: -relevance(s, d))]
        cands = [(by[sid], n) for sid, n in CAL["retrieval"][d.determination_id]]
        if cfg.policy_filter:
            cands = [(s, n) for s, n in cands if applies(s, d)]
        return [ev_of(s, passage(s, n)) for s, n in cands[:cfg.k]]

    def build_packet(self, d: Determination) -> tuple[Packet, dict]:
        ev = self.evidence(d)
        pk = Packet(determination_id=d.determination_id, instructions=instructions_for(d), evidence=ev,
                    question=d.question)
        return pk, {"spans": len(ev)}

    def render(self, d: Determination) -> Rendered:
        return Assembler(self.cfg.fmt, self.cfg.order).render(self.build_packet(d)[0], claim_facts(d))

    def run(self, d: Determination) -> Trace:
        r = self.render(d)
        a = self.model.generate(r, key=d.determination_id, topic=d.topic)
        cached = self.cache.lookup([(s.text, s.end - s.start) for s in r.segments]) if self.cache else 0
        gold = d.gold_span_id
        return Trace(d.determination_id, a.cited_span_id, gold, r.tokens, a.tokens_out, cached,
                     cost_usd(self.cfg.model, r.tokens, a.tokens_out, cached),
                     latency_s(self.cfg.model, r.tokens, a.tokens_out, cached) + CLIENT_S,
                     gold in {s.span_id for s in r.evidence})

    def run_many(self, dets) -> list[Trace]:
        return [self.run(d) for d in dets]


CORPUS = Corpus()
