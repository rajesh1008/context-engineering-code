"""Chapter 1 stand-in: Lakeshore's synthetic corpus, 500 determinations, the
applicability policy (Listing 1.2), a stand-in retriever and SimulatedModel.

STAND-IN NOTICE. Nothing here is a language model or a real search engine.
* The corpus is 378 synthetic spans (22,326 tokens): three families of forms
  (HO-3, a personal auto policy, a boat policy), endorsements, state bulletins,
  claims guidance, 14 underwriting notes and 13 claim records.
* The retriever ranks spans by a simulated relevance score carried in the
  fixtures (topic match, family match, a seeded jitter and, for the enriched
  query, a bonus for documents the query names). B's "keyword" ranking is the
  same score without the enrichment; it stands in for BM25.
* SimulatedModel("sim-large") scores every evidence span in the rendered packet
  from four things: how much of the question the span covers (a fixture value),
  whether the span's document is named in the claim header, authority under the
  rules (endorsements and bulletins outrank the form; underwriting notes are not
  binding), and position (attention dips in the middle of packets longer than
  the 24,000-token onset). It adds seeded noise that grows past the onset, and
  it misreads the topic of one question in a hundred (it then cites its second
  choice). It cites the top-scoring span.
* calibrate.py chose, per determination, the fixture details that the profile
  depends on (which rivals a question has and how close they read, where the
  governing span ranks, the question's length and a jitter seed) so that this
  mechanism reproduces the chapter's illustrative profile exactly. The
  mechanism is real and inspectable; the profile is illustrative, not measured.
"""
from __future__ import annotations

import hashlib, json, math, random
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

HERE = Path(__file__).resolve().parent
SEED = 2026
N_SPANS, CORPUS_TOKENS = 378, 22_326
PROV_TOTAL = 13_589            # provenance-header tokens over the whole corpus
INSTR = 53                     # role + task sentences
RULES_TOKENS = 50              # precedence, applicability, exact-citation sentences
HEADER = 32                    # claim header: policy, form, endorsements, state, date of loss
FIX_STAMP = 5                  # line the required-endorsement check adds to a packet
ROLE = "claims_assistant"


def rng(*key) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, (SEED,) + key)).encode()).hexdigest()[:12], 16))


# ---------------------------------------------------------------- types
class Kind(str, Enum):
    FORM = "form"
    ENDORSEMENT = "endorsement"
    BULLETIN = "bulletin"
    GUIDANCE = "guidance"
    NOTE = "underwriting note"
    CLAIM_RECORD = "claim record"


BINDING = {Kind.FORM, Kind.ENDORSEMENT, Kind.BULLETIN, Kind.GUIDANCE}


@dataclass(frozen=True)
class Provenance:
    doc_id: str
    label: str                       # e.g. "HO-3 (2024-06)"
    version: str
    jurisdiction: str = "ALL"
    entitlement: frozenset = frozenset({ROLE, "underwriter"})
    superseded_by: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None

    def in_force(self, d: date) -> bool:
        return (self.valid_from is None or self.valid_from <= d) and (self.valid_to is None or d <= self.valid_to)


@dataclass(frozen=True, eq=False)
class Span:
    span_id: str
    kind: Kind
    provenance: Provenance
    section: str
    page: int
    topic: str
    family: str
    conclusion: str
    tokens: int = 0
    header_tokens: int = 0           # provenance header the renderer puts above the span
    meta: dict = field(default_factory=dict)

    @property
    def cite(self) -> str:
        return f"{self.provenance.label}, {self.section}, page {self.page}"


class Determination(BaseModel):
    model_config = ConfigDict(frozen=True)
    determination_id: str
    holder: str
    state: str
    family: str
    form: str                        # label of the policy form edition
    endorsements: list[str]          # labels on the declarations
    policy_number: str
    date_of_loss: date
    topic: str
    question: str
    q_tokens: int
    governs: str                     # form | endorsement | bulletin
    gold_span: str
    gold: str


@dataclass
class Verdict:
    ok: bool
    reason: str


# ---------------------------------------------------------------- corpus
STATES = ["WI", "IL", "IA", "MN", "MI", "OH", "IN"]
WATER = {"sewer_backup", "surface_water", "burst_pipe", "ice_dam", "ice_damage", "hull"}
FAMILIES = {
    "HO-3": dict(form="HO-3", editions=[("2019-03", date(2019, 3, 1), date(2022, 5, 31)),
                                        ("2022-06", date(2022, 6, 1), date(2024, 5, 31)),
                                        ("2024-06", date(2024, 6, 1), None)],
                 topics=["sewer_backup", "surface_water", "burst_pipe", "wind_hail", "fire", "theft",
                         "roof", "mold", "ice_dam", "liability"],
                 endorsements=[("HO 04 95", "sewer_backup", ["2019-03", "2024-06"]),
                               ("LK 04 11", "mold", ["2024-06"]), ("LK 04 12", "roof", ["2024-06"]),
                               ("LK 04 13", "theft", ["2024-06"]), ("LK 04 14", "ice_dam", ["2024-06"])],
                 bulletins=["wind_hail", "roof"]),
    "PAP": dict(form="PP 00 01", editions=[("2018-03", date(2018, 3, 1), date(2023, 12, 31)),
                                           ("2024-01", date(2024, 1, 1), None)],
                topics=["collision", "comprehensive", "rental", "towing", "glass", "uninsured_motorist", "liability"],
                endorsements=[("LK 05 21", "rental", ["2024-01"]), ("LK 05 22", "towing", ["2024-01"]),
                              ("LK 05 23", "glass", ["2024-01"])],
                bulletins=[]),
    "BOAT": dict(form="BT 00 01", editions=[("2019-01", date(2019, 1, 1), date(2022, 12, 31)),
                                            ("2023-01", date(2023, 1, 1), None)],
                 topics=["hull", "ice_damage", "theft", "navigation", "liability"],
                 endorsements=[("LK 06 31", "ice_damage", ["2023-01"]), ("LK 06 32", "navigation", ["2023-01"])],
                 bulletins=[]),
}
GENERAL = ["Definitions", "Conditions", "Declarations", "Loss Settlement", "Duties After Loss", "Other Insurance",
           "Policy Period", "Concealment or Fraud", "Appraisal", "Mortgage Clause", "Cancellation", "Nonrenewal",
           "Subrogation", "Suit Against Us", "Liberalization", "Waiver or Change", "Assignment", "Death",
           "Loss Payment", "Abandonment of Property", "Recovered Property", "Volcanic Eruption Period"]


def _conclusion(topic: str, effect: str, label: str) -> str:
    if label == "HO 04 95 (2024-06)":
        return "covered up to $10,000"
    if label == "HO 04 95 (2019-03)":
        return "covered up to $5,000"
    return {"grant": "covered", "exclusion": "excluded unless endorsed",
            "endorsement": "covered as endorsed", "bulletin": "covered per bulletin"}.get(effect, "see clause")


def build_corpus() -> list[Span]:
    spans: list[Span] = []
    add = spans.append
    for fam, f in FAMILIES.items():
        eds = f["editions"]
        for i, (ed, vf, vt) in enumerate(eds):
            label = f"{f['form']} ({ed})"
            sup = f"{f['form']} ({eds[i + 1][0]})" if i + 1 < len(eds) else None
            prov = Provenance(f"{fam}-FORM", label, ed, superseded_by=sup, valid_from=vf, valid_to=vt)
            for j, t in enumerate(f["topics"]):
                add(Span(f"{label}|{t}|grant", Kind.FORM, prov, "Section I Perils Insured Against" if t != "liability"
                         else "Section II Coverage E", 1, t, fam, _conclusion(t, "grant", label), meta={"effect": "grant"}))
                sec = "Section I Exclusions A.3" if t == "sewer_backup" else f"Section I Exclusions A.{4 + j}"
                add(Span(f"{label}|{t}|exclusion", Kind.FORM, prov, sec, 2, t, fam,
                         _conclusion(t, "exclusion", label), meta={"effect": "exclusion"}))
        for code, t, eds_e in f["endorsements"]:
            for k, ed in enumerate(eds_e):
                label = f"{code} ({ed})"
                vf = next(v for e, v, _ in eds if e == ed) if any(e == ed for e, _, _ in eds) else date(2024, 6, 1)
                vt = date(2024, 5, 31) if k + 1 < len(eds_e) else None
                sup = f"{code} ({eds_e[k + 1]})" if k + 1 < len(eds_e) else None
                prov = Provenance(f"{fam}-END-{code}", label, ed, superseded_by=sup,
                                  valid_from=date(2019, 3, 1) if vt else vf, valid_to=vt)
                for n, sec in enumerate(["Coverage", "Limit of Liability", "Exclusions", "Conditions", "Schedule",
                                         "Definitions"]):
                    add(Span(f"{label}|{sec}", Kind.ENDORSEMENT, prov, sec, 1 + n // 2, t if n == 0 else "general",
                             fam, _conclusion(t, "endorsement", label) if n == 0 else "see clause",
                             meta={"effect": "endorsement" if n == 0 else "general"}))
        if fam in ("HO-3", "PAP"):
            for st in STATES:
                label = f"{f['form']} {st} Amendatory (2024-06)"
                prov = Provenance(f"{fam}-AMD-{st}", label, "2024-06", jurisdiction=st)
                add(Span(f"{label}|Amendments", Kind.ENDORSEMENT, prov, "Amendments", 1, "general", fam, "see clause"))
        for t in f["bulletins"]:
            for st in STATES:
                for yr, vf, vt in (("2024", date(2024, 1, 1), date(2025, 2, 28)), ("2025", date(2025, 3, 1), None)):
                    label = f"Bulletin {st}-{yr}-{t}"
                    prov = Provenance(f"BUL-{st}-{t}", label, yr, jurisdiction=st,
                                      superseded_by=f"Bulletin {st}-2025-{t}" if yr == "2024" else None,
                                      valid_from=vf, valid_to=vt)
                    add(Span(f"{label}|Guidance", Kind.BULLETIN, prov, "Guidance", 1, t, fam,
                             _conclusion(t, "bulletin", label), meta={"effect": "bulletin"}))
    for n in range(11):
        t = ["sewer_backup", "wind_hail", "collision", "hull", "roof", "theft", "glass", "fire", "mold",
             "towing", "surface_water"][n]
        fam = "PAP" if t in ("collision", "glass", "towing") else "BOAT" if t == "hull" else "HO-3"
        prov = Provenance(f"GUIDE-{n:02d}", f"Claims Manual G-{n + 1:02d} (2025-01)", "2025-01")
        add(Span(f"G-{n + 1:02d}|{t}", Kind.GUIDANCE, prov, "Handling", 1, t, fam, "see clause"))
    note_topics = ["sewer_backup", "surface_water", "burst_pipe", "wind_hail", "roof", "mold", "ice_dam",
                   "theft", "collision", "rental", "glass", "hull", "ice_damage", "fire"]
    for n, t in enumerate(note_topics):
        fam = "PAP" if t in ("collision", "rental", "glass") else "BOAT" if t in ("hull", "ice_damage") else "HO-3"
        when = "October 2022" if t == "sewer_backup" else f"{['March', 'June', 'September', 'November'][n % 4]} {2021 + n % 4}"
        prov = Provenance(f"UW-{n:02d}", f"Underwriting note UW-{n + 1:02d} ({when})", "note",
                          entitlement=frozenset({"underwriter"}))
        concl = "pays $5,000" if t == "sewer_backup" else "see note"
        add(Span(f"UW-{n + 1:02d}|{t}", Kind.NOTE, prov, "Note", 1, t, fam, concl))
    # pad with general clauses round-robin over form editions, then claim records are added per policy
    editions = [(fam, f["form"], e) for fam, f in FAMILIES.items() for e in f["editions"]]
    g = 0
    while len(spans) < N_SPANS - 13:
        fam, form, (ed, vf, vt) = editions[g % len(editions)]
        label = f"{form} ({ed})"
        sec = GENERAL[(g // len(editions)) % len(GENERAL)]
        i = [e[0] for e in FAMILIES[fam]["editions"]].index(ed)
        eds = FAMILIES[fam]["editions"]
        sup = f"{form} ({eds[i + 1][0]})" if i + 1 < len(eds) else None
        prov = Provenance(f"{fam}-FORM", label, ed, superseded_by=sup, valid_from=vf, valid_to=vt)
        add(Span(f"{label}|{sec}", Kind.FORM, prov, sec, 3 + g // len(editions), "general", fam, "see clause"))
        g += 1
    return spans


# ---------------------------------------------------------------- calibration file
CAL_PATH = HERE / "calibration.json"


def load_cal() -> dict:
    return json.loads(CAL_PATH.read_text())


def finish_corpus(spans: list[Span], sizes: dict, heads: dict, records: list[dict]) -> list[Span]:
    from dataclasses import replace
    out = [replace(s, tokens=sizes[s.span_id], header_tokens=heads[s.span_id]) for s in spans]
    for r in records:
        prov = Provenance(f"CLM-{r['policy_number']}", f"Claim record {r['claim_id']}", "record")
        out.append(Span(f"CLM|{r['claim_id']}", Kind.CLAIM_RECORD, prov, "Claim history", 1, r["topic"],
                        r["family"], r["text"], tokens=r["tokens"], header_tokens=r["header_tokens"],
                        meta={"policy_number": r["policy_number"]}))
    return out


class Corpus:
    def __init__(self, spans: list[Span]):
        self.spans = spans
        self.by_id = {s.span_id: s for s in spans}

    def history_for(self, det: Determination) -> list[Span]:
        return [s for s in self.spans if s.kind == Kind.CLAIM_RECORD
                and s.meta.get("policy_number") == det.policy_number]


# ---------------------------------------------------------------- applicability policy (Listing 1.2)
class Policy:
    def __init__(self, role: str = ROLE):
        self.role = role

    def applies(self, span: Span, det: Determination) -> Verdict:
        p = span.provenance
        if self.role not in p.entitlement:
            return Verdict(False, f"not entitled ({self.role})")
        if not p.version:
            return Verdict(False, "no version: fail closed")
        if p.jurisdiction not in ("ALL", det.state):
            return Verdict(False, f"wrong state ({p.jurisdiction})")
        if span.kind == Kind.FORM and p.label != det.form:
            return Verdict(False, "superseded" if p.superseded_by else "not on policy")
        if span.kind == Kind.ENDORSEMENT:
            amendatory = p.jurisdiction == det.state and det.family in p.doc_id
            if not amendatory and p.label not in det.endorsements:
                return Verdict(False, "endorsement not on policy")
        if span.kind == Kind.BULLETIN and not p.in_force(det.date_of_loss):
            return Verdict(False, "bulletin not in force on date of loss")
        if span.kind == Kind.CLAIM_RECORD and span.meta.get("policy_number") != det.policy_number:
            return Verdict(False, "another policy's claim")
        return Verdict(True, "applies")


# ---------------------------------------------------------------- fixture relevance and coverage
def related(a: str, b: str) -> bool:
    return a != b and a in WATER and b in WATER


def _unit(*key) -> float:
    h = hashlib.blake2b("|".join(map(str, (SEED,) + key)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


@lru_cache(maxsize=64)
def _jitters(did: str, seed: int) -> dict:
    return {}


def base_relevance(det: Determination, s: Span, seed: int) -> float:
    r = 0.6 if s.topic == det.topic else 0.35 if related(s.topic, det.topic) else 0.0
    r += 0.15 if s.family == det.family else 0.0
    cache = _jitters(det.determination_id, seed)
    j = cache.get(s.span_id)
    if j is None:
        j = cache[s.span_id] = _unit("jit", det.determination_id, seed, s.span_id)
    return r + j * 0.35


def on_query(det: Determination, s: Span) -> bool:
    """Does the enriched query (form edition, endorsements, state) name this span's document?"""
    p = s.provenance
    if s.kind == Kind.FORM:
        return p.label == det.form
    if s.kind == Kind.ENDORSEMENT:
        return p.label in det.endorsements
    if s.kind == Kind.BULLETIN:
        return p.jurisdiction == det.state
    return False


def default_cov(det: Determination, s: Span, by_id: dict) -> float:
    g = by_id[det.gold_span]
    if s.span_id == det.gold_span:
        return 1.0
    if s.topic == det.topic and s.kind == g.kind and s.provenance.doc_id == g.provenance.doc_id \
            and s.section == g.section:
        return 0.97                                       # another edition of the governing clause
    if s.topic == det.topic:
        if s.kind == Kind.NOTE:
            return 0.85
        if s.kind == Kind.CLAIM_RECORD:
            return 0.60
        if det.governs == "endorsement" and s.kind == Kind.FORM:
            return 0.85 if s.meta.get("effect") == "exclusion" else 0.75
        return 0.80
    if s.kind == Kind.CLAIM_RECORD:
        return 0.40
    if related(s.topic, det.topic):
        return 0.70 if s.kind == Kind.FORM and s.family != det.family else 0.55
    if s.topic == "general":
        return 0.30
    return 0.15


# ---------------------------------------------------------------- packets and rendering
@dataclass
class Packet:
    det: Determination
    spans: list[Span]
    rules: bool
    header: bool = True
    stamp: bool = False

    @property
    def tokens(self) -> int:
        return (INSTR + (RULES_TOKENS if self.rules else 0) + (HEADER if self.header else 0) + self.det.q_tokens
                + sum(s.tokens + s.header_tokens for s in self.spans) + (FIX_STAMP if self.stamp else 0))

    def named(self, s: Span) -> bool:
        if not self.header:
            return False
        p = s.provenance
        if s.kind == Kind.FORM:
            return p.label == self.det.form
        if s.kind == Kind.ENDORSEMENT:
            return p.label in self.det.endorsements or (p.jurisdiction == self.det.state and self.det.family in p.doc_id)
        if s.kind == Kind.BULLETIN:
            return p.jurisdiction == self.det.state and p.in_force(self.det.date_of_loss)
        if s.kind == Kind.CLAIM_RECORD:
            return s.meta.get("policy_number") == self.det.policy_number
        return False


class Assembler:
    """Renders a packet as text. render(pk, "") drops the claim header."""
    def __init__(self, fmt: str = "markdown"):
        self.fmt = fmt

    def render(self, pk: Packet, header: str | None = None) -> Packet:
        return Packet(pk.det, pk.spans, pk.rules, header=(header is None or header != ""), stamp=pk.stamp)

    def text(self, pk: Packet) -> str:
        d = pk.det
        out = ["## Instructions", "You are Lakeshore's claims assistant. Answer the coverage question from the "
               "evidence below and cite the span you rely on."]
        if pk.rules:
            out += ["Precedence: an endorsement or bulletin outranks the form it amends.",
                    "Applicability: use only documents named in the claim header.",
                    "Citation: cite form, edition, section and page exactly."]
        if pk.header:
            out += ["## Claim", f"Policy {d.policy_number} | {d.form} | endorsements: "
                    f"{', '.join(d.endorsements) or 'none'} | {d.state} | loss {d.date_of_loss}"]
        out += ["## Question", d.question, "## Evidence"]
        for s in pk.spans:
            out += [f"[{s.cite} | {s.kind.value} | {s.provenance.jurisdiction}]", f"({s.topic}) {s.conclusion}"]
        return "\n".join(out)


# ---------------------------------------------------------------- the model
@dataclass
class Answer:
    conclusion: str
    span: Span | None

    @property
    def abstained(self) -> bool:
        return self.span is None


class SimulatedModel:
    """Deterministic stand-in for a large model (window 200,000 tokens)."""
    H, R_AUTH, R_NONBINDING = 0.30, 0.25, -0.20
    EPS = 0.01

    def __init__(self, name: str = "sim-large", window: int = 200_000, onset: int = 24_000,
                 slope: float = 0.0, dip: float = 0.0):
        self.name, self.window, self.onset = name, window, onset
        cal = load_cal()["model"] if CAL_PATH.exists() else {}
        self.slope = cal.get("slope", slope)
        self.dip = cal.get("dip", dip)

    def reads_topic(self, det: Determination) -> bool:
        """The stand-in misreads the topic of one question in a hundred (fixture list)."""
        return det.determination_id not in MISREAD

    def scores(self, pk: Packet, cov: dict) -> list[tuple[float, Span]]:
        n, toks = len(pk.spans), pk.tokens
        over = max(0.0, (toks - self.onset) / 1000)
        amp = self.EPS + self.slope * over
        out = []
        for i, s in enumerate(pk.spans):
            v = cov.get(s.span_id, 0.15)
            v += self.H if pk.named(s) else 0.0
            if pk.rules:
                v += self.R_AUTH if s.kind in (Kind.ENDORSEMENT, Kind.BULLETIN) else 0.0
                v += self.R_NONBINDING if s.kind in (Kind.NOTE,) else 0.0
            if over > 0 and n > 1:
                v -= self.dip * math.sin(math.pi * i / (n - 1))
            v += amp * (2 * _unit("noise", pk.det.determination_id, s.span_id) - 1)
            out.append((v, s))
        return sorted(out, key=lambda t: -t[0])

    def answer(self, pk: Packet, cov: dict) -> Answer:
        if not pk.spans:
            return Answer("none (abstained)", None)
        ranked = self.scores(pk, cov)
        pick = ranked[1][1] if (not self.reads_topic(pk.det) and len(ranked) > 1) else ranked[0][1]
        return Answer(pick.conclusion, pick)




# ---------------------------------------------------------------- retrieval and the world
@dataclass
class Query:
    det: Determination
    enriched: bool              # True: query carries form edition, endorsements and state


class Retriever:
    """Stand-in retriever: ranks spans by the fixture's simulated relevance."""
    ENRICH = 0.13

    def __init__(self, world: "World"):
        self.w = world

    def score(self, q: Query, s: Span) -> float:
        fx = self.w.fx[q.det.determination_id]
        b = base_relevance(q.det, s, fx["seed"])
        if q.enriched:
            if s.span_id in fx["rel"]:
                return fx["rel"][s.span_id]
            return b + (self.ENRICH if on_query(q.det, s) else 0.0)
        return b + fx["kw"].get(s.span_id, 0.0)

    def search(self, q: Query, k: int = 40) -> list[tuple[Span, float]]:
        scored = [(s, self.score(q, s)) for s in self.w.corpus.spans if s.kind != Kind.CLAIM_RECORD]
        return sorted(scored, key=lambda t: (-t[1], t[0].span_id))[:k]


class Pipeline:
    def __init__(self, world: "World"):
        self.retriever = Retriever(world)

    def query(self, view: Determination) -> Query:
        return Query(view, enriched=True)

    def keyword(self, det: Determination) -> Query:
        return Query(det, enriched=False)


MISREAD: set = set()


@dataclass
class World:
    corpus: Corpus
    dets: list
    fx: dict
    cov: dict
    model: SimulatedModel


def det_from(d: dict) -> Determination:
    d = dict(d)
    d["date_of_loss"] = date.fromisoformat(d["date_of_loss"])
    return Determination(**{k: v for k, v in d.items() if k in Determination.model_fields})


def build_world(cal: dict | None = None) -> World:
    cal = cal or load_cal()
    MISREAD.clear()
    MISREAD.update(cal["misread"])
    corpus = Corpus(finish_corpus(build_corpus(), cal["sizes"], cal["heads"], cal["records"]))
    dets = [det_from(d) for d in cal["dets"]]
    fx = {d["determination_id"]: d["fx"] for d in cal["dets"]}
    cov = {}
    for det in dets:
        c = {s.span_id: default_cov(det, s, corpus.by_id) for s in corpus.spans}
        c.update(fx[det.determination_id]["cov"])
        cov[det.determination_id] = c
    model = SimulatedModel("sim-large")
    model.slope, model.dip = cal["model"]["slope"], cal["model"]["dip"]
    Retriever.ENRICH = cal.get("enrich", Retriever.ENRICH)
    return World(corpus, dets, fx, cov, model)
