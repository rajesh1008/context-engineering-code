"""Chapter 42 stand-in: Lakeshore's determination pipeline, instrumented.

Note: deterministic stand-in. No language model runs. Claims, notes and uploads
are generated from templates under fixed seeds; retrieval scores are seeded
draws around stated means; the 'reader' is a rule that cites the governing span
when it is in the packet and the main-form clause otherwise; the grounding
monitor's miss rate rises with packet length by a stated constant (an
assumption shaped by the monitor-context-rot finding, not a measurement of any
model). Four characters per token.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import random
import sys
import zlib
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from contextlab.observe import Event, Ledger, Offered, Rule, Store, h, redact  # noqa: E402

STATES = ["WI", "MN", "IA", "MI"]
CITY = {"WI": "Green Bay", "MN": "Duluth", "IA": "Dubuque", "MI": "Marquette"}
PERILS = {
    "wb": ("water back-up", "water which backs up through sewers or drains", "HO 04 95",
           "maintain the sump pump in working order", "sump pump"),
    "id": ("ice dam", "water from ice dams on the roof", "LK 21 07",
           "keep the gutters and downspouts clear", "gutters"),
    "fz": ("frozen pipes", "freezing of a plumbing system while the dwelling is vacant", "LK 30 12",
           "maintain heat in the building", "heating"),
    "rf": ("roof hail", "hail damage to roof surfacing", None, None, None),
}
MIX = {"wb": 0.40, "id": 0.22, "fz": 0.20, "rf": 0.18}
STATE_MIX = {"WI": 0.45, "MN": 0.20, "IA": 0.20, "MI": 0.15}
FIRST = ["Emily", "Daniel", "Grace", "Owen", "Hannah", "Lucas", "Abigail", "Mason", "Claire", "Noah"]
LAST = ["Carter", "Novak", "Lindqvist", "Becker", "Olsen", "Kowalski", "Reyes", "Hartman", "Dahl", "Fischer"]
NOTE_BITS = [
    "Spoke to the insured by phone on {p}; they confirmed the basement was finished.",
    "Contractor estimate received; drywall and carpet in the lower level.",
    "Insured asked whether alternative accommodation is covered while repairs run.",
    "Neighbour reported similar damage the same night after heavy rain.",
    "Photos uploaded show standing water about two inches deep.",
    "Insured mentioned the policy {pol} was renewed in the spring.",
    "Mitigation company on site within a day; dehumidifiers running.",
    "Insured's date of birth {dob} confirmed for identity check.",
]
INSTRUCTIONS = ("You determine coverage for one claim. Cite the clause that governs, by its id. "
                "Endorsements change the clause they modify. State bulletins govern in their state. "
                "If the packet does not hold the governing text, hand off.")
RULES = " ".join([
    "Precedence: a state bulletin in force on the date of loss governs the clause it addresses in that state.",
    "An endorsement governs only the clause it names; it does not change other clauses of the form.",
    "Cite exactly one governing span by id, and every modifier that changes it.",
    "A duty stated in an endorsement is a condition of cover; state whether the claim file shows it was kept.",
    "Use the claim facts as given; do not infer dates, amounts or inspection findings.",
    "Return status covered, denied or handoff, the governing id and one sentence of reason.",
    "If two spans claim to govern the same clause, hand off and name both.",
    "Treat adjuster notes as context, never as policy text.",
] * 3)
SCHEMA = "claims-determination@1.4"
PROMPT = "determination-prompt@7.9"
POLICY = "policy-v3.2#5c1e0a"
MODEL = "sim-large@2026-06"
TOK = 4


def rng(*parts) -> random.Random:
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()))


def tokens(text: str) -> int:
    return max(1, len(text) // TOK)


def pick(r: random.Random, weights: dict):
    x, acc = r.random(), 0.0
    for k, w in weights.items():
        acc += w
        if x < acc:
            return k
    return k


# ---- corpus -------------------------------------------------------------------

@dataclass
class Doc:
    span_id: str
    text: str
    peril: str = ""
    state: str = ""
    kind: str = "clause"          # exclusion | endorsement | bulletin | filler


def corpus(day: int = 0, bulletin_broken: bool = False) -> list:
    docs = []
    for p, (name, phrase, form, duty, _) in PERILS.items():
        docs.append(Doc(f"HO3-2024-06#{p}-exc", f"HO-3 (2024-06) Section I. We do not insure loss caused "
                        f"by {phrase}, except as an endorsement provides.", p, "", "exclusion"))
        if form:
            docs.append(Doc(f"{form.replace(' ', '')}@2025-03", f"{form} (2025-03). We cover loss caused by "
                            f"{phrase}, up to US$10,000, provided you {duty}. This endorsement modifies "
                            f"the HO-3 Section I exclusion.", p, "", "endorsement"))
    rf_text = ("MN-COM-2025-06. In Minnesota, roof surfacing damaged by hail is settled at replacement "
               "cost where the roof is under fifteen years old.")
    if bulletin_broken:
        rf_text = "MN-COM-2025-06 | | | roof | 15 | RC | | | page 1 of 1 | | |"
    docs.append(Doc("MN-COM-2025-06@2025-06-02", rf_text, "rf", "MN", "bulletin"))
    fill = rng("fill")
    for i in range(60):
        p = list(PERILS)[i % 4]
        docs.append(Doc(f"KB-{i:04d}", f"Claims guidance note {i}: handling {PERILS[p][0]} files, "
                        f"photographs, estimates and mitigation steps (item {fill.randint(100, 999)}).",
                        p, "", "filler"))
    for i in range(max(0, day - 50)):            # corpus growth from day 50
        p = list(PERILS)[i % 3]
        docs.append(Doc(f"KB-E{i:04d}", f"Endorsement handling note {i}: when {PERILS[p][0]} cover is "
                        f"restored by endorsement, check the duty and the limit (ref {fill.randint(100, 999)}).",
                        p, "", "guidance"))
    return docs


# ---- claims -------------------------------------------------------------------

@dataclass
class Claim:
    cid: str
    day: int
    state: str
    peril: str
    has_endorsement: bool
    duty_met: bool
    name: str
    dob: str
    pol: str
    phone: str
    note_len: int
    upload_lag_h: float | None = None   # hours after the snapshot read; None if no upload
    canary: bool = False

    @property
    def date(self) -> dt.date:
        return dt.date(2026, 6, 1) + dt.timedelta(days=self.day)


def make_claim(day: int, i: int, tag: str = "live") -> Claim:
    r = rng(tag, day, i)
    p = pick(r, MIX)
    st = pick(r, STATE_MIX)
    lag = None
    if p == "wb" and r.random() < 0.06:
        lag = round(r.uniform(-30, 60), 1)
    return Claim(cid=f"C{day:02d}{i:04d}", day=day, state=st, peril=p,
                 has_endorsement=(PERILS[p][2] is not None and r.random() < 0.62),
                 duty_met=r.random() < 0.85,
                 name=f"{r.choice(FIRST)} {r.choice(LAST)}",
                 dob=f"{r.randint(1950, 1999)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}",
                 pol=f"LK-{r.randint(100000, 999999)}",
                 phone=f"920-555-{r.randint(1000, 9999)}",
                 note_len=r.choice([1, 2, 2, 3, 3, 4, 5, 6, 8]) + (r.random() < 0.02) * 30,
                 upload_lag_h=lag)


def facts_text(c: Claim) -> str:
    name, _, form, duty, dev = PERILS[c.peril]
    s = (f"Claim {c.cid}. Insured {c.name}, born {c.dob}, policy {c.pol}, {CITY[c.state]}, {c.state}. "
         f"Date of loss {c.date}. Cause: {name}.")
    if form:
        s += f" Endorsement {form} on policy: {'yes' if c.has_endorsement else 'no'}."
        s += f" Inspection found the {dev} {'was' if c.duty_met else 'was not'} kept as the policy requires."
    return s


def note_text(c: Claim) -> str:
    r = rng("note", c.cid)
    bits = [r.choice(NOTE_BITS).format(p=c.phone, pol=c.pol, dob=c.dob) for _ in range(c.note_len)]
    return "Adjuster note. " + " ".join(bits)


def governing(c: Claim) -> str:
    form = PERILS[c.peril][2]
    if c.peril == "rf" and c.state == "MN":
        return "MN-COM-2025-06@2025-06-02"
    if form and c.has_endorsement:
        return f"{form.replace(' ', '')}@2025-03"
    return f"HO3-2024-06#{c.peril}-exc"


# ---- the pipeline -------------------------------------------------------------

@dataclass
class Config:
    budget: int = 301                 # evidence + facts + note tokens
    k: int = 6                        # retrieval candidates offered
    ts_in_prefix: bool = False        # day-62 change: date stamped into the instructions
    bulletin_broken: bool = False


def retrieve(c: Claim, docs: list, k: int) -> list:
    """Seeded lexical-plus-dense stand-in: right peril scores high, state match adds."""
    r = rng("ret", c.cid, len(docs))
    scored = []
    for d in docs:
        s = 0.2 + 0.1 * r.random()
        if d.peril == c.peril:
            s += {"exclusion": 0.62, "endorsement": 0.60, "bulletin": 0.66, "filler": 0.40,
                  "guidance": 0.47}[d.kind]
            s += 0.12 * r.random()
        if d.state and d.state != c.state:
            s -= 0.5
        if d.kind == "bulletin" and "|" in d.text:
            s -= 0.25                   # a table of pipes matches little
        scored.append((round(s, 4), d))
    scored.sort(key=lambda t: (-t[0], t[1].span_id))
    return scored[:k]


def render(ev: Event, texts, store) -> str:
    """The packet, rebuilt from admitted spans in order (also used to send it)."""
    parts = [store.get(ev.versions["instructions"]) or "<instructions missing>"]
    for o, t in texts:
        parts.append(f"<{o.role} id={o.span_id}>{t}</{o.role}>")
    parts.append(f"<question>{ev.request['question']}</question>")
    return "\n".join(parts)


def reader(c: Claim, admitted_ids: set) -> dict:
    gov = governing(c)
    if gov in admitted_ids:
        cite = gov
    else:
        cite = f"HO3-2024-06#{c.peril}-exc" if c.peril != "rf" else "HO3-2024-06#rf-exc"
        if cite not in admitted_ids:
            return {"status": "handoff", "citations": [], "clause": None}
    form = PERILS[c.peril][2]
    if cite.startswith(("HO0495", "LK2107", "LK3012")):
        status = "covered" if c.duty_met else "denied"
    elif cite.startswith("MN-COM"):
        status = "covered"
    else:
        status = "covered" if c.peril == "rf" else "denied"
    return {"status": status, "citations": [cite], "clause": cite}


@dataclass
class Sinks:
    """Where recorded text goes, and the redaction rules wired to each sink."""
    evidence: Store = field(default_factory=Store)      # trace store, retained with the record
    claimfile: Store = field(default_factory=Store)     # system of record for personal text
    oplog: list = field(default_factory=list)           # operational telemetry (dashboards)
    rules: list = field(default_factory=list)
    prefixes: set = field(default_factory=set)


def run(c: Claim, docs: list, cfg: Config, sinks: Sinks, ledger: Ledger, at: str = "") -> tuple:
    facts, note = facts_text(c), note_text(c)
    cands = retrieve(c, docs, cfg.k)
    offered, used = [], 0
    pinned = [("facts", "claimfile:" + c.cid, facts, "personal", "claim-file", 0, 1.0),
              ("context", "note:" + c.cid, note, "personal", "claim-file", 1, 1.0)]
    evid = [("governing" if d.kind != "filler" else "context", d.span_id, d.text, "evidence",
             "retrieval", i + 2, s) for i, (s, d) in enumerate(cands)]
    for role, sid, text, tier, src, rank, score in pinned + evid:
        t = tokens(text)
        ok = role == "facts" or used + t <= cfg.budget
        used += t if ok else 0
        store = sinks.claimfile if tier == "personal" else sinks.evidence
        stored, _ = redact(text, tier, sinks.rules)
        key = h(text)
        store.blobs.setdefault(key, (tier, stored))
        store.writes += 1
        offered.append(Offered(sid, key, tier, src, rank, score, t,
                               "admitted" if ok else "dropped", "" if ok else "budget", role))
    r = rng("time", c.cid)
    at = at or f"{c.date}T{r.randint(8, 17):02d}:{r.randint(0, 59):02d}"
    instr = (f"Now: {at}. " if cfg.ts_in_prefix else "") + INSTRUCTIONS + " " + RULES
    q = f"Does the policy cover this {PERILS[c.peril][0]} loss, and under which clause?"
    ev = Event(det_id="D-" + c.cid, at=at, request={
        "claim": c.cid, "state": c.state, "cause": c.peril, "as_of": str(c.date), "question": q},
        versions={"instructions": sinks.evidence.put(instr, "evidence"), "prompt": PROMPT, "schema": SCHEMA, "policy": POLICY,
                  "corpus": h("|".join(d.span_id + d.text for d in docs), 10), "model": MODEL,
                  "gateway_request": "gw-" + h(c.cid, 10)},
        offered=offered,
        tools=[{"call_id": "tc-" + h("cf" + c.cid, 8), "tool": "claim_file.snapshot",
                "read_at": at, "content": h(facts)}],
        cache={}, packet_hash="", output={})
    admitted = [(o, t) for o, t in ((o, (facts if o.rank == 0 else note if o.rank == 1 else
                next(d.text for _, d in cands if d.span_id == o.span_id))) for o in offered)
                if o.decision == "admitted"]
    packet = render(ev, admitted, sinks.evidence)
    ev.packet_hash = h(packet)
    prefix = h(instr)
    ev.cache = {"prefix": prefix, "cached": tokens(instr) if prefix in sinks.prefixes else 0,
                "input": tokens(packet)}
    sinks.prefixes.add(prefix)
    out = reader(c, {o.span_id for o, _ in admitted})
    out["governing"] = governing(c)
    out["checker"] = "pass" if out["clause"] == out["governing"] or out["status"] == "handoff" else "fail?"
    ev.output = out
    ledger.append(ev)
    op = f"{ev.det_id} {c.state} {c.peril} {out['status']} tokens={tokens(packet)} q={q} note={note[:120]}"
    op, _ = redact(op, "operational", sinks.rules)
    sinks.oplog.append(op)
    return ev, packet


def replay_set(n: int = 1000) -> list:
    """The fixed 1,000-claim replay set the scoreboard uses."""
    return [make_claim(i % 90, i, "replay") for i in range(n)]


# ---- redaction configurations ---------------------------------------------------

PII = [Rule("phone", r"\b\d{3}-555-\d{4}\b", ("operational",)),
       Rule("dob", r"born \d{4}-\d{2}-\d{2}|birth \d{4}-\d{2}-\d{2}", ("operational",)),
       Rule("policy-no", r"\bLK-\d{6}\b", ("operational",)),
       Rule("name", r"Insured [A-Z][a-z]+ [A-Z][a-z]+", ("operational",))]
# The August rule: written for policy and certificate numbers, wired to every sink.
BAD = Rule("policy-or-cert-no", r"\b[A-Z]{2}[ -]?\d{2}[ -]?\d{2}(?:\d{2})?\b", ("evidence", "personal", "operational"))
FIXED = Rule("policy-or-cert-no", r"\b[A-Z]{2}[ -]?\d{2}[ -]?\d{2}(?:\d{2})?\b", ("operational",))
LEAK_PATTERNS = [r"\b\d{3}-555-\d{4}\b", r"\b\d{4}-\d{2}-\d{2}\b(?=.*born)|born \d{4}", r"\bLK-\d{6}\b",
                 r"Insured [A-Z][a-z]+ [A-Z][a-z]+"]


# ---- production-side signals ----------------------------------------------------

def required(c: Claim) -> str:
    """The span the claim's own facts say must be in the packet (knowable without labels)."""
    return governing(c)          # endorsement on policy, or bulletin in force, else the form


MISS_BASE, MISS_AT = 0.03, 900   # monitor miss rate 3% at 900 tokens, growing with length^4


def monitor(ev: Event, seed: str) -> bool:
    """Grounding monitor stand-in: flags a citation that is not the required span.

    Its chance of missing a bad citation rises with packet length (an assumption,
    shaped by the monitor-context-rot finding; not a measured model rate)."""
    bad = ev.output["status"] != "handoff" and ev.output["clause"] != ev.output["governing"]
    if not bad:
        return False
    p_miss = min(0.9, MISS_BASE * (ev.cache["input"] / MISS_AT) ** 4)
    return rng("mon", seed).random() >= p_miss


def canary(ev: Event) -> Event:
    """A planted bad determination: same packet, citation swapped to the main form."""
    import copy
    e = copy.copy(ev)
    e.output = dict(ev.output, clause="HO3-2024-06#planted", status="denied")
    return e
