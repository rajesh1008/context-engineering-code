"""The Chapter 12 retrieval corpus and the full-mix query set (synthetic).

Spans carry provenance (document, edition, state, issued date, section) and
links (endorsement -> clause it modifies, bulletin -> clause, superseded ->
superseding). Queries are the 1,000 full-mix determinations: the 200-
determination evaluation slice, the 300 holdout and 500 new determinations
in Chapter 9's strata. Everything is seeded.
"""
import random
from dataclasses import dataclass, field

from . import lakeshore_clauses as C

STATES = ["WI", "IL", "MN", "MI", "IA", "IN"]
EDITIONS = ["2022-01", "2024-06"]
PRECEDENCE = {"bulletin": 3, "endorsement": 2, "amendatory": 2, "form": 1,
              "guideline": 0, "underwriting": -1}


@dataclass
class Span:
    span_id: str
    doc: str
    doc_type: str
    family: str
    section: str
    heading: str
    text: str
    clause: str               # topic key the span is about
    edition: str = ""
    state: str = ""           # "" means every state
    issued: str = ""
    superseded_by: str = ""
    links: list = field(default_factory=list)

    @property
    def full(self) -> str:
        return f"{self.section}. {self.heading}. {self.text}"


@dataclass
class Query:
    qid: str
    split: str                # eval, holdout, new
    stratum: str
    style: str                # kw, lay, identifier, numeric, negation, multihop
    text: str
    family: str
    state: str
    edition: str
    endorsements: tuple
    clause: str
    gold: dict                # span id -> grade
    governing: str


def _fill(text: str, ed: str) -> str:
    slots = {k: v[EDITIONS.index(ed)] for k, v in C.HO3_EDITION_SLOTS.items()}
    slots.update(f="40", g="1,200")
    return text.format(**slots)


def _underwriting(rng: random.Random) -> list[Span]:
    topics = [
        ("roofs", "roof age over 20 years, hail and windstorm history, shingle condition, require inspection photos before binding"),
        ("seasonal", "seasonal and vacant dwellings, winter heat, frozen pipes, water shut-off valves and low-temperature alarms"),
        ("sump", "basements with sump pumps, sewer back-up history, battery back-up pumps and the water back-up endorsement"),
        ("stoves", "wood stoves and fireplaces, chimney inspection, smoke and fire losses"),
        ("dogs", "dog bite liability, personal liability limits and medical payments to others"),
        ("homebiz", "home-based business exposures, customers on the premises, business exclusion and endorsements"),
        ("tnc", "drivers who deliver food or carry passengers for a fee, livery exclusion and the transportation network endorsement"),
        ("boats", "watercraft lay-up dates, winterizing, navigation limits on the Great Lakes"),
        ("restaurants", "restaurants and grocers, refrigeration, spoilage, business income and employee dishonesty"),
        ("trees", "older trees near dwellings, storm limbs, debris removal and tree limits"),
        ("mould", "older homes with mould history, moisture, leaks and the limited fungi endorsement"),
        ("claims", "claims frequency, prior water losses, roof claims and theft losses in the past five years"),
    ]
    out = []
    for i in range(132):
        name, words = topics[i % len(topics)]
        year = 2021 + (i % 5)
        st = rng.choice(STATES)
        text = (f"Underwriting bulletin for {st} agents: new business and renewals involving {words}. "
                f"Agents should collect the supporting documents listed in the appendix and refer exceptions "
                f"to the regional underwriting manager. Effective for policies written after {year}-{1 + i % 12:02d}-01.")
        out.append(Span(f"UW-{year}-{i:03d}", f"UB-{year}-{i:03d}", "underwriting", "ALL",
                        "Underwriting", f"Underwriting bulletin {name}", text, f"uw_{name}",
                        state=st, issued=f"{year}-{1 + i % 12:02d}-01"))
    return out


SIBLINGS = [  # other forms in the docstore that share HO-3 wording
    ("HO5", "HO-5", "2024-06", [("We do not insure", "We do not cover"), ("We cover", "We insure")],
     "Open perils form HO-5. "),
    ("HO4", "HO-4", "2023-01", [("building", "unit"), ("dwelling", "rented premises")],
     "Renters form HO-4. "),
    ("DP3", "DP-3", "2021-09", [("you", "the named insured"), ("We will pay", "The company will pay")],
     "Dwelling property form DP-3. "),
]
OTHER_BULLETINS = [
    ("catastrophe claim reporting", "insurers must report catastrophe claim counts weekly and assign an adjuster within 7 days of notice"),
    ("public adjusters", "a public adjuster contract must be disclosed to the insurer, and payment may name the public adjuster as payee"),
    ("contractor licensing", "after storms, insurers should remind policyholders to verify contractor licences and avoid assignment of benefits"),
    ("auto glass", "the policyholder may choose the glass repair shop and the insurer may not require a specific vendor"),
    ("total loss valuation", "vehicle total loss valuations must include taxes and fees and identify the comparable vehicles used"),
    ("grace period", "insurers must give a 30 day grace period for premium payments after a declared disaster"),
]
EXTRA_GUIDELINES = [
    ("CG-HO-003", "theft", "Claims guideline: theft losses. Obtain the police report number, proof of ownership and receipts for items over $1,000."),
    ("CG-HO-005", "fire", "Claims guideline: fire losses. Engage an origin and cause investigator when the fire started at night in an unoccupied room."),
    ("CG-HO-009", "liability", "Claims guideline: liability claims. Take a recorded statement from the insured within 48 hours and send a reservation of rights letter where coverage is in question."),
    ("CG-HO-011", "med_pay", "Claims guideline: medical payments to others. Pay reasonable bills promptly without an admission of liability."),
    ("CG-HO-017", "debris", "Claims guideline: debris removal. Obtain the hauling invoice and photographs of the debris before payment."),
    ("CG-HO-026", "appraisal", "Claims guideline: appraisal demands. Acknowledge the demand within 10 days and name our appraiser from the approved panel."),
    ("CG-HO-033", "loss_settlement", "Claims guideline: replacement cost holdback. Pay actual cash value first and release the depreciation holdback when repairs are complete."),
    ("CG-PA-002", "pap_otc", "Claims guideline: animal collisions. Deer strikes are paid under other than collision, not collision."),
    ("CG-PA-004", "pap_collision", "Claims guideline: collision losses. Obtain photographs, the repair estimate and the police report for losses over $5,000."),
    ("CG-PA-011", "pap_rental", "Claims guideline: rental while the car is repaired. Authorise rental in blocks of seven days."),
]


def _siblings() -> list[Span]:
    out = []
    for prefix, fam, ed, swaps, lead in SIBLINGS:
        for key, sec, head, text, pol, _, _ in C.HO3:
            t = _fill(text, "2024-06")
            for a, b in swaps:
                t = t.replace(a, b)
            out.append(Span(f"{prefix}-{ed}-{key}", f"{fam} ({ed})", "form", fam, sec, head,
                            lead + t, key, edition=ed))
    return out


def _other_bulletins() -> list[Span]:
    agency = {"WI": "WI-OCI", "IL": "IL-DOI", "MN": "MN-COM", "MI": "MI-DIFS",
              "IA": "IA-ID", "IN": "IN-IDOI"}
    out = []
    for j, st in enumerate(STATES):
        for i, (topic, rule) in enumerate(OTHER_BULLETINS):
            bid = f"{agency[st]}-{2023 + (i + j) % 3}-{20 + i:02d}"   # 20+ avoids real bulletin numbers
            fam = "PAP" if topic in ("auto glass", "total loss valuation") else "HO-3"
            out.append(Span(bid, bid, "bulletin", fam, "State bulletin", f"Bulletin {bid}",
                            f"{bid}. Bulletin on {topic}: {rule}.", f"other_{i}", state=st,
                            issued=f"{2023 + (i + j) % 3}-{1 + i:02d}-15"))
    return out


def _definitions() -> list[Span]:
    text = ("In this policy, you and your refer to the named insured shown in the Declarations and the "
            "spouse if a resident of the same household. Bodily injury means bodily harm, sickness or disease. "
            "Business includes a trade, profession or occupation. Residence premises means the one family "
            "dwelling where you reside. Water, as used in the water exclusion, includes flood water, surface water, "
            "water that backs up through sewers or drains, water from a sump, and water below the surface of the "
            "ground; water does not include water discharged from a plumbing system, which is covered by the water "
            "discharge peril, nor water from a heating system, nor water from a household appliance, nor water from "
            "a sprinkler system. Plumbing system means pipes, water heater, water supply lines and drains inside the "
            "dwelling. A household appliance is an appliance that uses water. Occurrence means an accident, including "
            "continuous or repeated exposure to substantially the same harmful conditions.")
    return [Span(f"HO3-{ed}-definitions", f"HO-3 ({ed})", "form", "HO-3", "Definitions",
                 "Definitions", text, "definitions", edition=ed) for ed in EDITIONS]


def build_corpus() -> dict[str, Span]:
    rng = random.Random(12)
    spans: list[Span] = []
    for ed in EDITIONS:
        for key, sec, head, text, pol, _, _ in C.HO3:
            spans.append(Span(f"HO3-{ed}-{key}", f"HO-3 ({ed})", "form", "HO-3", sec, head,
                              _fill(text, ed), key, edition=ed))
    for fam, bank, ed in (("PAP", C.PAP, "2023-01"), ("BOP", C.BOP, "2024-01"),
                          ("BOAT", C.BOAT, "2022-05")):
        for key, sec, head, text, pol, _, _ in bank:
            spans.append(Span(f"{fam}-{ed}-{key}", f"{fam} ({ed})", "form", fam, sec, head,
                              _fill(text, "2024-06"), key, edition=ed))
    spans += _definitions()
    for eid, fam, title, text, mod, _, _ in C.ENDORSEMENTS:
        spans.append(Span(eid.replace(" ", "-"), eid, "endorsement", fam, "Endorsement", title,
                          f"{eid}. {text}", mod))
    for st, (eid, title, text) in C.STATE_AMENDATORY.items():
        spans.append(Span(eid.replace(" ", "-"), eid, "amendatory", "HO-3", "Endorsement", title,
                          f"{eid}. {text}", "suit", state=st))
    for bid, st, key, issued, text, sup in C.BULLETINS:
        spans.append(Span(bid, bid, "bulletin", "HO-3", "State bulletin", f"Bulletin {bid}",
                          f"{bid}. {text}", key, state=st, issued=issued))
    for gid, key, issued, text, sup in C.GUIDELINES:
        fam = "PAP" if key.startswith("pap") else "HO-3"
        spans.append(Span(gid.replace(" ", "-"), gid, "guideline", fam, "Claims guideline",
                          f"Guideline {gid}", f"{gid}. {text}", key, issued=issued))
    for gid, key, text in EXTRA_GUIDELINES:
        fam = "PAP" if key.startswith("pap") else "HO-3"
        spans.append(Span(gid, gid, "guideline", fam, "Claims guideline", f"Guideline {gid}",
                          f"{gid}. {text}", key, issued="2024-03-01"))
    spans += _siblings() + _other_bulletins()
    spans += _underwriting(rng)
    corpus = {s.span_id: s for s in spans}
    for s in spans:                              # supersession
        for bid, _, _, _, _, sup in C.BULLETINS:
            if sup and s.span_id == sup:
                s.superseded_by = bid
        for gid, _, _, _, sup in C.GUIDELINES:
            if sup and s.span_id == sup.replace(" ", "-"):
                s.superseded_by = gid.replace(" ", "-")
    for s in spans:                              # links, both directions
        if s.doc_type in ("endorsement", "amendatory", "bulletin", "guideline"):
            for t in spans:
                if t.doc_type == "form" and t.clause == s.clause:
                    s.links.append(t.span_id)
                    t.links.append(s.span_id)
    return corpus


def admitted(span: Span, q) -> bool:
    """The admission check from earlier chapters: the policy record decides."""
    if span.doc_type == "underwriting":
        return False
    if span.family not in (q.family, "ALL"):
        return False
    if span.doc_type == "form" and span.family == "HO-3" and span.edition != q.edition:
        return False
    if span.state and span.state != q.state:
        return False
    if span.doc_type == "endorsement" and span.doc not in q.endorsements:
        return False
    return True
