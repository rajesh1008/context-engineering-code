"""Chapter 17 fixtures: Chapter 15's documents plus Lakeshore's endorsement chains.

Stand-in data. The documents are fixtures written for the lab; the extractor
is a seeded simulation of a model reading each span (sim-extract). The gold
edges, the multi-hop set and the control set are written by hand.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from datetime import date

from contextlab.graph import Candidate
from contextlab.policy import Request, Span
from corpus15 import ADJ, context_spans, governing_spans

NEW = [  # (sid, kind, topic, text, extra)
 ("LM-HO 0495E@2024-06#water_backup", "water_backup", "Water back-up limit enhancement. This endorsement modifies HO 04 95, Water Back-up and Sump Discharge or Overflow. It applies only if HO 04 95 is also shown in the declarations. The limit of liability stated in HO 04 95 is increased to US$25,000; the deductible stated in HO 04 95 continues to apply.", {}),
 ("LM-HO 2311@2025-01#mould", "mould", "Limited fungi, wet or dry rot coverage, 2025 edition. This endorsement modifies Section I Exclusions 2.c of HO-3. We will pay up to US$5,000 in the aggregate for loss caused by fungi or rot resulting from a peril insured against. Where the fungi or rot result from water which backs up through sewers or drains, payment is made within, and not in addition to, the limit of HO 04 95.", {}),
 ("LM-HO 0455@2024-06#roof", "roof", "Roof surface payment schedule. This endorsement modifies Section I Conditions D.2, loss settlement for roof surfaces, of HO-3. Roof surfaces more than 15 years old on the date of loss are settled at actual cash value using the schedule below; younger roof surfaces are settled as the policy provides.", {}),
]
REWRITE = {  # Chapter 15 spans whose text now states the relationship
 "LM-HO 2311@2022-01#mould": "Limited fungi, wet or dry rot coverage. This endorsement modifies Section I Exclusions 2.c of HO-3. We will pay up to US$5,000 in the aggregate for loss caused by fungi or rot resulting from a peril insured against. Where the fungi or rot result from water which backs up through sewers or drains, this limit is in addition to the limit of HO 04 95.",
 "FVB-HO 01@2023-01#water_backup": "Fox Valley Bank homeowners programme, water back-up. This endorsement replaces HO 04 95 for policies issued under the programme. We cover water which backs up through sewers or drains or overflows from a sump, up to US$15,000, deductible US$1,000.",
 "NWS-HO 02@2023-01#freezing": "Northwoods Savings seasonal dwelling. This endorsement modifies Section I Perils 4, freezing, of HO-3. Freezing of plumbing in a seasonal dwelling is covered only if the water supply was shut off and the system drained before the dwelling was closed for the season.",
 "LM-HO 0122 MN@2021-01#notice": "Special provisions, Minnesota. This endorsement modifies Section I Conditions B.1, duties after loss, of HO-3: late notice bars coverage only if it prejudiced our investigation.",
}


def spans():
    out = []
    for s in governing_spans():
        out.append(replace(s, text=REWRITE[s.sid]) if s.sid in REWRITE else s)
    for sid, topic, text, extra in NEW:
        doc, ver = sid.split("#")[0].split("@")
        out.append(Span(sid=sid, doc_id=doc, version=ver, kind="endorsement", topic=topic,
                        scope=topic, text=text, edition_family=doc, roles=ADJ, page=1,
                        effective_from=date(int(ver[:4]), int(ver[5:]), 1), **extra))
    return out + context_spans()


# Section references and names a reader resolves to nodes (entity resolution).
ALIASES = [
 (r"HO 04 95|Water Back-up and Sump Discharge", "HO 04 95"),
 (r"Section I Exclusions 2\.c", "clause:mould"), (r"Section I Conditions D\.2", "clause:roof"),
 (r"Section I Perils 4", "clause:freezing"), (r"Section I Conditions B\.1", "clause:notice"),
 (r"LM-HO 2311", "LM-HO 2311"),
]
PATTERNS = [  # what a model is asked to find, as (relation, cue)
 ("MODIFIES", r"modifies (.+?)(?: of HO-3|\. (?=[A-Z]))"), ("REQUIRES", r"applies only if ([^.]+?) is also shown"),
 ("REPLACES", r"replaces ([^.]+?) for"), ("LIMIT_WITHIN", r"within, and not in addition to, the limit of ([^.]+)"),
 ("LIMIT_IN_ADDITION", r"this limit is in addition to the limit of ([^.]+)"),
 ("SUPERSEDES", r"Supersedes ([^.(]+)"),
]


def resolve_alias(phrase):
    for pat, node in ALIASES:
        if re.search(pat, phrase):
            return node
    return None


def sim_extract(sp, seed=17, miss=0.04, wobble=0.18):
    """Stand-in for a model reading each span and proposing typed edges."""
    rng = random.Random(seed)
    out = []
    for s in sp:
        if s.kind == "form":
            out.append(Candidate(s.doc_id, "HAS_CLAUSE", f"clause:{s.topic}", s.sid, 0.99, s.text[:40]))
        if s.kind == "bulletin":
            out.append(Candidate(s.doc_id, "CONSTRAINS", f"clause:{s.topic}", s.sid,
                                 round(rng.uniform(0.82, 0.99), 2), s.text[:60]))
        for rel, cue in PATTERNS:
            for m in re.finditer(cue, s.text):
                dst = resolve_alias(m.group(1))
                if dst is None or rng.random() < miss:
                    continue                      # the model missed it or could not name it
                conf = round(min(0.99, rng.uniform(0.86, 0.99) - (wobble if rng.random() < 0.12 else 0)), 2)
                out.append(Candidate(s.doc_id, rel, dst, s.sid, conf, m.group(0)[:80]))
        if s.kind in ("guideline", "advisory", "note") and "endorsement" in s.text:
            out.append(Candidate(s.doc_id, "MODIFIES", "HO 04 95", s.sid, 0.74, s.text[:60]))
        if s.kind == "bulletin" and rng.random() < 0.35:   # a plausible over-reach
            out.append(Candidate(s.doc_id, "MODIFIES", f"clause:{s.topic}", s.sid, 0.81, s.text[:60]))
        if s.kind == "endorsement" and s.topic == "roof":
            out.append(Candidate(s.doc_id, "MODIFIES", "clause:debris", s.sid, 0.72, "settled as the policy provides"))
    return out


GOLD_EDGES = {
 ("LM-HO 0495E", "MODIFIES", "HO 04 95"), ("LM-HO 0495E", "REQUIRES", "HO 04 95"),
 ("LM-HO 2311", "MODIFIES", "clause:mould"), ("LM-HO 2311", "LIMIT_WITHIN", "HO 04 95"),
 ("LM-HO 2311", "LIMIT_IN_ADDITION", "HO 04 95"),
 ("LM-HO 0455", "MODIFIES", "clause:roof"), ("FVB-HO 01", "REPLACES", "HO 04 95"),
 ("NWS-HO 02", "MODIFIES", "clause:freezing"), ("LM-HO 0122 MN", "MODIFIES", "clause:notice"),
}


@dataclass(frozen=True)
class Q:
    qid: str
    template: str
    question: str
    req: Request
    gold: tuple                       # every span the answer needs
    inapplicable: tuple = ()          # spans that must be cited as not applying
    relation: str | None = None       # the limit relation the answer depends on
    asked: str | None = None          # the clause the question is about


SAYS = {
 "backup": ["sewer backed up into the finished basement, what is the most we pay",
            "floor drain overflowed in the storm, which limit and deductible apply",
            "sump pit overflowed and soaked the carpet, how much is covered"],
 "mould": ["black mould on the drywall two weeks after the sewer backup, how much can we pay for the mould",
           "musty smell and rot behind the panelling after the drain overflowed, what limit applies to the mould",
           "fungus growing in the basement after the back-up, is the mould paid on top of the back-up limit"],
 "roof": ["hail on a nineteen-year-old roof and the shingles are discontinued, what do we pay",
          "wind tore shingles off an old roof and the colour cannot be matched, full or partial replacement and at what value",
          "storm damage to a roof installed in 2007, matching shingles unavailable, how is the roof settled"],
 "cabin": ["pipes froze at the lake cabin after a power cut and the owners had not drained it, is it covered",
           "seasonal cottage, utility outage in January, burst pipes, was draining required",
           "frozen plumbing at a closed-up summer home after the electricity failed, do we pay"],
}
STATES = ["WI", "MN", "IL", "IA", "MI", "IN"]


def _req(tenant, state, edition, endos, on, cause, risk=None):
    return Request(role="adjuster", tenant=tenant, purpose="coverage-determination", line="HO",
                   policy_state=state, risk_state=risk or state, garaging_state=state,
                   editions={"HO-3": edition, "PAP": "2018-01"}, endorsements=frozenset(endos),
                   date_of_loss=on, cause=cause)


def _day(rng, lo, hi):
    return date.fromordinal(rng.randrange(lo.toordinal(), hi.toordinal()))


def multi_hop(seed=17, per=8):
    """Forty-eight determinations where more than one span decides the answer."""
    rng, out = random.Random(seed), []
    for i in range(per):
        st = rng.choice(STATES)
        on = _day(rng, date(2024, 7, 1), date(2026, 9, 30))
        out.append(Q(f"T1-{i}", "enhancement", rng.choice(SAYS["backup"]),
                     _req("lakeshore", st, "2024-06", {"HO 04 95@2024-06", "LM-HO 0495E@2024-06"}, on, "water_backup"),
                     ("HO 04 95@2024-06#water_backup", "LM-HO 0495E@2024-06#water_backup"), asked="water_backup"))
        on = _day(rng, date(2024, 7, 1), date(2026, 9, 30))
        out.append(Q(f"T2-{i}", "requirement unmet", rng.choice(SAYS["backup"]) + ", Fox Valley Bank policy",
                     _req("fvb", st, "2024-06", {"FVB-HO 01@2023-01", "LM-HO 0495E@2024-06"}, on, "water_backup"),
                     ("FVB-HO 01@2023-01#water_backup", "LM-HO 0495E@2024-06#water_backup"),
                     ("LM-HO 0495E@2024-06#water_backup",), asked="water_backup"))
        on = _day(rng, date(2025, 2, 1), date(2026, 9, 30))
        cause = "water_backup" if i % 2 == 0 else "mould"   # coded on the first peril, or recoded
        out.append(Q(f"T3-{i}", "shared limit", rng.choice(SAYS["mould"]),
                     _req("lakeshore", st, "2024-06", {"HO 04 95@2024-06", "LM-HO 2311@2025-01"}, on, cause),
                     ("HO-3@2024-06#mould", "HO 04 95@2024-06#water_backup", "LM-HO 2311@2025-01#mould"),
                     relation="LIMIT_WITHIN", asked="mould"))
        on = _day(rng, date(2023, 1, 1), date(2024, 12, 31))
        out.append(Q(f"T5-{i}", "additional limit", rng.choice(SAYS["mould"]),
                     _req("lakeshore", st, "2022-01", {"HO 04 95@2019-03", "LM-HO 2311@2022-01"}, on, cause),
                     ("HO-3@2022-01#mould", "HO 04 95@2019-03#water_backup", "LM-HO 2311@2022-01#mould"),
                     relation="LIMIT_IN_ADDITION", asked="mould"))
        st4 = "WI" if i % 2 == 0 else "MN"
        on = _day(rng, date(2025, 6, 2), date(2026, 9, 30))
        bul = "WI-OCI-2025-03@2025-03-15#roof" if st4 == "WI" else "MN-COM-2025-06@2025-06-02#roof"
        out.append(Q(f"T4-{i}", "endorsement and bulletin", rng.choice(SAYS["roof"]),
                     _req("lakeshore", st4, "2024-06", {"LM-HO 0455@2024-06"}, on, "roof"),
                     ("HO-3@2024-06#roof", "LM-HO 0455@2024-06#roof", bul), asked="roof"))
        on = _day(rng, date(2025, 5, 1), date(2026, 9, 30))
        ed = rng.choice(["2022-01", "2024-06"])
        out.append(Q(f"T6-{i}", "programme and bulletin", rng.choice(SAYS["cabin"]),
                     _req("nws", "IL", ed, {"NWS-HO 02@2023-01"}, on, "freezing"),
                     (f"HO-3@{ed}#freezing", "NWS-HO 02@2023-01#freezing", "IL-DOI-2025-05@2025-05-01#freezing"),
                     asked="freezing"))
    return out


CONTROL_SAYS = {
 "water_backup": SAYS["backup"], "roof": ["hail damaged the roof, partial or full replacement",
                                          "wind lifted shingles on one slope, how do we settle the roof"],
 "debris": ["derecho dropped a tree on the garage, is removal paid as debris"],
 "payment": ["the insured agreed the amount last week, when must payment issue"],
 "freezing": ["pipes froze while the family was away and the heat was on, covered"],
}


def control(seed=171, n=48):
    """Single-span determinations, where a graph should change nothing."""
    from corpus15 import reviewer_answer
    rng, sp, out = random.Random(seed), spans(), []
    cases = [("water_backup", None), ("roof", "IL"), ("roof", "WI"), ("debris", "IA"),
             ("payment", "IN"), ("freezing", "WI")]
    for i in range(n):
        topic, st = cases[i % len(cases)]
        st = st or rng.choice(STATES)
        on = _day(rng, date(2025, 6, 2), date(2026, 9, 30))
        endos = {"HO 04 95@2024-06"} if topic == "water_backup" else set()
        gold = reviewer_answer(sp, "HO", topic, st, st, "2024-06", endos, on, "lakeshore")
        out.append(Q(f"C-{i}", "single span", rng.choice(CONTROL_SAYS[topic]),
                     _req("lakeshore", st, "2024-06", endos, on, topic), (gold,), asked=topic))
    return out
