"""Chapter 21 fixtures: claimant and adjuster histories for the full mix.

Everything here is synthetic. The 1,000 determinations are Chapter 15's full
mix (seed 15). Each is assigned a claimant; about three in ten claimants have
claimed with Lakeshore before, and their history is a stream of episodes
(calls, emails, adjuster notes, letters, corrections) of the kind Lakeshore's
claims system and CRM hold. An extractor stand-in proposes memories from each
episode; the write policy decides what is kept.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from contextlab.memory import Candidate

GO_LIVE = date(2026, 10, 5)          # memory switched on for the replay
WATER = {"freezing", "water_backup", "mould"}
SURNAMES = ["Novak", "Olson", "Schmidt", "Larson", "Vang", "Kowalski", "Garcia", "Johnson",
            "Peterson", "Nguyen", "Hansen", "Moua", "Becker", "Lindqvist", "Ortiz", "Wagner",
            "Thao", "Fischer", "Miller", "Anderson", "Lee", "Brooks", "Kaur", "Svoboda"]
FIRST = ["Ann", "Mark", "Rosa", "Dale", "Pao", "Greta", "Luis", "Karen", "Tom", "Mai",
         "Erik", "Joy", "Hugo", "Lena", "Sam", "Nora", "Yer", "Paul", "Ruth", "Ivan"]

# semantic fields, their values, and the chance a later episode changes them
FIELDS = {
    "contact.channel": (["phone", "email", "text message"], 0.22),
    "contact.time": (["mornings", "after 5 pm", "weekends only"], 0.18),
    "letters.language": (["English", "Spanish", "Hmong"], 0.03),
    "basement": (["finished basement", "unfinished basement", "no basement"], 0.06),
    "sump_pump": (["sump pump installed", "battery back-up sump pump fitted", "no sump pump"], 0.25),
    "backflow_valve": (["backflow valve fitted", "no backflow valve"], 0.20),
    "roof.replaced": (["roof replaced 2016", "roof replaced 2021", "roof replaced 2024"], 0.12),
    "occupancy": (["occupied all year", "seasonal cabin, closed November to April"], 0.04),
    "authorised_contact": (["spouse authorised to discuss the claim", "no one else authorised"], 0.10),
}
CTYPE_OF = {"contact.channel": "preference", "contact.time": "preference",
            "letters.language": "preference", "authorised_contact": "household"}
LABEL = {"contact.channel": "contact preference", "contact.time": "best time to call",
         "letters.language": "language for letters", "basement": "basement",
         "sump_pump": "sump pump", "backflow_valve": "backflow valve",
         "roof.replaced": "roof replaced", "occupancy": "occupancy",
         "authorised_contact": "authorised to discuss the claim"}
ASK = {"contact.channel": "how does {s} prefer to be contacted",
       "contact.time": "what is the best time to call {s}",
       "letters.language": "which language does {s} want for letters",
       "basement": "what kind of basement does the {s} house have",
       "sump_pump": "does the {s} house have a sump pump",
       "backflow_valve": "is a backflow valve fitted at the {s} house",
       "roof.replaced": "when was the roof replaced at the {s} house",
       "occupancy": "what is the occupancy of the {s} property",
       "authorised_contact": "who is authorised to discuss the claim for {s}"}
NOISE = {  # what an extractor also proposes, by type, with example text
    "policy_fact": ["deductible is US${d:,}", "back-up limit is US${l:,} under HO 04 95",
                    "HO 04 95 is on the policy"],
    "status": ["payment of US${p:,} pending", "claim reopened for a supplement",
               "reserve raised on the claim"],
    "inference": ["claimant likely to dispute the settlement", "coverage probably applies",
                  "basement prone to flooding", "claimant seems to exaggerate damage"],
    "small_talk": ["mentioned a grandson's hockey game", "talked about the Packers",
                   "said the weather has been strange"],
    "sensitive": ["mentioned a recent hospital stay", "said money is tight this year",
                  "mentioned a divorce in progress"],
}


def h(*parts) -> float:
    x = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(x[:8], 16) / 2**32


@dataclass
class Claimant:
    cid: str
    name: str
    policy: str
    tenant: str
    repeat: bool
    facts: dict = field(default_factory=dict)        # field -> [(date, value), ...]
    deductible: tuple = (1000, 1000)                 # at the prior claim, now
    limit: tuple = (5000, 10000)
    prior: list = field(default_factory=list)        # (claim id, date, cause)
    other_tenant: str | None = None                  # a second policy under another brand


@dataclass(frozen=True)
class Episode:
    eid: str
    cid: str
    when: date
    kind: str            # fnol_call, claimant_call, email, adjuster_note, letter, correction
    author: str          # claimant, adjuster, system


def claimants(dets, seed=21):
    rng = random.Random(seed)
    out = {}
    for d in dets:
        cid = f"C{int(d.did[1:]):04d}"
        name = f"{rng.choice(FIRST)} {rng.choice(SURNAMES)}"
        st = d.req.policy_state or "WI"
        c = Claimant(cid, name, f"LMH-{st}-{rng.randrange(10**6, 10**7):07d}", d.req.tenant,
                     repeat=rng.random() < 0.31)
        if c.repeat:
            n = 1 if rng.random() < 0.7 else 2
            for i in range(n):
                when = date(2021, 3, 1) + timedelta(days=rng.randrange(0, 1900))
                c.prior.append((f"CL-{cid[1:]}-{i}", when, rng.choice(
                    ["water_backup", "freezing", "roof", "debris", "mould", "animal"])))
            c.prior.sort(key=lambda p: p[1])
            for f, (vals, change) in FIELDS.items():
                if rng.random() < 0.55:
                    v0 = rng.choice(vals)
                    v1 = rng.choice([v for v in vals if v != v0]) if rng.random() < change else None
                    c.facts[f] = [(None, v0)] + ([(None, v1)] if v1 else [])
            first = c.prior[0][1]
            d0 = rng.choice([500, 1000, 1000, 2500])
            c.deductible = (d0, d0 if rng.random() < 0.7 else rng.choice([1000, 2500, 5000]))
            l0 = 5000 if first < date(2024, 6, 1) else 10000
            c.limit = (l0, 10000 if rng.random() < 0.8 else l0)
            if d.req.tenant != "lakeshore" or rng.random() < 0.03:
                c.other_tenant = "fvb" if d.req.tenant != "fvb" else "lakeshore"
        out[d.did] = c
    return out


def episodes(c: Claimant):
    """The claimant's history before go-live, in date order."""
    rng = random.Random(c.cid)
    eps = []
    for cl, when, _ in c.prior:
        plan = ["fnol_call"] + ["adjuster_note"] * rng.randint(2, 5) + \
               [rng.choice(["claimant_call", "email"]) for _ in range(rng.randint(1, 4))] + ["letter"]
        for i, k in enumerate(plan):
            author = {"adjuster_note": "adjuster", "letter": "system"}.get(k, "claimant")
            eps.append(Episode(f"{cl}/E{i:02d}", c.cid, when + timedelta(days=i * 3), k, author))
    return sorted(eps, key=lambda e: (e.when, e.eid))


def observe(c: Claimant, eps):
    """Date each fact by the episode in which Lakeshore was told it: the first value early
    in the first claim, a changed value in a later episode the claimant started."""
    told = [e for e in eps if e.author == "claimant"]
    for f, vals in c.facts.items():
        i = int(h(c.cid, f) * min(2, len(told)))
        dated = [(told[i].when, vals[0][1], told[i].eid)]
        later = [e for e in told if e.when > told[i].when]
        if len(vals) > 1 and later:
            e = later[int(h(c.cid, f, "later") * len(later))]
            dated.append((e.when, vals[1][1], e.eid))
        c.facts[f] = dated


def extract(c: Claimant, eps):
    """sim-extract-mem: what an extractor proposes from each episode."""
    cands, subj = [], f"claimant:{c.cid}"
    at = defaultdict(list)
    for f, vals in c.facts.items():
        for when, v, eid in vals:
            at[eid].append((f, v))
    for e in eps:
        src = {"claimant": "claimant", "adjuster": "adjuster", "system": "system"}[e.author]
        for f, v in at[e.eid]:
            ct = CTYPE_OF.get(f, "property")
            sj = subj if ct != "property" else f"policy:{c.policy}"
            cands.append(Candidate(sj, ct, f"{c.name.split()[1]}: {LABEL[f]}: {v}",
                                   src, e.eid, e.when, c.tenant, f, v))
        if e.kind == "fnol_call":
            cl = e.eid.split("/")[0]
            cause = next(p[2] for p in c.prior if p[0] == cl)
            cands.append(Candidate(subj, "event", f"{c.name.split()[1]}: prior claim {cl} on "
                                   f"{e.when.isoformat()}, {cause.replace('_', ' ')}; amounts in the claims system",
                                   "system", cl, e.when, c.tenant))
        for ct, opts in NOISE.items():
            p = {"policy_fact": 0.30, "status": 0.25, "inference": 0.22, "small_talk": 0.35,
                 "sensitive": 0.05}[ct]
            if h(e.eid, ct) < p:
                t = opts[int(h(e.eid, ct, "t") * len(opts))].format(
                    d=c.deductible[0], l=c.limit[0], p=int(500 + 4000 * h(e.eid, "p")))
                fld = ("deductible" if "deductible" in t else "limit" if "limit" in t
                       else None) if ct == "policy_fact" else None
                val = (c.deductible[0] if fld == "deductible" else c.limit[0]) if fld else None
                srcx = "model" if ct == "inference" else src
                cands.append(Candidate(subj, ct, f"{c.name.split()[1]}: {t}", srcx, e.eid, e.when,
                                       c.tenant, fld, str(val) if val else None))
    return cands


# Corrections adjusters made to the assistant before go-live; consolidation turns them into lessons.
CORRECTIONS = [
    ("tbrennan", date(2026, 2, 17), ("HO", "WI", "water"),
     "Not a back-up. Frozen supply line split inside the north wall of a 1950s house in Green Bay "
     "after the February thaw. On older Green Bay houses after a thaw, check the north wall for a "
     "frozen line before coding back-up.", "freezing"),
]
GOOD = [  # (adjuster, date, scope, lesson text): handling advice that proposes no cause
    ("tbrennan", ("HO", "*", "roof"), "Ask for photos of every slope visible from the street before settling a matching claim."),
    ("tbrennan", ("HO", "WI", "roof"), "Hail claims after a June storm in Brown County: check the date against the storm report."),
    ("jkoenig", ("HO", "MN", "water"), "Seasonal cabins: confirm who checked the heat and when before deciding a freezing claim."),
    ("jkoenig", ("HO", "MN", "roof"), "Ice dams on low-slope roofs: ask whether the gutters were heated."),
    ("mruiz", ("HO", "IL", "water"), "Chicago two-flats: ask which unit's drain backed up first."),
    ("mruiz", ("HO", "IL", "debris"), "Parkway trees belong to the city; ask before paying debris removal."),
    ("akhan", ("HO", "IA", "debris"), "After a derecho, photograph the stump before the tree service removes it."),
    ("akhan", ("HO", "*", "notice"), "Late notice: record the date the insured first knew of the damage, not the date of loss."),
    ("dpoole", ("HO", "MI", "water"), "Ask how long the leak ran; the mould sub-limit depends on it."),
    ("dpoole", ("PAP", "*", "animal"), "Deer strikes: ask for the police report number if one was taken."),
    ("lsato", ("HO", "IN", "payment"), "Confirm the mortgagee on the payment before the agreed amount is issued."),
    ("lsato", ("PAP", "*", "oos_limits"), "Out-of-state accidents: record the state of the accident in the claim before asking."),
    ("tbrennan", ("HO", "WI", "water"), "Ask whether the sump pump has a battery back-up and when it was last tested."),
]


def family(cause):
    return "water" if cause in WATER else cause


# Adjusters' own preferences, written from the settings screen (source: the adjuster).
ADJUSTERS = {
    "tbrennan": [("letter.order", "state the deductible before the limit"),
                 ("history.format", "list prior claims with dates and causes, newest first")],
    "jkoenig": [("letter.order", "state the limit before the deductible"),
                ("summary.length", "three sentences at most")],
    "mruiz": [("letter.language", "offer a Spanish copy when the claimant asked for one")],
    "akhan": [("summary.length", "one paragraph, then the open questions")],
    "dpoole": [("history.format", "prior claims only if within five years")],
    "lsato": [("letter.order", "state the payment date first")],
}


def adjuster_for(did):
    names = sorted(ADJUSTERS)
    return names[int(h("assign", did) * len(names))]
