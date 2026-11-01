"""Chapter 18 stand-in: Lakeshore's endorsements, bulletins, policy records and claims.

Everything here is synthetic and deterministic. The corpus has the shape the
chapter needs: endorsements that refer to other endorsements (sometimes by
identifier, sometimes only by name), editions that differ in the number that
matters, state bulletins that look alike across states and supersede one
another, claims-manual pages that share the vocabulary without the authority,
and policy records that say which edition of each endorsement a policy carries.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date

STATES = {"WI": "Wisconsin", "IA": "Iowa", "MN": "Minnesota",
          "IL": "Illinois", "MI": "Michigan", "IN": "Indiana"}
GRAPH_BUILT = date(2026, 6, 30)          # Chapter 17's last full graph build

# family -> (name, what it covers, base clause it modifies, peril words)
FAM = {
    "WB": ("water back-up and sump discharge", "direct physical loss caused by water which backs up through sewers or drains or overflows or is discharged from a sump",
           "HO-EXC-WATER", "sewer drain sump backup basement water overflow"),
    "SL": ("service line", "physical damage to underground piping and wiring that serve the dwelling, including excavation",
           "HO-EXC-EARTH", "lateral pipe line underground excavation yard collapse"),
    "EB": ("equipment breakdown", "mechanical or electrical breakdown of covered home equipment, including sump pumps and furnaces",
           "HO-EXC-MECH", "pump motor breakdown electrical furnace failed"),
    "OL": ("ordinance or law", "the increased cost to repair or rebuild the dwelling to meet a building code or ordinance",
           "HO-EXC-ORD", "code ordinance rebuild permit upgrade city requires"),
    "MD": ("limited fungi, wet rot and mould", "fungi, wet rot and mould that result from a covered water loss",
           "HO-EXC-FUNGI", "mould mold rot fungi moisture drywall"),
    "RS": ("roof surfacing payment schedule", "roof surfacing, paid by a schedule that reduces payment with the age of the roof",
           "HO-SET-LOSS", "roof shingle hail age surfacing"),
    "HS": ("home-sharing host activities", "the dwelling and contents while rented to guests through a home-sharing platform",
           "HO-EXC-BUSINESS", "guest rental host platform short-term stay"),
    "SP": ("scheduled personal property", "scheduled articles such as jewellery, watches and fine arts",
           "HO-LIM-SPECIAL", "jewellery ring watch art scheduled stolen"),
    "IR": ("identity recovery", "expenses to recover from identity fraud",
           "HO-LIM-SPECIAL", "identity fraud stolen credit"),
    "TP": ("trees, shrubs and plants increased limit", "trees, shrubs and plants on the residence premises",
           "HO-LIM-TREES", "tree shrub plant storm fell"),
}
# family -> references in its text: (target family, sentence pattern)
REFS = {
    "MD": [("WB", "This endorsement applies only to mould that results from a loss covered under {ref}.")],
    "WB": [("EB", "Where a sump pump has failed, the failure of the pump itself is governed by {ref}."),
           ("SL", "The limit for this endorsement is shared with, and not in addition to, the limit of {ref}.")],
    "SL": [("OL", "The increased cost of excavation to meet a code is paid only under {ref}.")],
    "RS": [("OL", "Code-required roof upgrades are paid only under {ref}.")],
    "HS": [("SP", "Scheduled articles belonging to guests are excluded unless scheduled under {ref}.")],
    "TP": [], "EB": [], "OL": [], "SP": [], "IR": [],
}
EDITIONS = {f: ["2021-07", "2024-04"] for f in FAM}
EDITIONS.update({"WB": ["2020-01", "2022-10", "2025-03"], "SL": ["2021-07", "2025-03"]})
ENTRY = ["WB", "MD", "SL", "OL", "RS", "HS", "TP"]       # perils a claim starts from


@dataclass
class Span:
    sid: str
    kind: str                  # form | endorsement | bulletin | manual
    family: str
    text: str
    edition: str = ""
    state: str = ""
    effective: date | None = None
    until: date | None = None  # bulletins: superseded on this date
    refs: list = field(default_factory=list)   # [(family, by_id)]

    @property
    def tokens(self) -> int:
        return int(len(self.text.split()) * 1.3)


def _limit(f, ed):
    base = {"WB": 5000, "SL": 10000, "EB": 50000, "OL": 10, "MD": 5000,
            "RS": 0, "HS": 25000, "SP": 0, "IR": 15000, "TP": 5}[f]
    return base * (1 + EDITIONS[f].index(ed))


def _endorsement(f, ed, rng):
    name, cover, clause, _ = FAM[f]
    lim = _limit(f, ed)
    lim_txt = (f"The limit of liability for this coverage is US${lim:,}." if f not in ("OL", "TP", "RS", "SP")
               else f"The limit is {lim}% of the Coverage A limit." if f in ("OL", "TP")
               else "Payment follows the schedule printed in this endorsement.")
    body = [f"Endorsement LK-{f} edition {ed}, {name}.",
            f"This endorsement modifies {clause} of the homeowners form.",
            f"We cover {cover}.", f"Typical losses: {FAM[f][3]}.", lim_txt]
    refs = []
    for tgt, pat in REFS[f]:
        by_id = ed == EDITIONS[f][-1] and rng.random() > 0.25
        ref = (f"endorsement LK-{tgt}" if by_id else f"the {FAM[tgt][0]} endorsement, if attached")
        body.append(pat.format(ref=ref)); refs.append((tgt, by_id))
    body.append("All other provisions of the policy apply.")
    return Span(f"LK-{f}@{ed}", "endorsement", f, " ".join(body), edition=ed, refs=refs)


def _forms():
    seen, out = set(), []
    for f, (name, cover, clause, words) in FAM.items():
        if clause in seen:
            continue
        seen.add(clause)
        out.append(Span(clause, "form", clause,
                        f"Homeowners form clause {clause}. We do not cover loss caused by "
                        f"{words.split()[0]} or related causes except as provided by an attached "
                        f"endorsement. Section I conditions apply to {name} losses."))
    return out


def _bulletins(rng):
    out = []
    for st, sname in STATES.items():
        for f in rng.sample(list(FAM), 5):
            old = date(2022, rng.randint(1, 12), 1)
            new = date(rng.choice([2025, 2026]), rng.randint(1, 10), 1)
            for i, (eff, until) in enumerate([(old, new), (new, None)]):
                n = 5000 * (i + 2)
                sid = f"BUL-{st}-{f}-{eff:%Y%m}"
                txt = (f"Bulletin {sid}. For homeowners policies written in {sname}, the "
                       f"{FAM[f][0]} endorsement (LK-{f}) may not apply a deductible or sublimit "
                       f"that reduces payment below US${n:,} for losses on or after "
                       f"{eff:%d %B %Y}. Insurers must apply this to {FAM[f][3].split()[0]} claims.")
                if i:
                    txt += f" This bulletin supersedes BUL-{st}-{f}-{old:%Y%m}."
                out.append(Span(sid, "bulletin", f, txt, state=st, effective=eff, until=until))
    return out


def _manual(rng):
    out = []
    for f, (name, cover, clause, words) in FAM.items():
        for i in range(2):
            st = rng.choice(list(STATES.values()))
            out.append(Span(f"MAN-{f}-{i}", "manual", f,
                            f"Claims manual: handling {name} claims. Adjusters in {st} often see "
                            f"{words} losses. Confirm the endorsement edition on the declarations, "
                            f"photograph the damage and note whether the {FAM[f][3].split()[0]} cause "
                            f"was sudden. See the endorsement and any state bulletin for limits."))
    return out


def build(seed: int = 18):
    rng = random.Random(seed)
    spans = _forms() + [_endorsement(f, ed, rng) for f in FAM for ed in EDITIONS[f]]
    spans += _bulletins(rng) + _manual(rng)
    return {s.sid: s for s in spans}


@dataclass
class Policy:
    pid: str
    state: str
    issued: date
    schedule: dict             # family -> edition on the declarations


def policies(n: int = 400, seed: int = 7):
    rng, out = random.Random(seed), []
    for i in range(n):
        st = rng.choice(list(STATES))
        issued = date(rng.choice([2021, 2022, 2023, 2024, 2025]), rng.randint(1, 12), 1)
        sched = {}
        for f in FAM:
            if rng.random() < 0.7:
                eds = [e for e in EDITIONS[f] if date(int(e[:4]), int(e[5:]), 1) <= issued]
                sched[f] = eds[-1] if eds else EDITIONS[f][0]
        out.append(Policy(f"P-{40000 + i}", st, issued, sched))
    return out


# Adjuster-style descriptions of each entry peril, with the shorthand adjusters use.
PHRASE = {
    "WB": "sump quit in the storm, 6in of water in finished bsmt, WBU?",
    "MD": "mould behind bsmt drywall 3 wks after the drain backed up",
    "SL": "sewer lateral collapsed under front yard, needs excavating",
    "RS": "hail on a 14yr old shingle roof",
    "HS": "guest's watch went missing during a short-term stay",
    "OL": "kitchen fire, city requires permit upgrades on rebuild, o&l?",
    "TP": "storm brought down two maples in the yard",
}
SHORTHAND = {"bsmt": "basement", "wbu": "water back-up sump discharge endorsement",
             "o&l": "ordinance or law endorsement", "dol": "date of loss",
             "wks": "weeks", "14yr": "fourteen year", "6in": "six inches"}


@dataclass
class Claim:
    cid: str
    policy: Policy
    loss: date
    entries: list              # entry families, in the order the adjuster wrote them
    question: str
    required: set              # governing spans (the policy record is always present)
    cause_codes: list          # structured loss-cause codes on the claim record


def closure(pol, entries):
    seen, todo = [], list(entries)
    while todo:
        f = todo.pop(0)
        if f in seen or f not in pol.schedule:
            continue
        seen.append(f)
        todo += [t for t, _ in REFS[f]]
    return seen


def in_force(b, on):
    return b.effective <= on and (b.until is None or on < b.until)


def governing(pol, entries, loss, corpus):
    fams = closure(pol, entries)
    req = {f"LK-{f}@{pol.schedule[f]}" for f in fams}
    req |= {FAM[f][2] for f in entries if f in pol.schedule}
    req |= {s.sid for s in corpus.values() if s.kind == "bulletin" and s.state == pol.state
            and s.family in fams and in_force(s, loss)}
    return req


def claims(corpus, n, hard, seed):
    """Hard claims start from a peril whose endorsement refers onwards; easy ones do not."""
    rng, pols, out = random.Random(seed), policies(), []
    pool = ["WB", "MD", "SL", "RS", "HS"] if hard else ["OL", "TP"]
    while len(out) < n:
        pol = rng.choice(pols)
        k = 2 if hard and rng.random() < 0.2 else 1
        entries = rng.sample(pool, k)
        if any(f not in pol.schedule for f in entries):
            continue
        loss = date(rng.choice([2025, 2026]), rng.randint(1, 10), rng.randint(1, 28))
        if loss < pol.issued:
            continue
        req = governing(pol, entries, loss, corpus)
        if hard and len(req) < 4:
            continue
        q = (f"{pol.state} {pol.pid}, DOL {loss:%m/%d/%y}: " + "; also ".join(PHRASE[f] for f in entries)
             + ". Covered, and what limit applies?")
        codes = list(entries) if rng.random() > 0.06 else entries[:1]   # some records carry one code
        out.append(Claim(f"{'H' if hard else 'E'}{seed}-{len(out):03d}", pol, loss, entries, q, req, codes))
    return out
