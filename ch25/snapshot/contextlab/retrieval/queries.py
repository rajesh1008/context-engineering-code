"""The 1,000 full-mix determinations as retrieval queries (seeded)."""
import random

from . import lakeshore_clauses as C
from .corpus import EDITIONS, PRECEDENCE, STATES, Query

CITY = {"WI": "Green Bay", "IL": "Rockford", "MN": "Duluth", "MI": "Grand Rapids",
        "IA": "Cedar Rapids", "IN": "Fort Wayne"}
BANKS = {"HO-3": C.HO3, "PAP": C.PAP, "BOP": C.BOP, "BOAT": C.BOAT}
FORM_ED = {"PAP": "2023-01", "BOP": "2024-01", "BOAT": "2022-05"}
ENDORSE = {e[0]: e for e in C.ENDORSEMENTS}
MULTIHOP = {  # bulletin id -> adjuster question that needs the bulletin
 "WI-OCI-2025-03": "Wisconsin frozen pipe claim: how many days to acknowledge and inspect, and what heating evidence must a denial document",
 "IA-ID-2024-08": "Iowa derecho, fallen tree on the garage: does the extra percentage for hauling apply before the tree cap",
 "MN-COM-2025-06": "Minnesota hail on the roof and the new shingles will not match: must we replace the whole slope facing the street",
 "IL-DOI-2025-02": "Illinois renewal: sewer back-up is excluded, must the back-up rider be offered and disclosed",
 "MI-DIFS-2024-06": "Michigan insured told us about the loss four months late: can we deny for late notice",
 "IN-IDOI-2025-01": "Indiana agreed claim: how many days to send payment before interest starts",
 "WI-OCI-2024-11": "Wisconsin denial for a slow leak: what evidence must the letter state",
 "IA-ID-2023-04": "Iowa mold denial: what must the letter reference",
}
NUMERIC = [  # (family, clause, query, needs endorsement, governing override)
 ("HO-3", "freezing", "Wisconsin frozen pipe claims must be acknowledged within 10 days, which bulletin says so", None, "WI-OCI-2025-03"),
 ("HO-3", "water_exclusion", "$10,000 per occurrence limit for sump discharge", "LM-HO 2307", "LM-HO-2307"),
 ("HO-3", "mould", "$5,000 aggregate for fungi and rot", "LM-HO 2311", "LM-HO-2311"),
 ("HO-3", "trees", "$2,500 for any one tree and 10% of Coverage A", "LM-HO 2320", "LM-HO-2320"),
 ("HO-3", "loss_settlement", "up to 25% of Coverage A for code upgrades", "LM-HO 2330", "LM-HO-2330"),
 ("HO-3", "loss_settlement", "insurance is 80% or more of full replacement cost", None, None),
 ("HO-3", "appraisal", "choose an appraiser within 20 days of a written request", None, None),
 ("HO-3", "med_pay", "expenses incurred within three years of the accident, guest injured at the house", None, None),
 ("HO-3", "suit", "action must be started within two years after the date of loss", None, None),
 ("BOAT", "boat_layup", "out of commission from November 1 until April 15", None, None),
 ("BOAT", "boat_tow", "up to $1,000 for towing on the lake", None, None),
 ("PAP", "pap_rental", "$40 per day and $1,200 total while the car is repaired", None, None),
]
NEGATION = [  # (clause, question that asks where cover is taken away, endorsement on file)
 ("mould", "mould that is not hidden in the walls and not from a covered leak: where does it say we do not pay", "LM-HO 2311"),
 ("water_exclusion", "water rising through the basement floor from the ground, not from the sump: which wording excludes it", "LM-HO 2307"),
 ("pap_hire", "driver was carrying a paying passenger, not just logged in to the app: where does it say liability is not covered", "LM-PA 0411"),
 ("seepage", "the leak was not sudden, it went on for months: which clause says it is not covered", None),
 ("trees", "wind blew the tree over and there is no tree rider: where does it say wind is not a peril for trees", None),
 ("motor_excl", "the injury did not happen on the property, it was a car crash: which exclusion applies", None),
 ("business_excl", "customer hurt at the home bakery: which exclusion says it is not covered", None),
 ("earth", "sinkhole cracked the foundation: which clause says it is not covered", None),
 ("wear", "roof simply wore out with age: where does it say we will not pay", None),
 ("bop_freeze", "store pipes froze and the heat was not kept on: which clause says we will not pay", None),
 ("boat_ice", "engine block cracked from freezing, boat not winterized: where does it say it is not covered", None),
]
TWINS = {  # the failure injection: same words, opposite answers (Section 12.8)
 "neg": ("mould that is not hidden in the walls and not from a covered leak: where does it say we do not pay", "HO3-2024-06-mould"),
 "pos": ("mould that is hidden in the walls and from a covered leak: where does it say we pay", "LM-HO-2311"),
}
IDENT = {  # clause -> identifier-only queries
 "freezing": ["Section I Perils Insured Against 12"], "loss_payment": ["Section I Conditions B.9"],
 "water_exclusion": ["Section I Exclusions A.3", "LM-HO 2307"], "mould": ["LM-HO 2311"],
 "trees": ["LM-HO 2320", "Section I Additional Coverages E.3"], "loss_settlement": ["LM-HO 2330"],
 "pap_hire": ["Part A Exclusions A.5", "LM-PA 0411"], "duties": ["Section I Conditions B.2"],
 "debris": ["Section I Additional Coverages E.1"], "windstorm": ["MN-COM-2025-06"],
}


def _base(family, ed, clause):
    return f"HO3-{ed}-{clause}" if family == "HO-3" else f"{family}-{FORM_ED[family]}-{clause}"


def build_queries(corpus, seed: int = 9) -> list[Query]:
    rng = random.Random(seed)
    strata = (("docstore", 160), ("policy-admin", 130), ("scanned", 90),
              ("multi-state", 60), ("guidance", 40), ("third-party", 20))
    plan = ([("eval", "docstore")] * 200 + [("holdout", "docstore")] * 300 +
            [("new", s) for s, n in strata for _ in range(n)])
    out = []
    for i, (split, stratum) in enumerate(plan):
        if split == "new":
            style = rng.choices(["kw", "lay", "identifier", "numeric", "negation", "multihop"],
                                [30, 25, 10, 10, 10, 15])[0]
        else:
            style = rng.choices(["kw", "lay"], [95, 5] if split == "eval" else [85, 15])[0]
        out.append(_make(i, split, stratum, style, rng, corpus))
    return out


def _make(i, split, stratum, style, rng, corpus):
    family = rng.choices(["HO-3", "PAP", "BOP", "BOAT"], [70, 15, 10, 5])[0]
    state, ed = rng.choice(STATES), rng.choices(EDITIONS, [40, 60])[0]
    ends = {e for e, p in (("LM-HO 2307", .3), ("LM-HO 2311", .2), ("LM-HO 2320", .15),
                           ("LM-HO 2330", .2), ("LM-PA 0411", .2)) if rng.random() < p}
    governing = None
    if style == "numeric":
        family, clause, text, need, governing = rng.choice(NUMERIC)
        if need:
            ends.add(need)
        if governing == "WI-OCI-2025-03":
            state = "WI"
    elif style == "negation":
        clause, text, need = rng.choice(NEGATION)
        family = ("PAP" if clause.startswith("pap") else "BOP" if clause.startswith("bop")
                  else "BOAT" if clause.startswith("boat") else "HO-3")
        if need:
            ends.add(need)
    elif style == "multihop":
        bid = rng.choice(sorted(MULTIHOP))
        family, state, text = "HO-3", bid.split("-")[0], MULTIHOP[bid]
        clause = next(b[2] for b in C.BULLETINS if b[0] == bid)
        governing = bid
    elif style == "identifier":
        clause = rng.choice(sorted(IDENT))
        text = rng.choice(IDENT[clause])
        family = "PAP" if clause.startswith("pap") else "HO-3"
        if text.startswith("LM-"):
            ends.add(text)
            governing = text.replace(" ", "-")
        elif text.startswith("MN-"):
            state, governing = "MN", text
    else:
        entry = rng.choice(BANKS[family])
        clause = entry[0]
        text = rng.choice(entry[5] if style == "kw" else entry[6])
    if family != "HO-3":
        ends = {e for e in ends if e.startswith("LM-PA")} if family == "PAP" else set()
    else:
        ends = {e for e in ends if e.startswith("LM-HO")}
    if style == "lay" and rng.random() < 0.6:
        text = f"{CITY[state]} claim, loss in {rng.choice(['January', 'March', 'July', 'October'])}: {text}"
    base = _base(family, ed, clause)
    gold = {base: 2}
    if governing is None:
        governing = base
        for e in ends:                           # an attached endorsement governs
            if ENDORSE[e][4] == clause and style not in ("negation",):
                governing = e.replace(" ", "-")
        if clause == "suit" and family == "HO-3":
            governing = [s.span_id for s in corpus.values()
                         if s.doc_type == "amendatory" and s.state == state][0]
    if governing != base:
        gold[base] = 1
        gold[governing] = 2
    return Query(f"Q{i + 1:04d}", split, stratum, style, text, family, state, ed,
                 tuple(sorted(ends)), clause, gold, governing)
