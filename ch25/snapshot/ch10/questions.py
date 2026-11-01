"""Fixture questions for the Chapter 10 lab, each with a gold clause and a policy.

The wording is an adjuster's, not the form's, on purpose: "frozen pipe burst"
against "freezing of a plumbing system". Gold clauses come from truth.json.
"""
from __future__ import annotations
import random
import re
import clauses as C

PERIL_Q = {
    1: "Kitchen fire spread to the living room. Is the fire damage covered?",
    2: "Hail dented the siding and strong winds tore shingles off the roof. Is that covered?",
    3: "The gas grill exploded on the deck and scorched the back wall.",
    4: "Looters broke in during street unrest and smashed the front windows.",
    5: "A small plane clipped the chimney on approach to the airfield.",
    6: "The neighbour's car rolled down the hill into the garage wall.",
    7: "Furnace puffback covered the walls and ceilings in soot.",
    8: "Vandals spray painted the siding and broke the porch light.",
    9: "Burglar stole a laptop and a camera from the house.",
    10: "A large oak branch fell through the roof onto the sofa.",
    11: "Heavy wet snow on the roof cracked the rafters and damaged the attic.",
    13: "The hot water heating system boiler split open and bulged.",
    14: "A washing machine hose let go and flooded the laundry room.",
    15: "A power surge from the grid fried the wiring in the walls.",
    16: "Volcanic ash damaged the roof during a trip abroad.",
}
FROZEN = [
    ("DET-0022", "Frozen pipe burst during the January cold snap after the furnace went out. Do we pay?"),
    ("frozen-2", "A frozen pipe split in the upstairs bathroom while the family was away for a week in February."),
    ("frozen-3", "Frozen pipe in the basement burst after a minus twenty night; water damage to the finished rec room."),
    ("frozen-4", "The frozen pipe under the kitchen sink cracked and the water ruined the cabinets."),
    ("frozen-5", "The attic sprinkler pipe was frozen solid and burst, soaking the ceiling below."),
]
ENDO_Q = [
    ("HO 04 95", "2024-06", "coverage", "Sewer backed up into the finished basement after a storm and the sump pump failed. Is it covered and up to how much?"),
    ("HO 04 95", "2019-03", "coverage", "Sewer backed up into the basement through the floor drain. What is the limit for this policy?"),
    ("HO 04 95", "2024-06", "deductible", "What deductible applies to a sewer backup claim under the water back-up endorsement?"),
    ("HO 04 95", "2024-06", "exclusions", "River flooding came into the basement. Does the water back-up endorsement pay?"),
    ("HO 04 54", "2022-01", "coverage", "An earthquake cracked the foundation and the chimney. Is that covered?"),
    ("HO 04 54", "2022-01", "deductible", "How is the earthquake deductible calculated?"),
    ("HO 04 98", "2024-06", "coverage", "The freezer lost power for three days and all the food spoiled."),
    ("HO 04 90", "2022-01", "coverage", "Is the stolen television paid at full replacement cost or depreciated value?"),
    ("HO 04 61", "2022-01", "coverage", "The engagement ring listed on the schedule was lost at a restaurant."),
    ("HO 04 35", "2024-06", "coverage", "The condo association assessed every owner for a hail-damaged clubhouse roof."),
    ("HO 04 16", "2024-06", "conditions", "The customer removed the burglar alarm. Do they have to tell us?"),
    ("LM HO 17", "2024-06", "coverage", "How are hail-damaged shingles paid under the roof surface payment schedule?"),
]
BULLETIN_Q = [
    ("WI-OCI-2025-03", "2025-06-10", "Hail damaged siding on one side of the house and the old siding no longer matches. Do we replace the other sides?"),
    ("WI-OCI-2023-07", "2024-08-02", "Hail damaged siding on one side of the house and the old siding no longer matches. Do we replace the other sides?"),
    ("WI-OCI-2024-11", "2025-04-03", "Sewer surcharge and sump discharge happened together and there was standing water in the yard. Can we deny for surface water?"),
    ("WI-OCI-2022-05", "2023-03-15", "The sump pump failed mechanically and water came in. Is it surface water?"),
    ("IL-DOI-2025-05", "2025-12-20", "A two-day power outage left the house without heat and the pipes froze. Did the insured fail to keep heat on?"),
    ("IL-DOI-2023-10", "2024-05-02", "Mold grew after a covered pipe leak. Does the mold exclusion apply to the cleanup?"),
    ("IA-ID-2024-08", "2025-07-01", "Hail left cosmetic dents on a metal roof. Can we apply the cosmetic damage exclusion?"),
    ("IA-ID-2025-06", "2025-08-11", "A derecho damaged the roof and the shed on the same day. How many deductibles apply?"),
    ("MN-COMM-2026-01", "2026-02-14", "We paid the roof at actual cash value and the customer asks how depreciation was worked out."),
    ("MN-COMM-2024-04", "2025-01-09", "The house is being renovated with contractors there every week. Is it vacant?"),
    ("MI-DIFS-2025-08", "2025-12-01", "An ice dam leaked water into the bedroom ceiling. The roof did not sag. Is it covered?"),
    ("MI-DIFS-2023-09", "2024-02-01", "An ice dam leaked water into the bedroom ceiling and tore the gutters off. Is it covered?"),
    ("IN-DOI-2025-02", "2025-05-05", "The declarations page shows a 2% wind deductible but no dollar amount. Is that allowed?"),
    ("IN-DOI-2024-03", "2024-09-09", "A catalytic converter was stolen from the car in the driveway. Is it a homeowners theft claim?"),
]
THING_SYN = {"detached garage": "garage behind the house", "fence": "backyard fence", "swimming pool": "in-ground pool",
    "outdoor antenna": "TV antenna", "satellite dish": "dish on the roof", "lawn and shrubs": "landscaping and lawn",
    "trees": "maple trees", "jewellery": "wedding ring jewellery", "firearms": "hunting rifle firearms",
    "electronic data": "computer files", "money and bank notes": "cash", "boat and trailer": "fishing boat",
    "hot tub": "spa tub", "kitchen cabinets": "cabinets", "septic system": "septic tank", "bicycles": "bikes"}
CAUSE_SYN = {"wear and tear": "age and deterioration", "mechanical breakdown": "motor breakdown",
    "birds or rodents": "squirrels and rodents", "insects": "termites", "mould or wet rot": "mold and rot",
    "earth movement": "sinkhole earth movement", "flood": "river flood", "surface water": "rainwater pooling surface water",
    "power failure": "blackout", "neglect": "lack of upkeep", "faulty workmanship": "bad contractor workmanship",
    "continuous seepage": "slow leak seepage", "domestic animals": "the family dog", "ice damming": "ice dams", "hail": "hailstorm",
    "windstorm": "high winds"}


GOVERNING = {"form", "endorsement", "bulletin", "schedule"}


def policy_allows(policy: dict, doc: dict) -> bool:
    """Chapter 2's applicability filter plus Chapter 9's evidence classes: a
    coverage question ranks governing documents only; a procedure question
    ranks the manuals only."""
    k = doc["kind"]
    if policy.get("procedural"):
        return k in ("manual", "guideline")
    if k == "form":
        return doc["edition"] == policy["form"]
    if k == "endorsement":
        if doc["label"].startswith("HO 01"):
            return doc["label"].endswith(policy["state"])
        return (doc["label"], doc["edition"]) in policy["endorsements"]
    if k == "bulletin":
        b = next(b for b in C.BULLETINS if b[0] == doc["label"])
        return b[1] == policy["state"] and b[2] <= policy["dol"] and (b[3] is None or policy["dol"] < b[3])
    if k == "schedule":
        return policy["form"] == "2024-06"
    return False                                         # manuals never govern coverage


def build(truth: dict, seed: int = 10) -> list[dict]:
    rng = random.Random(seed)
    cl = truth["clauses"]
    qs = []

    def add(qid, text, gold_id, policy, group, gold=None):
        g = dict(gold or cl[gold_id]); g["id"] = gold_id
        qs.append(dict(qid=qid, text=text, gold=g, policy=policy, group=group))

    states = list(C.STATES)
    for ed in ("2022-01", "2024-06"):
        for n, q in PERIL_Q.items():
            st = rng.choice(states)
            add(f"peril-{n}-{ed}", q, f"HO-3@{ed}#perils-{n}", dict(form=ed, endorsements=[], state=st, dol="2025-06-01"), "form")
    for qid, q in FROZEN:
        add(qid, q, "HO-3@2022-01#perils-12", dict(form="2022-01", endorsements=[], state="WI", dol="2025-01-20"), "frozen-2022")
    add("frozen-2024", FROZEN[0][1], "HO-3@2024-06#perils-12", dict(form="2024-06", endorsements=[], state="WI", dol="2025-01-20"), "form")
    for ed in ("2022-01", "2024-06"):
        roof = next(cid for cid, c in cl.items() if cid.startswith(f"HO-3@{ed}#cond-") and c["text"].startswith("Roof Surfacing"))
        add(f"roof-{ed}", "Wind tore up a 17-year-old roof. Do we pay replacement cost or actual cash value for the shingles?",
            roof, dict(form=ed, endorsements=[], state="MN", dol="2025-08-14"), "form")
    for form, ed, sec, q in ENDO_Q:
        cid = f"{form}@{ed}#{sec}"
        add(f"endo-{form}-{ed}-{sec}".replace(" ", ""), q, cid,
            dict(form="2024-06" if ed >= "2024" else "2022-01", endorsements=[(form, ed), ("HO 04 90", "2022-01")]
                 if form != "HO 04 90" else [(form, ed)], state="WI", dol="2025-05-01"), "endorsement")
    for bid, dol, q in BULLETIN_Q:
        st = next(b[1] for b in C.BULLETINS if b[0] == bid)
        add(f"bul-{bid}", q, f"{bid}#guidance", dict(form="2024-06", endorsements=[], state=st, dol=dol), "bulletin")
    tab = cl["LM DED@2024-06#table-1"]
    for i, row in enumerate(C.DEDUCTIBLE_ROWS):
        st = next(k for k, v in C.STATES.items() if v == row[0])
        g = dict(tab); g["text"] = " ".join(C.DEDUCTIBLE_HEAD) + " " + " ".join(row)
        g["pages"] = [2] if i < 3 else [3]; g["row"] = list(row)
        add(f"ded-{st}", f"What wind or hail deductible applies to a {row[0]} homeowners policy on the 2024 form?",
            "LM DED@2024-06#table-1", dict(form="2024-06", endorsements=[], state=st, dol="2025-06-01"), "table", g)
    sch = cl["LM HO 17@2024-06#schedule"]
    for age, pct in C.ROOF_SCHEDULE[2:4]:
        g = dict(sch); g["text"] = f"Age of roof surfacing Percentage of replacement cost paid {age} {pct}"; g["row"] = [age, pct]
        add(f"roofsched-{age.split()[0]}", f"Hail damaged a roof that is {age.split()[0]} or so years old under the roof surface payment schedule. What percentage do we pay?",
            "LM HO 17@2024-06#schedule", dict(form="2024-06", endorsements=[("LM HO 17", "2024-06")], state="WI", dol="2025-06-01"), "table", g)
    # generated clauses with a unique (thing, cause) pair inside the policy's documents
    pat = re.compile(r"loss to (.+?) caused by (.+?) unless")
    docs = {d["doc_id"]: d for d in truth["docs"]}
    cands = []
    for cid, c in cl.items():
        m = pat.search(c["text"])
        if m and c["kind"] in ("form", "endorsement"):
            cands.append((cid, m.group(1), m.group(2)))
    rng.shuffle(cands)
    taken = 0
    for cid, thing, cause in cands:
        c = cl[cid]
        st = c["label"][-2:] if c["label"].startswith("HO 01") else rng.choice(states)
        pol = dict(form=c["edition"] if c["kind"] == "form" else "2024-06", endorsements=[], state=st, dol="2025-06-01")
        same = [k for k, o in cl.items() if f"loss to {thing} caused by {cause}" in o["text"] and policy_allows(pol, docs[o["doc_id"]])]
        if len(same) != 1:
            continue
        t, ca = THING_SYN.get(thing, thing), CAUSE_SYN.get(cause, cause)
        add(f"gen-{taken}", f"Is damage to the {t} from {ca} covered?", cid, pol, "generated")
        taken += 1
        if taken == 60:
            break
    mpat = re.compile(r"^Handling (.+?) claims involving (.+)$")
    mc = [(cid, c) for cid, c in cl.items() if c["label"] == "LM CLAIMS MANUAL"]
    rng.shuffle(mc)
    seen = 0
    for cid, c in mc:
        m = mpat.match(c["path"][-1].split(". ", 1)[-1])
        if not m:
            continue
        same = [k for k, o in mc if o["path"][-1].split(". ", 1)[-1] == c["path"][-1].split(". ", 1)[-1]]
        if len(same) != 1:
            continue
        t, ca = THING_SYN.get(m.group(1), m.group(1)), CAUSE_SYN.get(m.group(2), m.group(2))
        add(f"man-{seen}", f"What does the claims manual say about a {t} claim involving {ca}?", cid,
            dict(form="2024-06", endorsements=[], state="WI", dol="2025-06-01", procedural=True), "manual")
        seen += 1
        if seen == 30:
            break
    return qs
