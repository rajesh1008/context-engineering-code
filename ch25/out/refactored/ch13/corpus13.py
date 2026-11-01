"""A production-shaped Lakeshore knowledge corpus with three tenants (synthetic).

Lakeshore sells homeowners policies directly through its agents and, under
two private-label programmes, through Fox Valley Bank and Northwoods
Savings. Each programme has its own copy of the HO-3 form, its own
endorsements and its own manuals; state bulletins are public. A partner's
material must never appear on another tenant's claim. Everything is seeded.
"""
import random
from dataclasses import dataclass

from contextlab.retrieval import lakeshore_clauses as C

TENANTS = {"lakeshore": "Lakeshore Mutual", "foxvalley": "Fox Valley Bank",
           "northwoods": "Northwoods Savings"}
BULK = {"lakeshore": 46_000, "foxvalley": 2_900, "northwoods": 950}
DETS = {"lakeshore": 890, "foxvalley": 80, "northwoods": 30}
EDITIONS = ["2022-01", "2024-06"]
SCOPE = {t: (t, "public") for t in TENANTS}

PARTNER_END = [  # tenant, id, title, text, modifies, lay
 ("foxvalley", "FVB-HO 01", "Water back-up, Fox Valley Bank Homeowners Programme",
  "We insure for direct physical loss caused by water which backs up through sewers or drains or "
  "overflows or is discharged from a sump or sump pump. The most we pay for this coverage is $15,000 "
  "per occurrence, and the deductible is $1,000. The Water exclusion is amended accordingly.",
  "water_exclusion", "sump pump failed and sewage came up through the floor drain"),
 ("foxvalley", "FVB-HO 02", "Mortgage clause, Fox Valley Bank as lender",
  "Loss payable to Fox Valley Bank as mortgagee is paid jointly to you and the bank. The bank will be "
  "given 30 days' notice of cancellation or non-renewal.",
  "mortgage", "the lender wants to be on the cheque for the repairs"),
 ("northwoods", "NWS-HO 01", "Roof surfacing settled at actual cash value",
  "Loss to roof surfacing caused by windstorm or hail is settled at actual cash value, meaning the cost "
  "to repair or replace less depreciation for the age of the shingles.",
  "windstorm", "hail took off part of the roof and the shingles were fifteen years old"),
 ("northwoods", "NWS-HO 02", "Seasonal dwelling, freezing condition",
  "For a seasonal dwelling shown in the Schedule, freezing of a plumbing or heating system is covered "
  "only if a low-temperature alarm was installed and monitored throughout the period of absence.",
  "freezing", "pipes froze at the lake cabin over the winter while nobody was there"),
]

FILLER = ("file review supervisor document record letter agent customer schedule system step "
          "policyholder upload photo estimate reserve contact workflow queue note template call "
          "inspection vendor invoice assignment diary checklist referral training section page").split()
KINDS = ["procedure manual", "training note", "letter template", "underwriting guide",
         "rating manual", "agent bulletin"]


@dataclass
class Span:
    span_id: str
    tenant: str
    doc_type: str      # form, endorsement, bulletin, guideline, manual
    clause: str
    text: str


@dataclass
class Det:
    qid: str
    tenant: str
    text: str
    gold: str          # the governing span
    style: str = "lay"


def _fill(text, ed):
    slots = {k: v[EDITIONS.index(ed)] for k, v in C.HO3_EDITION_SLOTS.items()}
    slots.update(f="40", g="1,200")
    return text.format(**slots)


def _forms(tenant, label, editions):
    out = []
    for ed in editions:
        for key, sec, head, text, *_ in C.HO3:
            sid = f"{tenant[:2].upper()}-HO3-{ed}-{key}"
            out.append(Span(sid, tenant, "form", key,
                            f"{label} HO-3 ({ed}) {sec}. {head}. {_fill(text, ed)}"))
    return out


def build(seed: int = 13):
    rng = random.Random(seed)
    spans = _forms("lakeshore", "Lakeshore Mutual", EDITIONS)
    for fam, bank in (("PAP", C.PAP), ("BOP", C.BOP), ("BOAT", C.BOAT)):
        for key, sec, head, text, *_ in bank:
            spans.append(Span(f"LA-{fam}-{key}", "lakeshore", "form", key,
                              f"Lakeshore Mutual {fam} {sec}. {head}. {_fill(text, '2024-06')}"))
    for eid, fam, title, text, mod, *_ in C.ENDORSEMENTS:
        spans.append(Span(eid.replace(" ", "-"), "lakeshore", "endorsement", mod,
                          f"{eid}. {title}. {text}"))
    for gid, key, issued, text, sup in C.GUIDELINES:
        spans.append(Span(gid.replace(" ", "-"), "lakeshore", "guideline", key, text))
    for bid, st, key, issued, text, sup in C.BULLETINS:
        spans.append(Span(bid, "public", "bulletin", key, text))
    for t in ("foxvalley", "northwoods"):
        spans += _forms(t, TENANTS[t] + " Homeowners Programme", ["2024-06"])
    for t, eid, title, text, mod, _ in PARTNER_END:
        spans.append(Span(eid.replace(" ", "-"), t, "endorsement", mod,
                          f"{eid}. {title}. {text}"))
    topics = [(e[0], [e[2]]) for e in C.HO3] + [(e[0], [e[2]]) for e in C.PAP + C.BOP + C.BOAT]
    for t, n in BULK.items():
        pool = topics if t == "lakeshore" else topics[:len(C.HO3)]
        for i in range(n):
            key, lay = rng.choice(pool)
            body = rng.sample(FILLER, 12)
            if rng.random() < 0.3:
                body.append(rng.choice(lay[0].split()).strip(",").lower())
            rng.shuffle(body)
            spans.append(Span(f"{t[:2].upper()}-MAN-{i:05d}", t, "manual", key,
                              f"{TENANTS[t]} {rng.choice(KINDS)} {rng.randint(1, 400)}. "
                              + " ".join(body) + "."))
    return {s.span_id: s for s in spans}, determinations(rng)


def determinations(rng):
    ho3 = {e[0]: e for e in C.HO3}
    lam = {e[4]: e[0].replace(" ", "-") for e in C.ENDORSEMENTS if e[1] == "HO-3"}
    other = [(f"LA-{fam}-{e[0]}", e[6]) for fam, b in (("PAP", C.PAP), ("BOP", C.BOP),
                                                     ("BOAT", C.BOAT)) for e in b]
    pend = {(t, mod): (eid.replace(" ", "-"), lay) for t, eid, _, _, mod, lay in PARTNER_END}
    dets = []
    for t, n in DETS.items():
        for i in range(n):
            if t == "lakeshore" and rng.random() < 0.25:
                gold, lays = rng.choice(other)
                text = rng.choice(lays)
            else:
                amended = [m for (pt, m) in pend if pt == t]
                key = (rng.choice(amended) if t != "lakeshore" and rng.random() < 0.35
                       else rng.choice(list(ho3)))
                text = rng.choice(ho3[key][6])
                if t == "lakeshore":
                    ed = rng.choice(EDITIONS)
                    gold = f"LA-HO3-{ed}-{key}"
                    if key in lam and rng.random() < 0.4:
                        gold = lam[key]
                else:
                    gold = f"{t[:2].upper()}-HO3-2024-06-{key}"
                    if (t, key) in pend:
                        gold, text = pend[(t, key)][0], text
            dets.append(Det(f"{t[:2].upper()}-D{i:04d}", t, text, gold))
    return dets
