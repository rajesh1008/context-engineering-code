"""Chapter 28 stand-in: Lakeshore's forms, 300 determinations and a writer.

Synthetic and seeded. Form text is Lakeshore's stand-in wording, not any
bureau's. The writer is a deterministic stand-in for the drafting model: it
writes fixed sentences per claim type and makes mistakes at the stated RATES.
Those rates are parameters of the stand-in, not measurements of any model.
"""
import random
import zlib
from dataclasses import dataclass, field

from contextlab.citations import Entry, Manifest, NOSOURCE

RATES = {"neighbour": 0.05, "addition": 0.03, "number": 0.015,
         "missing": 0.03, "stale": 0.01}
CONCLUSION_RATES = {"polarity": 0.03, "stacking": 0.05, "exception": 0.25, "reason": 0.06, "detail": 0.05}


def rng(tag):
    return random.Random(zlib.crc32(tag.encode()))


HO3 = {  # page in the 2024-06 edition: spans
    5: ["We insure for direct physical loss to the dwelling caused by windstorm or hail.",
        "We do not insure for loss to the interior caused by rain or snow unless the wind or hail first makes an opening in a roof or wall.",
        "We insure for direct physical loss caused by the weight of ice, snow or sleet."],
    6: ["We insure for accidental discharge of water from a plumbing or heating system, including the cost to tear out and replace part of the building to reach the system.",
        "We do not insure for freezing of a plumbing system in an unoccupied dwelling unless you maintained heat in the building or shut off the water supply and drained the system.",
        "We do not insure for continuous or repeated seepage or leakage of water over a period of 14 or more days."],
    7: ["We do not insure for loss caused by water which backs up through sewers or drains or overflows from a sump, unless an endorsement provides that coverage.",
        "We do not insure for loss caused by flood, surface water, waves or overflow of a body of water, whether or not driven by wind.",
        "We do not insure for loss caused by water below the surface of the ground which seeps through a foundation, wall or floor."],
    8: ["We do not insure for loss caused by mould, fungus or wet rot.",
        "We do insure for mould, fungus or wet rot hidden within walls or floors that results from accidental discharge of water from a plumbing system."],
    11: ["After a loss you must give prompt notice to us and protect the property from further damage.",
         "You must show the damaged property and submit a signed proof of loss within 60 days of our request."],
}
HO495 = {
    "2024-06": {1: ["Water Back-Up and Sump Discharge or Overflow. This endorsement changes the policy and modifies the Homeowners 3 Special Form.",
                    "We will pay up to $10,000 for direct physical loss caused by water or waterborne material which backs up through sewers or drains or overflows or is discharged from a sump, sump pump or related equipment, even if caused by mechanical breakdown of the sump pump.",
                    "A $1,000 deductible applies.",
                    "We do not pay for loss caused by your negligent failure to maintain the sump pump."],
                2: ["The limit for this coverage is part of, and does not increase, the Coverage A limit of liability."]},
    "2019-03": {1: ["Water Back-Up and Sump Discharge or Overflow. This endorsement changes the policy and modifies the Homeowners 3 Special Form.",
                    "We will pay up to $5,000 for direct physical loss caused by water or waterborne material which backs up through sewers or drains or overflows or is discharged from a sump, sump pump or related equipment.",
                    "A $500 deductible applies."],
                2: ["The limit for this coverage is part of, and does not increase, the Coverage A limit of liability."]},
}


def ho3_page(edition, page):
    """The 2022-01 edition carries the same clauses one page later."""
    return page + 1 if edition == "2022-01" else page


def form_entries(doc, edition):
    if doc == "HO-3":
        return [Entry("HO-3", edition, ho3_page(edition, p), i + 1, t)
                for p, ts in HO3.items() for i, t in enumerate(ts)]
    return [Entry("HO 04 95", edition, p, i + 1, t)
            for p, ts in HO495[edition].items() for i, t in enumerate(ts)]


def library():
    """Every edition of every form, as the document store holds them."""
    return [e for ed in ("2024-06", "2022-01") for e in form_entries("HO-3", ed)] + \
           [e for ed in ("2024-06", "2019-03") for e in form_entries("HO 04 95", ed)]


# Claim types. Each sentence: (kind, text, support) where support names spans as
# (doc, page_2024, n) for forms, ("CLAIM", n) for claim facts, ("DECS", n).
TYPES = {
    "backup_endorsed": dict(share=0.30, decision="covered", facts=[
        "Reported cause: the sump pump failed during heavy rain and water backed up into the basement.",
        "Dwelling occupied at time of loss: yes."],
        decs="Endorsements on this policy: HO 04 95 ({e495}), water back-up limit ${lim:,}.",
        sentences=[
            ("fact", "The reported cause is a sump pump failure during heavy rain that let water back up into the basement.", [("CLAIM", 1)]),
            ("policy", "The HO-3 form does not insure loss caused by water which backs up through sewers or drains or overflows from a sump unless an endorsement provides that coverage.", [("HO-3", 7, 1)]),
            ("policy", "The HO 04 95 endorsement will pay up to ${lim:,} for direct physical loss caused by water which backs up through drains or is discharged from a sump{breakdown}.", [("HO 04 95", 1, 2)]),
            ("policy", "A ${ded:,} deductible applies.", [("HO 04 95", 1, 3)]),
            ("conclusion", "The loss is covered by the endorsement up to ${lim:,} for water discharged from the sump, and a ${ded:,} deductible applies.", [("HO 04 95", 1, 2), ("HO 04 95", 1, 3)]),
            ("note", "Please send photographs of the damaged items and the failed pump.", []),
        ]),
    "backup_plain": dict(share=0.15, decision="excluded", facts=[
        "Reported cause: water backed up through the floor drain during heavy rain.",
        "Dwelling occupied at time of loss: yes."],
        decs="Endorsements on this policy: none for water back-up.",
        sentences=[
            ("fact", "The reported cause is water that backed up through the floor drain during heavy rain.", [("CLAIM", 1)]),
            ("fact", "The declarations list no endorsement for water back-up.", [("DECS", 1)]),
            ("policy", "The HO-3 form does not insure loss caused by water which backs up through sewers or drains unless an endorsement provides that coverage.", [("HO-3", 7, 1)]),
            ("conclusion", "The loss is not insured because water backed up through a drain and no endorsement provides that coverage.", [("HO-3", 7, 1), ("DECS", 1)]),
            ("note", "You may ask your agent about adding water back-up coverage at renewal.", []),
        ]),
    "frozen_pipe": dict(share=0.20, decision="covered", facts=[
        "Reported cause: a pipe froze and burst in the kitchen wall during a cold snap.",
        "Dwelling occupied and heated at time of loss: yes."],
        decs="Endorsements on this policy: none relevant.",
        sentences=[
            ("fact", "A pipe froze and burst in the kitchen wall during a cold snap while the dwelling was occupied and heated.", [("CLAIM", 1), ("CLAIM", 2)]),
            ("policy", "The form insures accidental discharge of water from a plumbing system, including the cost to tear out and replace part of the building to reach the system.", [("HO-3", 6, 1)]),
            ("policy", "The freezing exclusion applies only to an unoccupied dwelling where heat was not maintained and the water supply was not shut off.", [("HO-3", 6, 2)]),
            ("conclusion", "The burst pipe is insured, including the cost to tear out and replace the kitchen wall to reach the plumbing system.", [("HO-3", 6, 1)]),
            ("policy", "You must submit a signed proof of loss within 60 days of our request.", [("HO-3", 11, 2)]),
        ]),
    "mould_hidden": dict(share=0.08, decision="covered", facts=[
        "Reported cause: mould found inside the bathroom wall after a supply line leaked.",
        "Leak duration estimated by the plumber: about 5 days."],
        decs="Endorsements on this policy: none relevant.",
        sentences=[
            ("fact", "Mould was found inside the bathroom wall after a supply line leaked for about 5 days.", [("CLAIM", 1), ("CLAIM", 2)]),
            ("policy", "The form does not insure loss caused by mould, fungus or wet rot.", [("HO-3", 8, 1)]),
            ("policy", "The form does insure mould hidden within walls that results from accidental discharge of water from a plumbing system.", [("HO-3", 8, 2)]),
            ("conclusion", "The mould is insured because it was hidden within the wall and resulted from accidental discharge of water from a plumbing system.", [("HO-3", 8, 2), ("CLAIM", 1)]),
        ]),
    "mould_seepage": dict(share=0.07, decision="excluded", facts=[
        "Reported cause: mould behind the shower tiles from a slow leak.",
        "Leak duration estimated by the plumber: about 6 weeks."],
        decs="Endorsements on this policy: none relevant.",
        sentences=[
            ("fact", "Mould was found behind the shower tiles from a slow leak that lasted about 6 weeks.", [("CLAIM", 1), ("CLAIM", 2)]),
            ("policy", "The form does not insure continuous or repeated seepage or leakage of water over a period of 14 or more days.", [("HO-3", 6, 3)]),
            ("policy", "The form does not insure loss caused by mould, fungus or wet rot.", [("HO-3", 8, 1)]),
            ("conclusion", "The mould is not insured because the leakage continued for 14 or more days.", [("HO-3", 6, 3), ("CLAIM", 2)]),
        ]),
    "hail_interior": dict(share=0.20, decision="covered", facts=[
        "Reported cause: hail broke a skylight and rain entered the kitchen.",
        "Roof inspection: skylight glazing broken by hail."],
        decs="Endorsements on this policy: none relevant.",
        sentences=[
            ("fact", "Hail broke a skylight and rain entered the kitchen.", [("CLAIM", 1)]),
            ("policy", "The form insures direct physical loss to the dwelling caused by windstorm or hail.", [("HO-3", 5, 1)]),
            ("policy", "Interior damage caused by rain is insured only if the wind or hail first makes an opening in a roof or wall.", [("HO-3", 5, 2)]),
            ("conclusion", "The interior rain damage is insured because the hail first made an opening in the roof.", [("HO-3", 5, 2), ("CLAIM", 2)]),
            ("note", "An adjuster will contact you to arrange an inspection.", []),
        ]),
}

# Wrong conclusions the stand-in writes: (text, pointers or None to keep, span supports text).
H = lambda p, n: ("HO-3", p, n)
WRONG = {
    "polarity": {
        "backup_endorsed": ("The loss is not covered by the endorsement for water discharged from the sump.", None, False),
        "frozen_pipe": ("The burst pipe is not insured because the pipe froze.", None, False),
        "hail_interior": ("The interior rain damage is not insured because rain entered the kitchen.", None, False)},
    "stacking": {
        "backup_endorsed": ("The ${lim:,} back-up limit is paid in addition to the dwelling limit of liability.", None, False)},
    "exception": {
        "mould_seepage": ("The mould is insured because the form does insure mould hidden within walls that results from accidental discharge of water from a plumbing system.",
                          [H(8, 2)], True)},
    "reason": {
        "backup_plain": ("The loss is not insured because the form does not insure loss caused by surface water.", [H(7, 2)], True),
        "backup_endorsed": ("The loss is covered because the form insures accidental discharge of water from a plumbing system.", [H(6, 1)], True),
        "mould_seepage": ("The mould is not insured because the form does not insure loss caused by mould, fungus or wet rot.", [H(8, 1)], True)},
    "detail": {
        "hail_interior": ("The interior rain damage is insured because the hail first made an opening in a wall.", None, True),
        "frozen_pipe": ("The burst pipe is insured, including the cost to tear out and replace part of the building to reach the system, because the water supply was shut off.",
                        [H(6, 1), H(6, 2)], True)},
}
WRONG_DECISION = {"polarity": True, "stacking": False, "exception": True, "reason": False, "detail": False}
ADDITION = " provided the loss was reported within 72 hours"


@dataclass
class Sentence:
    kind: str
    text: str
    support: list             # gold: Entry objects that support the sentence
    pointers: list            # Entry objects the writer points to
    mistake: str | None = None


@dataclass
class Determination:
    det_id: str
    claim_id: str
    ctype: str
    state: str
    ho3: str
    e495: str | None
    decision: str             # what the rules give
    written_decision: str     # what the writer put in the structured field
    packet: list              # Entry objects in assembly order
    sentences: list = field(default_factory=list)


def _resolve(spec, ho3, e495, claim, decs):
    if spec[0] == "CLAIM":
        return claim[spec[1] - 1]
    if spec[0] == "DECS":
        return decs
    doc, page, n = spec
    ed = ho3 if doc == "HO-3" else e495
    pg = ho3_page(ed, page) if doc == "HO-3" else page
    return next(e for e in form_entries(doc, ed) if e.page == pg and e.n == n)


def build(n=300, seed="ch28"):
    r = rng(seed)
    types = list(TYPES)
    weights = [TYPES[t]["share"] for t in types]
    out = []
    for i in range(n):
        ct = r.choices(types, weights)[0]
        if i == 187:                       # Emily Carter's claim (Chapter 1)
            ct = "backup_endorsed"
        T = TYPES[ct]
        st = r.choice(["WI", "WI", "IA", "MN", "IL", "IN", "MI"])
        ho3 = "2024-06" if r.random() < 0.55 else "2022-01"
        e495 = ("2024-06" if r.random() < 0.7 else "2019-03") if ct == "backup_endorsed" else None
        if i == 187:
            st, ho3, e495 = "WI", "2024-06", "2024-06"
        lim, ded = (10000, 1000) if e495 == "2024-06" else (5000, 500)
        cid = f"CLM-2026-{1000 + i:05d}"
        claim = [Entry(cid, "record", 1, k + 1, t) for k, t in enumerate(T["facts"])]
        pol = "DECS-LMH-WI-0048213" if i == 187 else f"DECS-LMH-{st}-{48000 + i:07d}"
        decs = Entry(pol, "2025-06", 1, 1, T["decs"].format(e495=e495, lim=lim))
        fmt = dict(lim=lim, ded=ded,
                   breakdown=", even if caused by mechanical breakdown of the sump pump" if e495 == "2024-06" else "")
        sents = []
        for kind, text, sup in T["sentences"]:
            gold = [_resolve(s, ho3, e495, claim, decs) for s in sup]
            sents.append(Sentence(kind, text.format(**fmt), gold, list(gold)))
        # assemble the packet: needed spans, their page neighbours, a distractor page
        need = []
        for s in sents:
            need += [e for e in s.support if e not in need]
        forms = form_entries("HO-3", ho3) + (form_entries("HO 04 95", e495) if e495 else [])
        pages = {(e.doc, e.page) for e in need}
        neigh = [e for e in forms if (e.doc, e.page) in pages and e not in need]
        extra = [p for m in WRONG.values() if ct in m and m[ct][1] for p in m[ct][1]]
        neigh += [x for x in (_resolve(w, ho3, e495, claim, decs) for w in extra)
                  if x not in need + neigh]          # retrieved as related, so in the packet
        other = [e for e in forms if (e.doc, e.page) not in pages]
        r.shuffle(other)
        packet = need + neigh + other[: max(0, 10 - len(need) - len(neigh))]
        packet = sorted(packet, key=lambda e: (-(e in need) * r.random(), r.random()))
        packet += [e for e in claim + [decs] if e not in packet]
        written = T["decision"]
        for s in sents:
            mistakes(r, s, ct, packet, fmt, ho3)
            if s.mistake in WRONG_DECISION and WRONG_DECISION[s.mistake]:
                written = "excluded" if written == "covered" else "covered"
        out.append(Determination(f"DET28-{i + 1:04d}", cid, ct, st, ho3, e495,
                                 T["decision"], written, packet, sents))
    return out


def mistakes(r, s, ct, packet, fmt, ho3):
    """Apply at most one stand-in mistake to a sentence, in place."""
    if s.kind == "note":
        return
    if s.kind == "conclusion":
        for m, p in CONCLUSION_RATES.items():
            if ct in WRONG[m] and r.random() < p:
                text, ptrs, ok = WRONG[m][ct]
                s.text, s.mistake = text.format(**fmt), m
                if ptrs:
                    s.pointers = [_resolve(x, ho3, None, None, None) for x in ptrs]
                s.support = list(s.pointers) if ok else []
                return
        return
    u, acc = r.random(), 0.0
    for m, p in RATES.items():
        acc += p
        if u < acc:
            break
    else:
        return
    s.mistake = m
    if m == "neighbour":
        tgt = s.pointers[0]
        cands = [e for e in packet if e.doc == tgt.doc and e.page == tgt.page and e not in s.support] \
            or [e for e in packet if e not in s.support and e.edition != "record"]
        s.pointers = [r.choice(cands)]
    elif m == "addition":
        s.text, s.support = s.text.rstrip(".") + ADDITION + ".", []
    elif m == "number":
        swap = {"$10,000": "$5,000", "$5,000": "$10,000", "$1,000": "$500", "$500": "$1,000",
                "60 days": "30 days", "14 or more": "30 or more", "5 days": "15 days"}
        for a, b in swap.items():
            if a in s.text:
                s.text, s.support = s.text.replace(a, b, 1), []
                break
        else:
            s.mistake = None
    elif m == "missing":
        s.pointers = []
    elif m == "stale":
        tgt = s.pointers[0]
        if tgt.doc == "HO-3":
            old = "2022-01" if tgt.edition == "2024-06" else "2024-06"
            s.pointers = [Entry("HO-3", old, ho3_page(old, tgt.page - (tgt.edition == "2022-01")),
                                tgt.n, tgt.text)]
        else:
            s.mistake = None


def render(s: Sentence, packet_order: list, fmt: str) -> str:
    """Write a sentence with its pointers in one citation format."""
    if s.kind == "note":
        return f"{s.text} {NOSOURCE}"
    if not s.pointers:
        return s.text
    tags = []
    for e in s.pointers:
        if fmt == "handle":
            tags.append(f"[[{e.handle}]]")
        elif fmt == "index":
            tags.append(f"[{packet_order.index(e) + 1}]" if e in packet_order else "[99]")
        elif fmt == "page":
            tags.append(f"[{e.doc} ({e.edition}), page {e.page}]")
        else:
            tags.append(f"[{e.doc} ({e.edition})]")
    return f"{s.text} {' '.join(dict.fromkeys(tags))}"


def rerank(packet, seed):
    """The injected fault: a reranker that reorders spans after indices exist."""
    r = rng(seed)
    return sorted(packet, key=lambda e: r.random())


def manifest(packet):
    return Manifest(list(packet))
