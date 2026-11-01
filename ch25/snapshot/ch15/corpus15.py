"""Chapter 15 corpus: governing documents, context documents and the full mix.

Stand-in data. Documents are fixtures written for the lab; the full mix of
1,000 determinations keeps Chapter 9's strata and carries the errors that
Chapters 10 to 14 left, so the Chapter 14 configuration starts where the
scoreboard does. The rules that decide authority are real code.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from datetime import date, timedelta

from contextlab.policy import Span, Request

ADJ = frozenset({"adjuster", "supervisor"})
UW = frozenset({"underwriter"})
SIU = frozenset({"siu"})
ALLR = frozenset({"adjuster", "supervisor", "underwriter"})
STATES = ["WI", "MN", "IL", "IA", "MI", "IN"]
HO3 = ["2019-03", "2022-01", "2024-06"]

HO3_TEXT = {
 "freezing": ("Section I Perils 4", "We do not cover loss caused by freezing of a plumbing, heating or air conditioning system or of a household appliance while the dwelling is unoccupied, unless you have used reasonable care to maintain heat in the building or to shut off the water supply and drain the system."),
 "water_backup": ("Section I Exclusions A.3", "We do not cover loss caused by water or waterborne material which backs up through sewers or drains or which overflows or is discharged from a sump, sump pump or related equipment."),
 "roof": ("Section I Conditions D.2", "Loss settlement for roof surfaces: we will pay the cost to repair or replace the damaged part of the roof with material of like kind and quality. We do not pay to replace undamaged parts to achieve a uniform appearance."),
 "debris": ("Section I Additional Coverages 1", "Debris removal: we will pay your reasonable expense for the removal of debris of covered property and of a tree that damages a covered structure, up to 5% of the limit of liability."),
 "notice": ("Section I Conditions B.1", "Duties after loss: you must give prompt notice to us or our agent. Failure to give prompt notice may bar coverage for the loss."),
 "payment": ("Section I Conditions J", "Loss payment: we will pay you within 60 days after we receive your proof of loss and reach agreement with you or there is an appraisal award."),
 "mould": ("Section I Exclusions 2.c", "We do not cover loss caused by mould, fungus or wet rot unless hidden within the walls and resulting from the accidental discharge of water from a plumbing system."),
}
HO3_PAGES = {"2019-03": 0, "2022-01": 1, "2024-06": 2}
PAP_TEXT = {
 "animal": ("Part D Other than collision", "Loss caused by contact with a bird or animal is considered other than collision. Collision means the upset of your covered auto or its impact with another vehicle or object."),
 "oos_limits": ("Part A Out of state coverage", "If an auto accident occurs in any state other than the one in which your covered auto is principally garaged, we will interpret your policy for that accident to provide the higher limit required by that state's financial responsibility law."),
 "collision": ("Part D Collision", "We will pay for direct and accidental loss to your covered auto caused by collision, minus any deductible shown in the declarations."),
}


def S(**kw):
    kw.setdefault("roles", ADJ)
    return Span(**kw)


def governing_spans():
    out = []
    for ed in HO3:
        for i, (topic, (sec, text)) in enumerate(HO3_TEXT.items()):
            out.append(S(sid=f"HO-3@{ed}#{topic}", doc_id="HO-3", version=ed, kind="form",
                         topic=topic, text=f"{sec}. {text}", edition_family="HO-3",
                         effective_from=date(int(ed[:4]), int(ed[5:]), 1),
                         page=8 + 2 * i + HO3_PAGES[ed]))
    for i, (topic, (sec, text)) in enumerate(PAP_TEXT.items()):
        out.append(S(sid=f"PAP@2018-01#{topic}", doc_id="PAP", version="2018-01", kind="form",
                     topic=topic, text=f"{sec}. {text}", edition_family="PAP",
                     effective_from=date(2018, 1, 1), page=4 + 3 * i))
    E = lambda fam, ed, scope, text, **k: S(
        sid=f"{fam}@{ed}#{scope}", doc_id=fam, version=ed, kind="endorsement", topic=scope,
        scope=scope, text=text, edition_family=fam,
        effective_from=date(int(ed[:4]), int(ed[5:]), 1), page=1, **k)
    out += [
     E("HO 04 95", "2019-03", "water_backup", "Water back-up and sump discharge or overflow: we cover direct physical loss caused by water which backs up through sewers or drains or overflows from a sump, up to US$5,000, deductible US$500."),
     E("HO 04 95", "2024-06", "water_backup", "Water back-up and sump discharge or overflow: we cover direct physical loss caused by water which backs up through sewers or drains or overflows from a sump, up to US$10,000, deductible US$1,000."),
     E("LM-HO 2311", "2022-01", "mould", "Limited fungi, mould, wet or dry rot coverage: we will pay up to US$5,000 in the aggregate for mould, fungus or rot resulting from a peril insured against, such as a plumbing leak."),
     E("FVB-HO 01", "2023-01", "water_backup", "Fox Valley Bank homeowners programme, water back-up: we cover water which backs up through sewers or drains or overflows from a sump, up to US$15,000, deductible US$1,000.", tenant="fvb"),
     E("NWS-HO 02", "2023-01", "freezing", "Northwoods Savings seasonal dwelling: freezing of plumbing in a seasonal dwelling is covered only if the water supply was shut off and the system drained before the dwelling was closed for the season.", tenant="nws"),
     E("LM-HO 0122 MN", "2021-01", "notice", "Special provisions, Minnesota: duties after loss are amended so that late notice bars coverage only if it prejudiced our investigation.", jurisdiction="MN"),
     E("LM-HO 0121 MI", "2021-01", "notice", "Special provisions, Michigan: suit must be started within one year after the date of loss; notice is amended to allow notice within one year.", jurisdiction="MI"),
    ]
    B = lambda bid, st, topic, eff, text, **k: S(
        sid=f"{bid}@{eff}#{topic}", doc_id=bid, version=eff, kind="bulletin", topic=topic,
        jurisdiction=st, text=text, effective_from=date.fromisoformat(eff), page=1, **k)
    out += [
     B("WI-OCI-2023-07", "WI", "roof", "2023-07-01", "Wisconsin bulletin on hail and wind roof claims: where replacement shingles do not match, the insurer should replace the damaged roof slope so the repaired roof has a reasonably uniform appearance.", effective_to=date(2025, 3, 15), superseded_by="WI-OCI-2025-03"),
     B("WI-OCI-2025-03", "WI", "roof", "2025-03-15", "Wisconsin bulletin on roof loss settlement: when matching materials are unavailable the insurer must replace the entire roof covering on the affected structure.", supersedes="WI-OCI-2023-07"),
     B("MN-COM-2023-02", "MN", "roof", "2023-02-10", "Minnesota bulletin on hail roof claims: insurers should replace damaged roofing so that the repaired roof shingles match and have a reasonably uniform appearance."),
     B("MN-COM-2025-06", "MN", "roof", "2025-06-02", "Minnesota bulletin on roof loss settlement: where materials cannot reasonably be matched, replace every slope visible from the street. Supersedes MN-COM-2023-02.", supersedes="MN-COM-2023-02"),
     B("IL-DOI-2025-05", "IL", "freezing", "2025-05-01", "Illinois bulletin on frozen pipes: where heat was lost because of a utility interruption, insurers may not deny for failure to maintain heat in the dwelling."),
     B("IA-ID-2024-08", "IA", "debris", "2024-08-15", "Iowa bulletin after the derecho: removal of trees that fell on a covered structure is payable as debris removal in addition to the limit for trees."),
     B("MI-DIFS-2024-06", "MI", "notice", "2024-06-24", "Michigan bulletin: an insurer may not deny a residential property claim for late notice unless it shows the delay prejudiced its investigation."),
     B("IN-IDOI-2025-01", "IN", "payment", "2025-01-15", "Indiana bulletin: payment on an agreed residential property claim must issue within 30 days of agreement; interest accrues after that."),
     B("WI-OCI-2024-11", "WI", "animal", "2024-11-01", "Wisconsin bulletin on deer: a collision with an object while swerving to avoid a deer is treated as contact with an animal and paid as other than collision."),
    ]
    return out


def context_spans():
    C = lambda sid, kind, topic, text, **k: S(sid=sid, doc_id=sid.split("@")[0],
                                              version=sid.split("@")[1], kind=kind,
                                              topic=topic, text=text, **k)
    return [
     C("CG-HO-014@v2", "guideline", "freezing", "Claims guideline frozen pipes: request utility bills or thermostat logs for the 30 days before the loss before deciding whether reasonable care was used to maintain heat."),
     C("CG-HO-031@v1", "guideline", "roof", "Claims guideline hail roof claims: photograph test squares, record shingle age, and check the state bulletin on matching before settling a partial roof replacement."),
     C("CG-HO-022@v1", "guideline", "water_backup", "Claims guideline basement water: determine the source first, sewer or drain back-up, sump overflow or surface water, and check the declarations for the water back-up endorsement."),
     C("NAIC-UCSP@2025", "advisory", "payment", "Model guidance on unfair claims settlement practices: insurers should pay claims promptly once liability is reasonably clear and should not delay payment of agreed amounts.", roles=ALLR),
     C("NAIC-UCSP@2025b", "advisory", "notice", "Model guidance: an insurer should not deny a claim for late notice without a reasonable basis showing the delay mattered to the investigation.", roles=ALLR),
     C("UW-NOTE-0002@2018", "note", "water_backup", "Underwriting note: sewer backup is never covered on HO-3; do not quote the back-up rider on older dwellings.", roles=UW),
     C("UW-NOTE-0019@2024", "note", "roof", "Underwriting note: roofs over 15 years old on hail-exposed risks in Minnesota and Wisconsin to be written at actual cash value at renewal.", roles=UW),
    ]


@dataclass(frozen=True)
class Det:
    did: str
    stratum: str
    question: str
    req: Request
    gold: str | tuple            # sid, or a pair of sids that must both be cited
    carried: str | None = None   # an upstream cause no Chapter 15 rule can reach
    upstream_withheld: bool = False
    notes: tuple = ()            # claim-file note spans for this claim only


QUESTIONS = {
 "freezing": ["pipes froze in the {place} while the heat was off, is the freezing of the plumbing covered",
              "frozen plumbing burst in the {place}, did they maintain heat, freezing loss"],
 "water_backup": ["sewer backed up into the {place} after the storm, is water that backs up through drains covered",
                  "sump pump overflowed and water backed up the drain into the {place}"],
 "roof": ["hail damaged the roof shingles and the replacement shingles do not match, do we replace the whole roof for a uniform appearance",
          "wind tore shingles off the roof, matching shingles unavailable, replace part of the roof or all of it"],
 "debris": ["derecho dropped a tree on the {place}, is removal of the tree paid as debris removal",
            "storm knocked a tree onto the {place}, who pays for removal of the debris"],
 "notice": ["insured gave notice of the loss four months late, can we deny for late notice",
            "late notice of the {place} loss, does failure to give prompt notice bar coverage"],
 "payment": ["we reached agreement on the {place} claim amount, how many days until loss payment",
             "agreed claim amount, when must loss payment issue to the insured"],
 "mould": ["mould found behind the {place} wall after a plumbing leak, how much is covered",
           "fungus and wet rot after water leaked from the plumbing in the {place}"],
 "animal": ["insured swerved to avoid a deer, an animal, and hit a guard rail: collision or other than collision",
            "hit a deer on the highway at night, is contact with an animal other than collision"],
 "oos_limits": ["accident in another state with a higher financial responsibility law limit, which liability limit applies",
                "insured's auto principally garaged at home crashed in another state, is the higher limit provided"],
}
PLACES = ["basement", "kitchen", "cabin", "garage", "bathroom", "laundry room"]
HO_TOPICS = ["freezing", "water_backup", "roof", "debris", "notice", "payment", "mould"]
PAP_TOPICS = ["animal", "oos_limits"]


def bulletin_for(spans, state, topic, on):
    """The reviewers' reading: the state's bulletin in force on the date of loss."""
    hits = [s for s in spans if s.kind == "bulletin" and s.jurisdiction == state
            and s.topic == topic and s.effective_from <= on]
    nxt = {s.supersedes: s for s in spans if s.supersedes}
    hits = [s for s in hits if not (s.doc_id in nxt and nxt[s.doc_id].effective_from <= on)]
    return hits[0].sid if hits else None


def reviewer_answer(spans, line, topic, pstate, rstate, edition, endos, on, tenant):
    if line == "PAP":
        b = bulletin_for(spans, pstate, topic, on) if topic != "oos_limits" else None
        return b or f"PAP@2018-01#{topic}"
    b = bulletin_for(spans, rstate, topic, on)
    if b:
        return b
    for s in spans:
        if s.kind == "endorsement" and s.scope == topic:
            if f"{s.edition_family}@{s.version}" in endos or s.jurisdiction == pstate:
                return s.sid
    return f"HO-3@{edition}#{topic}"


def endorsements_for(rng, tenant, edition):
    e = set()
    if rng.random() < 0.45:
        e.add("HO 04 95@2024-06" if edition == "2024-06" else "HO 04 95@2019-03")
    if rng.random() < 0.2:
        e.add("LM-HO 2311@2022-01")
    if tenant == "fvb":
        e.discard("HO 04 95@2024-06"); e.discard("HO 04 95@2019-03"); e.add("FVB-HO 01@2023-01")
    if tenant == "nws":
        e.add("NWS-HO 02@2023-01")
    return frozenset(e)


def rand_date(rng, lo=date(2023, 1, 1), hi=date(2026, 9, 30)):
    return lo + timedelta(days=rng.randrange((hi - lo).days))


STRATA = [("reviewed", 500), ("docstore-new", 160), ("policy-admin", 130),
          ("scanned", 90), ("multi-state", 60), ("guidance", 40), ("third-party", 20)]


def full_mix(seed=15):
    rng = random.Random(seed)
    spans = governing_spans()
    dets, n = [], 0
    plan = []
    for stratum, k in STRATA:
        plan += [stratum] * k
    # the 24 multi-state claims whose governing rule belongs to the property's state
    cross = [("MN", "WI", "roof"), ("IL", "WI", "roof"), ("WI", "MN", "roof"), ("IA", "MN", "roof"),
             ("MN", "IA", "debris"), ("WI", "IL", "freezing"), ("IN", "MI", "notice"), ("IL", "IN", "payment")]
    cross_i = 0
    flags = {"multi-hop": 6, "no-risk-state": 4}
    carried_left = {"reviewed": 2, "policy-admin": 4, "scanned": 1, "third-party": 1}
    withheld_left = 11
    regress_left = 2
    for stratum in plan:
        n += 1
        tenant = "lakeshore"
        if stratum in ("docstore-new", "policy-admin") and rng.random() < 0.18:
            tenant = rng.choice(["fvb", "nws"])
        line = "PAP" if (stratum != "multi-state" and rng.random() < 0.15) else "HO"
        pstate = rng.choice(STATES)
        rstate = pstate
        on = rand_date(rng)
        edition = rng.choice(HO3) if on >= date(2024, 6, 1) else rng.choice(HO3[:2])
        topic = rng.choice(PAP_TOPICS if line == "PAP" else HO_TOPICS)
        if stratum == "guidance":
            topic = rng.choice(["freezing", "roof", "water_backup", "payment"])
            line = "HO"
        if stratum == "third-party" and rng.random() < 0.5:
            line, topic = "HO", "water_backup"
        carried, rec_rstate = None, None
        if stratum == "multi-state":
            if cross_i < 24:
                pstate, rstate, topic = cross[cross_i % len(cross)]
                on = rand_date(rng, date(2025, 6, 2) if topic in ("roof", "freezing") else date(2025, 1, 15))
                cross_i += 1
                if flags["multi-hop"]:
                    flags["multi-hop"] -= 1; carried = "needs two governing spans (Chapter 17)"
                elif flags["no-risk-state"]:
                    flags["no-risk-state"] -= 1; rec_rstate = "missing"
            else:
                rstate = rng.choice([s for s in STATES if s != pstate])
                topic = rng.choice(["water_backup", "mould", "oos_limits"])
                if rng.random() < 0.25:            # the contract's state still governs notice
                    pstate, rstate, topic = "MN", rng.choice(["WI", "IA"]), "notice"
                line = "PAP" if topic == "oos_limits" else "HO"
        endos = endorsements_for(rng, tenant, edition) if line == "HO" else frozenset()
        gold = reviewer_answer(spans, line, topic, pstate, rstate, edition, endos, on, tenant)
        if carried == "needs two governing spans (Chapter 17)":
            gold = (gold, f"HO-3@{edition}#{topic}")   # the bulletin and the settlement clause
        rec_endos = endos
        upstream = False
        if stratum == "scanned" and withheld_left and rng.random() < 0.2:
            withheld_left -= 1; upstream = True
        if stratum == "docstore-new" and regress_left and line == "HO" and topic == "roof" and pstate in ("IL", "IN", "IA", "MI"):
            regress_left -= 1; rec_rstate = rng.choice(["WI", "MN"])  # geocoded across a state line
        q = rng.choice(QUESTIONS[topic]).format(place=rng.choice(PLACES))
        req = Request(role="adjuster", tenant=tenant, purpose="coverage-determination", line=line,
                      policy_state=pstate,
                      risk_state=None if rec_rstate == "missing" else (rec_rstate or rstate),
                      garaging_state=pstate, editions={"HO-3": edition, "PAP": "2018-01"},
                      endorsements=rec_endos, date_of_loss=on, cause=topic)
        dets.append(Det(f"D{n:04d}", stratum, q, req, gold, carried, upstream))
    return add_claim_notes(carry_record_errors(dets, carried_left), rng)


REASONS = {"reviewed": "endorsement schedule misread in the policy summary",
           "policy-admin": "endorsement missing from the policy record",
           "scanned": "OCR misread the endorsement on the declarations",
           "third-party": "licensed feed misdated the policy change"}


def carry_record_errors(dets, left):
    """Errors upstream of retrieval: the record omits an endorsement the policy has."""
    out = []
    for d in dets:
        e = [x for x in d.req.endorsements
             if x.split("@")[0] in ("HO 04 95", "FVB-HO 01", "LM-HO 2311")
             and (d.req.cause == "mould") == x.startswith("LM-HO 2311")]
        if (left.get(d.stratum) and d.carried is None and not d.upstream_withheld
                and d.req.cause in ("water_backup", "mould") and e):
            left[d.stratum] -= 1
            d = replace(d, carried=REASONS[d.stratum],
                        req=replace(d.req, endorsements=d.req.endorsements - set(e)))
        out.append(d)
    return out


def add_claim_notes(dets, rng):
    """Claim-file notes: adjuster-visible by source, a few restricted by content."""
    uw = [s for s in context_spans() if s.kind == "note"]
    out = []
    for i, d in enumerate(dets):
        notes = [S(sid=f"NOTE-{d.did}-1@1", doc_id=f"NOTE-{d.did}-1", version="1", kind="note",
                   topic=d.req.cause, text="Adjuster note: called the insured; photos and a contractor estimate requested.")]
        r = rng.random()
        if r < 0.007:
            notes.append(S(sid=f"NOTE-{d.did}-2@1", doc_id=f"NOTE-{d.did}-2", version="1", kind="note",
                           topic=d.req.cause, roles=SIU,
                           text="Special investigations referral: prior losses at this address; do not discuss the referral with the insured."))
        elif d.req.cause in ("water_backup", "roof") and r < 0.09:
            src = uw[0] if d.req.cause == "water_backup" else uw[1]
            copied = rng.random() < 0.6
            body = src.text if copied else ("Underwriting says sewer back-up is not covered on HO-3 at all."
                                            if d.req.cause == "water_backup" else
                                            "Per underwriting, older hail-exposed roofs go to ACV at renewal.")
            notes.append(S(sid=f"NOTE-{d.did}-2@1", doc_id=f"NOTE-{d.did}-2", version="1", kind="note",
                           topic=d.req.cause, text="Adjuster note, pasted: " + body))
        out.append(replace(d, notes=tuple(notes)))
    return out


def injected_copy():
    """A re-scan of the 2023 Wisconsin bulletin, ingested without its end date."""
    old = next(s for s in governing_spans() if s.doc_id == "WI-OCI-2023-07")
    return replace(old, sid="WI-OCI-2023-07@2023-07-01#roof-rescan", effective_to=None,
                   superseded_by=None)


def corpus_version(spans):
    return "corpus-" + hashlib.sha256("|".join(sorted(s.sid for s in spans)).encode()).hexdigest()[:10]
