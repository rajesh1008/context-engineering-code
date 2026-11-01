"""Chapter 33 stand-in: Lakeshore packets, a reader, and three compressors.

Note: deterministic stand-in. No language model runs. Packets are generated
from templates; the 'reader' is a rule-based parser that turns packet text
into a determination; the abstractive compressor is a seeded paraphraser
whose keep and inversion rates are stated constants (assumptions, not
measurements of any model); the token scorer stands in for a learned
token-classification compressor. Four characters per token.
"""
from __future__ import annotations

import datetime as dt
import random
import re
import sys
import pathlib
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from contextlab.compress import Sentence, NEG, tokens, negation_profile  # noqa: E402

PERILS = [
    dict(key="wb", name="water back-up", phrase="water which backs up through sewers or drains",
         kw="sewer", form="HO 04 95", dev="sump pump",
         duty="maintain the sump pump in working order",
         find="Inspection on {d} found the sump pump was {n}maintained in working order.",
         para_exc="Sewer back-up is excluded in the main form.",
         para_res="The {form} endorsement restores sewer back-up cover."),
    dict(key="id", name="ice dam", phrase="water from ice dams on the roof",
         kw="ice dam", form="LK 21 07", dev="gutters",
         duty="keep the gutters and downspouts clear",
         find="Inspection on {d} found the gutters were {n}kept clear of debris.",
         para_exc="Ice dam water is excluded in the main form.",
         para_res="The {form} endorsement restores ice dam cover."),
    dict(key="fz", name="frozen pipes", phrase="freezing of a plumbing system while the dwelling is vacant",
         kw="freezing", form="LK 30 12", dev="heat",
         duty="maintain heat in the building",
         find="Inspection on {d} found heat was {n}maintained in the building.",
         para_exc="Freezing in a vacant dwelling is excluded in the main form.",
         para_res="The {form} endorsement restores freezing cover while vacant."),
]
NOTICE_DAYS = 60

POLICY_FILL = [
    "We insure against direct physical loss to property described in Coverage A.",
    "The most we will pay for loss to trees, shrubs and plants is US${a} for any one tree.",
    "Additional living expense is paid for the shortest time required to repair the dwelling.",
    "Collapse of a building caused by hidden decay is an additional coverage.",
    "Debris removal expense is included in the limit of liability for the damaged property.",
    "We may inspect the property at any reasonable time while this policy is in force.",
    "Smoke damage from a sudden and accidental event is covered under Coverage C.",
    "Property removed to protect it from a covered peril is covered for thirty days.",
    "Theft of property from a dwelling under construction is a matter for Section I conditions.",
    "Glass breakage is covered when it is part of a covered building.",
    "Fire department service charges are paid up to US${a} without a deductible.",
    "Loss to a pair or set is settled by repairing or replacing a part of the set.",
    "Appraisal may be demanded in writing by either party if the amount of loss is disputed.",
    "Mortgage clause rights apply to the mortgagee named in the Declarations.",
    "Credit card and counterfeit money coverage is limited to US${a}.",
    "Reasonable repairs to protect the property from further damage are covered.",
    "Ordinance or law coverage pays up to ten percent of the Coverage A limit.",
    "Volcanic eruption is covered when it causes direct physical loss.",
    "Our payment for loss is due sixty days after we receive your proof of loss.",
    "Lightning damage to electrical equipment is covered under Coverage C.",
    "Loss to outdoor antennas is subject to the Coverage C special limits.",
    "Watercraft and their trailers are limited to US${a} under Coverage C.",
    "Grave markers are covered up to US${a} for loss caused by a covered peril.",
    "Concealment or fraud by any insured voids the policy for all insureds.",
    "Suit against us must be brought within two years after the date of loss.",
    "Section II conditions apply to liability claims and do not change Section I.",
    "Business property on the residence premises is limited to US${a}.",
    "Landlord furnishings in an apartment on the premises are covered up to US${a}.",
    "Loss assessment coverage applies to assessments charged by an association.",
    "Personal property usually located at a secondary residence is limited to ten percent.",
]
END_FILL = [
    "All other provisions of the policy apply.",
    "This endorsement is effective from the date shown in the Declarations.",
    "Terms in bold have the meanings given in the definitions section.",
    "This endorsement does not increase the limits shown for Coverage A.",
    "Premium for this endorsement is shown in the Declarations.",
    "The endorsement applies only to the residence premises.",
    "Loss settlement follows the Section I conditions of the policy.",
    "Coverage under this endorsement is excess over any other valid insurance.",
    "Reference copies of this endorsement are filed with the state regulator.",
]
NOTE_FILL = [
    "Called the insured on {d} to arrange a visit; left a voicemail.",
    "Insured sent eleven photos of the affected area on {d}.",
    "Contractor estimate received on {d}: US${a} for clean-up and drywall.",
    "Mitigation company attended and set up two dehumidifiers.",
    "Insured asked about the timeline for payment; explained the process.",
    "Neighbour reported similar damage on the same street.",
    "Insured says the carpet was replaced two years ago.",
    "Requested receipts for the damaged furniture by email.",
    "Spoke with the mortgage company about endorsement of the cheque.",
    "Desk adjuster reviewed the photos; water line visible at about 20 centimetres.",
    "Insured prefers contact by text after five in the evening.",
    "Second contractor quote on {d}: US${a} including flooring.",
    "Moisture readings were taken in three rooms on {d}.",
    "Insured confirmed no other insurance applies to the contents.",
    "Diary set for follow-up on the outstanding receipts.",
    "Team lead reviewed the reserve on {d}; no change.",
    "Insured mentioned heavy rain in the area that week.",
    "Contents inventory received with forty two lines.",
    "Asked the insured whether any items were saved; some boxes were dried out.",
    "Photos show staining on the lower drywall in the family room.",
]
LETTER_FILL = [
    "Thank you for reporting your claim; your adjuster will contact you shortly.",
    "Please keep receipts for any emergency repairs you make.",
    "We have received your photographs and are reviewing them.",
    "Your claim number should appear on all correspondence with us.",
    "We are waiting for the contractor's itemised estimate.",
    "If you have questions, please call the number on your policy documents.",
    "We acknowledge your letter dated {d} about temporary accommodation.",
    "Please send the contents inventory when it is complete.",
]


def fmt(d: dt.date) -> str:
    return f"{d.day} {d.strftime('%B')} {d.year}"


@dataclass
class Packet:
    pid: str
    peril: dict
    attached: bool
    maintained: bool
    late: bool
    limit: int
    ded: int
    cov_a: int
    loss: int
    d_loss: dt.date
    d_rep: dt.date
    sents: list[Sentence]
    facts: dict = field(default_factory=dict)      # kind -> handle
    residual: bool = False

    @property
    def question(self) -> str:
        return f"Is the {self.peril['name']} loss on {self.pid} covered, and what is payable?"

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.sents)

    def gold(self):
        if not self.attached:
            return ("deny", "excluded")
        if self.late:
            return ("deny", "late notice")
        if not self.maintained:
            return ("deny", "condition")
        return ("pay", min(self.loss, self.limit) - self.ded)


def make_packet(i: int, seed: int = 33) -> Packet:
    r = random.Random(f"{seed}-{i}")
    p = r.choices(PERILS, weights=[5, 3, 2])[0]
    attached = r.random() < 0.7
    maintained = r.random() >= 0.22
    late = r.random() < 0.05
    limit = r.choice([5000, 10000, 15000, 25000])
    ded = r.choice([500, 1000, 2500])
    cov_a = r.choice([240000, 310000, 425000])
    loss = r.randrange(3000, 30000, 100)
    d_loss = dt.date(2026, 1, 1) + dt.timedelta(days=r.randrange(0, 200))
    d_rep = d_loss + dt.timedelta(days=r.randrange(70, 120) if late else r.randrange(0, 20))
    d_ins = d_rep + dt.timedelta(days=r.randrange(2, 9))
    pid = f"D{i:04d}"
    S, facts = [], {}

    def add(span, kind, text, fact=None):
        h = f"{pid}-{span}#s{sum(1 for s in S if s.handle.startswith(pid + '-' + span)) + 1}"
        S.append(Sentence(h, text, kind))
        if fact:
            facts[fact] = h

    def amt():
        return f"{r.randrange(5, 60) * 100:,}"

    def day():
        return fmt(d_loss + dt.timedelta(days=r.randrange(1, 40)))

    forms = ([p["form"]] if attached else []) + ["HO 04 90"]
    add("DEC", "declarations", f"Declarations for policy {pid[1:]}-HO3, Wisconsin, term 2026.")
    add("DEC", "declarations", f"Coverage A limit of liability: US${cov_a:,}.", "cov_a")
    add("DEC", "declarations", f"Endorsements shown: {', '.join(forms)}.")
    pol = r.sample(POLICY_FILL, 26)
    k = r.randrange(8, 20)
    for j, t in enumerate(pol):
        if j == k:
            add("POL", "policy", f"We do not insure for loss caused by {p['phrase']}.", "exclusion")
        if j == 4:
            add("POL", "policy", f"You must give us notice of loss within {NOTICE_DAYS} days.", "notice")
        add("POL", "policy", t.format(a=amt()))
    if attached:
        f = p["form"]
        core = [(f"This endorsement removes the exclusion of loss caused by {p['phrase']}.", "exception"),
                (f"The most we will pay under {f} is US${limit:,} for any one loss.", "limit"),
                (f"A deductible of US${ded:,} applies to each loss under {f}.", "deductible"),
                (f"Coverage under {f} does not apply if the loss results from your failure to {p['duty']}.",
                 "condition")]
        fill = r.sample(END_FILL, 7)
        for j, t in enumerate(fill):
            add("END", "endorsement", t)
            if j in (0, 2, 3, 5):
                add("END", "endorsement", *core.pop(0))
    notes = r.sample(NOTE_FILL, 18)
    for j, t in enumerate(notes):
        if j == 0:
            add("NTS", "notes", f"Loss occurred on {fmt(d_loss)}.", "d_loss")
            add("NTS", "notes", f"Loss reported on {fmt(d_rep)}.", "d_rep")
        if j == r.randrange(6, 14) and "finding" not in facts:
            add("NTS", "notes", p["find"].format(d=fmt(d_ins), n="" if maintained else "not "), "finding")
        add("NTS", "notes", t.format(d=day(), a=amt()))
    if "finding" not in facts:
        add("NTS", "notes", p["find"].format(d=fmt(d_ins), n="" if maintained else "not "), "finding")
    add("NTS", "notes", f"Adjusted loss amount agreed at US${loss:,}.", "loss")
    for t in r.sample(LETTER_FILL, 6):
        add("LTR", "letters", t.format(d=day()))
    return Packet(pid, p, attached, maintained, late, limit, ded, cov_a, loss, d_loss, d_rep, S, facts)


def build(n: int, seed: int = 33, residual: tuple = ()) -> list[Packet]:
    out = [make_packet(i, seed) for i in range(1, n + 1)]
    for pk in out:
        pk.residual = pk.pid in residual
    return out


# ------------------------------------------------------------------ the reader
MONEY = re.compile(r"US\$([\d,]+)")
DATE = re.compile(r"(\d{1,2}) (January|February|March|April|May|June|July|August|September|"
                   r"October|November|December) (\d{4})")


def money(s):
    m = MONEY.search(s)
    return int(m.group(1).replace(",", "")) if m else None


def date(s):
    m = DATE.search(s)
    return dt.datetime.strptime(" ".join(m.groups()), "%d %B %Y").date() if m else None


def read_facts(texts: list[str], p: dict) -> dict:
    """What a reader can establish from packet text. Polarity comes from cues."""
    f = dict(excl=None, exc=False, limit=None, ded=None, cov_a=None, cond=None,
             maint=None, d_loss=None, d_rep=None, notice=False, loss=None)
    form, kw, dev = p["form"].lower(), p["kw"], p["dev"]
    for t in texts:
        s = t.lower()
        neg = negation_profile(t) > 0
        if kw in s and ("removes" in s or "restores" in s):
            f["exc"] = True
        elif "deductible" in s and money(t) and form in s:
            f["ded"] = money(t)
        elif form in s and money(t):
            f["limit"] = money(t)
        elif "coverage a" in s and money(t):
            f["cov_a"] = money(t)
        elif kw in s and (neg or any(w in s for w in ("insure", "cover"))):
            f["excl"] = neg
        elif "fail" in s and dev in s and form in s:
            f["cond"] = neg                      # True: failure voids the cover
        elif ("inspection" in s or "found" in s) and dev in s:
            f["maint"] = not neg
        elif "notice" in s and "days" in s:
            f["notice"] = True
        elif "occurred" in s and date(t):
            f["d_loss"] = date(t)
        elif "reported" in s and date(t):
            f["d_rep"] = date(t)
        elif "agreed" in s and money(t):
            f["loss"] = money(t)
    return f


def determine(texts: list[str], p: dict):
    f = read_facts(texts, p)
    if f["excl"] and not f["exc"]:
        return ("deny", "excluded")
    if f["notice"]:
        if not (f["d_loss"] and f["d_rep"]):
            return ("withheld", "dates")
        if (f["d_rep"] - f["d_loss"]).days > NOTICE_DAYS:
            return ("deny", "late notice")
    if f["cond"] and f["maint"] is False:
        return ("deny", "condition")
    if f["loss"] is None:
        return ("withheld", "amount")
    limit = f["limit"] or f["cov_a"] or 0
    return ("pay", min(f["loss"], limit) - (f["ded"] or 0))


def grade(pk: Packet, texts: list[str]) -> str:
    got = determine(texts, pk.peril)
    if pk.residual:
        return "wrong"
    if got[0] == "withheld":
        return "withheld"
    return "right" if got == pk.gold() else "wrong"


# ------------------------------------------------------------------ compressors
STOP = set("a an the of to in on at by for and or is was were be been this that which with as "
           "we you your our any it its from are has have under if".split())


def stem(w):
    return w.lower().rstrip("s")


def qterms(pk: Packet) -> set:
    return {stem(w) for w in re.findall(r"[A-Za-z]+", pk.question) if w.lower() not in STOP}


def ext_score(pk: Packet, terms: set):
    prior = {"endorsement": 0.4, "policy": 0.2, "declarations": 0.3, "notes": 0.1, "letters": 0.0}

    def score(s: Sentence) -> float:
        ws = {stem(w) for w in re.findall(r"[A-Za-z]+", s.text) if w.lower() not in STOP}
        jitter = (hash_(s.handle) % 100) / 1000
        return len(ws & terms) + prior[s.kind] + 0.3 * bool(money(s.text)) + jitter
    return score


def hash_(s: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(s)) % 9973


def token_score(tok: str, sent: str = "") -> float:
    """Stand-in for a learned token scorer. Words in a sentence that carries a
    number or date score higher (it sees context). Negations get a middling
    score: an assumption about a generic scorer, not a measurement."""
    w = tok.strip(".,;:").lower()
    lift = 0.0 if any(c.isdigit() for c in sent) and w not in STOP else 0.0
    return min(0.96, _base(tok, w) + lift)


def _base(tok: str, w: str) -> float:
    if any(c.isdigit() for c in w) or w.startswith("us$"):
        return 0.97
    if tok[:1].isupper() and len(w) <= 3:
        return 0.9
    if tok[:1].isupper():
        return 0.8
    if w in ROLE:
        return 0.86
    if w in ("not", "no", "unless", "except"):
        return 0.55
    if w in STOP or len(w) <= 2:
        return 0.1 + (hash_(w) % 10) / 100
    return 0.45 + 0.04 * min(len(w), 10) + (hash_(w) % 50) / 1000


ROLE = {"occurred", "reported", "agreed", "deductible", "most", "pay", "exclusion", "removes",
        "insure", "found", "inspection", "failure", "maintained", "notice", "days", "coverage",
        "loss", "sewers", "ice", "dams", "freezing", "vacant", "pump", "gutters", "heat"}
FORCE = NEG + ("removes", "fail", "failure")

# abstractive stand-in: keep probability by fact, at ratio 0.5 (assumptions)
KEEP = dict(exclusion=0.97, exception=0.92, limit=0.94, deductible=0.8, condition=0.82,
            finding=0.88, loss=0.95, d_loss=0.55, d_rep=0.55, notice=0.6, cov_a=0.7)
KEEP_GUIDED = dict.fromkeys(KEEP, 0.985)
P_INVERT = 0.03          # chance a kept condition comes out with its polarity reversed
FILLER_SUMMARY = [
    "The insured reported water damage and a contractor attended.",
    "Photos and estimates were received and reviewed.",
    "The insured has been kept informed of progress.",
    "Receipts and an inventory were requested.",
    "Mitigation equipment was installed.",
    "Correspondence acknowledged the claim and requested documents.",
]


def paraphrase(pk: Packet, kind: str, invert: bool) -> str:
    p, f = pk.peril, pk.peril["form"]
    if kind == "exclusion":
        return p["para_exc"]
    if kind == "exception":
        return p["para_res"].format(form=f)
    if kind == "limit":
        return f"Cover under {f} is capped at US${pk.limit:,} per loss."
    if kind == "deductible":
        return f"The {f} deductible is US${pk.ded:,}."
    if kind == "condition":
        if invert:
            return f"Cover under {f} applies even if you fail to {p['duty']}."
        return f"Cover under {f} is void if you fail to {p['duty']}."
    if kind == "finding":
        return f"Inspection found the {p['dev']} was {'' if pk.maintained else 'not '}maintained."
    if kind == "loss":
        return f"Loss amount agreed at US${pk.loss:,}."
    if kind == "d_loss":
        return f"Loss occurred on {fmt(pk.d_loss)}."
    if kind == "d_rep":
        return f"Loss reported on {fmt(pk.d_rep)}."
    if kind == "notice":
        return f"Notice is due within {NOTICE_DAYS} days."
    if kind == "cov_a":
        return f"Coverage A limit is US${pk.cov_a:,}."
    raise KeyError(kind)


def abstractive(pk: Packet, ratio: float, guided: bool = False, invert_rate: float = P_INVERT,
                seed: str = "a", only: set | None = None) -> list[str]:
    """Seeded paraphraser. `only` limits it to sentences of these kinds; the
    rest of the packet passes verbatim (used for the incident replay)."""
    r = random.Random(f"{seed}-{pk.pid}-{ratio}-{guided}")
    keep = KEEP_GUIDED if guided else KEEP
    squeeze = {0.5: 1.0, 0.33: 0.92, 0.2: 0.78}.get(ratio, 1.0)
    h2k = {h: k for k, h in pk.facts.items()}
    scope = [s for s in pk.sents if only is None or s.kind in only]
    out, budget = [], ratio * sum(s.tokens for s in scope)
    for s in pk.sents:
        if only is not None and s.kind not in only:
            out.append(s.text)
            continue
        kind = h2k.get(s.handle)
        sq = 1 - (1 - squeeze) / 4 if guided else squeeze
        if kind and r.random() < min(1.0, keep[kind] * sq):
            inv = kind == "condition" and r.random() < (invert_rate / 6 if guided else invert_rate)
            out.append(paraphrase(pk, kind, inv))
    used = sum(tokens(t) for t in out if t not in [s.text for s in pk.sents])
    # the rest of the budget goes on the gist of the other sentences, in order:
    # a short paraphrase standing for each, the way a summary condenses detail
    for s in scope:
        if used >= budget * 0.9:
            break
        if s.handle in h2k:
            continue
        w = s.text.rstrip(".").split()
        t = " ".join(w[:6]) + "."
        out.append(t); used += tokens(t)
    for t in FILLER_SUMMARY:
        if used >= budget * 0.9:
            break
        out.append(t); used += tokens(t)
    return out
