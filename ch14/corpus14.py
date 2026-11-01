"""Chapter 14 stand-in world: Lakeshore spans, the gold set, the claims
glossary, a public 'leaderboard' set and four embedding models.

Every number in the chapter's lab comes from this file plus
contextlab/embeddings.py. Seeds are fixed. Nothing calls a provider.
"""
import random

from contextlab.embeddings import SimEmbedder

# concept: (plain synonyms, adjuster jargon). The concept name is the formal term.
CONCEPTS = {
    "depreciation": (["wear", "aging"], ["acv", "holdback"]),
    "replacement": (["replace", "rebuild"], ["rcv", "rc"]),
    "ordinance": (["code", "upgrade"], ["o&l"]),
    "concurrent": (["combined", "together"], ["acc"]),
    "matching": (["match", "mismatch"], ["los"]),
    "cosmetic": (["dent", "dimple"], ["cosmetic-only"]),
    "roof": (["shingles", "roofing"], ["rsp"]),
    "backup": (["drain", "sump"], ["wbu"]),
    "water": (["leak", "wet"], ["wd"]),
    "livingexpense": (["hotel", "displaced"], ["ale"]),
    "injury": (["medical", "hurt"], ["pip", "medpay"]),
    "uninsured": (["hit-and-run"], ["um", "uim"]),
    "bodilyinjury": (["injured", "victim"], []),
    "businessincome": (["revenue", "shutdown"], []),
    "subrogation": (["recover", "reimburse"], ["subro"]),
    "reservation": (["reserve"], ["ror"]),
    "examination": (["interview", "statement"], ["euo"]),
    "appraisal": (["dispute", "umpire"], []),
    "theft": (["stolen", "burglary"], []),
    "vacancy": (["empty", "unoccupied"], ["vacant-60"]),
    "hull": (["boat", "vessel"], ["h&m"]),
    "indemnity": (["crew", "pollution"], ["p&i"]),
    "layup": (["storage", "winterised"], ["port-risk"]),
    "navigation": (["cruising", "waters"], ["nav-limits"]),
    "wreck": (["salvage", "sunk"], ["wreck-removal"]),
    "collision": (["crash", "accident"], ["coll"]),
    "comprehensive": (["glass", "deer"], ["comp", "otc"]),
    "rental": (["loaner"], ["rental-reimb"]),
    "equipment": (["machinery", "breakdown"], ["eb"]),
    "spoilage": (["perishable", "freezer"], []),
    "deductible": (["out-of-pocket"], ["ded", "aop"]),
    "hail": (["hailstorm", "ice"], []),
    "wind": (["gust", "tornado"], ["w&h"]),
    "fungi": (["mould", "rot"], []),
    "notice": (["report", "late"], ["fnol"]),
    "cancellation": (["cancel", "nonrenewal"], ["nonrenew"]),
    "labour": (["workmanship", "contractor"], ["o&p"]),
}
AMBIGUOUS = {"bi"}  # adjusters: bodily injury on auto, business income on commercial
FILLER = ["insured", "policy", "coverage", "property", "loss", "damage", "pay",
          "section", "condition", "provided", "applies", "amount", "limit", "premises"]
BOILER = ("in this and the of to as it is by any all which when for on or with "
          "from that these under into at be an its will we you your our has have "
          "warranty assured underwriters").split()
STATES = {"WI": "wisconsin", "MN": "minnesota", "IL": "illinois",
          "IA": "iowa", "MI": "michigan", "IN": "indiana"}
FAMILIES = {"HO-3": "homeowners", "PAP": "auto", "BOP": "businessowners",
            "BOAT": "watercraft"}
FAMILY_CONCEPTS = {
    "HO-3": ["water", "backup", "theft", "wind", "hail", "fungi", "vacancy",
             "livingexpense", "replacement", "depreciation", "ordinance",
             "concurrent", "notice", "appraisal", "examination", "deductible",
             "roof", "matching"],
    "PAP": ["collision", "comprehensive", "injury", "uninsured", "bodilyinjury",
            "rental", "deductible", "notice", "subrogation", "cancellation",
            "examination", "theft"],
    "BOP": ["businessincome", "equipment", "spoilage", "theft", "water", "wind",
            "ordinance", "vacancy", "deductible", "notice", "fungi", "backup"],
    "BOAT": ["hull", "indemnity", "layup", "navigation", "wreck", "theft",
             "deductible", "notice", "collision", "subrogation"],
}
BULLETIN_TOPICS = [("matching", "replacement"), ("cosmetic", "roof"),
                   ("ordinance", "replacement"), ("concurrent", "water"),
                   ("backup", "water"), ("livingexpense", "vacancy"),
                   ("depreciation", "labour"), ("notice", "hail"),
                   ("depreciation", "roof"), ("matching", "roof"),
                   ("appraisal", "depreciation"), ("examination", "notice")]

LEXICON = {}
JARGON_TOKENS = set()
for c, (plain, jargon) in CONCEPTS.items():
    LEXICON[c] = c
    for t in plain:
        LEXICON[t] = c
    for t in jargon:
        LEXICON[t] = c
        JARGON_TOKENS.add(t)
for code, name in STATES.items():
    LEXICON[name] = name
    LEXICON[code.lower()] = name
FAM_WORD = {"HO-3": "home", "PAP": "auto", "BOP": "shop", "BOAT": "boat"}


class Span:
    def __init__(self, sid, family, kind, state, concepts, text, status="current"):
        self.sid, self.family, self.kind, self.state = sid, family, kind, state
        self.concepts, self.text, self.status = concepts, text, status

    @property
    def slice(self):
        return "bulletins" if self.kind in ("bulletin", "manual") else (
            "marine" if self.family == "BOAT" else self.kind + "s")


def _clause_text(rng, family, concepts, kind, sid):
    body = []
    for c in concepts:
        body += [c] * 3 + rng.sample(FILLER, 4)
    rng.shuffle(body)
    head = f"{sid} {FAMILIES[family]} {kind}"
    if family == "BOAT":  # long marine clauses: definitions first, the rule last
        pre = [rng.choice(BOILER) for _ in range(400)]
        return " ".join([head] + pre + body)
    return " ".join([head] + body)


def build_spans(seed=14):
    rng = random.Random(seed)
    spans = []
    for fam, cs in FAMILY_CONCEPTS.items():
        for i, c in enumerate(cs):
            other = cs[(i * 5 + 3) % len(cs)]
            sid = f"{fam}#c{i + 1:02d}"
            spans.append(Span(sid, fam, "form", None, [c, other],
                              _clause_text(rng, fam, [c, other], "form", sid)))
        for j in range(6):
            pair = rng.sample(cs, 2)
            sid = f"{fam}-END#{j + 1:02d}"
            spans.append(Span(sid, fam, "endorsement", None, pair,
                              _clause_text(rng, fam, pair, "endorsement", sid)))
    names = list(CONCEPTS)
    for m in range(120):  # claims-manual pages: same vocabulary, no authority
        pair = rng.sample(names, 2)
        sid = f"MAN#{m + 1:03d}"
        spans.append(Span(sid, "HO-3", "manual", None, pair,
                          _clause_text(rng, "HO-3", pair, "manual", sid)))
    for code, name in STATES.items():
        for k, topic in enumerate(BULLETIN_TOPICS):
            sid = f"BUL-{code}-{k + 1:02d}"
            fam = "HO-3"
            text = _clause_text(rng, fam, list(topic), "bulletin", sid)
            spans.append(Span(sid, fam, "bulletin", name, list(topic),
                              f"{name} department bulletin {name} " + text))
            if (k + len(code)) % 5 == 0:  # an older edition of the same bulletin
                old = _clause_text(rng, fam, list(topic), "bulletin", sid + "-2023")
                spans.append(Span(sid + "-2023", fam, "bulletin", name, list(topic),
                                  f"{name} department bulletin {name} " + old,
                                  status="superseded"))
    return spans


def _render(rng, concept, p_jargon):
    plain, jargon = CONCEPTS[concept]
    if jargon and rng.random() < p_jargon:
        return rng.choice(jargon)
    return rng.choice(plain + [concept]) if rng.random() < 0.6 else rng.choice(plain)


P_JARGON = {"bulletins": 0.75, "forms": 0.35, "endorsements": 0.45, "marine": 0.5}


def _query(rng, s):
    words = [_render(rng, c, P_JARGON[s.slice]) for c in s.concepts]
    if s.kind == "bulletin" and rng.random() < 0.5:
        words = words[:1]
    if s.kind == "bulletin":
        words.append(rng.choice([s.state, [k for k, v in STATES.items()
                                           if v == s.state][0].lower()]))
    else:
        words.append(FAM_WORD[s.family])
    if s.family == "PAP" and "bodilyinjury" in s.concepts:
        words = [("bi" if w in CONCEPTS["bodilyinjury"][0] + ["bodilyinjury"]
                  and rng.random() < 0.7 else w) for w in words] + ["bi"]
    if s.family == "BOP" and "businessincome" in s.concepts:
        words.append("bi")
    rng.shuffle(words)
    return " ".join(words)


def gold_set(spans, seed=1401):
    """240 adjuster-written questions, each labelled with one governing span."""
    rng = random.Random(seed)
    cur = [s for s in spans if s.status == "current" and s.kind != "manual"]
    plan = {"forms": 95, "endorsements": 45, "bulletins": 80, "marine": 20}
    out = []
    for sl, n in plan.items():
        pool = [s for s in cur if s.slice == sl]
        for i in range(n):
            s = pool[i % len(pool)]
            out.append((_query(rng, s), s.sid, sl))
    return out


def query_log(spans, n=1200, seed=1402):
    """Replayed production questions for shadow traffic (labels kept for analysis)."""
    rng = random.Random(seed)
    cur = [s for s in spans if s.status == "current" and s.kind != "manual"]
    mix = [("forms", 0.58), ("endorsements", 0.15), ("bulletins", 0.23), ("marine", 0.04)]
    out = []
    for _ in range(n):
        r, acc = rng.random(), 0.0
        for sl, p in mix:
            acc += p
            if r < acc:
                break
        s = rng.choice([x for x in cur if x.slice == sl])
        out.append((_query(rng, s), s.sid, sl))
    return out


# The claims glossary: Lakeshore's own jargon list. It misses some terms and
# its commercial team wrote "bi = business income" for everybody.
GLOSSARY = {t: c for t, c in LEXICON.items() if t in JARGON_TOKENS
            and t not in {"holdback", "nav-limits", "otc", "o&p"}}
GLOSSARY["bi"] = "businessincome"


def synthetic_pairs(spans, train_states=("WI", "MN", "IL"), per_span=6, seed=1403,
                    glossary=None, families=("BOP",)):
    """Stand-in for LLM-generated training questions, built from span text and
    the glossary only. No gold question is used; bulletins from IA, MI and IN
    are held out entirely."""
    glossary = glossary if glossary is not None else GLOSSARY
    rng = random.Random(seed)
    inv = {}
    for t, c in glossary.items():
        inv.setdefault(c, []).append(t)
    keep = {STATES[c] for c in train_states}
    out = []
    for s in spans:
        if s.status != "current" or (s.kind == "bulletin" and s.state not in keep):
            continue
        if s.kind in ("form", "endorsement", "manual") and s.family not in families:
            continue
        if s.kind == "manual":
            continue
        for _ in range(per_span):
            q = []
            for c in s.concepts:
                q.append(rng.choice(inv[c]) if c in inv and rng.random() < 0.7 else c)
            q.append(s.state if s.kind == "bulletin" else FAMILIES[s.family])
            if s.family == "BOP" and "businessincome" in s.concepts:
                q.append("bi")
            out.append((" ".join(q), s.sid))
    return out


def leaderboard_set(seed=1404, n=300):
    """A public-style benchmark: plain-English paraphrase retrieval, no jargon."""
    rng = random.Random(seed)
    names = list(CONCEPTS)
    docs, qs = [], []
    for i in range(n):
        cs = rng.sample(names, 3)
        docs.append(" ".join(sum(([c] * 2 + rng.sample(FILLER, 3) for c in cs), [])))
        qs.append((" ".join(rng.choice(CONCEPTS[c][0]) for c in cs), i))
    return docs, qs


def models(provider_update=False):
    kw = dict(lexicon=LEXICON, jargon_tokens=JARGON_TOKENS)
    return {
        "sim-embed": SimEmbedder("sim-embed", "1", 256, 0.55, 0.10, **kw),
        "gen-small": SimEmbedder("gen-small", "2026-03", 384, 0.75, 0.15, **kw),
        "gen-large": SimEmbedder("gen-large", "2026-03", 1024, 1.00, 0.30,
                                 matryoshka=True, **kw,
                                 space="gen-large@2026-10" if provider_update else None,
                                 drift=0.9 if provider_update else 0.0),
        "open-base": SimEmbedder("open-base", "1.2", 768, 0.85, 0.45,
                                 max_input_tokens=512, **kw),
    }


PROBES = ["water back-up endorsement sump overflow limit",
          "matching undamaged siding line of sight wisconsin",
          "actual cash value depreciation of labour",
          "protection and indemnity crew pollution",
          "ordinance or law increased cost of construction",
          "bodily injury limits per person per accident",
          "business income period of restoration",
          "notice of loss within sixty days"]


def respan(spans, max_tokens=480):
    """Split overlong spans into pieces that fit, each keeping its heading."""
    from contextlab.embeddings import n_tokens
    out = []
    for s in spans:
        if n_tokens(s.text) <= max_tokens:
            out.append(s)
            continue
        ws = s.text.split()
        head, rest = ws[:3], ws[3:]
        step = int(max_tokens / 1.3) - len(head)
        for i in range(0, len(rest), step):
            p = Span(f"{s.sid}#p{i // step + 1}", s.family, s.kind, s.state,
                     s.concepts, " ".join(head + rest[i:i + step]), s.status)
            out.append(p)
    return out
