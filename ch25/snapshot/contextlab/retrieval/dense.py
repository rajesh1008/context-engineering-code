"""A deterministic bi-encoder stand-in ("sim-embed") and a flat dense index.

sim-embed learns nothing. It maps words to concepts through a fixed lexicon,
gives every concept a fixed random direction (related concepts share part of
a group direction), weights the pieces and mean-pools them into one unit
vector. That reproduces the behaviour of a small bi-encoder that matters in
Chapter 12: paraphrases land close together; pooling discards word order and
nearly all of a negation; numbers and rare identifiers carry little weight.
The weights below are the whole model, stated so you can argue with them.
"""
import hashlib

import numpy as np

from .text import tokenize, stem

DIM = 256
W_CONCEPT, W_UNKNOWN, W_NUMBER, W_NEGATION = 1.0, 0.35, 0.12, 0.08
NEGATION = {"not", "no", "never", "without", "except", "unless", "nor",
            "doesn't", "isn't", "won't", "don't", "didn't", "aren't", "can't"}
STOP = set("a an and are as at be by for from in into is it its of on or "
           "such that the their then there these this to was will with we you "
           "your our us i my me he she they what which who does do did has "
           "have had can would should could if when where how any all".split())

GROUPS = {  # concept -> group; concepts in a group share 60% of a direction
    "water": "WET", "flood": "WET", "sewer": "WET", "plumbing": "WET",
    "seepage": "WET", "freeze": "COLD", "ice": "COLD", "snow": "COLD",
    "wind": "STORM", "hail": "STORM", "roof": "STORM", "tree": "STORM",
    "mould": "WET", "fire": "HEAT", "smoke": "HEAT", "theft": "CRIME",
    "vandal": "CRIME", "vehicle": "MOTOR", "collision": "MOTOR",
    "glass": "MOTOR", "animal": "MOTOR", "hire": "MOTOR", "rental": "MOTOR",
    "boat": "MARINE", "layup": "MARINE", "navigation": "MARINE",
    "business": "TRADE", "spoilage": "TRADE", "income": "TRADE",
    "cover": "PAY", "pay": "PAY", "limit": "PAY", "exclude": "PAY",
}


def _lexicon() -> dict[str, str]:
    raw = {
        "water": "water wet leak leaking leaked drip dripping moisture",
        "flood": "flood flooding flooded surface overflow tidal river rising",
        "sewer": "sewer sewage drain drains backup back-up backed sump "
                 "septic",
        "plumbing": "plumbing pipe pipes piping sprinkler heating boiler "
                    "radiator appliance heater faucet tap fixture",
        "rupture": "burst bursting split cracking cracked crack tearing torn "
                   "rupture ruptured bulging",
        "seepage": "seepage seep seeping continuous repeated slow weeks",
        "freeze": "freezing freeze frozen froze cold sub-zero winter snap",
        "ice": "ice icy ice-dam ice-dams icicle",
        "snow": "snow sleet weight",
        "wind": "wind windstorm tornado derecho gust gusts gale storm",
        "hail": "hail hailstone hailstones hailstorm",
        "roof": "roof roofing shingle shingles gutter",
        "tree": "tree trees shrub shrubs plants lawn limb branch",
        "debris": "debris removal cleanup clean-up haul",
        "mould": "mould mold fungus fungi mildew rot spores",
        "fire": "fire burning burned blaze flames lightning",
        "smoke": "smoke soot",
        "theft": "theft stolen stole steal burglary break-in burglar burglars robbery larceny",
        "vandal": "vandalism vandal malicious graffiti",
        "vacant": "vacant vacancy unoccupied empty away absent",
        "vehicle": "vehicle car auto automobile truck van suv",
        "collision": "collision crash accident upset rear-ended",
        "glass": "glass windshield windscreen window",
        "animal": "animal deer bird",
        "hire": "hire rideshare delivery deliveries livery fee courier",
        "rental": "rental rent substitute loaner",
        "boat": "boat watercraft vessel yacht outboard",
        "layup": "lay-up layup winterized stored ashore",
        "navigation": "navigation navigate navigating waters lake michigan",
        "business": "business commercial premises shop store tenant bakery customer customers trade occupation",
        "spoilage": "spoilage spoil spoiled perishable refrigeration",
        "income": "income earnings revenue",
        "cover": "cover covers covered coverage insure insures insured "
                 "protect protected",
        "pay": "pay pays paid payment payable settle settlement reimburse",
        "limit": "limit limits maximum cap capped",
        "exclude": "exclude excludes excluded exclusion exclusions",
        "notice": "notice notify notified report reported reporting tell told",
        "deadline": "deadline within prompt promptly late fast long wait",
        "appraisal": "appraisal appraiser umpire dispute disputes disagree estimate impartial",
        "suit": "suit lawsuit sue legal action court",
        "mortgage": "mortgage mortgagee lender bank lienholder",
        "medical": "medical injury injured hurt hospital doctor treatment er bills funeral",
        "liability": "liability liable sued negligence",
        "neglect": "neglect neglected preserve save worse further",
        "wear": "wear wore tear deterioration deteriorated age old rust rusted corrosion",
        "earth": "earth earthquake sinkhole subsidence settling landslide mudslide",
        "collapse": "collapse collapsed sagging cave",
        "depreciation": "depreciation depreciate replacement replace rebuild underinsured",
        "intent": "intentional intent purpose deliberately arson",
        "edition": "edition version form forms",
        "bulletin": "bulletin bulletins guidance commissioner department",
        "endorse": "endorsement endorsements rider amendatory attached",
    }
    lex = {}
    for concept, words in raw.items():
        for w in words.split():
            lex[w] = concept
            lex[stem(w)] = concept
    return lex


LEXICON = _lexicon()


def _direction(name: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    v = np.random.default_rng(seed).standard_normal(DIM)
    return v / np.linalg.norm(v)


class SimEncoder:
    name = "sim-embed"

    def __init__(self):
        self._cache: dict[str, np.ndarray] = {}

    def piece(self, tok: str) -> tuple[np.ndarray, float] | None:
        if tok in STOP:
            return None
        if tok in NEGATION:
            return _direction("tok:" + tok), W_NEGATION
        if any(c.isdigit() for c in tok):
            v = 0.8 * _direction("NUMBER") + 0.6 * _direction("tok:" + tok)
            return v / np.linalg.norm(v), W_NUMBER
        concept = LEXICON.get(tok) or LEXICON.get(stem(tok))
        if concept:
            v = _direction("concept:" + concept)
            if concept in GROUPS:
                v = 0.8 * v + 0.6 * _direction("group:" + GROUPS[concept])
            return v / np.linalg.norm(v), W_CONCEPT
        return _direction("tok:" + stem(tok)), W_UNKNOWN

    def encode(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        acc = np.zeros(DIM)
        for tok in tokenize(text):
            p = self.piece(tok)
            if p:
                acc += p[1] * p[0]
        n = np.linalg.norm(acc)
        v = acc / n if n else acc
        self._cache[text] = v
        return v


class DenseIndex:
    """Flat (exact) cosine search. Section 12.7 replaces it with HNSW."""

    def __init__(self, encoder: SimEncoder):
        self.encoder = encoder
        self.ids: list[str] = []
        self.rows: list[np.ndarray] = []
        self.matrix = None

    def add(self, doc_id: str, text: str) -> None:
        self.ids.append(doc_id)
        self.rows.append(self.encoder.encode(text))
        self.matrix = None

    def search(self, query: str, allowed=None) -> dict[str, float]:
        if self.matrix is None:
            self.matrix = np.vstack(self.rows)
        sims = self.matrix @ self.encoder.encode(query)
        return {d: float(s) for d, s in zip(self.ids, sims)
                if allowed is None or d in allowed}
