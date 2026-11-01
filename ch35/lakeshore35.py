"""Chapter 35 fixtures: one simulated day of Lakeshore determinations.

Note: deterministic stand-in. 3,200 determinations in arrival order (the
daily volume used since Chapter 26). Every 16th-of-50 slot is in the reviewed
full mix of 1,000 (`reviewed`), which carries the four record-layer errors
the scoreboard has carried since Chapter 26. Hardness `h` is latent: part of
it comes from features the system can see before drafting, part from the
notes themselves, which no router sees.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

DAY = 3200
REVIEWED = 1000
INTENTS_HO = ("water", "wind", "hail", "theft", "fire", "liability")
INTENTS_PA = ("collision", "comprehensive", "liability", "glass")
STATES = (("WI", .45), ("MN", .20), ("IL", .15), ("IA", .10), ("MI", .10))
BULLETIN_CHANGE_AT = 1900   # WI water bulletin revised mid-afternoon
RECORD_ERRORS = 4


@dataclass
class Det:
    i: int
    line: str
    state: str
    intent: str
    form: str
    endorsements: tuple
    multi_state: int
    bulletin: int
    causes: int
    prior: int
    photos: int
    amount: float
    deductible: float
    policy: int
    notes_tokens: int
    hidden: float
    ambig: float = 0.0
    h: float = 0.0
    reviewed: bool = False
    record_error: bool = False
    u_flag: float = 0.0
    u_flag2: float = 0.0
    u_rerank: float = 0.0
    u_extract: float = 0.0

    def feats(self) -> dict:
        return {"causes2": int(self.causes >= 2), "causes3": int(self.causes >= 3),
                "n_end": len(self.endorsements), "multi_state": self.multi_state,
                "bulletin": self.bulletin, "prior": self.prior,
                "photos": min(self.photos, 10) / 10, "auto": int(self.line == "PA"),
                "amount_k": min(self.amount, 40000) / 40000,
                "ambiguity": self.ambig}

    @property
    def band(self) -> int:              # the true difficulty class
        return 0 if self.h < 0.35 else (1 if self.h < 0.70 else 2)

    @property
    def mechanical(self) -> bool:       # outcome fixed by rule, no reading needed
        return (self.amount < self.deductible and self.causes == 1
                and not self.bulletin and not self.multi_state)

    def retrieval_key(self, versioned: bool = True, bulletin_version: int = 1):
        k = (self.intent, self.form, self.endorsements, self.state)
        return k + (bulletin_version,) if versioned else k


def _pick(rng, pairs):
    x, acc = rng.random(), 0.0
    for v, p in pairs:
        acc += p
        if x < acc:
            return v
    return pairs[-1][0]


def day(seed: int = 35) -> list:
    rng = random.Random(seed)
    out = []
    for i in range(DAY):
        line = "HO" if rng.random() < 0.68 else "PA"
        state = _pick(rng, STATES)
        intent = rng.choice(INTENTS_HO if line == "HO" else INTENTS_PA)
        form = (rng.choice(("HO-3 2024-06", "HO-3 2024-06", "HO-3 2022-01", "HO-5 2024-06"))
                if line == "HO" else rng.choice(("PP 00 01 2018-09", "PP 00 01 2018-09", "PP 00 01 2005-01")))
        pool = ("HO 04 95", "HO 04 90", "HO 04 61", "HO 23 82", "HO 04 16") if line == "HO" \
            else ("PP 03 06", "PP 03 13", "PP 04 01")
        n_end = min(int(rng.expovariate(0.9)), len(pool))
        ends = tuple(sorted(rng.sample(pool, n_end)))
        causes = _pick(rng, ((1, .64), (2, .27), (3, .09)))
        multi = int(rng.random() < 0.08)
        bulletin = int(state == "WI" and intent == "water") or int(rng.random() < 0.05)
        prior = _pick(rng, ((0, .62), (1, .25), (2, .09), (3, .04)))
        photos = int(rng.expovariate(0.25))
        amount = round(math.exp(rng.gauss(8.6, 1.1)), -1)
        deductible = rng.choice((500, 1000, 1000, 2500))
        hidden = rng.betavariate(1.2, 4.0)
        d = Det(i, line, state, intent, form, ends, multi, bulletin, causes, prior, photos,
                amount, deductible, rng.randrange(6000), int(900 + rng.expovariate(1 / 700)),
                hidden)
        d.h = min(1.0, max(0.0, -0.05 + 0.26 * (causes >= 2) + 0.18 * (causes >= 3)
                           + 0.04 * n_end + 0.22 * multi + 0.14 * bulletin + 0.03 * prior
                           + 0.75 * hidden + rng.gauss(0, 0.03)))
        d.ambig = min(1.0, max(0.0, hidden + rng.gauss(0, 0.04)))
        d.u_flag, d.u_flag2, d.u_rerank, d.u_extract = (rng.random() for _ in range(4))
        out.append(d)
    rev = random.Random(seed + 1).sample(range(DAY), REVIEWED)
    for k in rev:
        out[k].reviewed = True
    for k in sorted(rev)[::250][:RECORD_ERRORS]:
        out[k].record_error = True
    return out


if __name__ == "__main__":
    D = day()
    from collections import Counter
    print(Counter(d.band for d in D), sum(d.mechanical for d in D))
