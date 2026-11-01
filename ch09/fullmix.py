"""The full-mix evaluation set and a deterministic stand-in model.

Note: no language model runs here. Outcomes come from stated rules plus
seeded noise (sim-large behaviour, as in earlier chapters). The 500
reviewed determinations keep the outcomes Chapter 8 measured.
"""
import random
from dataclasses import dataclass
from datetime import date, timedelta

# share of daily determinations by what the adjuster had to open
# (90 days of claims-system access logs, modelled)
DAILY_MIX = {"docstore": 0.66, "policy-admin": 0.13, "scanned": 0.09,
             "multi-state": 0.06, "guidance": 0.04, "third-party": 0.02}
STRATA_N = {"docstore": 160, "policy-admin": 130, "scanned": 90,
            "multi-state": 60, "guidance": 40, "third-party": 20}


@dataclass
class Det:
    det_id: str
    stratum: str
    split: str                 # eval, holdout or new
    u: float                   # seeded noise for the stand-in
    pointer_wrong: bool = False
    edition_printed: bool = True
    carried_wrong: bool = False
    sql_fault: bool = False


def build() -> list[Det]:
    rng = random.Random(9)
    dets = []
    # the 500 reviewed determinations: born-digital, single state
    for i in range(500):
        split = "eval" if i < 200 else "holdout"
        wrong = (split == "eval" and i < 5) or (split == "holdout" and i < 202)
        dets.append(Det(f"DET-{i+1:04d}", "docstore", split, rng.random(),
                        carried_wrong=wrong))
    n = 500
    for stratum, k in STRATA_N.items():
        for _ in range(k):
            n += 1
            d = Det(f"DET-{n:04d}", stratum, "new", rng.random())
            if stratum == "policy-admin":
                d.pointer_wrong = rng.random() < 0.22
                d.sql_fault = rng.random() < 0.03
            if stratum == "scanned":
                d.edition_printed = rng.random() < 0.58
            dets.append(d)
    return dets


def outcome(d: Det, config: str) -> str:
    """Return 'right', 'wrong' or 'withheld' for one determination."""
    if d.split != "new":
        return "wrong" if d.carried_wrong else "right"
    u, ch9 = d.u, config == "ch09"
    if d.stratum == "docstore":            # chunking misses: Chapter 10
        return "wrong" if u < 0.06 else "right"
    if d.stratum == "policy-admin":
        if not ch9:                        # header comes from the pointer
            p = 0.70 if d.pointer_wrong else 0.03
            return "wrong" if u < p else "right"
        if d.sql_fault:                    # verifier rejects, adjuster decides
            return "withheld"
        return "wrong" if u < 0.03 else "right"
    if d.stratum == "scanned":
        if not ch9:                        # scan never indexed: base form cited
            return "wrong" if u < 0.22 else "right"
        if not d.edition_printed:          # fail closed
            return "withheld"
        return "wrong" if u < 0.12 else "right"   # OCR noise: Chapter 11
    if d.stratum == "multi-state":         # jurisdiction: Chapter 15
        return "wrong" if u < 0.40 else "right"
    if d.stratum == "guidance":
        return "wrong" if u < 0.06 else "right"
    if d.stratum == "third-party":         # licence gate removes the feed
        return "wrong" if u < (0.05 if ch9 else 0.20) else "right"
    raise ValueError(d.stratum)


TOKENS_CH08 = 1540
EXTRA = {  # extra packet tokens by stratum (headers, query spans, scans)
    "ch08": {"docstore": 0, "policy-admin": 0, "scanned": 0,
             "multi-state": 140, "guidance": 0, "third-party": 90},
    "ch09": {"docstore": 36, "policy-admin": 104, "scanned": 212,
             "multi-state": 176, "guidance": 36, "third-party": 36},
}
USD_PER_DET_CH08 = 0.00548


def lookup_minutes(d: Det, result: str) -> float:
    if result == "withheld":
        return 9.0 + 4.0 * d.u             # adjuster finds the page by hand
    return 1.0 + 0.6 * d.u


def score(dets, config):
    res = [(d, outcome(d, config)) for d in dets]
    wrong = sum(r == "wrong" for _, r in res)
    held = sum(r == "withheld" for _, r in res)
    mins = sorted(lookup_minutes(d, r) for d, r in res)
    n = len(res)
    median = (mins[n // 2 - 1] + mins[n // 2]) / 2 if n % 2 == 0 else mins[n // 2]
    tokens = sum(TOKENS_CH08 + EXTRA[config][d.stratum]
                 if d.split == "new" else
                 TOKENS_CH08 + (36 if config == "ch09" else 0)
                 for d, _ in res) / n
    return {"n": n, "wrong": wrong, "rate": wrong / n, "withheld": held,
            "median_min": median, "mean_min": sum(mins) / n,
            "tokens": tokens, "usd": USD_PER_DET_CH08 * tokens / TOKENS_CH08}
