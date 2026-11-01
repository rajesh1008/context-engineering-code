"""Chapter 41 stand-in world: Lakeshore's claims mix drifting through a storm season.

Deterministic; no language model runs. Each determination is right or wrong by
segment with a stand-in probability per pipeline version, drawn with a uniform
fixed per claim (common random numbers, so versions are compared on the same
claims). Overrides, categories and coders' notes are generated from the truth.
"""
import random

SEGMENTS = ["plain", "limit", "conditions", "deductible", "withhold", "edition",
            "conflict", "multi_peril"]
MONTHS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
_MIX = {  # production shares by month, in SEGMENTS order
    "Apr": (.40, .14, .14, .10, .08, .07, .05, .02),
    "May": (.39, .14, .14, .10, .08, .07, .05, .03),
    "Jun": (.37, .14, .13, .12, .08, .06, .05, .05),
    "Jul": (.35, .13, .13, .13, .08, .06, .05, .07),
    "Aug": (.33, .13, .13, .14, .07, .06, .05, .09),
    "Sep": (.33, .13, .13, .14, .07, .06, .05, .09),
    "Oct": (.33, .13, .13, .14, .07, .06, .05, .09),
}
MIX = {m: dict(zip(SEGMENTS, v)) for m, v in _MIX.items()}
# gold-v1.1 (Chapter 39): gold-v1, 1,000 single-governing-clause claims, plus 40 Illinois new-edition claims; lab mix set close to April
# (one governing span per item, so no multi-peril claims were eligible)
GOLD = {"plain": 424, "limit": 149, "conditions": 149, "deductible": 106,
        "withhold": 85, "edition": 74, "conflict": 53}
# probability the determination is right, as an adjuster would judge it
ACC = {
    "before":   dict(plain=.970, limit=.930, conditions=.920, deductible=.910,
                     withhold=.950, edition=.930, conflict=.920, multi_peril=.780),
    "released": dict(plain=.990, limit=.960, conditions=.950, deductible=.930,
                     withhold=.960, edition=.960, conflict=.950, multi_peril=.450),
    "per_peril": dict(plain=.990, limit=.960, conditions=.950, deductible=.930,
                      withhold=.960, edition=.960, conflict=.950, multi_peril=.930),
}
CAUSE = {  # first failure when a determination in this segment is wrong
    "plain": [("extraction", .6), ("condition_missed", .4)],
    "limit": [("wrong_limit_line", 1.0)], "conditions": [("condition_missed", 1.0)],
    "deductible": [("deductible_misapplied", 1.0)], "withhold": [("should_withhold", 1.0)],
    "edition": [("superseded_edition", 1.0)], "conflict": [("precedence_missed", 1.0)],
    "multi_peril": [("second_peril", 1.0)],
}
OVERRIDE_IF_WRONG = {"multi_peril": .80}   # adjusters often fix the second peril in notes
OVERRIDE_IF_WRONG_DEFAULT, OVERRIDE_IF_RIGHT = .90, .02
WEEKLY = 2400
STATES, FORMS = ["WI", "IL", "MI", "MN", "IA"], ["HO-3", "HO-5", "DP-3", "HO 04 90"]


def _pick(r, table):
    x, acc = r, 0.0
    for name, p in table:
        acc += p
        if x < acc:
            return name
    return table[-1][0]


_WEEKS = ["Jul"] * 4 + ["Aug"] * 5 + ["Sep"] * 4 + ["Oct"] * 4


def week_month(w):          # week 0 starts Mon 6 Jul; the release lands in week 9 (7 Sep)
    return _WEEKS[w]


def production(weeks=range(17), seed=41):
    """One dict per determination: id, week, month, segment, u, state, form, w (cause draw)."""
    rng, out = random.Random(seed), []
    for w in weeks:
        mix = MIX[week_month(w)]
        table = list(mix.items())
        for i in range(WEEKLY):
            out.append(dict(id=f"W{w:02d}-{i:04d}", week=w, month=week_month(w),
                            segment=_pick(rng.random(), table), u=rng.random(),
                            v=rng.random(), o=rng.random(),
                            state=rng.choice(STATES), form=rng.choice(FORMS)))
    return out


def gold_set(seed=4):
    rng, out = random.Random(seed), []
    for seg, n in GOLD.items():
        out += [dict(id=f"G-{seg}-{i:03d}", segment=seg, u=rng.random()) for i in range(n)]
    return out


def right(d, version):
    return d["u"] < ACC[version][d["segment"]]


def outcome(d, version):
    """(right, overridden, first-failure category or None)."""
    ok = right(d, version)
    if ok:
        over = d["o"] < OVERRIDE_IF_RIGHT
        return ok, over, ("not_a_failure" if over else None)
    p = OVERRIDE_IF_WRONG.get(d["segment"], OVERRIDE_IF_WRONG_DEFAULT)
    return ok, d["o"] < p, _pick(d["v"], CAUSE[d["segment"]])


def released(week):
    return "released" if week >= 9 else "before"
