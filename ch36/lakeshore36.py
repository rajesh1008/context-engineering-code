"""Lakeshore stand-in for Chapter 36: the determination module and its metric.

The 'model' is a deterministic function of the prompt and the claim. It is not
a language model. It encodes four behaviours the chapter needs: rules help the
cases they name, demonstrations help cases like them, item-specific rules fix
the items they were written from and misfire on look-alikes, and any change to
the prompt re-rolls a little noise on every item (churn).
"""
import hashlib
import random
from dataclasses import dataclass

CASES = {  # share of the full mix, base error rate with no rule
    "plain": (0.40, 0.08), "limit": (0.14, 0.35), "conditions": (0.14, 0.30),
    "deductible": (0.10, 0.28), "withhold": (0.08, 0.30),
    "edition": (0.07, 0.25), "conflict": (0.07, 0.25)}
STATES = ["WI", "MN", "IA", "IL", "IN", "MI"]
FORMS = ["HO-3", "HO 04 95", "HO-5", "DP-3"]

RULES = {
    "P1": "Cite the policy edition in force on the date of loss.",
    "P2": "Where an endorsement and the base form conflict, the endorsement governs.",
    "P1r": "Cite the edition attached at the latest renewal.",
    "G1": "Every rationale sentence must be supported by the cited span; omit what is not.",
    "G2": "Quote limits from the endorsement schedule, not the declarations summary.",
    "G3": "State a deductible only when the cited span states it.",
    "G4": "If the governing span is not in the packet, say so and refer.",
    "G5": "Name every condition attached to the cover the span grants.",
    "N1": "Think step by step.",
    "N2": "Be concise.",
    "N3": "Answer as an experienced claims adjuster.",
    "N4": "Be accurate and cite carefully.",
}
FIXES = {"G1": "*", "G2": "limit", "G3": "deductible", "G4": "withhold", "G5": "conditions"}
WEIGHT = {"G2": 0.45, "G3": 0.75, "G5": 0.75}
HAND_WRITTEN = (("P1", "P2"), ("G1", "G2", "G4"))
POOL = ["G1", "G2", "G3", "G4", "G5", "N1", "N2", "N3"]


@dataclass(frozen=True)
class Item:
    id: str
    case: str
    state: str
    form: str
    limit: int
    renewal_mismatch: bool
    retrieval_miss: bool


def u(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16) / 16 ** 12


def make(prefix, n, seed, misses=0):
    rng, names, weights = random.Random(seed), list(CASES), [CASES[c][0] for c in CASES]
    out = []
    for i in range(n):
        case = rng.choices(names, weights)[0]
        form = rng.choice(FORMS)
        limit = rng.choice([10000, 10000, 5000, 25000]) if form == "HO 04 95" else 0
        out.append(Item(f"{prefix}{i:04d}", case, rng.choice(STATES), form, limit,
                        case == "edition" and rng.random() < 0.15, i < misses))
    return out


def narrow_rule(item):
    """What a reflector writes when it generalises from one failure."""
    if item.case == "limit" and item.form == "HO 04 95":
        return f"X:{item.state}:limit:{item.limit}"
    return f"K:{item.id}"


def p_error(prompt, it):
    rules = set(prompt.rules) | set(prompt.pinned)
    if it.case in ("edition", "conflict"):
        need = "P1" if it.case == "edition" else "P2"
        if it.case == "edition" and "P1r" in rules and it.renewal_mismatch:
            return 1.0
        if need in rules or (it.case == "edition" and "P1r" in rules):
            return 0.0
    p = CASES[it.case][1]
    for r, c in FIXES.items():
        if r in rules and c == it.case:
            p *= WEIGHT.get(r, 0.55)
    if "G1" in rules:
        p *= 0.8
    if "N2" in rules and it.case == "conditions":
        p += 0.08
    narrow = [r for r in rules if r[:2] in ("K:", "X:")]
    p += CLUTTER * len(narrow)
    shown = sum(1 for d in prompt.demos if d.split(":")[1] == it.case)
    p *= 0.8 ** min(shown, 2)
    for r in sorted(rules):
        if r.startswith("K:") and r[2:] == it.id and u("K", it.id) < 0.5:
            return 0.0
        if r.startswith("X:"):
            _, st, case, lim = r.split(":")
            if it.state == st and it.case == case and it.form == "HO 04 95":
                return 0.0 if it.limit == int(lim) else 0.9
    return p


def program(prompt, it):
    """Return a prediction: did the module cite the governing span, and is every
    rationale sentence supported? Noise depends on the whole prompt (churn)."""
    if it.retrieval_miss:
        return {"clause_ok": False, "supported": False, "why": "governing span not retrieved"}
    p = p_error(prompt, it)
    roll = u("item", it.id) + JITTER * (u(prompt.key, it.id) - 0.5)
    bad = p > 0 and roll < p
    clause_bad = bad and it.case in ("edition", "conflict")
    return {"clause_ok": not clause_bad, "supported": not bad,
            "why": f"{it.case} sentence not supported by cited span" if bad else ""}


JITTER = 0.10
CLUTTER = 0.004
METRIC_VERSION = "faithfulness-v2"


def faithfulness(it, pred):
    """Metric as code: 1 only if the governing span is cited and every rationale
    sentence is supported by it; feedback names the case and the state."""
    if not pred["clause_ok"]:
        return 0.0, {"case": it.case, "state": it.state, "form": it.form,
                     "msg": "wrong clause cited", "item": it}
    if not pred["supported"]:
        return 0.0, {"case": it.case, "state": it.state, "form": it.form,
                     "msg": pred["why"], "item": it}
    return 1.0, {}


def wrong_clause(prompt, items):
    return sum(1 for it in items if not program(prompt, it)["clause_ok"])


def tokens(prompt):
    return 1800 + 25 * (len(prompt.rules) + len(prompt.pinned)) + 420 * len(prompt.demos)


def rule_text(r):
    if r.startswith("K:"):
        return f"Claim {r[2:]}: match the rationale to its span."
    if r.startswith("X:"):
        _, st, _, lim = r.split(":")
        return f"{st} HO 04 95 back-up claims carry a US${int(lim):,} limit."
    return RULES[r]


def is_narrow(r):
    return r[:2] in ("K:", "X:")


def needs_human(r):
    """Anything that states an amount, an outcome or an edition goes to Laura."""
    return r.startswith("X:") or r.startswith("P")


def propose(parent, fails, r):
    """Stand-in reflector. A failure pattern seen across three or more states
    becomes a general rule; otherwise it writes a rule from one failure."""
    from collections import Counter
    count = Counter(f["case"] for f in fails.values())
    order = [k for k, _ in count.most_common()]
    order = order[r % len(order):] + order[:r % len(order)]
    for case in order:
        states = {f["state"] for f in fails.values() if f["case"] == case}
        general = [k for k, v in FIXES.items() if v == case and k not in parent.rules]
        if general and len(states) >= 3:
            return parent.with_rule(general[0])
        for f in fails.values():
            if f["case"] == case and narrow_rule(f["item"]) not in parent.rules:
                return parent.with_rule(narrow_rule(f["item"]))
    return None


def as_demo(it):
    return f"D:{it.case}:{it.id}"
