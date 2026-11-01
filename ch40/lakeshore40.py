"""Chapter 40 stand-in world: Lakeshore's context layer as components with fault
switches, a claims mix to run them on, and the checks of every layer.

Deterministic. No language model runs. Layer-one checks exercise real (tiny) component
code. Layers two to five are simulated: each fault has a stated effect on the claims it
touches, and drafting and judging draw fixed pseudo-random numbers keyed by claim and
run, so a rerun with a new seed behaves like a rerun against a live model. Every
probability, price and duration below is a stand-in parameter.
"""
import hashlib
import math
import random
from dataclasses import dataclass, field

from contextlab.pyramid import Suite

# ---------------- prices and durations (stand-ins, US$) ----------------
DET_USD = 0.0348          # cost per determination after Chapter 35
JUDGE_USD = 0.0015        # one judged sentence
EXTRACT_USD = 0.0040      # one extraction call scored on a gold item

# ---------------- the components (real code, fault switches) ----------------
EDITIONS = {   # form -> (edition, effective from, uploaded to the document store)
    "HO-3": [("2019-03", "2019-03-01", "2026-08-14"),     # old edition re-scanned in 2026
             ("2024-06", "2024-06-01", "2024-05-20")],
    "HO 04 95": [("2019-03", "2019-03-01", "2019-02-11"),
                 ("2024-06", "2024-06-01", "2024-05-20")],
}


@dataclass(frozen=True)
class System:
    faults: frozenset = frozenset()
    judge_prompt: str = "judge-v3"
    checker_threshold: float = 0.80

    def has(self, f):
        return f in self.faults


def resolve_edition(sys_, form, loss_date):
    eds = EDITIONS[form]
    if sys_.has("F1"):                                   # newest upload wins
        return max(eds, key=lambda e: e[2])[0]
    live = [e for e in eds if e[1] <= loss_date]
    return max(live, key=lambda e: e[1])[0] if live else None


def join_pages(sys_, pages):
    out = [pages[0]]
    for p in pages[1:]:
        if not sys_.has("F2") and not out[-1].rstrip().endswith((".", ":")) and p[:1].islower():
            out[-1] = out[-1].rstrip() + " " + p
        else:
            out.append(p)
    return out


def chunk_table(sys_, header, rows, per=2):
    chunks = [rows[i:i + per] for i in range(0, len(rows), per)]
    return [([header] if (i == 0 or not sys_.has("F3")) else []) + c for i, c in enumerate(chunks)]


def budget(sys_, spans, limit):
    """spans: (id, kind, tokens, score). Governing and schedule spans go first."""
    must = () if sys_.has("F4") else ("governing", "schedule")
    order = sorted(spans, key=lambda s: (s[1] not in must, -s[3]))
    kept, used = [], 0
    for s in order:
        if used + s[2] <= limit:
            kept.append(s); used += s[2]
    return kept


def dedupe(sys_, spans):
    """spans: (id, form family, is_endorsement). One span per family, endorsements kept."""
    seen, out = set(), []
    for sid, fam, endo in spans:
        key = fam if (sys_.has("F6") or not endo) else sid
        if key not in seen:
            seen.add(key); out.append(sid)
    return out


def cache_key(sys_, claim_id, form_edition, bulletin_version, watermark):
    parts = [claim_id, form_edition, watermark] + ([] if sys_.has("F11") else [bulletin_version])
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def entitled(sys_, span_tenant, caller_tenant):
    return sys_.has("F10") or span_tenant in (None, caller_tenant)


def draft_prompt(sys_):
    rules = "Cite the governing span by document, edition and page."
    if sys_.has("F9"):
        rules += " Be thorough. Mention every condition that might apply."
    return rules


JUDGE_PROMPT_SHA = hashlib.sha256(b"judge-v3").hexdigest()[:12]


# ---------------- the claims mix ----------------
CASES = [("plain", .40), ("limit", .14), ("conditions", .14), ("deductible", .10),
         ("withhold", .08), ("edition", .07), ("conflict", .07)]
UNSUP = {"plain": .010, "limit": .035, "conditions": .030, "deductible": .030,
         "withhold": 0.0, "edition": .020, "conflict": .025}      # per sentence


@dataclass(frozen=True)
class Claim:
    id: str
    case: str
    n_sent: int
    partner: bool
    bulletin: bool
    stale: bool           # a bulletin for this claim is republished during the run
    late: bool            # loss after a bulletin published in production this week


def pick(rng, table):
    x, acc = rng.random(), 0.0
    for name, p in table:
        acc += p
        if x < acc:
            return name
    return table[-1][0]


def claims(prefix, n, seed, partner=.08, bulletin=.12, stale=0.0, late=0.0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        case, n_sent = pick(rng, CASES), rng.randint(3, 6)
        p, b, st, lt = (rng.random() for _ in range(4))
        out.append(Claim(f"{prefix}{i:04d}", case, n_sent, p < partner, b < bulletin,
                         b < bulletin and st < stale, lt < late))
    return out


GOLD = claims("G", 400, 4001, partner=.02)               # frozen September 2026
L3SET = claims("J", 400, 4003)
REPLAY = claims("R", 1000, 4004, stale=.5)                # Ch39 full mix; bulletin change mid-run
SHADOW = claims("S", 2000, 4005, stale=.5, late=.10)      # one week of live claims, mirrored
SAMPLE = SHADOW[:50]                                      # adjuster-reviewed each week


def u(*key):
    """A fixed uniform draw for a key: same key, same number, on every run."""
    h = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


# ---------------- what each fault does to the claims it touches ----------------
# (kind, {case or attribute: probability per claim}); "wrong" cites the wrong clause,
# "insuff" drops a span the determination needs, "unsup" raises unsupported sentences.
EFFECTS = {
    "F1": ("wrong", {"edition": .90}),
    "F2": ("insuff", {"conditions": .25}),
    "F3": ("insuff", {"limit": .30, "deductible": .30}),
    "F4": ("insuff", {"limit": .35, "deductible": .35}),
    "F5": ("wrong", {"edition": .25, "conflict": .25}),
    "F6": ("insuff", {"limit": .30, "conditions": .20}),
    "F7": ("unsup", {"*": 1.8}),
    "F9": ("unsup", {"limit": 3.0, "conditions": 3.0, "deductible": 3.0}),
    "F10": ("wrong", {"partner": .50}),
    "F11": ("wrong", {"stale": .90}),
    "F12": ("wrong", {"late": .90}),
}


def hit(sys_, c, kind, only=None):
    """Does a fault of this kind change claim c's outcome under sys_?"""
    for f in sys_.faults:
        k, eff = EFFECTS.get(f, (None, {}))
        if k != kind or (only and f not in only):
            continue
        p = eff.get(c.case, 0)
        for attr in ("partner", "stale", "late"):
            if getattr(c, attr):
                p = max(p, eff.get(attr, 0))
        if p and u(f, c.id) < p:
            return True
    return False


def unsup_rate(sys_, c):
    p = UNSUP[c.case]
    for f in sys_.faults:
        k, eff = EFFECTS.get(f, (None, {}))
        if k == "unsup":
            p *= eff.get(c.case, eff.get("*", 1.0))
    if hit(sys_, c, "insuff"):
        p = max(p, .20)
    return min(p, .9)


def draft(sys_, c, run):
    """Per-sentence truth for one drafted determination (redrawn on every run)."""
    p = unsup_rate(sys_, c)
    return [u("d", c.id, j, run) >= p for j in range(c.n_sent)]


JUDGES = {"judge-v3": (.08, .015), "judge-v4": (.55, .005)}      # (fp, fn) per sentence


def judge(sys_, c, sents, run):
    fp, fn = JUDGES[sys_.judge_prompt]
    out = []
    for j, t in enumerate(sents):
        e = fn if t else fp
        out.append(t if u("j", c.id, j, run, sys_.judge_prompt) >= e else not t)
    return out


def faith(sys_, items, run):
    return [all(judge(sys_, c, draft(sys_, c, run), run)) for c in items]


def wrong_count(sys_, items, base=3):
    """Wrong-clause count on the full mix: the three Chapter 39 left, plus faults."""
    return base + sum(hit(sys_, c, "wrong") for c in items)


# ---------------- the suite: every check, tagged ----------------
suite = Suite()
A = suite.add


# Layer 1: unit tests on real component code (seconds are declared per test)
@A("Chapter 2 smoke tests (ten, run as a block)", 1, "evidence", "platform", seconds=40)
def t_smoke2(s, seed):
    return True          # well-formedness: passes on a pipeline that is wrong in a tidy way


@A("edition in force on date of loss", 1, "evidence", "platform", components=("resolver",))
def t_edition(s, seed):
    return resolve_edition(s, "HO 04 95", "2025-05-02") == "2024-06"


@A("re-uploaded old scan does not win", 1, "evidence", "platform", components=("resolver",))
def t_rescan(s, seed):
    return resolve_edition(s, "HO-3", "2025-05-02") == "2024-06"


@A("no edition before the first fails closed", 1, "evidence", "platform", components=("resolver",))
def t_before(s, seed):
    return resolve_edition(s, "HO-3", "2018-01-01") is None


@A("clause split across a page break is joined", 1, "evidence", "documents", components=("parser",))
def t_pagebreak(s, seed):
    return join_pages(s, ["12. Frozen pipes, if you have", "maintained heat."]) == \
        ["12. Frozen pipes, if you have maintained heat."]


@A("new clause after a page break stays separate", 1, "evidence", "documents", components=("parser",))
def t_pagebreak2(s, seed):
    return len(join_pages(s, ["12. Frozen pipes.", "13. Ice damage."])) == 2


@A("every schedule chunk carries its header", 1, "evidence", "documents", components=("chunker",))
def t_header(s, seed):
    ch = chunk_table(s, "Coverage | Limit", ["A | 300,000", "B | 30,000", "E | 10,000"])
    return all(c[0] == "Coverage | Limit" for c in ch)


@A("budget never exceeded", 1, "cost", "platform", components=("budgeter",))
def t_budget(s, seed):
    sp = [("g", "governing", 900, .9), ("sch", "schedule", 400, .2), ("r1", "related", 700, .8)]
    return sum(x[2] for x in budget(s, sp, 1500)) <= 1500


@A("endorsement on the declarations survives dedupe", 1, "evidence", "platform",
   components=("assembler",))
def t_dedupe(s, seed):
    return "HO 04 95" in dedupe(s, [("HO-3", "HO-3", False), ("HO 04 95", "HO-3", True)])


@A("cache key names the bulletin version", 1, "freshness", "platform", components=("cache",))
def t_cache(s, seed):
    return cache_key(s, "C1", "2024-06", "WI-b7", "w41") != cache_key(s, "C1", "2024-06", "WI-b8", "w41")


@A("cache key names the watermark", 1, "freshness", "platform", components=("cache",))
def t_cache2(s, seed):
    return cache_key(s, "C1", "2024-06", "WI-b7", "w41") != cache_key(s, "C1", "2024-06", "WI-b7", "w42")


@A("partner endorsement hidden from other tenants", 1, "entitlement", "platform",
   components=("retrieval",))
def t_tenant(s, seed):
    return not entitled(s, "partner-a", "partner-b")


@A("direct-book span visible to every tenant", 1, "entitlement", "platform", components=("retrieval",))
def t_tenant2(s, seed):
    return entitled(s, None, "partner-b")


@A("citation checker threshold pinned", 1, "grounding", "platform", components=("checker",))
def t_threshold(s, seed):
    return s.checker_threshold == 0.80


@A("judge prompt hash pinned", 1, "grounding", "evaluation", components=("judge",))
def t_judgehash(s, seed):
    return hashlib.sha256(s.judge_prompt.encode()).hexdigest()[:12] == JUDGE_PROMPT_SHA


@A("draft prompt asks for a citation", 1, "grounding", "platform", components=("prompt",))
def t_prompt(s, seed):
    return "Cite the governing span" in draft_prompt(s)


# Layer 2: gold-set evals, deterministic retrieval and assembly on 400 frozen claims
GOLD_EDITION = [c for c in GOLD if c.case in ("edition", "conflict")]
GOLD_SCHED = [c for c in GOLD if c.case in ("limit", "deductible", "conditions")]


@A("gold: governing hit, all 400", 2, "evidence", "evaluation", seconds=90, sample=400,
   components=("resolver", "parser", "retrieval", "chunker"))
def g_hit(s, seed):
    return not any(hit(s, c, "wrong") for c in GOLD)


@A("gold: governing hit, edition and conflict slice", 2, "evidence", "evaluation", seconds=20,
   sample=len(GOLD_EDITION), components=("resolver", "retrieval"))
def g_edition(s, seed):
    return not any(hit(s, c, "wrong") for c in GOLD_EDITION)


@A("gold: packet sufficiency, schedule slice", 2, "evidence", "evaluation", seconds=60,
   sample=len(GOLD_SCHED), components=("budgeter", "assembler", "chunker", "parser"))
def g_suff(s, seed):
    return not any(hit(s, c, "insuff") for c in GOLD_SCHED)


CHECKER_SET = [(u("lab", i) < .85, u("score", i)) for i in range(300)]   # (supported, draw)


@A("gold: citation checker on 300 labelled sentences", 2, "grounding", "evaluation", seconds=30,
   sample=300, components=("checker",))
def g_checker(s, seed):
    scores = [(t, .6 + .4 * x if t else .75 * x) for t, x in CHECKER_SET]
    bad = [sc for t, sc in scores if not t]
    return sum(sc >= s.checker_threshold for sc in bad) / len(bad) <= .05


@A("gold: extraction fields, 300 notes", 2, "evidence", "claims-data", seconds=240,
   dollars=300 * EXTRACT_USD, sample=300, components=("extraction",))
def g_extract(s, seed):
    return True


# Layer 3: judge-scored evals on 400 determinations, candidate against baseline
def paired_drop(base, cand, z=1.96):
    d = [int(b) - int(c) for b, c in zip(base, cand)]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) or 1e-9
    return m - z * sd / math.sqrt(len(d)) > 0          # candidate significantly worse


BASE = System()


@A("judge: faithfulness, 400 determinations", 3, "grounding", "evaluation", seconds=900,
   dollars=400 * DET_USD + 1800 * JUDGE_USD, deterministic=False, sample=400,
   components=("prompt", "budgeter", "assembler", "checker", "model"))
def j_faith(s, seed):
    return not paired_drop(faith(BASE, L3SET, ("b", seed)), faith(s, L3SET, ("c", seed)))


@A("judge: agreement with 120 adjuster labels", 3, "human", "evaluation", seconds=120,
   dollars=120 * JUDGE_USD, deterministic=False, sample=120, components=("judge",))
def j_valid(s, seed):
    fp, fn = JUDGES[s.judge_prompt]
    return fp <= 0.20                                   # Chapter 39's TNR >= 0.8 rule


# Layer 4: end-to-end on the reviewed full mix, with a bulletin change mid-replay
@A("end-to-end: wrong-clause on 1,000", 4, "evidence", "evaluation", seconds=3600,
   dollars=1000 * DET_USD, sample=1000)
def e_wrong(s, seed):
    return wrong_count(s, REPLAY) <= 3


@A("end-to-end: cost per determination", 4, "cost", "platform", seconds=30, sample=1000)
def e_cost(s, seed):
    return True


@A("end-to-end: tenancy on partner claims", 4, "entitlement", "platform", seconds=30, sample=1000)
def e_tenant(s, seed):
    # cross-tenant citations: a span from another partner's book in the packet
    return not any(hit(s, c, "wrong", only=("F10",)) for c in REPLAY if c.partner)


# Layer 5: shadow and production sampling, per week
@A("shadow: agreement by case type, 2,000 mirrored", 5, "evidence", "claims-ops",
   seconds=7 * 86400, dollars=2000 * DET_USD, deterministic=False, sample=2000)
def p_shadow(s, seed):
    return sum(hit(s, c, "wrong") for c in SHADOW) < 3


@A("production: 50 determinations reviewed by adjusters", 5, "human", "claims-ops",
   seconds=7 * 86400, deterministic=False, sample=50)
def p_sample(s, seed):
    return not any(hit(s, c, "wrong") for c in SAMPLE)


@A("production: bulletin feed watermark lag", 5, "freshness", "platform", seconds=3600, sample=1)
def p_fresh(s, seed):
    return not s.has("F12")


HUMAN_MINUTES = {"production: 50 determinations reviewed by adjusters": 50 * 6,
                 "judge: agreement with 120 adjuster labels": 0}

FAULTS = {
    "F1": "edition chosen by upload date (the 2019 scan wins)",
    "F2": "page-break join removed from the parser",
    "F3": "schedule header only on the first chunk",
    "F4": "budgeter ranks by score alone",
    "F5": "retriever weights a lookalike edition up",
    "F6": "dedupe folds endorsements into the base form",
    "F7": "citation checker threshold lowered to 0.65",
    "F8": "judge prompt edited (judge-v4)",
    "F9": "generic rule appended to the draft prompt",
    "F10": "tenant filter skipped on partner spans",
    "F11": "cache key without the bulletin version",
    "F12": "bulletin feed stalled in production",
}


def system_with(f):
    if f == "F7":
        return System(frozenset({f}), checker_threshold=0.65)
    if f == "F8":
        return System(frozenset({f}), judge_prompt="judge-v4")
    return System(frozenset({f}))
