"""Chapter 23 stand-in: a two-hour adjuster session on the workbench, and four ways to hold it.

Everything here is a deterministic stand-in, not a real model. Shapes follow the research the
chapter cites (context rot past an onset, look-alike pull, compaction dropping process
constraints about six times as often as facts, constraint violations near 40% once the rule is
gone and 0% while it is visible); the exact values belong to this file.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

# ----------------------------------------------------------------- stand-in constants
WINDOW = 128_000          # the workbench model's context window, tokens
BUDGET = 40_000           # Lakeshore's working budget for a managed session window
TAIL = 10_000             # tokens of recent history summarisation leaves verbatim
SYS_TOKENS = 2_400        # instructions: Rulebook excerpt, Laura Jensen's two rules, tools
ONSET, ROT_SLOPE = 32_000, 0.0016   # error added per 1,000 tokens past the onset
PULL = 0.006              # error added per look-alike packet (same form family, other claim)
BASE = 0.012              # answer error with every needed fact visible, short window
P_WRONG_MISSING = 0.55    # a needed fact is not in the window: the model guesses
P_WRONG_DISTORTED = 0.85  # the summary rewrote the fact
P_VIOLATE_DROPPED = 0.40  # a constraint is gone and the action it forbade comes up
P_KEEP_FACT, P_KEEP_RULE, P_DISTORT = 0.94, 0.64, 0.03   # per fact or rule, per compaction
P_NOTE_CAPTURED = 0.80    # a free-text note's fact is captured into state when it arrives
STUB = 18                 # tokens of a cleared tool result: call, ids, "re-fetch by id"
FOLD = 70                 # tokens of a folded claim's outcome note
PRICE_IN, PRICE_OUT, CACHE_READ = 3.00, 12.00, 0.10   # US$ per million; cached share of input price

FAMILIES = ["HO-3", "HO-3", "HO-3", "HO-3", "BOP", "PAP", "HO-3", "BOAT", "HO-3", "BOP"]


@dataclass
class Item:
    idx: int
    minute: float
    role: str                 # user | tool | assistant
    claim: str
    tokens: int
    kind: str = ""            # record | packet | notes | photos | decl | ask | draft | answer | rule
    facts: tuple = ()         # fact keys this item carries (tool) or needs (assistant)
    rule: str | None = None   # constraint id stated in this user turn
    action: str | None = None # action the assistant takes in this turn, if any
    structured: bool = True   # tool output is a structured record (facts extract cleanly)


@dataclass
class Session:
    sid: str
    adjuster: str
    items: list[Item]
    rules: dict[str, tuple[str, str, str]]    # cid -> (text, action it forbids, claim or "*")
    big: set[str]                            # claims with reserve over US$25,000
    family: dict[str, str]


def _u(*key) -> float:
    h = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def build_session(seed: int, adjuster: str = "tbrennan") -> Session:
    """One afternoon: 8 to 10 claims, constraints stated as it goes, one or two returns."""
    rng = random.Random(seed)
    items: list[Item] = []
    minute = 0.0
    rules = {"K1": ("Any claim with a reserve over US$25,000: draft only and route to Margaret "
                    "Hale's queue; never mark it ready.", "mark_ready", "*")}
    n = rng.randint(8, 10)
    claims = [f"C{i + 1:02d}" for i in range(n)]
    big = {c for c in claims if rng.random() < 0.3} | {claims[0]}
    family = {c: FAMILIES[i % len(FAMILIES)] for i, c in enumerate(claims)}

    def add(role, claim, tokens, kind, facts=(), rule=None, action=None, structured=True, dt=0.3):
        nonlocal minute
        minute += dt
        items.append(Item(len(items), round(minute, 1), role, claim, tokens, kind, tuple(facts),
                          rule, action, structured))

    add("user", "*", 70, "rule", rule="K1", dt=1.0)
    seen: dict[str, list[str]] = {}
    hold_on = rng.sample(claims[:3], 1)[0]
    counsel_on = rng.sample(claims[3:6], 1)[0]
    rules["K2"] = (f"{hold_on} is on legal hold: do not quote the recorded statement in any draft.",
                   "quote_statement", hold_on)
    rules["K3"] = (f"{counsel_on}: the claimant is represented; all contact through counsel, no "
                   "direct letters.", "contact_letter", counsel_on)
    returns = sorted(rng.sample(range(5, n), 2 if n > 8 else 1))
    for i, c in enumerate(claims):
        f = seen.setdefault(c, [])
        add("user", c, rng.randint(40, 70), "ask", dt=rng.uniform(0.6, 1.6))
        rec = [f"{c}.amount", f"{c}.dol", f"{c}.edition", f"{c}.schedule"]
        add("tool", c, rng.randint(1000, 1400), "record", rec, dt=0.1)
        pk = [f"{c}.span{j}" for j in range(rng.randint(3, 5))]
        add("tool", c, rng.randint(1700, 2600), "packet", pk, dt=0.1)
        f += rec + pk
        add("assistant", c, rng.randint(500, 700), "draft", rng.sample(f, 3), action="draft")
        if c == hold_on:
            add("user", c, 60, "rule", rule="K2", dt=1.2)
        if c == counsel_on:
            add("user", c, 55, "rule", rule="K3", dt=1.2)
        for k in range(rng.randint(5, 8)):
            add("user", c, rng.randint(30, 90), "ask", dt=rng.uniform(0.5, 1.7))
            if rng.random() < 0.65:
                kind = rng.choice(["notes", "notes", "photos", "decl", "prior"])
                nf = [f"{c}.{kind}{k}"]
                add("tool", c, rng.randint(700, 3000), kind, nf, structured=kind != "notes", dt=0.1)
                f += nf
            need = rng.sample(f, min(len(f), rng.choice([1, 2, 2])))
            act = None
            if k == 2:
                act = "mark_ready"
            elif c == hold_on and k == 4:
                act = "quote_statement"
            elif c == counsel_on and k == 3:
                act = "contact_letter"
            add("assistant", c, rng.randint(180, 380), "answer", need, action=act)
        if i in returns:
            back = claims[0] if i == returns[0] else hold_on
            add("user", back, rng.randint(40, 80), "ask", dt=rng.uniform(1.0, 2.0))
            add("assistant", back, rng.randint(220, 400), "answer", rng.sample(seen[back], 2),
                action="quote_statement" if back == hold_on else "mark_ready")
    return Session(f"S{seed:03d}", adjuster, items, rules, big, family)


# ----------------------------------------------------------------- window strategies

@dataclass
class Summary:
    facts: dict[str, str] = field(default_factory=dict)   # key -> "ok" | "distorted"
    rules: set[str] = field(default_factory=set)
    n: int = 0

    def tokens(self) -> int:
        return 80 + 26 * len(self.facts) + 34 * len(self.rules) if self.n else 0


@dataclass
class Result:
    strategy: str
    sid: str
    answers: int = 0
    wrong: int = 0
    withheld: int = 0
    violations: int = 0
    lost: int = 0                 # calls refused because the window overflowed
    overflow_minute: float | None = None
    refetches: int = 0
    compactions: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    peak: int = 0
    trace: list = field(default_factory=list)   # (minute, window tokens, outcome)
    by_hour: dict = field(default_factory=lambda: {0: [0, 0], 1: [0, 0]})  # hour -> [answers, wrong]

    def cost(self) -> float:
        paid_in = (self.input_tokens - self.cached_tokens) + CACHE_READ * self.cached_tokens
        return (paid_in * PRICE_IN + self.output_tokens * PRICE_OUT) / 1e6


def _prefix(a: list, b: list) -> int:
    t = 0
    for (ka, na), (kb, nb) in zip(a, b):
        if ka != kb:
            break
        t += na
    return t


def run(sess: Session, strategy: str) -> Result:
    """Replay one session under one strategy: full | truncate | summarise | structured."""
    from contextlab.statestore import StateStore, Fact

    res = Result(strategy, sess.sid)
    hist: list[Item] = []            # what the strategy currently keeps verbatim
    summ = Summary()
    store = StateStore(sess.sid, sess.adjuster)
    cleared: set[int] = set()        # structured: tool items cleared to stubs
    active = None
    prev_render: list = []
    dead = False

    def render() -> list:
        out = [("SYS", SYS_TOKENS)]
        if strategy == "summarise" and summ.n:
            out.append((f"SUM{summ.n}", summ.tokens()))
        if strategy == "structured":
            out += [(f"PIN{len(store.pinned)}", 40 * len(store.pinned))]
            out += [(f"FOLD{c}", FOLD) for c, s in store.claims.items() if s.outcome and c != active]
            st = store.claims.get(active)
            out.append((f"STATE{active}{len(st.facts) if st else 0}", 60 + 20 * (len(st.facts) if st else 0)))
            out += [(f"I{i.idx}{'s' if i.idx in cleared else ''}", STUB if i.idx in cleared else i.tokens)
                    for i in hist if i.claim in (active, "*") and i.role != "user" or
                    (i.role == "user" and i.claim == active)]
            return out
        return out + [(f"I{i.idx}", i.tokens) for i in hist]

    def visible_fact(key: str) -> str:
        """'raw' | 'state' | 'summary-ok' | 'summary-distorted' | 'stub' | 'absent'."""
        for i in hist:
            if i.role == "tool" and key in i.facts:
                if strategy == "structured" and (i.claim != active or i.idx in cleared):
                    break
                return "raw"
        if strategy == "summarise" and key in summ.facts:
            return "summary-" + summ.facts[key]
        if strategy == "structured":
            c = key.split(".")[0]
            if c in store.claims and key in store.claims[c].facts:
                return "state"
            if any(i.role == "tool" and key in i.facts for i in hist):
                return "stub"
        return "absent"

    def rule_visible(cid: str) -> bool:
        if strategy == "structured":
            return cid in {c for c, _ in store.pinned}
        if any(i.rule == cid for i in hist):
            return True
        return strategy == "summarise" and cid in summ.rules

    def lookalikes() -> int:
        fam = sess.family.get(active)
        keep = [i for i in hist if i.kind == "packet" and i.claim != active and sess.family[i.claim] == fam]
        if strategy == "structured":
            return 0
        return len(keep)

    def compact():
        nonlocal hist
        tail, t = [], 0
        for i in reversed(hist):
            if t + i.tokens > TAIL:
                break
            tail.insert(0, i)
            t += i.tokens
        region = hist[: len(hist) - len(tail)]
        summ.n += 1
        res.compactions += 1
        res.input_tokens += sum(i.tokens for i in region) + summ.tokens() + SYS_TOKENS
        for k in list(summ.facts):
            if _u(sess.sid, "keep", summ.n, k) > P_KEEP_FACT:
                del summ.facts[k]
        for r in list(summ.rules):
            if _u(sess.sid, "keepr", summ.n, r) > P_KEEP_RULE:
                summ.rules.discard(r)
        for i in region:
            for k in i.facts if i.role == "tool" else ():
                u = _u(sess.sid, "keep", summ.n, k)
                if u < P_KEEP_FACT:
                    summ.facts[k] = "distorted" if u < P_DISTORT else "ok"
            if i.rule and _u(sess.sid, "keepr", summ.n, i.rule) < P_KEEP_RULE:
                summ.rules.add(i.rule)
        res.output_tokens += summ.tokens()
        hist = tail

    for it in sess.items:
        if it.role != "assistant":
            if strategy == "structured":
                if it.claim not in ("*", active) and it.claim:
                    if active and store.claims.get(active):
                        store.fold(active, _outcome(store.claims[active], sess))
                    active = it.claim
                    st = store.claim(active)
                    if st.outcome:
                        st.outcome = ""          # reloaded: the claim is active again
                if it.rule:
                    store.pin(it.rule, sess.rules[it.rule][0])
                if it.role == "tool":
                    for k in it.facts:
                        if it.structured or _u(sess.sid, "cap", k) < P_NOTE_CAPTURED:
                            store.record(it.claim, Fact(k, "v", f"{k}#src"))
            elif it.claim != "*":
                active = it.claim
            hist.append(it)
            continue

        # ---- an assistant turn is one model call
        if strategy == "structured":
            if it.claim != active:
                if active and store.claims.get(active):
                    store.fold(active, _outcome(store.claims[active], sess))
                active = it.claim
            raw = [i for i in hist if i.role == "tool" and i.claim == active and i.idx not in cleared]
            if sum(i.tokens for i in raw) > 8_000 or sum(t for _, t in render()) > BUDGET:
                for i in raw[:-2]:
                    cleared.add(i.idx)
            for k in it.facts:
                if visible_fact(k) == "stub":
                    res.refetches += 1
                    src = next(i for i in hist if i.role == "tool" and k in i.facts)
                    store.record(src.claim, Fact(k, "v", f"{k}#src"))
        if strategy == "truncate":
            while sum(t for _, t in render()) > BUDGET and hist:
                hist.pop(0)
        if strategy == "summarise" and sum(t for _, t in render()) > BUDGET:
            compact()

        win = render()
        tokens = sum(t for _, t in win)
        res.peak = max(res.peak, tokens)
        hour = 0 if it.minute < 60 else 1
        if dead or tokens > WINDOW:
            if not dead:
                res.overflow_minute = it.minute
            dead = True
            res.lost += 1
            res.trace.append((it.minute, tokens, "overflow"))
            continue
        res.input_tokens += tokens
        res.cached_tokens += _prefix(prev_render, win)
        res.output_tokens += it.tokens
        prev_render = win

        # ---- outcome
        u = _u(sess.sid, "ans", it.idx)
        p = BASE + ROT_SLOPE * max(0, tokens - ONSET) / 1000 + PULL * lookalikes()
        states = [visible_fact(k) for k in it.facts]
        outcome = "ok"
        if "summary-distorted" in states:
            p = max(p, P_WRONG_DISTORTED)
        if "absent" in states:
            p = max(p, P_WRONG_MISSING)
        if u < p:
            outcome = "wrong"
        elif "absent" in states or any(s.startswith("summary") for s in states):
            outcome = "withheld"      # no source to cite: the citation check holds it back
        res.answers += 1
        res.by_hour[hour][0] += 1
        if outcome == "wrong":
            res.wrong += 1
            res.by_hour[hour][1] += 1
        elif outcome == "withheld":
            res.withheld += 1
        if it.action:
            for cid, (_, act, scope) in sess.rules.items():
                if act != it.action or (scope != "*" and scope != it.claim):
                    continue
                if cid == "K1" and it.claim not in sess.big:
                    continue
                stated = any(x.rule == cid and x.idx < it.idx for x in sess.items)
                if stated and not rule_visible(cid) and _u(sess.sid, "viol", it.idx, cid) < P_VIOLATE_DROPPED:
                    res.violations += 1
                    outcome += "+violation"
        if strategy == "structured" and it.action == "draft":
            store.claim(it.claim).status = "drafted"
        if strategy == "structured" and it.action == "mark_ready":
            store.claim(it.claim).status = "routed" if it.claim in sess.big else "ready"
        res.trace.append((it.minute, tokens, outcome))
        hist.append(it)
    res.store, res.active, res.window = store, active, render()
    return res


def _outcome(c, sess) -> str:
    where = "Margaret Hale's queue" if c.claim_id in sess.big else "ready for Tom's sign-off"
    return f"{c.status}; {len(c.facts)} facts with sources; {where}"


STRATEGIES = ["full", "truncate", "summarise", "structured"]
