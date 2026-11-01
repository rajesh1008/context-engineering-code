"""contextlab.budgeter (v7.2, Chapter 32): choose what makes the window.

Scoring (relevance, authority, freshness), four packers (top-k, fill by score,
greedy by value per token, exact 0/1 knapsack), and the Budgeter: pins, bundles
for dependencies, one span per redundancy group, a relevance floor and
hierarchical (document, then span) headers.  Also a small prefix-cache cost
model (v7.3) for laying the packet out around cache boundaries.  Deterministic;
no model runs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date

FLAT_HEADER = 30      # document, edition, page and handle on every span
DOC_HEADER = 26       # document and edition, once per document (hierarchical)
HANDLE = 17           # span handle (Chapter 28)
TIER_SCORE = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.35, 5: 0.2}


@dataclass(frozen=True)
class Candidate:
    span_id: str
    doc_id: str
    tokens: int
    relevance: float          # retriever and reranker score, 0..1
    tier: int                 # authority tier, 1 (bulletin) .. 5 (claim note)
    effective: date
    kind: str = "clause"
    requires: tuple = ()      # span_ids this span cannot be read without
    group: str = ""           # redundancy group: same text, other rendering
    cited_90d: int = 0        # usage history (used only by the faulty scorer)
    section: str = ""         # sections at or under whole_section tokens pack whole


@dataclass
class Weights:
    relevance: float = 0.60
    authority: float = 0.25
    freshness: float = 0.15
    half_life_days: float = 730.0


def freshness(c: Candidate, today: date, w: Weights) -> float:
    age = max(0, (today - c.effective).days)
    return 0.5 ** (age / w.half_life_days)


def score(c: Candidate, today: date, w: Weights = Weights()) -> float:
    return (w.relevance * c.relevance + w.authority * TIER_SCORE[c.tier]
            + w.freshness * freshness(c, today, w))


def score_usage(c: Candidate, today: date, w: Weights = Weights()) -> float:
    """The faulty variant: authority learned from how often a span was cited."""
    usage = min(1.0, c.cited_90d / 40.0)
    return w.relevance * c.relevance + (w.authority + w.freshness) * usage


def cost_flat(spans) -> int:
    return sum(s.tokens + FLAT_HEADER for s in spans)


def cost_hier(spans) -> int:
    return sum(s.tokens + HANDLE for s in spans) + DOC_HEADER * len({s.doc_id for s in spans})


class Calibrator:
    """Logistic map from score to the probability a span is needed.

    Greedy and knapsack packers add values, so the values must be on a common,
    additive scale; a reranker score is not.  Fit on labelled pools."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = a, b

    def fit(self, xs, ys, steps=3_000, lr=5.0):
        import math
        a = b = 0.0
        n = len(xs)
        for _ in range(steps):
            ga = gb = 0.0
            for x, y in zip(xs, ys):
                p = 1 / (1 + math.exp(-(a * x + b)))
                ga += (p - y) * x; gb += p - y
            a -= lr * ga / n; b -= lr * gb / n
        self.a, self.b = a, b
        return self

    def __call__(self, x: float) -> float:
        import math
        return 1 / (1 + math.exp(-(self.a * x + self.b)))


# ---- the four packers ----------------------------------------------------
def top_k(cands, k=10, key=None):
    return sorted(cands, key=key, reverse=True)[:k]


def fill_by_score(cands, budget, key):
    out, used = [], 0
    for c in sorted(cands, key=key, reverse=True):
        t = c.tokens + FLAT_HEADER
        if used + t <= budget:
            out.append(c); used += t
    return out


def greedy_density(cands, budget, key, floor=0.0):
    out, used = [], 0
    for c in sorted(cands, key=lambda c: key(c) / (c.tokens + FLAT_HEADER), reverse=True):
        t = c.tokens + FLAT_HEADER
        if key(c) >= floor and used + t <= budget:
            out.append(c); used += t
    return out


def knapsack(cands, budget, key, floor=0.0, quantum=10):
    """Exact 0/1 knapsack by dynamic programming over tokens (in quanta)."""
    items = [c for c in cands if key(c) >= floor]
    cap = budget // quantum
    wts = [-(-(c.tokens + FLAT_HEADER) // quantum) for c in items]
    best = [0.0] * (cap + 1)
    keep = [[False] * (cap + 1) for _ in items]
    for i, c in enumerate(items):
        for b in range(cap, wts[i] - 1, -1):
            if best[b - wts[i]] + key(c) > best[b]:
                best[b] = best[b - wts[i]] + key(c); keep[i][b] = True
    out, b = [], cap
    for i in range(len(items) - 1, -1, -1):
        if keep[i][b]:
            out.append(items[i]); b -= wts[i]
    return out[::-1]


# ---- the Budgeter ----------------------------------------------------------
@dataclass
class Budgeter:
    budget: int = 1_000           # target for everything optional
    hard_cap: int = 2_400         # ceiling for pins and document slots
    floor: float = 0.60
    redundancy: float = 1.0          # share of value lost by a second copy
    pin_kinds: tuple = ("bulletin",)
    slot_tier: int = 2
    whole_section: int = 340
    today: date = date(2026, 11, 1)
    weights: Weights = field(default_factory=Weights)
    scorer: object = None

    def _v(self, c):
        return (self.scorer or score)(c, self.today, self.weights)

    def bundles(self, cands):
        """Each span with the spans it requires: packed together or not at all."""
        by_id = {c.span_id: c for c in cands}
        out = []
        for c in cands:
            need, todo = {c.span_id: c}, list(c.requires)
            while todo:
                r = by_id.get(todo.pop())
                if r and r.span_id not in need:
                    need[r.span_id] = r; todo.extend(r.requires)
            out.append((c, list(need.values())))
        return out

    def pack(self, cands, pins=()):
        chosen, groups, trace = {}, set(), []

        def add(spans, why):
            for s in spans:
                if s.span_id not in chosen:
                    chosen[s.span_id] = s
                    if s.group:
                        groups.add(s.group)
                    trace.append((s.span_id, why))

        pinned = [c for c in cands if c.kind in self.pin_kinds or c.span_id in pins]
        bundles = dict((c.span_id, b) for c, b in self.bundles(cands))
        for c in pinned:
            add(bundles[c.span_id], "pinned")
        # hierarchical stage: one slot per contract document (tier 1-2), best span first
        docs, support = {}, {r for c in cands for r in c.requires}
        for c in cands:   # a span that others require rides with them; it is never a slot head
            if c.span_id not in support and c.tier <= self.slot_tier and self._v(c) >= self.floor:
                if c.doc_id not in docs or self._v(c) > self._v(docs[c.doc_id]):
                    docs[c.doc_id] = c
        for c in sorted(docs.values(), key=self._v, reverse=True):
            if c.group and c.group in groups: continue
            unit = list(bundles[c.span_id])
            if c.section:
                sec = [s for s in cands if s.doc_id == c.doc_id and s.section == c.section]
                if sum(s.tokens for s in sec) <= self.whole_section:
                    unit += sec
            new = list({s.span_id: s for s in unit if s.span_id not in chosen}.values())
            if new and cost_hier(list(chosen.values()) + new) <= self.hard_cap:
                add(new, "document slot")
            elif new:   # fail closed: the trace says what did not fit, and the
                trace.append((c.span_id, "over cap"))   # completeness check withholds
        while True:
            best, best_d = None, 0.0
            for c, need in bundles.items():
                head = need[0]
                if head.span_id in chosen or self._v(head) < self.floor:
                    continue
                new = [s for s in need if s.span_id not in chosen]
                v = sum(self._v(s) * (1 - self.redundancy if s.group in groups and s.group else 1)
                        for s in new)
                t = cost_hier(list(chosen.values()) + new) - cost_hier(list(chosen.values()))
                if cost_hier(list(chosen.values()) + new) > self.budget or t <= 0:
                    continue
                d = v / t
                if d > best_d:
                    best, best_d = new, d
            if best is None:
                break
            add(best, "value per token")
        return list(chosen.values()), trace


# ---- per-step budgets (v7.3) -------------------------------------------------
@dataclass
class StepBudget:
    role: str
    evidence: str = "index"  # "full", "index" or "none"; a new step fails small
    tools: bool = True
    thinking: int = 0        # reasoning tokens allowed for this step


# ---- prefix-cache cost model (v7.3) ------------------------------------------
@dataclass
class CachePrices:
    per_million: float = 3.00   # US$ per million input tokens (stand-in)
    write: float = 1.25
    read: float = 0.10
    ttl_s: int = 300
    min_tokens: int = 1_024


class PrefixCache:
    """Caches prompt prefixes at breakpoints; a hit needs an identical prefix."""

    def __init__(self, prices: CachePrices = CachePrices()):
        self.p, self.store = prices, {}
        self.tokens = {"read": 0, "write": 0, "plain": 0}

    def call(self, t: float, blocks, breakpoints):
        """blocks: list of (key, tokens); breakpoints: indices after which to cache."""
        keys, cum = [], []
        run = 0
        for i, (k, n) in enumerate(blocks):
            keys.append(hash(tuple(b[0] for b in blocks[: i + 1])))
            run += n; cum.append(run)
        hit = 0
        for i in range(len(blocks) - 1, -1, -1):
            k = keys[i]
            if k in self.store and t - self.store[k] <= self.p.ttl_s and cum[i] >= self.p.min_tokens:
                hit = cum[i]; self.store[k] = t; break
        last_bp = max([b for b in breakpoints if cum[b] >= self.p.min_tokens], default=None)
        write = 0
        if last_bp is not None and cum[last_bp] > hit:
            write = cum[last_bp] - hit
            for b in breakpoints:
                if cum[b] >= self.p.min_tokens:
                    self.store[keys[b]] = t
        plain = cum[-1] - hit - write
        self.tokens["read"] += hit; self.tokens["write"] += write; self.tokens["plain"] += plain
        return hit, write, plain

    def dollars(self):
        p, tk = self.p, self.tokens
        return (tk["read"] * p.read + tk["write"] * p.write + tk["plain"]) * p.per_million / 1e6

    def hit_rate(self):
        tot = sum(self.tokens.values())
        return self.tokens["read"] / tot if tot else 0.0
