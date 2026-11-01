"""contextlab v7.6: costs, routing, cascades and the frontier (Chapter 35).

Every model call is recorded as a `Call` with its stage, tier and four token
counts (uncached input, cached input, output, thinking). Prices are per
million tokens and illustrative. Thinking tokens are billed as output and are
never cached. `Ledger` answers "where do the tokens go" and "where do the
dollars go", which are different questions.

`Router` scores difficulty from features the system can see before it drafts,
and maps the score to a tier and a reasoning budget. `cascade` runs the
cheapest acceptable draft first and stops at the first draft the check
passes. `CacheLayer` is a keyed cache whose key must name every version the
cached value depends on. `pareto` returns the frontier of (cost, quality)
points: no other point is both cheaper and at least as good.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tier:
    name: str
    price_in: float        # US$ per million uncached input tokens
    price_out: float       # US$ per million output (and thinking) tokens
    cache_read: float = 0.10   # cached input billed at this share of price_in


TIERS = {
    "large": Tier("large", 3.00, 12.00),
    "mid": Tier("mid", 0.80, 3.20),
    "small": Tier("small", 0.20, 0.80),
}
BUDGETS = {"none": 0, "low": 800, "medium": 4000, "high": 9000}   # thinking caps


@dataclass(frozen=True)
class Call:
    stage: str
    tier: str
    uncached: int
    cached: int = 0
    output: int = 0
    thinking: int = 0

    @property
    def tokens(self) -> int:
        return self.uncached + self.cached + self.output + self.thinking

    def dollars(self) -> dict:
        t = TIERS[self.tier]
        return {"uncached": self.uncached * t.price_in / 1e6,
                "cached": self.cached * t.price_in * t.cache_read / 1e6,
                "output": self.output * t.price_out / 1e6,
                "thinking": self.thinking * t.price_out / 1e6}

    def cost(self) -> float:
        return sum(self.dollars().values())


@dataclass
class Ledger:
    calls: list = field(default_factory=list)
    n: int = 0                       # determinations recorded

    def add(self, calls) -> None:
        self.calls.extend(calls)
        self.n += 1

    def by(self, key: str, what: str = "cost") -> dict:
        out = defaultdict(float)
        for c in self.calls:
            k = getattr(c, key)
            out[k] += c.cost() if what == "cost" else c.tokens
        return {k: v / max(self.n, 1) for k, v in out.items()}

    def by_component(self) -> dict:
        out = defaultdict(float)
        for c in self.calls:
            for k, v in c.dollars().items():
                out[k] += v
        return {k: v / max(self.n, 1) for k, v in out.items()}

    def per_determination(self) -> float:
        return sum(c.cost() for c in self.calls) / max(self.n, 1)

    def tokens_per_determination(self) -> float:
        return sum(c.tokens for c in self.calls) / max(self.n, 1)


@dataclass(frozen=True)
class Route:
    tier: str
    budget: str
    exit_check: bool       # may the draft exit on a passed check?


class Router:
    """Difficulty score from visible features -> (tier, budget)."""

    def __init__(self, weights: dict, cuts=(0.30, 0.62), bias: float = 0.0,
                 routes=None, strict: bool = True):
        self.weights, self.cuts, self.bias, self.strict = weights, cuts, bias, strict
        self.routes = routes or (Route("small", "low", True),
                                 Route("large", "low", False),
                                 Route("large", "high", False))

    def score(self, feats: dict) -> float:
        missing = [k for k in self.weights if k not in feats]
        if missing and self.strict:
            raise KeyError(f"router feature(s) missing: {missing}")
        s = self.bias + sum(w * float(feats.get(k, 0)) for k, w in self.weights.items())
        return min(max(s, 0.0), 1.0)

    def band(self, feats: dict) -> int:
        s = self.score(feats)
        return 0 if s < self.cuts[0] else (1 if s < self.cuts[1] else 2)

    def route(self, feats: dict) -> Route:
        return self.routes[self.band(feats)]


def cascade(steps, check):
    """Run (name, draft_fn) steps in order; stop at the first draft that passes.

    Returns (draft, name, attempts). The last step's draft is returned even if
    it fails, so the caller can flag it for a human.
    """
    attempts = []
    for name, draft_fn in steps:
        d = draft_fn()
        attempts.append(name)
        if check(d):
            return d, name, attempts
    return d, name, attempts


class CacheLayer:
    """A keyed cache. The key function must include every version the value
    depends on; `stats` reports hits, misses and hit rate."""

    def __init__(self, name: str, key_fn):
        self.name, self.key_fn = name, key_fn
        self.store, self.hits, self.misses = {}, 0, 0

    def get_or(self, item, compute):
        k = self.key_fn(item)
        if k in self.store:
            self.hits += 1
            return self.store[k], True
        self.misses += 1
        v = compute()
        self.store[k] = v
        return v, False

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"layer": self.name, "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / n if n else 0.0}


def pareto(points):
    """points: iterable of (name, cost, quality). Lower cost, higher quality
    is better. Returns the non-dominated points sorted by cost."""
    pts = sorted(points, key=lambda p: (p[1], -p[2]))
    front, best = [], float("-inf")
    for p in pts:
        if p[2] > best:
            front.append(p)
            best = p[2]
    return front
