"""contextlab.caching (v7.5, Chapter 34): cache-aware packet layout.

A deterministic stand-in for a provider prompt cache. It is not any vendor's
implementation: it models the behaviour the vendor guides document (exact
prefix match, a minimum cacheable length, a time to live refreshed on use,
machine-local caches with overflow routing) so that layouts can be compared.
Successor to the simulated cache of Chapter 5.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field

VOLATILITY = {"static": 0, "daily": 1, "route": 2, "claim": 3, "call": 4}


@dataclass(frozen=True)
class Segment:
    name: str
    text: str
    tokens: int
    volatility: str = "static"
    breakpoint: bool = False

    @property
    def digest(self):
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


def cache_layout(segments, route_order=None):
    """Stable sort by volatility; route segments in the given name order."""
    rank = {n: i for i, n in enumerate(route_order or [])}
    return sorted(segments, key=lambda s: (VOLATILITY[s.volatility],
                                           rank.get(s.name.split(":")[0], 0)))


def canonical_json(obj):
    """Deterministic serialisation: sorted keys, fixed separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass
class Prices:
    """Stand-in tariff in US$ per million tokens (not any vendor's list)."""
    input: float = 3.00
    output: float = 15.00
    read: float = 0.10      # multiplier on input for cached reads
    write: float = 1.00     # multiplier for writes (1.0 = implicit caching)

    def cost(self, uncached, cached, written, output):
        tok = uncached + self.read * cached + self.write * written
        return (tok * self.input + output * self.output) / 1e6


@dataclass
class CallResult:
    input_tokens: int
    cached: int
    written: int
    output: int
    cost: float
    shard: int
    hit_boundary: str = ""


class PrefixCache:
    """Exact-prefix cache with entries at segment boundaries.

    explicit=False: implicit caching, an entry at every segment end.
    explicit=True: entries only at segments marked breakpoint=True, and the
    lookup walks back over at most `lookback` boundaries.
    """

    def __init__(self, prices=None, min_tokens=1024, ttl_s=300, shards=1,
                 overflow_rpm=15, explicit=False, lookback=20):
        self.prices = prices or Prices()
        self.min_tokens, self.ttl_s = min_tokens, ttl_s
        self.shards, self.overflow_rpm = shards, overflow_rpm
        self.explicit, self.lookback = explicit, lookback
        self.store = [dict() for _ in range(shards)]
        self.rate = {}

    def _keys(self, model, segments):
        h, cum, out = hashlib.sha256(model.encode()), 0, []
        for s in segments:
            h.update(s.digest.encode())
            cum += s.tokens
            out.append((h.hexdigest()[:20], cum, s))
        return out

    def _shard(self, keys, t, routing_key):
        if self.shards == 1:
            return 0
        head = routing_key or next((k for k, c, _ in keys
                                    if c >= self.min_tokens), keys[-1][0])
        base = int(hashlib.sha256(head.encode()).hexdigest(), 16) % self.shards
        slot = (head, int(t // 60))
        n = self.rate.get(slot, 0)
        self.rate[slot] = n + 1
        spill = n // self.overflow_rpm          # overflow above the rate
        return (base + spill) % self.shards

    def call(self, t, segments, output_tokens, model="sim-large",
             routing_key=None):
        keys = self._keys(model, segments)
        shard = self._shard(keys, t, routing_key)
        store = self.store[shard]
        marks = [i for i, (_, _, s) in enumerate(keys)
                 if (s.breakpoint or not self.explicit)]
        cached, hit_name = 0, ""
        for i in reversed(marks[-self.lookback:] if self.explicit else marks):
            k, cum, s = keys[i]
            if store.get(k, -1) >= t:
                cached, hit_name = cum, s.name
                store[k] = t + self.ttl_s        # refreshed on use
                break
        top = 0
        for i in marks:
            k, cum, _ = keys[i]
            if cum >= self.min_tokens:
                store[k] = max(store.get(k, -1), t + self.ttl_s)
                top = max(top, cum)
        written = max(0, top - cached) if self.prices.write != 1.0 else 0
        total = keys[-1][1]
        uncached = total - cached - written
        cost = self.prices.cost(uncached, cached, written, output_tokens)
        return CallResult(total, cached, written, output_tokens, cost,
                          shard, hit_name)


def hit_rate(results):
    """Token-weighted: cached input tokens over all input tokens."""
    inp = sum(r.input_tokens for r in results)
    return sum(r.cached for r in results) / inp if inp else 0.0


TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|\b\d{2}:\d{2}:\d{2}\b")
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-")


def lint_packet(segments, tool_orders_seen=1, memo_keys=None, min_tokens=1024):
    """CacheLint-style checks on one rendered packet (rule ids are ours)."""
    findings, seen_vol, cum = [], 0, 0
    first_break = None
    for i, s in enumerate(segments):
        cum += s.tokens
        v = VOLATILITY[s.volatility]
        if v < seen_vol:
            findings.append(("C1-volatile-before-stable", s.name))
        seen_vol = max(seen_vol, v)
        if s.volatility in ("static", "daily", "route"):
            if TIMESTAMP.search(s.text) or UUID.search(s.text):
                findings.append(("C2-volatile-value-in-prefix", s.name))
        if s.breakpoint and first_break is None:
            first_break = cum
    if tool_orders_seen > 1:
        findings.append(("C3-tool-order-varies", str(tool_orders_seen)))
    if first_break is not None and first_break < min_tokens:
        findings.append(("C4-breakpoint-below-minimum", str(first_break)))
    for name, key in (memo_keys or {}).items():
        if "watermark" not in key:
            findings.append(("C5-memo-without-watermark", name))
    return findings


@dataclass
class ResultCache:
    """Exact-key cache of finished determinations (Chapter 34)."""
    store: dict = field(default_factory=dict)

    @staticmethod
    def key(claim_id, question, policy_snapshot, watermark, fingerprint, model):
        q = " ".join(question.lower().split())
        raw = canonical_json([claim_id, q, policy_snapshot, watermark,
                              fingerprint, model])
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def get(self, k):
        return self.store.get(k)

    def put(self, k, value):
        self.store[k] = value
