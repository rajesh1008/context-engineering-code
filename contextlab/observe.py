"""contextlab.observe: recording, redacting and reconstructing the packet (Chapter 42).

contextlab v9.2. One immutable answer event per determination: what was asked,
which versions produced the answer, every span offered to the packet with its
provenance and the budget decision taken on it, the cache state, the tool reads
with their read times, the hash of the rendered packet and the output. Span text
is not copied into the event. It lives once in a content-addressed store, keyed
by the hash of the text, under a retention tier. Events are hash-chained so a
missing or edited event is detectable. `reconstruct` rehydrates the packet from
the store, re-renders it and checks the hash against the event.

Stdlib only. Chapter 15's audit record, Chapter 26's tool-call ids, Chapter 31's
envelope hash and Chapter 38's gateway request id all become fields here.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict

VERSION = "9.2"
TIERS = ("evidence", "personal", "operational")


def h(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


@dataclass
class Offered:
    """One span offered to the packet, and what the budgeter did with it."""
    span_id: str
    content: str            # content hash in the store
    tier: str               # evidence | personal
    source: str             # retrieval route or tool name
    rank: int
    score: float
    tokens: int
    decision: str           # admitted | dropped
    reason: str = ""        # why dropped (budget, gate, duplicate)
    role: str = "context"


class Store:
    """Content-addressed blobs with a tier; the same text is stored once."""

    def __init__(self):
        self.blobs: dict[str, tuple[str, str]] = {}
        self.writes = 0

    def put(self, text: str, tier: str) -> str:
        if tier not in TIERS:
            raise ValueError(tier)
        key = h(text)
        self.writes += 1
        self.blobs.setdefault(key, (tier, text))
        return key

    def get(self, key: str):
        item = self.blobs.get(key)
        return None if item is None else item[1]

    def bytes(self) -> int:
        return sum(len(t.encode()) + 24 for _, t in self.blobs.values())


@dataclass
class Event:
    """The immutable answer event for one determination."""
    det_id: str
    at: str
    request: dict
    versions: dict          # prompt, schema, policy, corpus, model, gateway request id
    offered: list           # list[Offered]
    tools: list             # [{call_id, tool, read_at, content}]
    cache: dict             # prefix hash, cached and uncached tokens
    packet_hash: str
    output: dict            # status, clause, citations, checker
    prev: str = ""
    hash: str = ""

    def body(self) -> dict:
        d = asdict(self)
        d.pop("hash")
        return d

    def seal(self, prev: str) -> "Event":
        self.prev = prev
        self.hash = h(json.dumps(self.body(), sort_keys=True), 24)
        return self


class Ledger:
    """Append-only, hash-chained events."""

    def __init__(self):
        self.events: list[Event] = []
        self.index: dict[str, int] = {}

    def append(self, ev: Event) -> Event:
        prev = self.events[-1].hash if self.events else "genesis"
        self.index[ev.det_id] = len(self.events)
        self.events.append(ev.seal(prev))
        return ev

    def get(self, det_id: str) -> Event:
        return self.events[self.index[det_id]]

    def verify(self) -> int | None:
        """Index of the first event whose hash or link does not hold, or None."""
        prev = "genesis"
        for i, ev in enumerate(self.events):
            body = json.dumps(ev.body(), sort_keys=True)
            if ev.prev != prev or h(body, 24) != ev.hash:
                return i
            prev = ev.hash
        return None

    def bytes(self) -> int:
        return sum(len(json.dumps(asdict(e))) for e in self.events)


@dataclass
class Rule:
    """A redaction rule and the tiers it may touch."""
    name: str
    pattern: str
    tiers: tuple = ("personal", "operational")
    token: str = "[REDACTED]"

    def apply(self, text: str, tier: str) -> tuple[str, int]:
        if tier not in self.tiers:
            return text, 0
        return re.subn(self.pattern, self.token, text)


def redact(text: str, tier: str, rules: list) -> tuple[str, list]:
    hits = []
    for r in rules:
        text, n = r.apply(text, tier)
        if n:
            hits.append((r.name, n))
    return text, hits


def rehydrate(ev: Event, store: Store) -> tuple[list, list]:
    """Admitted span texts in packet order, and the content keys that are missing."""
    texts, missing = [], []
    for o in sorted((o for o in ev.offered if o.decision == "admitted"), key=lambda o: o.rank):
        t = store.get(o.content)
        if t is None or h(t) != o.content:
            missing.append(o.span_id)
        else:
            texts.append((o, t))
    return texts, missing


def reconstruct(ev: Event, store: Store, render) -> dict:
    """What the model saw, rebuilt from the store and checked against the event."""
    texts, missing = rehydrate(ev, store)
    packet = render(ev, texts, store)
    return {"det_id": ev.det_id, "missing": missing,
            "exact": not missing and h(packet) == ev.packet_hash,
            "packet": packet,
            "dropped": [(o.span_id, o.reason) for o in ev.offered if o.decision == "dropped"],
            "cited": ev.output.get("citations", []),
            "tools": [(t["call_id"], t["tool"], t["read_at"]) for t in ev.tools]}


def reconstructability(events, store: Store, render) -> dict:
    """The redaction gate: every sampled event must rebuild exactly."""
    res = [reconstruct(e, store, render) for e in events]
    bad = [r["det_id"] for r in res if not r["exact"]]
    return {"n": len(res), "exact": len(res) - len(bad), "failed": bad}


# ---- metrics and alerts -------------------------------------------------------

def p_limits(p: float, n: int, z: float = 3.0) -> tuple[float, float]:
    """p-chart control limits for a proportion with baseline p over n trials."""
    sd = (p * (1 - p) / max(n, 1)) ** 0.5
    return max(0.0, p - z * sd), min(1.0, p + z * sd)


@dataclass
class Alarm:
    """Alert when a slice's daily rate falls below its lower limit `run` days running."""
    name: str
    baseline: float
    run: int = 2
    z: float = 3.0
    low: int = 0
    fired: list = field(default_factory=list)

    def observe(self, day: int, hits: int, n: int) -> bool:
        if n == 0:
            return False
        lcl, _ = p_limits(self.baseline, n, self.z)
        self.low = self.low + 1 if hits / n < lcl else 0
        if self.low == self.run:
            self.fired.append(day)
            return True
        return False


class Ewma:
    """Exponentially weighted mean of a daily rate with a fixed lower bound."""

    def __init__(self, baseline: float, sd: float, lam: float = 0.3, k: float = 3.0):
        self.m, self.lam = baseline, lam
        self.lcl = baseline - k * sd * (lam / (2 - lam)) ** 0.5
        self.fired: list = []

    def observe(self, day: int, rate: float) -> bool:
        self.m = self.lam * rate + (1 - self.lam) * self.m
        if self.m < self.lcl:
            self.fired.append(day)
            return True
        return False
