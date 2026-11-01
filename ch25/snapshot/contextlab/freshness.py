"""contextlab.freshness: change capture, time-to-visible and cache epochs (Chapter 24, v5.6).

Chapter 15 decides, at query time, which version of a document was in force
on the date that matters. This module covers the other clock: when a change
committed in a source system becomes visible to retrieval, and what must be
forgotten when it does. It knows nothing about any particular database; a
Change is what a log-based capture tool (one row image per committed change,
with its log position) would hand to a consumer.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta


@dataclass(frozen=True)
class Change:
    source: str                     # docstore | policy_admin | claims
    key: str                        # doc_id, policy number or claim id
    seq: int                        # log position in the source (LSN, SCN, offset)
    committed: datetime             # commit time in the source system
    version: str | None = None      # the full row image's version, not a delta
    scope: tuple = ()               # (state, line) the change can affect
    supersedes: str | None = None
    effective_from: date | None = None


class Applier:
    """Idempotent, per-key ordered application of full row images.

    At-least-once delivery means duplicates; partitioned transport can mean a
    later image of a key arrives before an earlier one. Keeping the highest
    applied position per key makes both harmless.
    """

    def __init__(self):
        self.pos: dict = {}
        self.state: dict = {}
        self.duplicates = 0
        self.out_of_order = 0

    def apply(self, ch: Change) -> bool:
        k = (ch.source, ch.key)
        last = self.pos.get(k, -1)
        if ch.seq <= last:
            if ch.seq == last:
                self.duplicates += 1
            else:
                self.out_of_order += 1
            return False
        self.pos[k], self.state[k] = ch.seq, ch
        return True


class NaiveApplier:
    """What a first streaming job usually does: apply in arrival order, append."""

    def __init__(self):
        self.state: dict = {}
        self.chunks: list = []

    def apply(self, ch: Change) -> bool:
        self.state[(ch.source, ch.key)] = ch
        self.chunks.append((ch.key, ch.version))
        return True


def next_nightly(t: datetime, at_hour: int = 2, build: timedelta = timedelta(hours=1, minutes=40)) -> datetime:
    """When a nightly rebuild that starts at at_hour makes a change committed at t visible."""
    start = t.replace(hour=at_hour, minute=0, second=0, microsecond=0)
    if start <= t:
        start += timedelta(days=1)
    return start + build


@dataclass
class Stage:
    name: str
    seconds: float


def time_to_visible(stages: list[Stage]) -> float:
    """Seconds from commit to retrievable: the sum of every stage on the path."""
    return sum(s.seconds for s in stages)


class Watermarks:
    """Highest source position each consumer has made visible, per scope."""

    def __init__(self):
        self.head: dict = defaultdict(int)          # scope -> latest committed seq
        self.seen: dict = defaultdict(lambda: defaultdict(int))  # consumer -> scope -> seq

    def committed(self, scope, seq):
        self.head[scope] = max(self.head[scope], seq)

    def visible(self, consumer, scope, seq):
        self.seen[consumer][scope] = max(self.seen[consumer][scope], seq)

    def behind(self, consumer, scope) -> bool:
        return self.seen[consumer][scope] < self.head[scope]


@dataclass
class Entry:
    value: object
    stored: datetime
    epoch: int
    tags: frozenset = field(default_factory=frozenset)


class ScopedCache:
    """A result cache with a TTL and two invalidation hooks.

    mode "ttl": entries expire only on the clock.
    mode "tags": a change invalidates entries whose tags name the changed or superseded doc.
    mode "epoch": a change bumps its scope's epoch; every entry in that scope misses.
    """

    def __init__(self, ttl: timedelta, mode: str = "epoch"):
        self.ttl, self.mode = ttl, mode
        self.entries: dict = {}
        self.epoch: dict = defaultdict(int)
        self.hits = self.misses = 0

    def get(self, key, scope, now: datetime):
        e = self.entries.get((key, scope))
        if e and now - e.stored < self.ttl and e.epoch == self.epoch[scope]:
            self.hits += 1
            return e.value
        self.misses += 1
        return None

    def put(self, key, scope, value, now, tags=frozenset()):
        self.entries[(key, scope)] = Entry(value, now, self.epoch[scope], frozenset(tags))

    def on_change(self, ch: Change):
        if self.mode == "epoch":
            for sc in ch.scope:
                self.epoch[sc] += 1
        elif self.mode == "tags":
            names = {ch.key, ch.supersedes} - {None}
            for k in [k for k, e in self.entries.items() if e.tags & names]:
                del self.entries[k]


def staleness_incident(decided: datetime, required: list, visible_at: dict) -> list:
    """Docs the decision needed that had committed but were not yet visible to it."""
    return [d for d, committed in required
            if committed <= decided and visible_at.get(d, datetime.max) > decided]
