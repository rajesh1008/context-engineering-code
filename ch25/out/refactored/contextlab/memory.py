# Copyright 2026 Lakeshore Mutual. Internal use only.
# SPDX-License-Identifier: LicenseRef-Lakeshore-Internal
# Generated header; do not edit by hand.

"""contextlab.memory: governed memory for the context layer (Chapter 21, v4.6).

A memory is a record with a subject, a kind, a source and a lifetime, not a
cache of chat. The write policy decides what may become memory and of which
kind; supersession keeps one current value per subject and field; the reader
routes by kind and scope before it ranks by relevance, trust and recency.
Version 1 logs every write with its provenance. It does not yet log reads:
Chapter 22 adds the read log, and Chapter 23 the lifecycle (retention,
erasure, review).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date

KINDS = ("working", "episodic", "semantic", "procedural")
HALF_LIFE_DAYS = {"working": 1, "episodic": 540, "semantic": 1460, "procedural": 365}
TRUST = {"system": 0.95, "adjuster": 0.9, "consolidation": 0.85,
         "claimant": 0.7, "model": 0.3}


@dataclass(frozen=True)
class Candidate:
    """Something an extractor proposes to remember, before the write policy."""
    subject: str                 # claimant:C0123, property:P0456, procedure:HO/WI/water
    ctype: str                   # preference, property, household, event, correction,
                                 # policy_fact, status, inference, small_talk, sensitive
    text: str
    source_type: str             # system, adjuster, claimant, model
    source_ref: str              # episode id in the system that holds the original
    when: date
    tenant: str = "lakeshore"
    field: str | None = None     # slot for a semantic fact, e.g. contact.channel
    value: str | None = None


@dataclass(frozen=True)
class MemoryRecord:
    mid: str
    kind: str
    subject: str
    text: str
    tenant: str
    source_type: str
    source_ref: str
    written_by: str
    written: date
    trust: float
    field: str | None = None
    value: str | None = None
    scope: tuple = ()            # for procedural: (line, state, cause family)
    supersedes: str | None = None
    superseded_by: str | None = None
    derived_from: tuple = ()
    expires: date | None = None
    roles: frozenset = frozenset({"adjuster", "supervisor"})


ROUTE = {"preference": "semantic", "property": "semantic", "household": "semantic",
         "event": "episodic", "correction": "episodic"}
REFUSE = {"policy_fact": "system of record owns it; fetch it live",
          "status": "volatile; fetch it live",
          "small_talk": "no later use",
          "sensitive": "not needed for the purpose"}


class WritePolicy:
    """Decides whether a candidate becomes memory, and of which kind."""

    def route(self, c: Candidate) -> tuple[str | None, str]:
        if c.ctype in REFUSE:
            return None, REFUSE[c.ctype]
        if c.source_type == "model" or c.ctype == "inference":
            return None, "model inference; a model may not certify its own output"
        if not c.source_ref:
            return None, "no source to point back to"
        return ROUTE.get(c.ctype), "ok" if c.ctype in ROUTE else "unknown type"


class WriteEverything(WritePolicy):
    """The baseline: every candidate is stored, as semantic unless it is an event."""

    def route(self, c):
        return ("episodic" if c.ctype in ("event", "correction") else "semantic"), "ok"


@dataclass
class Memory:
    policy: WritePolicy = field(default_factory=WritePolicy)
    records: dict = field(default_factory=dict)
    write_log: list = field(default_factory=list)
    _current: dict = field(default_factory=dict)      # (subject, field) -> mid

    def write(self, c: Candidate, by: str = "extractor") -> MemoryRecord | None:
        kind, reason = self.policy.route(c)
        self.write_log.append((c.when, c.subject, c.ctype, c.source_ref, kind, reason))
        if kind is None:
            return None
        rec = MemoryRecord(f"M{len(self.records) + 1:05d}", kind, c.subject, c.text,
                           c.tenant, c.source_type, c.source_ref, by, c.when,
                           TRUST[c.source_type], c.field, c.value)
        return self.put(rec)

    def put(self, rec: MemoryRecord) -> MemoryRecord:
        key = (rec.subject, rec.field)
        old = self._current.get(key) if rec.field else None
        if old and self.records[old].written <= rec.written:
            rec = replace(rec, supersedes=old)
            self.records[old] = replace(self.records[old], superseded_by=rec.mid)
        if rec.field and (not old or rec.supersedes):
            self._current[key] = rec.mid
        self.records[rec.mid] = rec
        return rec

    def current(self, subject: str, fld: str) -> MemoryRecord | None:
        mid = self._current.get((subject, fld))
        return self.records[mid] if mid else None

    def live(self):
        return [r for r in self.records.values() if r.superseded_by is None]


def recency(rec: MemoryRecord, as_of: date) -> float:
    age = max(0, (as_of - rec.written).days)
    return 0.5 ** (age / HALF_LIFE_DAYS[rec.kind])


def in_scope(rec: MemoryRecord, subjects: set, scope: tuple, tenant: str, role: str) -> bool:
    if rec.tenant != tenant or role not in rec.roles or rec.superseded_by:
        return False
    if rec.kind == "procedural":
        return all(have in ("*", want) for have, want in zip(rec.scope, scope))
    return rec.subject in subjects


def rank(rec: MemoryRecord, relevance: float, as_of: date) -> float:
    return relevance * rec.trust * recency(rec, as_of)


@dataclass
class ReadRequest:
    subjects: set                 # the claimant, policy and property on this claim
    scope: tuple                  # (line, state, cause family) for procedural lessons
    tenant: str
    role: str
    as_of: date
    budget_tokens: int = 160
    per_kind: dict = field(default_factory=lambda: {"episodic": 3, "semantic": 6,
                                                    "procedural": 2})


def tokens(text: str) -> int:
    return math.ceil(len(text.split()) * 1.33) + 4       # text plus a source tag


def read(mem: Memory, q: ReadRequest, relevance) -> list[MemoryRecord]:
    """Route by kind and scope first, then rank, then fit the budget."""
    chosen, used = [], 0
    for kind, k in q.per_kind.items():
        pool = [r for r in mem.records.values()
                if r.kind == kind and in_scope(r, q.subjects, q.scope, q.tenant, q.role)]
        pool.sort(key=lambda r: (-rank(r, relevance(r), q.as_of), r.mid))
        for r in pool[:k]:
            if used + tokens(r.text) <= q.budget_tokens:
                chosen.append(r); used += tokens(r.text)
    return chosen
