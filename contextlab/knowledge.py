"""contextlab.knowledge: sources, admission and evidence (Chapter 9, v1.7).

Chapter 2 introduced a Source that rejected spans with no version.
Chapter 9 widens it into an inventory entry: every source declares what
kind of evidence it gives, how it is reached, where its version comes
from, how fresh it must be, who may see it and what it is licensed for.
Admission fails closed: anything the inventory cannot vouch for is
withheld, stays visible to its owner, and never reaches a packet.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from enum import Enum


class EvidenceClass(str, Enum):
    GOVERNING = "governing"      # says what the contract or the law means
    FACTUAL = "factual"          # says what is true of one policy or claim
    ADVISORY = "advisory"        # interprets; may inform, never decides
    PROCEDURAL = "procedural"    # says how the work is done


class Access(str, Enum):
    RETRIEVE = "retrieve"        # curated, versioned, indexed as spans
    QUERY = "query"              # structured query against a system of record
    TOOL = "tool"                # fetched live at request time, never stored
    RULEBOOK = "rulebook"        # a person's knowledge, written down and owned
    EXCLUDE = "exclude"          # registered so that it can be refused


class Volatility(str, Enum):
    LOW = "low"                  # curate, version, cite
    MODERATE = "moderate"        # pre-warm with a time to live
    HIGH = "high"                # never store; fetch live with a timestamp


class VersionRule(str, Enum):
    EDITION = "edition"          # printed edition, e.g. HO 04 95 (2024-06)
    EFFECTIVE = "effective"      # effective-from / effective-to dates
    AS_OF = "as-of"              # timestamp of the read
    ROW = "row"                  # row version in the system of record
    NONE = "none"


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    owner: str | None
    system: str
    evidence_class: EvidenceClass
    access: Access
    volatility: Volatility
    version_rule: VersionRule
    version_origin: str = "document"     # where the version must come from
    max_age: timedelta | None = None     # freshness service level
    roles: frozenset = frozenset({"adjuster"})
    purposes: frozenset = frozenset({"claims_determination"})
    precedence: int = 0
    note: str = ""

    def admit(self, rec: "Record", *, on: date, now: datetime,
              role: str = "adjuster",
              purpose: str = "claims_determination") -> "Admission":
        """Fail closed: the first failed check names the reason."""
        checks = [
            (self.owner is not None, "source has no owner"),
            (self.access is not Access.EXCLUDE, "source is excluded"),
            (role in self.roles, f"role {role!r} not entitled"),
            (purpose in self.purposes, f"not licensed for {purpose}"),
            (self.version_rule is VersionRule.NONE or rec.version is not None,
             "no version"),
            (rec.version_origin in (None, self.version_origin)
             or self.version_rule is VersionRule.NONE,
             f"version came from {rec.version_origin}, "
             f"inventory requires {self.version_origin}"),
            (rec.in_force(on), f"not in force on {on.isoformat()}"),
            (self.max_age is None or now - rec.observed_at <= self.max_age,
             "older than the freshness limit"),
        ]
        for ok, reason in checks:
            if not ok:
                return Admission(False, reason)
        return Admission(True, "admitted")


@dataclass(frozen=True)
class Record:
    """An item offered by a source. It is not evidence until admitted."""
    record_id: str
    source: str
    text: str
    version: str | None
    version_origin: str | None = "document"
    effective_from: date | None = None
    effective_to: date | None = None     # exclusive
    superseded_by: str | None = None
    jurisdiction: str = "ALL"
    observed_at: datetime = datetime(2026, 10, 1)
    page: int | None = None

    def in_force(self, on: date) -> bool:
        if self.effective_from and on < self.effective_from:
            return False
        if self.effective_to and on >= self.effective_to:
            return False
        return True


@dataclass(frozen=True)
class Admission:
    ok: bool
    reason: str


@dataclass(frozen=True)
class Evidence:
    """A record plus the stamp that says why it was allowed in."""
    record: Record
    source: Source
    admitted_on: date
    inventory: str                       # inventory version and fingerprint

    def header(self) -> str:
        r, s = self.record, self.source
        ver = r.version or "unversioned"
        page = f" | page {r.page}" if r.page else ""
        return (f"{r.record_id} ({ver}){page} | {s.evidence_class.value} | "
                f"{s.system} | owner {s.owner} | as of {r.observed_at:%Y-%m-%d}")


@dataclass
class Inventory:
    version: str
    sources: list[Source] = field(default_factory=list)
    quarantine: list[tuple[str, str, str]] = field(default_factory=list)

    def by_name(self, name: str) -> Source:
        return next(s for s in self.sources if s.name == name)

    def fingerprint(self) -> str:
        rows = [{k: (sorted(v) if isinstance(v, frozenset) else str(v))
                 for k, v in asdict(s).items()} for s in self.sources]
        blob = json.dumps(rows, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def gaps(self) -> list[str]:
        out = []
        for s in self.sources:
            if s.owner is None:
                out.append(f"{s.name}: no owner")
            if (s.version_rule is VersionRule.NONE and s.access is Access.RETRIEVE
                    and s.evidence_class in (EvidenceClass.GOVERNING,
                                             EvidenceClass.FACTUAL)):
                out.append(f"{s.name}: retrieved as evidence with no version rule")
            if s.volatility is Volatility.HIGH and s.access is Access.RETRIEVE:
                out.append(f"{s.name}: high volatility but stored for retrieval")
        return out

    def offer(self, rec: Record, *, on: date, now: datetime,
              role: str = "adjuster",
              purpose: str = "claims_determination") -> Evidence | None:
        src = self.by_name(rec.source)
        adm = src.admit(rec, on=on, now=now, role=role, purpose=purpose)
        if adm.ok:
            return Evidence(rec, src, on, f"{self.version}#{self.fingerprint()}")
        # visible to the owner, absent from retrieval
        self.quarantine.append((rec.record_id, src.owner or "unowned", adm.reason))
        return None
