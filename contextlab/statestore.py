"""contextlab v5.3: StateStore, session state kept outside the window, and the memory lifecycle.

Three things are kept apart on purpose:
  state    what this working session needs now (claims open, facts with span ids, pinned
           constraints, decisions); lives in the StateStore, is rendered into the window
           on purpose, and is checkpointed by reference, not by rendered text;
  memory   records that outlive the session (Chapter 22's memguard.MemoryRecord, with scope,
           subject, origins, parents, valid_to and status); this module adds the lifecycle:
           retention, re-validation, read budget, tenancy and erasure;
  context  the packet assembled for one model call from the two above plus evidence.

Chapter 23 of The Context Engineering Handbook for LLMs and AI Agents. Plain Python, no dependencies.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta


# ---------------------------------------------------------------- session state

@dataclass
class Fact:
    key: str            # e.g. "C02.limit.HO0495"
    value: str          # e.g. "US$10,000"
    source: str         # span or record id the fact came from, e.g. "LK-HO0495@2024-06#p2"
    tokens: int = 20


@dataclass
class ClaimState:
    claim_id: str
    status: str = "open"            # open | drafted | routed | awaiting
    facts: dict[str, Fact] = field(default_factory=dict)
    outcome: str = ""               # the folded note once the adjuster moves on

    def note_tokens(self) -> int:
        return 30 + 8 * bool(self.outcome)


class StateStore:
    """Working state for one session, kept outside the window and reloaded by name."""

    def __init__(self, session_id: str, adjuster: str):
        self.session_id, self.adjuster = session_id, adjuster
        self.claims: dict[str, ClaimState] = {}
        self.pinned: list[tuple[str, str]] = []      # (constraint id, exact text)
        self.checkpoints: dict[str, dict] = {}

    def claim(self, claim_id: str) -> ClaimState:
        return self.claims.setdefault(claim_id, ClaimState(claim_id))

    def record(self, claim_id: str, fact: Fact) -> None:
        self.claim(claim_id).facts[fact.key] = fact

    def pin(self, cid: str, text: str) -> None:
        if cid not in {c for c, _ in self.pinned}:
            self.pinned.append((cid, text))

    def fold(self, claim_id: str, outcome: str) -> None:
        self.claim(claim_id).outcome = outcome

    def sheet(self) -> str:
        """The progress file: one line per claim, rendered at the top of every call."""
        return "\n".join(f"{c.claim_id}\t{c.status}\t{c.outcome or '-'}" for c in self.claims.values())

    # -- checkpointing the packet by reference
    def checkpoint(self, packet_ids: list[str], editions: dict[str, str], turn: int) -> str:
        snap = {"session": self.session_id, "turn": turn, "packet": sorted(packet_ids),
                "editions": dict(sorted(editions.items())), "pinned": self.pinned,
                "claims": {k: {"status": v.status, "outcome": v.outcome,
                               "facts": {f.key: [f.value, f.source] for f in v.facts.values()}}
                           for k, v in self.claims.items()}}
        blob = json.dumps(snap, sort_keys=True)
        cid = hashlib.sha256(blob.encode()).hexdigest()[:16]
        self.checkpoints[cid] = json.loads(blob)
        return cid

    @classmethod
    def restore(cls, snap: dict, adjuster: str) -> "StateStore":
        s = cls(snap["session"], adjuster)
        s.pinned = [tuple(p) for p in snap["pinned"]]
        for k, v in snap["claims"].items():
            c = s.claim(k)
            c.status, c.outcome = v["status"], v["outcome"]
            for key, (val, src) in v["facts"].items():
                c.facts[key] = Fact(key, val, src)
        return s


def checkpoint_id(snap: dict) -> str:
    return hashlib.sha256(json.dumps(snap, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- memory lifecycle
# Works on contextlab.memguard.MemoryRecord (Chapter 22). Status values used here:
#   live, held, quarantined, superseded (Chapter 22) and expired, erased (this chapter).

@dataclass
class RetentionRule:
    scope: str                    # policy | claimant | adjuster | team
    keep_days: int | None         # delete this long after written (None: no age limit)
    revalidate_days: int | None = None   # re-check against the system of record after this


@dataclass
class LifecyclePolicy:
    rules: dict[str, RetentionRule]
    read_budget: int = 6          # memories per packet
    min_origins_team: int = 3     # a team memory needs this many independent origins

    def state(self, m, today: date) -> str:
        if m.status != "live":
            return m.status
        r = self.rules.get(m.scope)
        age = (today - m.written).days
        if m.valid_to and today > m.valid_to:
            return "past validity"
        if r and r.keep_days is not None and age > r.keep_days:
            return "past retention"
        if r and r.revalidate_days is not None and age > r.revalidate_days:
            return "due for re-validation"
        return "live"


def sweep(store: dict, today: date, policy: LifecyclePolicy) -> dict[str, int]:
    """Count records by lifecycle state. Read-only."""
    out: dict[str, int] = {}
    for m in store.values():
        k = policy.state(m, today)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def expire(store: dict, today: date, policy: LifecyclePolicy, holds=frozenset()) -> list[str]:
    """Retention: mark records past their keep period expired, except legal holds."""
    gone = []
    for m in store.values():
        if policy.state(m, today) == "past retention" and m.subject not in holds:
            m.status, m.note = "expired", f"retention {today.isoformat()}"
            gone.append(m.mid)
    return gone


def revalidate(store: dict, today: date, policy: LifecyclePolicy, lookup, make) -> dict:
    """Re-check records due for re-validation against the system of record.

    lookup(subject) -> current value; make(old, value) -> the replacement record."""
    out = {"checked": 0, "unchanged": 0, "corrected": []}
    for m in list(store.values()):
        if policy.state(m, today) != "due for re-validation":
            continue
        out["checked"] += 1
        value = lookup(m.subject)
        new = make(m, value)
        m.status, m.note = "superseded", f"re-validated {today.isoformat()}"
        new.supersedes = m.mid
        store[new.mid] = new
        if m.asserts and m.asserts[1] != value:
            out["corrected"].append(m.mid)
        else:
            out["unchanged"] += 1
    return out


def may_read(m, reader_tenant: str, tenant_of) -> bool:
    """Tenancy: a record about a subject is readable only inside that subject's tenant."""
    if m.scope in ("policy", "claimant"):
        return tenant_of(m.subject) == reader_tenant
    return m.scope != "team" or getattr(m, "tenant", None) in (None, reader_tenant)


@dataclass
class ErasureReport:
    request: str
    deleted: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    rederived: list[str] = field(default_factory=list)
    dropped_derived: list[str] = field(default_factory=list)
    kept_redacted: list[str] = field(default_factory=list)
    checkpoints_redacted: int = 0
    tombstone: dict = field(default_factory=dict)


def erase(store: dict, subject: str, own_refs: set[str], request: str, today: date,
          policy: LifecyclePolicy, checkpoints: dict | None = None, holds=frozenset(),
          incident_ids=frozenset()) -> ErasureReport:
    """Erase one data subject from memory: direct records, derived records, checkpoints.

    own_refs: origin refs that belong to the subject (their determinations, notes, records).
    incident_ids: records kept as incident evidence; redacted, not deleted."""
    rep = ErasureReport(request)
    for m in sorted(store.values(), key=lambda x: x.mid):
        if m.status == "erased":
            continue
        direct = m.subject == subject
        mine = [o for o in m.origins if o.ref in own_refs or o.ref == subject]
        if not direct and not mine:
            continue
        if subject in holds:
            rep.held.append(m.mid)
            continue
        if direct:
            m.status, m.text, m.asserts = "erased", "", None
            rep.deleted.append(m.mid)
            continue
        m.origins = tuple(o for o in m.origins if o not in mine)
        if m.mid in incident_ids:
            rep.kept_redacted.append(m.mid)
        elif m.scope == "team" and len({o.ref for o in m.origins}) < policy.min_origins_team:
            m.status = "erased"
            rep.dropped_derived.append(m.mid)
        else:
            rep.rederived.append(m.mid)
    for snap in (checkpoints or {}).values():
        hit = False
        for c in snap.get("claims", {}).values():
            for k in [k for k, (v, src) in c["facts"].items() if src.startswith(subject)]:
                del c["facts"][k]
                hit = True
        rep.checkpoints_redacted += hit
    rep.tombstone = {"request": request, "date": today.isoformat(),
                     "subject_hash": hashlib.sha256(subject.encode()).hexdigest()[:12],
                     "deleted": len(rep.deleted), "held": len(rep.held),
                     "rederived": len(rep.rederived), "dropped": len(rep.dropped_derived),
                     "redacted_incident": len(rep.kept_redacted),
                     "checkpoints": rep.checkpoints_redacted}
    return rep
