"""contextlab.memguard (v4.7): provenance, read logs, tracing, quarantine, rollback.

Chapter 22. A memory record carries where it came from (its origins), what it
was derived from (its parents), what it asserts, who it is about (scope and
subject) and when it holds (valid_from, valid_to, supersedes). Every read is
logged against the determination that made it. With both, a bad memory can be
traced back from a wrong answer and forward to every answer it touched, and
removed by family rather than by guesswork. The module does not store or rank
memories; Chapter 21's memory store does that. It sits beside it.
"""
from dataclasses import dataclass, field
from datetime import date

GOVERNED = {"span", "record"}            # governed knowledge or a system of record
UNVERIFIED = {"determination", "note", "claimant", "summary"}


@dataclass(frozen=True)
class Origin:
    kind: str          # span, record, determination, note, claimant, summary
    ref: str           # identifier of the thing it came from


@dataclass
class MemoryRecord:
    mid: str
    scope: str                     # policy, claimant, adjuster, team
    subject: str                   # the policy, claimant or adjuster it is about; team: a state
    text: str
    written: date
    origins: tuple = ()
    parents: tuple = ()            # memory ids it was consolidated from
    asserts: tuple | None = None   # (topic, value) when the memory states a coverage position
    valid_from: date | None = None
    valid_to: date | None = None
    supersedes: str | None = None
    status: str = "live"           # live, quarantined, held, superseded
    note: str = ""

    def authority(self) -> str:
        """Authority is bound to origin at write time, never to wording."""
        return "governed" if any(o.kind in GOVERNED for o in self.origins) else "unverified"


@dataclass
class Read:
    det: str
    day: date
    mids: tuple
    subject_keys: tuple            # policy, claimant, adjuster, state of the determination


@dataclass
class ReadLog:
    reads: list = field(default_factory=list)

    def log(self, det, day, mids, subject_keys):
        self.reads.append(Read(det, day, tuple(mids), tuple(subject_keys)))

    def readers(self, mids) -> list:
        mids = set(mids)
        return [r for r in self.reads if mids & set(r.mids)]

    def read_by(self, det) -> tuple:
        return next((r.mids for r in self.reads if r.det == det), ())


def write_gate(m: MemoryRecord) -> str:
    """A coverage position may be remembered only from a governed origin."""
    if m.asserts and m.authority() != "governed":
        m.status = "held"
        m.note = "coverage assertion without a governed origin"
    return m.status


def family(store: dict, root: str) -> set:
    """The root and every memory consolidated from it, at any depth."""
    out, frontier = {root}, [root]
    while frontier:
        cur = frontier.pop()
        for m in store.values():
            if cur in m.parents and m.mid not in out:
                out.add(m.mid); frontier.append(m.mid)
    return out


def root_of(store: dict, mid: str) -> str:
    seen = mid
    while store[seen].parents:
        parent = [p for p in store[seen].parents if p in store]
        if not parent:
            break
        seen = parent[0]
    return seen


def culprits(det, read_log: ReadLog, rerun) -> list:
    """Leave-one-out: memories whose removal alone changes the answer."""
    mids = read_log.read_by(det)
    base = rerun(det, mask=())
    return [m for m in mids if rerun(det, mask=(m,)) != base]


def quarantine(store: dict, mids, reason: str) -> int:
    n = 0
    for mid in mids:
        if store[mid].status == "live":
            store[mid].status, store[mid].note = "quarantined", reason
            n += 1
    return n


def rollback(store: dict, before: date) -> int:
    """Restore the store as it stood before a date: everything later goes."""
    n = 0
    for m in store.values():
        if m.written >= before and m.status == "live":
            m.status, m.note = "quarantined", f"rolled back to {before}"
            n += 1
    return n


def scope_violations(read_log: ReadLog, store: dict) -> list:
    """Reads of a policy, claimant or adjuster memory by someone else's determination."""
    bad = []
    for r in read_log.reads:
        for mid in r.mids:
            m = store[mid]
            if m.scope != "team" and m.subject not in r.subject_keys:
                bad.append((r.det, mid))
    return bad


def stale(store: dict, record_lookup, on: date) -> list:
    """Live memories that the system of record now contradicts."""
    out = []
    for m in store.values():
        if m.status == "live" and m.scope == "policy":
            truth = record_lookup(m.subject, on)
            if truth is not None and m.asserts and m.asserts[1] != truth.get(m.asserts[0]):
                out.append(m.mid)
    return out
