"""contextlab.governance: who owns each copy, and how we know it matches (Chapter 44, v9.6).

Earlier chapters built mechanisms: upserts and deletion checks (13), an
embedding registry (14), authority rules (15), memory erasure (23), change
capture (24). This module adds the governance layer over them:

  * an asset register: every store the context layer writes, with an owner,
    a retention rule, an erasure answer and a tenant field;
  * a catalogue lifecycle with term-aware retirement;
  * reconciliation of every copy against the catalogue, on a schedule,
    whether or not anything announced a change;
  * change classes for embedding refresh, with gates and an impact plan;
  * erasure and partner-exit fan-out across the register;
  * policy-to-evidence coverage and a governance calendar.
Standard library only.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

from contextlab.policy import Policy, Request, Verdict

VERSION = "v9.6"

ERASURE_ANSWERS = ("delete", "redact", "age-out", "exempt", "none-held")
LIFECYCLE = ("draft", "published", "superseded", "retired", "void")


def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- register
@dataclass
class Asset:
    name: str
    kind: str                       # source, derived, log, evaluation, evidence, backup
    owner: str | None               # accountable role
    steward: str | None             # role that runs it day to day
    personal_data: bool
    tenant_field: bool
    retention_days: int | None      # None: kept while the source is
    erasure: str | None             # one of ERASURE_ANSWERS
    review: str | None              # cadence
    derived_from: tuple = ()
    exemption: str | None = None    # stated basis when erasure == "exempt"


class Register:
    def __init__(self, assets):
        self.assets = {a.name: a for a in assets}

    def gaps(self):
        out = []
        for a in self.assets.values():
            if not a.owner:
                out.append((a.name, "no owner"))
            if a.personal_data and a.erasure not in ERASURE_ANSWERS:
                out.append((a.name, "personal data with no erasure answer"))
            if a.erasure == "exempt" and not a.exemption:
                out.append((a.name, "exempt with no stated basis"))
            if a.erasure == "age-out" and not a.retention_days:
                out.append((a.name, "age-out with no period"))
            if not a.tenant_field and a.kind != "source":
                out.append((a.name, "no tenant field"))
            if not a.review:
                out.append((a.name, "no review cadence"))
        return out

    def downstream(self, name):
        """Every asset built, directly or not, from this one."""
        seen, todo = [], [name]
        while todo:
            cur = todo.pop(0)
            for a in self.assets.values():
                if cur in a.derived_from and a.name not in seen:
                    seen.append(a.name)
                    todo.append(a.name)
        return seen


# ---------------------------------------------------------------- catalogue
@dataclass(frozen=True)
class Doc:
    doc_id: str
    version: str
    kind: str
    state: str = "published"         # LIFECYCLE
    text_hash: str = ""
    effective_from: date | None = None
    effective_to: date | None = None  # end of supersession (by date of loss)
    retired_for_terms_from: date | None = None   # attaches to no term starting on/after
    tenant: str = "lakeshore"
    owner: str = "forms library"


@dataclass(frozen=True)
class TermRequest(Request):
    term_start: date | None = None


class TermAwarePolicy(Policy):
    """Chapter 15's Policy plus one gate: a retired document governs only the
    policy terms that began before it was retired."""
    version = "policy-v1.1"

    def __init__(self, spans, retired: dict, **kw):
        super().__init__(spans, **kw)
        self.retired = retired                    # doc_id -> first term start excluded

    def admit(self, s, r):
        cut = self.retired.get(s.doc_id)
        if cut and getattr(r, "term_start", None) is None:
            return Verdict(False, "retired document; no term start")
        if cut and r.term_start >= cut:
            return Verdict(False, f"retired; attaches to no term from {cut}")
        return super().admit(s, r)


# ---------------------------------------------------------------- reconciliation
@dataclass(frozen=True)
class Copy:
    """One row in a derived store, as that store describes itself."""
    store: str
    key: str                    # span id, cache key, label id, memory id
    doc_id: str
    version: str
    text_hash: str = ""
    model: str | None = None
    state: str = "published"
    effective_to: date | None = None
    tenant: str = "lakeshore"


FINDINGS = ("retired-open", "superseded-open", "orphan", "missing", "stale-hash",
            "wrong-model", "end-mismatch", "tenant-mismatch", "void-present")


def reconcile(catalogue, copies, live_model, expect_in=("passages", "vectors")):
    """Compare every copy with the catalogue. Returns a list of findings."""
    cat = {(d.doc_id, d.version): d for d in catalogue}
    out = []
    present = defaultdict(set)
    for c in copies:
        present[c.store].add((c.doc_id, c.version))
        d = cat.get((c.doc_id, c.version))
        if d is None:
            out.append((c.store, c.key, "orphan", "no catalogue record"))
            continue
        if d.state == "void":
            out.append((c.store, c.key, "void-present", "withdrawn in error"))
        if d.state == "retired" and c.state != "retired":
            out.append((c.store, c.key, "retired-open",
                        f"catalogue retired for terms from {d.retired_for_terms_from}"))
        if d.state == "superseded" and c.state == "published" and c.store in expect_in:
            out.append((c.store, c.key, "superseded-open", "copy still marked current"))
        if c.text_hash and d.text_hash and c.text_hash != d.text_hash:
            out.append((c.store, c.key, "stale-hash", "text changed since copy"))
        if c.model and c.model != live_model:
            out.append((c.store, c.key, "wrong-model", f"{c.model} not {live_model}"))
        if c.store == "passages" and c.effective_to != d.effective_to:
            out.append((c.store, c.key, "end-mismatch", f"{c.effective_to} vs {d.effective_to}"))
        if c.tenant != d.tenant:
            out.append((c.store, c.key, "tenant-mismatch", f"{c.tenant} vs {d.tenant}"))
    for d in catalogue:
        if d.state in ("published", "superseded", "retired"):
            for store in expect_in:
                if (d.doc_id, d.version) not in present[store]:
                    out.append((store, f"{d.doc_id}@{d.version}", "missing", "not in store"))
    return out


# ---------------------------------------------------------------- change control
@dataclass(frozen=True)
class Change:
    cid: str
    what: str
    touches: str      # content, metadata, spanning, template, model, tenant, purge
    spans: int        # rows affected
    forced_by: str | None = None    # a retire-by date, a legal deadline


GATES = {
    "standard": ("logged",),
    "normal": ("change record", "probe fingerprint", "per-slice gold delta",
               "rollback rehearsed", "owner sign-off"),
    "emergency": ("change record", "owner sign-off", "post-change review in 5 days"),
}


def classify(c: Change):
    """Change class, whether vectors are recomputed, and the gates that apply."""
    reembed = c.touches in ("content", "spanning", "template", "model")
    if c.touches == "purge":
        cls = "emergency"
    elif c.touches in ("metadata",) or (c.touches == "content" and c.spans <= 500):
        cls = "standard"
    else:
        cls = "normal"
    if c.forced_by and cls == "normal" and c.forced_by.startswith("missed"):
        cls = "emergency"
    return cls, reembed, GATES[cls]


def impact(old_to_new: dict, dependants: dict):
    """Restatement plan: for a change that renames span ids, which dependants
    point at a span that will no longer exist, and which can be remapped."""
    plan = {}
    for store, refs in dependants.items():
        moved = [r for r in refs if r in old_to_new and old_to_new[r]]
        lost = [r for r in refs if r in old_to_new and not old_to_new[r]]
        plan[store] = {"refs": len(refs), "remap": len(moved), "unmappable": len(lost)}
    return plan


# ---------------------------------------------------------------- erasure / exit
def fan_out(register: Register, holdings: dict):
    """holdings: asset -> rows held for one subject (or one tenant).
    Returns per-asset action and the assets with rows but no answer."""
    plan, unanswered = {}, []
    for name, a in register.assets.items():
        n = holdings.get(name, 0)
        if n == 0:
            plan[name] = ("none-held", 0)
            continue
        ans = a.erasure if a.erasure in ERASURE_ANSWERS else None
        if ans is None:
            unanswered.append(name)
        plan[name] = (ans or "NO ANSWER", n)
    return plan, unanswered


def exit_drill(register: Register, holdings: dict):
    """A partner leaves: every asset with its rows must be able to find them by tenant."""
    out = {}
    for name, a in register.assets.items():
        n = holdings.get(name, 0)
        if not n:
            continue
        if a.tenant_field:
            out[name] = ("select by tenant", n)
        else:
            out[name] = ("cannot select: flush or rebuild", n)
    return out


# ---------------------------------------------------------------- evidence
CHAIN = ("rule", "enforcement point", "telemetry", "evidence", "regression test")


def coverage(controls: dict):
    """controls: policy -> {link: artefact or None}. A control is complete when
    every link from the written rule to the regression test exists."""
    rows = {}
    for p, links in controls.items():
        missing = [k for k in CHAIN if not links.get(k)]
        rows[p] = missing
    complete = sum(1 for m in rows.values() if not m)
    return complete, rows


def calendar(year: int, partners: dict):
    """The governance calendar: recurring events plus contract dates."""
    ev = []
    d = date(year, 1, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)
    while d.year == year:
        ev.append((d, "weekly", "reconciliation report"))
        d += timedelta(days=7)
    for m in range(1, 13):
        ev.append((date(year, m, 15), "monthly", "lifecycle review with privacy office"))
    for m in (1, 4, 7, 10):
        ev.append((date(year, m, 20), "quarterly", "probe, retire-by and register re-walk"))
        ev.append((date(year, m, 25), "quarterly", "erasure and exit drill"))
    for m in (3, 9):
        ev.append((date(year, m, 10), "half-yearly", "examination readiness drill"))
    ev.append((date(year, 6, 1), "annual", "inventory re-walk from the work"))
    ev.append((date(year, 11, 1), "annual", "retention periods reviewed"))
    for name, when in partners.items():
        ev.append((when - timedelta(days=90), "contract", f"{name}: exit plan refreshed"))
        ev.append((when, "contract", f"{name}: contract date"))
    return sorted(ev)
