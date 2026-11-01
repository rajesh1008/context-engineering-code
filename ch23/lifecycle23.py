"""Chapter 23 stand-in: Chapter 22's full-mix replay with the memory lifecycle switched on.

Imports Chapter 22's stand-in unchanged (../ch22/lakeshore22.py, ../ch22/replay22.py) and
changes only the store it starts from and the read budget. Deterministic; no language model runs.
"""
from __future__ import annotations

import pathlib
import sys
from datetime import date

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

sys.path.insert(0, str(HERE.parent / "ch22"))
import lakeshore22 as L                       # noqa: E402
import replay22 as R                          # noqa: E402
from contextlab.memguard import MemoryRecord as M, Origin as O   # noqa: E402
from contextlab.statestore import (LifecyclePolicy, RetentionRule, expire, revalidate,  # noqa: E402
                                   sweep)

TODAY = L.DAYS[0]                    # Monday 5 October 2026: the policy runs before the replay
POLICY = LifecyclePolicy({
    "policy":   RetentionRule("policy", None, revalidate_days=180),  # earlier-claim records
    "claimant": RetentionRule("claimant", 730),                      # notes from closed claims
    "adjuster": RetentionRule("adjuster", 365),
    "team":     RetentionRule("team", 730, revalidate_days=180),
}, read_budget=6)
LEGAL_HOLDS: frozenset[str] = frozenset()     # policies under litigation hold (none in the mix)


def tenant_of(policy: str) -> str:
    """Two bank-partner programmes write policies on Lakeshore paper."""
    x = L.u(policy, "tenant")
    return "partner-a" if x < 0.11 else "partner-b" if x < 0.18 else "lakeshore"


POINTER = "policy-admin://{policy}/schedule#HO 04 95"


def pointer_text(policy: str) -> str:
    return (f"Earlier-claim record for {policy}: the water back-up endorsement is read from "
            f"{POINTER.format(policy=policy)} at use; no value is stored here.")


def is_pointer(m) -> bool:
    return m.scope == "policy" and m.asserts is None and "policy-admin://" in m.text


def resolve_pointer(m, policy_system: dict):
    """Re-read the endorsement from the policy system when the packet is built."""
    if not is_pointer(m):
        raise ValueError(f"{m.mid} is not a pointer")
    return policy_system[m.subject]


def record_truth(dets):
    return {d.policy: d.ho0495 for d in dets}


def lifecycle_store(dets, retention=True, revalidation=True, report=None):
    store = _ORIGINAL(dets)
    truth = record_truth(dets)
    before = sweep(store, TODAY, POLICY)
    out = {"before": before}
    if revalidation:
        def make(old, value):
            # The replacement is a pointer, not a copy: it names where the endorsement
            # lives and stores no value, so nothing in memory can go stale. The packet
            # builder resolves it against the policy system at use (resolve_pointer).
            return M(old.mid + "-r", "policy", old.subject, pointer_text(old.subject),
                     date(2026, 10, 2), (O("record", old.subject),), asserts=None)
        out["revalidated"] = revalidate(store, TODAY, POLICY, truth.get, make)
    if retention:
        out["expired"] = len(expire(store, TODAY, POLICY, LEGAL_HOLDS))
    out["after"] = sweep(store, TODAY, POLICY)
    if report is not None:
        report.update(out)
    return store


_ORIGINAL = L.seed_store


def replay(lifecycle: bool, k_max: int = L.K_MAX, **kw):
    """Chapter 22's close configuration (write gate on), optionally with the lifecycle."""
    L.seed_store = (lambda dets: lifecycle_store(dets, **kw)) if lifecycle else _ORIGINAL
    try:
        d, s, l, o, ev = R.run(gate=True, k_max=k_max)
    finally:
        L.seed_store = _ORIGINAL
    return d, s, l, o, ev


def lookup_saved(dets, log) -> float:
    """Chapter 22's measure: seconds of lookup a repeat-claimant packet saves the adjuster."""
    rep = [d for d in dets if d.repeat]
    tot = 0
    for d in rep:
        rs = log.read_by(d.did)
        tot += (45 if any(m.startswith("M-P") for m in rs) else 0) + 20 * min(2, sum(m.startswith("M-C") for m in rs))
    return tot / len(rep)
