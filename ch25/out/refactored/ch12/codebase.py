"""Second setting: retrieval over a small SaaS billing codebase (Austin).

Deterministic stand-in. Thirty functions live in src/billing/; a vendored
copy of an older release sits in vendor/billing/ with the same names,
the same docstrings and one flipped comparison each (a 'buggy twin').
Canonical functions are reachable from the API entry points; twins are not.
"""
import hashlib
import random

VERBS = ["prorate", "apply", "compute", "validate", "renew", "refund", "charge",
         "suspend", "reactivate", "round", "convert", "schedule", "settle",
         "cap", "waive"]
NOUNS = [("invoice", "an invoice"), ("seat_count", "the seat count"),
         ("trial", "a trial account"), ("coupon", "a coupon"),
         ("tax", "sales tax"), ("credit", "account credit"),
         ("usage", "metered usage"), ("plan_change", "a mid-cycle plan change")]
CONDS = ["grace period has ended", "seat count is below the plan minimum",
         "usage exceeds the included quota", "coupon has expired",
         "customer is tax exempt", "credit balance is negative"]


def build(seed: int = 12):
    rng = random.Random(seed)
    pairs = [(v, n) for v in VERBS for n in NOUNS]
    rng.shuffle(pairs)
    docs, queries, calls = {}, [], {}
    for i, (verb, (noun, phrase)) in enumerate(pairs[:30]):
        name, cond = f"{verb}_{noun}", CONDS[i % len(CONDS)]
        err = f"BillingError: cannot {verb} {phrase} when the {cond}"
        doc = f"{verb.capitalize()} {phrase}; refuse when the {cond}."
        body = f"if limit {'>='} value: raise BillingError('{err[14:]}')"
        twin = body.replace(">=", ">")
        docs[f"src/billing/{noun}.py::{name}"] = f"def {name}(): \"{doc}\" {body}"
        docs[f"vendor/billing/{noun}.py::{name}"] = f"def {name}(): \"{doc}\" {twin}"
        gold = f"src/billing/{noun}.py::{name}"
        queries += [("nl", f"where do we {verb} {phrase} and refuse if the {cond}", gold),
                    ("identifier", f"{name}", gold), ("error", err, gold)]
    canon = [d for d in docs if d.startswith("src/")]
    for d in canon:                                 # call graph from the API
        calls[d] = rng.sample([c for c in canon if c != d], 2)
    reachable = set(canon)                          # entry points reach every src/ fn
    return docs, queries, calls, reachable


def jitter(doc_id: str) -> float:
    """Near-twins score almost identically; which one wins is arbitrary."""
    return int(hashlib.sha256(doc_id.encode()).hexdigest()[:6], 16) * 1e-12


def passes_test(doc_id: str, docs) -> bool:
    """Execution check: the regression test fails on the flipped comparison."""
    return ">=" in docs[doc_id]
