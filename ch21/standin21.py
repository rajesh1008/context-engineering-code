"""Deterministic stand-ins for the Chapter 21 lab. No model is called.

sim-consolidate  turns episodes into notes and corrections into lessons. On a
                 review sample it is faithful most of the time; it drops a
                 qualifier, merges two people or widens a scope at fixed rates.
sim-cause        reads the evidence for a water claim. Where the evidence is
                 close ("open"), a lesson in the memory block that names a
                 cause adds weight to that cause.
sim-letter       drafts the letter; when the memory block holds a policy number
                 that disagrees with the live lookup, it uses the remembered one
                 at a fixed rate.
Every draw is a hash of the stand-in's name and the item, so runs are identical.
"""
from corpus21 import h

OPEN = 0.35          # share of water claims whose evidence is close between two causes
ACCEPT = 0.90        # share of proposed causes an adjuster confirms without reopening the file
STALE_USE = 0.30     # chance the letter uses a remembered number over the live one
REVIEW = {"faithful": 0.925, "dropped qualifier": 0.045, "merged two people": 0.01,
          "widened scope": 0.02}


def consolidate_note(subject, items):
    """Many episodic records about one subject become one semantic note per field."""
    out = {}
    for r in sorted(items, key=lambda r: r.written):
        if r.field:
            out[r.field] = r            # the latest statement wins; the older is superseded
    return out


def review(nid):
    x, acc = h("sim-consolidate", nid), 0.0
    for outcome, p in REVIEW.items():
        acc += p
        if x < acc:
            return outcome
    return "faithful"


def lesson(text, scope, cause):
    """sim-consolidate on an adjuster correction. Long notes lose their qualifiers."""
    if cause and len(text.split()) > 30:
        return (f"{'Wisconsin' if scope[1] == 'WI' else scope[1]} basement water: check for a "
                f"frozen supply line first, and code the loss as freezing unless back-up "
                f"residue is shown at the drain.", scope, cause)
    return text, scope, None


def cause_reading(did, true_cause, lessons):
    """Return the cause the determination proceeds on, and whether a lesson moved it."""
    if true_cause not in ("freezing", "water_backup", "mould"):
        return true_cause, False
    named = [c for _, c in lessons if c]
    if not named or h("sim-cause", did, "open") >= OPEN:
        return true_cause, False
    pick = named[0]
    if pick != true_cause and h("adjuster", did, "accept") < ACCEPT:
        return pick, True
    return true_cause, False


def letter_number(did, field, live, remembered):
    if remembered is None or remembered == live:
        return live
    return remembered if h("sim-letter", did, field) < STALE_USE else live
