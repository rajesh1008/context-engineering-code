"""contextlab.operations: running the context layer for a year (Chapter 45).

contextlab v9.8, standard library only. Six small tools for the work that only
appears once a context layer has been live for months:

* Job and Calendar: every scheduled or event-driven job is an unattended loop,
  and a loop needs a machine-checkable stop, persisted state, a verifier that
  is not the assembler, a bounded cost and a named human on the loop.
* TunedParam and stale_params: a tuned number records the conditions it was
  tuned under, so growth beyond a limit raises a re-tune instead of a surprise.
* headroom: the share of cited governing spans that came from the bottom of the
  candidate pool, a label-free leading indicator of corpus growth.
* Artefact and Inventory: the bill of materials of everything derived from one
  embedding model, which turns a migration runbook into a generated plan.
* replay_compare and recalibrate_floor: compare an old and a new stack on the
  same past determinations, slice by slice, and move score thresholds by
  quantile rather than by hand.
* mix_adjusted: a directly standardised rate, so a seasonal change in the claim
  mix is not read as drift.
* Instruction and prune: an instruction ledger with a rationale and a test per
  line, so lines can be removed one at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

VERSION = "9.8"
LOOP_PROPERTIES = ("stop", "state", "verifier", "budget", "human")


# --------------------------------------------------------------------- loops
@dataclass
class Job:
    name: str
    cadence: str                 # daily | weekly | monthly | quarterly | event
    assembler: str               # the component that does the work
    stop: str = ""               # machine-checkable stop condition
    state: str = ""              # where progress is persisted
    verifier: str = ""           # who checks the output (must differ from assembler)
    budget: str = ""             # bound on tokens, money or time
    human: str = ""              # named person on the loop, and when they are paged

    def gaps(self) -> list[str]:
        out = [p for p in LOOP_PROPERTIES if not getattr(self, p)]
        if self.verifier and self.verifier == self.assembler:
            out.append("verifier is the assembler")
        return out


PER_MONTH = {"daily": 30, "weekly": 4, "monthly": 1, "quarterly": 1 / 3, "event": 0}


@dataclass
class Calendar:
    jobs: list[Job] = field(default_factory=list)

    def audit(self) -> dict[str, list[str]]:
        return {j.name: j.gaps() for j in self.jobs if j.gaps()}

    def runs_per_year(self) -> int:
        return round(sum(PER_MONTH[j.cadence] * 12 for j in self.jobs))


# --------------------------------------------------------- tuned parameters
@dataclass
class TunedParam:
    name: str
    value: float
    tuned_at: dict               # e.g. {"corpus_pages": 40000, "embed_model": "open-base@1.3"}
    growth_limit: float = 0.25   # relative growth of any numeric condition that forces a re-tune


def stale_params(params: list[TunedParam], now: dict) -> list[tuple[str, str]]:
    """Return (name, reason) for every parameter whose tuning conditions no longer hold."""
    out = []
    for p in params:
        for key, then in p.tuned_at.items():
            cur = now.get(key)
            if cur is None:
                continue
            if isinstance(then, (int, float)) and not isinstance(then, bool):
                growth = (cur - then) / then if then else float("inf")
                if abs(growth) > p.growth_limit:
                    out.append((p.name, f"{key} {then:,} -> {cur:,} ({growth:+.0%})"))
            elif cur != then:
                out.append((p.name, f"{key} {then} -> {cur}"))
    return out


def headroom(ranks: list[int], k: int, tail: float = 0.25) -> float:
    """Share of in-pool governing spans ranked in the last `tail` of a pool of size k."""
    hits = [r for r in ranks if r <= k]
    if not hits:
        return 0.0
    edge = (1 - tail) * k
    return sum(r > edge for r in hits) / len(hits)


# ------------------------------------------------------- derived artefacts
@dataclass
class Artefact:
    name: str
    kind: str                    # index | adapter | threshold | baseline | probe
    derived_from: str            # embedding model id it was built or calibrated on
    rebuild: str                 # how to bring it onto a new model
    owner: str = "context platform"
    order: int = 5               # indexes first, thresholds after the replay


class Inventory:
    """Everything that depends on an embedding model, so a migration plan can be generated."""

    def __init__(self, items: list[Artefact] | None = None):
        self.items: list[Artefact] = list(items or [])

    def register(self, a: Artefact) -> None:
        self.items.append(a)

    def dependants(self, model: str) -> list[Artefact]:
        return [a for a in self.items if a.derived_from == model]

    def plan(self, old: str, new: str) -> list[str]:
        steps = sorted(self.dependants(old), key=lambda a: (a.order, a.name))
        return [f"{a.name} ({a.kind}): {a.rebuild} [{old} -> {new}]" for a in steps]

    def left_behind(self, old: str, done: set[str]) -> list[str]:
        return [a.name for a in self.dependants(old) if a.name not in done]


# ---------------------------------------------------------------- replays
def replay_compare(old: list[dict], new: list[dict], slice_key: str,
                   tolerances: dict[str, float]) -> dict:
    """Compare two runs over the same determinations, slice by slice.

    Each record is a dict with `slice_key` and numeric or boolean metrics. A tolerance
    is the largest allowed absolute change in a slice mean. Returns per-slice deltas
    and the list of (slice, metric, delta) that block the change.
    """
    assert len(old) == len(new), "replay must cover the same determinations"
    by: dict[str, list[tuple[dict, dict]]] = {}
    for a, b in zip(old, new):
        assert a["id"] == b["id"], "replay records out of order"
        by.setdefault(a[slice_key], []).append((a, b))
    deltas, blocks = {}, []
    for s, pairs in sorted(by.items()):
        n = len(pairs)
        deltas[s] = {"n": n}
        for m, tol in tolerances.items():
            d = sum(float(b[m]) for _, b in pairs) / n - sum(float(a[m]) for a, _ in pairs) / n
            deltas[s][m] = d
            if abs(d) > tol:
                blocks.append((s, m, round(d, 4)))
    return {"deltas": deltas, "blocks": blocks, "passed": not blocks}


def quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def recalibrate_floor(old_scores: list[float], new_scores: list[float], floor: float) -> float:
    """Move a score floor so it cuts the same share of the replay under the new model."""
    share = sum(s < floor for s in old_scores) / len(old_scores)
    return round(quantile(new_scores, share), 3)


# --------------------------------------------------------------- drift
def mix_adjusted(rate_by_stratum: dict[str, float], ref_weights: dict[str, float]) -> float:
    """Directly standardised rate: this period's stratum rates on a reference mix."""
    tot = sum(ref_weights.values())
    return sum(rate_by_stratum.get(s, 0.0) * w for s, w in ref_weights.items()) / tot


# -------------------------------------------------------- instructions
@dataclass
class Instruction:
    id: str
    text: str
    added: str
    rationale: str = ""          # why the line exists; empty means nobody remembers
    test: str = ""               # the check that fails if the line is removed and still needed


def prune(lines: list[Instruction], still_needed) -> tuple[list[Instruction], list[str], int]:
    """Remove documented lines whose test passes without them.

    `still_needed(line)` runs that line's test with the line removed and returns True
    if the test fails. Undocumented lines are kept: removing one safely would need
    every subset of them tested, 2**n runs. Returns (kept, removed ids, undocumented).
    """
    kept, removed = [], []
    undocumented = 0
    for ln in lines:
        if not (ln.rationale and ln.test):
            kept.append(ln)
            undocumented += 1
        elif still_needed(ln):
            kept.append(ln)
        else:
            removed.append(ln.id)
    return kept, removed, undocumented
