"""contextlab.erroranalysis: open coding, axial coding, drift and the stop rule (Chapter 41).

contextlab v9.0. Standard library only. The functions here do the bookkeeping of
error analysis; the judgement (what counts as a failure, what the first failure
in a trace was) stays with the people reading the traces.

- `OpenCode`: one reader's note on one trace, first failure only.
- `Codebook`: axial categories with a definition and the phrases that place a
  note in them; `apply` returns a category or None (a note the book cannot place).
- `prevalence`, `pareto`: counts and severity-weighted counts.
- `cohen_kappa`: chance-corrected agreement between two coders.
- `saturation`: new categories per trace read, and where a no-new-category run ends.
- `tvd`, `unsupported`, `reweight`: how far an eval set's mix is from production,
  which production segments it cannot speak for, and its score on the current mix.
- `wilson_upper`: upper confidence bound on a failure rate, for exit rules.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OpenCode:
    trace_id: str
    coder: str
    acceptable: bool
    note: str                 # free text: the first thing that went wrong
    segment: str = ""


@dataclass
class Category:
    name: str
    definition: str
    phrases: tuple            # any of these in a note places it here
    severity: int = 1         # 0 = not a failure of the system
    fix_owner: str = ""


@dataclass
class Codebook:
    version: str
    categories: list = field(default_factory=list)

    def apply(self, note: str):
        low = note.lower()
        for c in self.categories:
            if any(p in low for p in c.phrases):
                return c.name
        return None

    def severity(self, name):
        return next((c.severity for c in self.categories if c.name == name), 1)


def prevalence(labels):
    n = len(labels)
    return {k: v / n for k, v in Counter(labels).most_common()} if n else {}


def pareto(labels, book: Codebook):
    """(category, count, severity-weighted count), heaviest first."""
    cnt = Counter(l for l in labels if l is not None)
    rows = [(k, v, v * book.severity(k)) for k, v in cnt.items()]
    return sorted(rows, key=lambda r: (-r[2], -r[1], r[0]))


def cohen_kappa(a, b):
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def saturation(labels, run=30):
    """Cumulative distinct categories after each trace, and the index (1-based)
    at which `run` consecutive traces have added nothing new (None if never)."""
    seen, curve, last_new, stop = set(), [], 0, None
    for i, l in enumerate(labels, 1):
        if l is not None and l not in seen:
            seen.add(l); last_new = i
        curve.append(len(seen))
        if stop is None and i - last_new >= run:
            stop = i
    return curve, stop


def tvd(p: dict, q: dict):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def unsupported(eval_counts: dict, prod_mix: dict, min_items=20):
    """Production share carried by segments with fewer than `min_items` eval items."""
    return {k: s for k, s in prod_mix.items() if eval_counts.get(k, 0) < min_items and s > 0}


def reweight(seg_scores: dict, prod_mix: dict):
    """Eval score re-weighted to the production mix over the segments it covers,
    with the share it cannot cover reported beside it."""
    covered = {k: s for k, s in prod_mix.items() if k in seg_scores}
    mass = sum(covered.values())
    score = sum(seg_scores[k] * s for k, s in covered.items()) / mass if mass else float("nan")
    return score, 1.0 - mass


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + spread) / (1 + z * z / n)
