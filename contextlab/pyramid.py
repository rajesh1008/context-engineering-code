"""contextlab.pyramid (v8.8, Chapter 40): tag every check on the context layer with a
layer, a verification stream and an owner, run the checks by layer with their cost,
climb a release's gate ladder cheapest rung first, and lint the suite for checks that
sit on the wrong layer.

Standard library only. A check is a function of (system, seed) that returns True when
the system passes. Costs are declared per run, so a suite can be costed before it runs.
"""
from dataclasses import dataclass, field

LAYERS = {1: "unit", 2: "gold-set eval", 3: "judge-scored eval",
          4: "end-to-end", 5: "shadow and production"}
STREAMS = ("evidence", "grounding", "entitlement", "freshness", "cost", "human")


@dataclass(frozen=True)
class Check:
    name: str
    layer: int
    stream: str
    owner: str
    fn: object
    seconds: float = 1.0          # declared wall-clock per run
    dollars: float = 0.0          # declared model spend per run, US$
    deterministic: bool = True
    sample: int = 0               # items scored, for layers 2 to 5
    components: tuple = ()        # what a change must touch to select this check

    def run(self, system, seed=0):
        return Result(self, bool(self.fn(system, seed)))


@dataclass(frozen=True)
class Result:
    check: Check
    passed: bool


@dataclass
class Suite:
    checks: list = field(default_factory=list)

    def add(self, name, layer, stream, owner, **kw):
        def wrap(fn):
            self.checks.append(Check(name, layer, stream, owner, fn, **kw))
            return fn
        return wrap

    def layer(self, n):
        return [c for c in self.checks if c.layer == n]

    def select(self, changed):
        """Checks whose components intersect the changed ones, plus untagged ones."""
        changed = set(changed)
        return [c for c in self.checks if not c.components or changed & set(c.components)]

    def run(self, system, layers=(1, 2, 3, 4, 5), seed=0, checks=None):
        pool = checks if checks is not None else self.checks
        return [c.run(system, seed) for c in pool if c.layer in layers]


def cost(results_or_checks):
    items = [getattr(x, "check", x) for x in results_or_checks]
    return {"checks": len(items), "seconds": sum(c.seconds for c in items),
            "dollars": round(sum(c.dollars for c in items), 4)}


def by_layer(checks):
    return {n: cost([c for c in checks if c.layer == n]) for n in LAYERS}


def matrix(checks):
    """Counts of checks per (stream, layer); empty cells are decisions to make."""
    return {(s, n): sum(1 for c in checks if c.stream == s and c.layer == n)
            for s in STREAMS for n in LAYERS}


def ladder(suite, system, rungs, seed=0, order=None):
    """Climb rungs (tuples of layers) in order; stop at the first rung with a failure.
    Returns the rung reached, the failing checks and what was spent getting there."""
    spent = []
    for i, layers in enumerate(rungs):
        pool = [c for c in (order or suite.checks) if c.layer in layers]
        results = [c.run(system, seed) for c in pool]
        spent += results
        failed = [r.check.name for r in results if not r.passed]
        if failed:
            return {"rung": i, "failed": failed, **cost(spent)}
    return {"rung": None, "failed": [], **cost(spent)}


def first_red(suite, system, order, seed=0):
    """Run checks one at a time in the given order; report what was spent before
    the first failure (the time and money to the first red)."""
    spent = []
    for c in order:
        r = c.run(system, seed)
        spent.append(c)
        if not r.passed:
            return {"check": c.name, "layer": c.layer, **cost(spent)}
    return {"check": None, "layer": None, **cost(spent)}


def lint(checks, per_commit_seconds=120):
    """Rules for where a check may live. Returns (check name, rule) pairs."""
    out = []
    for c in checks:
        if c.layer == 1 and not c.deterministic:
            out.append((c.name, "L1 must be deterministic"))
        if c.layer == 1 and c.dollars > 0:
            out.append((c.name, "L1 must not spend on models"))
        if c.layer >= 2 and c.sample == 0:
            out.append((c.name, "L2+ must declare its sample size"))
        if not c.owner:
            out.append((c.name, "every check needs an owner"))
    l1 = sum(c.seconds for c in checks if c.layer == 1)
    if l1 > per_commit_seconds:
        out.append(("layer 1", f"per-commit layer takes {l1:.0f} s > {per_commit_seconds} s"))
    return out
