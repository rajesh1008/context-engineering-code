"""contextlab v12.0: finishing a context layer (Chapter 51).

Standard library only. Five small tools for the last twenty percent:
- EdgeCase and prioritise: an edge-case backlog ranked by expected harm that
  escapes the diagnostics, per day of work, not by frequency.
- Check and run_checklist: a hardening checklist whose items are code, so the
  build can fail on them.
- Runbook and drill: runbooks with a trigger, an owner and steps, timed in a drill.
- Diagnostic and compare: run a candidate model against the same diagnostics as
  production and report what each diagnostic saw.
- targets: the scoreboard read against the sponsor's targets.
"""
from dataclasses import dataclass, field

VERSION = "12.0"


@dataclass(frozen=True)
class EdgeCase:
    key: str
    name: str
    per_10k: float      # determinations affected per 10,000
    severity: int       # 1 (cosmetic) to 5 (wrong payment or a regulatory finding)
    caught: float       # share the existing diagnostics catch before harm, 0..1
    days: float         # engineering days to fix and test

    def escaped_harm(self):
        return self.per_10k * self.severity * (1 - self.caught)

    def priority(self):
        return self.escaped_harm() / self.days


def prioritise(cases, by="priority"):
    keys = {"priority": lambda c: -c.priority(),
            "frequency": lambda c: -c.per_10k,
            "harm": lambda c: -c.escaped_harm()}
    return sorted(cases, key=lambda c: (keys[by](c), c.key))


@dataclass(frozen=True)
class Check:
    key: str
    area: str
    text: str
    test: object        # callable(config) -> bool

    def run(self, config):
        try:
            return bool(self.test(config))
        except (KeyError, TypeError):
            return False    # a check that cannot read its evidence fails


def run_checklist(checks, config):
    results = {c.key: c.run(config) for c in checks}
    failed = [c for c in checks if not results[c.key]]
    return results, failed


@dataclass(frozen=True)
class Runbook:
    key: str
    trigger: str
    owner: str
    steps: tuple        # (step, minutes) pairs
    rollback: str

    def minutes(self):
        return sum(m for _, m in self.steps)


def drill(runbook, discovery_minutes, has_runbook=True):
    """Minutes from alert to contained: discovery plus the steps.

    Without a runbook the responder first works out the steps; the stand-in
    charges the discovery time once per step."""
    if has_runbook:
        return discovery_minutes + runbook.minutes()
    return runbook.minutes() + discovery_minutes * len(runbook.steps)


@dataclass(frozen=True)
class Diagnostic:
    key: str
    measure: object     # callable(results) -> float
    floor: float        # alarm when the measure falls below this
    higher_is_better: bool = True

    def alarm(self, results):
        v = self.measure(results)
        return (v < self.floor) if self.higher_is_better else (v > self.floor), v


def compare(diagnostics, baseline, candidate):
    rows = []
    for d in diagnostics:
        a_alarm, a = d.alarm(baseline)
        b_alarm, b = d.alarm(candidate)
        rows.append({"diagnostic": d.key, "production": a, "candidate": b,
                     "alarm": b_alarm and not a_alarm})
    return rows


def targets(scoreboard, goals):
    """goals: {row: (comparison, threshold)}; comparison is '<' or '>='."""
    out = {}
    for row, (op, t) in goals.items():
        v = scoreboard[row]
        out[row] = {"value": v, "target": f"{op} {t}",
                    "met": v < t if op == "<" else v >= t}
    return out
