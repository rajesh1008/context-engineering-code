"""Chapter 33 stand-in: a long-running agent under three context-management families.

Note: deterministic stand-in. No model runs. An episode is a 70-step
subrogation-file run: tool results of 400 to 1,800 tokens, a claims-system
hold that arrives inside a tool result early on, an action late in the run
that the hold forbids, and six facts needed for the final demand. Survival
odds per compaction are stated constants (0.94 for a fact, 0.64 for a
constraint, as in Chapter 23's stand-in); a dropped constraint is broken
40% of the time when its action comes up. These are assumptions.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

STEPS, SUB = 70, 10              # steps per episode; steps per sub-task
THRESH, KEEP_RECENT = 32_000, 6_000
P_FACT, P_RULE, P_ACTIVE = 0.94, 0.64, 0.60
P_BREAK, P_REFETCH = 0.40, 0.97
SUMMARY, DIGEST, STUB, SYSTEM = 2_000, 400, 30, 1_500


@dataclass
class Episode:
    sizes: list
    rule_step: int
    action_step: int
    fact_steps: list
    attack: bool


def episodes(n: int, seed: int = 33, attack_share: float = 0.0) -> list[Episode]:
    out = []
    for i in range(n):
        r = random.Random(f"{seed}-ep-{i}")
        out.append(Episode([r.randrange(400, 1800) for _ in range(STEPS)], r.randrange(4, 14),
                           r.randrange(45, 66), sorted(r.sample(range(2, 60), 6)),
                           r.random() < attack_share))
    return out


def run(ep: Episode, family: str, pin: bool = False, offload: bool = False, seed: str = "") -> dict:
    """family: full, threshold, on_demand, trim, isolate. `pin` keeps the hold
    outside the summarised region; `offload` writes facts to a progress file,
    so a compaction turns a fact into a stub that can be read back."""
    key = f"{seed}-{family}-{ep.rule_step}-{ep.action_step}-{sum(ep.sizes)}"
    r, rf = random.Random(key + "-rule"), random.Random(key + "-fact")
    window, peak, total, rule, violated = SYSTEM, 0, 0, None, False
    facts, refetch = {}, 0
    recent = []                                   # (step, tokens) still verbatim
    for t in range(STEPS):
        size = ep.sizes[t]
        if family == "isolate":                   # sub-agent reads; parent gets a digest
            size = DIGEST // SUB
        if t == ep.rule_step:
            p_carry = 0.7 if family == "isolate" else 1.0
            rule = "pinned" if pin else ("live" if r.random() < p_carry else None)
        if t in ep.fact_steps:
            facts[t] = "live" if (family != "isolate" or rf.random() < 0.95) else None
        recent.append((t, size + 150))
        window += size + 150
        if family == "trim":                      # clear all but the last five results
            if len(recent) > 5:
                old = recent.pop(0)
                window -= old[1] - STUB
                if rule == "live" and old[0] == ep.rule_step:
                    rule = None
                if old[0] in facts and facts[old[0]] == "live":
                    facts[old[0]] = "stub"
        compact = (family == "threshold" and window >= THRESH) or \
                  (family == "on_demand" and window >= THRESH and (t + 1) % SUB == 0)
        if compact:
            p_rule = 0.1 if ep.attack and t >= ep.rule_step else P_RULE
            if rule == "live" and r.random() >= p_rule:
                rule = None
            for k, v in facts.items():
                active = family == "threshold" and k > t - (t % SUB)
                if v == "live" and rf.random() >= (P_ACTIVE * P_FACT if active else P_FACT):
                    facts[k] = "stub" if offload else None
            window = SYSTEM + SUMMARY + KEEP_RECENT
            recent = []
        if t == ep.action_step and rule is None and r.random() < P_BREAK:
            violated = True
        peak = max(peak, window)
        total += window
    ok = True
    for k, v in facts.items():
        if v is None:
            ok = False
        elif v == "stub":
            refetch += 1
            total += ep.sizes[k]
            ok = ok and rf.random() < P_REFETCH
    if family == "full":
        ok = ok and rf.random() >= max(0.0, (window - 40_000) / 400_000)
    return dict(peak=peak, total=total, violated=violated, ok=ok, refetch=refetch)


def summarise(eps: list[Episode], family: str, pin: bool = False, offload: bool = False) -> dict:
    rs = [run(e, family, pin, offload) for e in eps]
    n = len(rs)
    return dict(family=family, pin=pin, offload=offload, peak=round(sum(x["peak"] for x in rs) / n),
                total=round(sum(x["total"] for x in rs) / n),
                violations=sum(x["violated"] for x in rs), success=round(100 * sum(x["ok"] for x in rs) / n, 1))
