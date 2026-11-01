"""Chapter 40 lab: the test pyramid for Lakeshore's context layer.

Note: deterministic stand-in; see lakeshore40.py for the components, the fault effects
and every stated parameter. Run from 04-code/: python3 ch40/lab.py > ch40/output.txt
"""
import json
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import lakeshore40 as L                                                  # noqa: E402
from contextlab.pyramid import (LAYERS, STREAMS, Check, by_layer, cost,   # noqa: E402
                                first_red, ladder, lint, matrix)

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
SEEDS = range(40)
CHECKS = L.suite.checks
BY = {c.name: c for c in CHECKS}
SMOKE_SET = L.L3SET[:20]
SMOKE_USD = 20 * L.DET_USD + sum(c.n_sent for c in SMOKE_SET) * L.JUDGE_USD


def section(t):
    print(f"\n== {t}")


# ---- the judge used as a unit test (the failure injection) ----
def smoke_pass(s, run, need=17):
    return sum(L.faith(s, SMOKE_SET, ("u", run))) >= need


def smoke_retry(s, run, tries=3):
    return any(smoke_pass(s, (run, t)) for t in range(tries))


SMOKE = Check("judge smoke: 20 determinations >= 85%", 1, "grounding", "platform",
              lambda s, seed: smoke_retry(s, seed), seconds=180, dollars=round(SMOKE_USD, 4),
              deterministic=False, components=("prompt", "model"))


def rate(fn, runs=400):
    return sum(1 for k in range(runs) if fn(k)) / runs


flake = {}
for label, s in (("clean", L.System()), ("F4", L.system_with("F4")), ("F9", L.system_with("F9"))):
    once = rate(lambda k: not smoke_pass(s, k))
    retried = rate(lambda k: not smoke_retry(s, k))
    flake[label] = (once, retried)
# the fix: a deterministic layer-one test for the September incident
def keeps_schedule(s, seed):
    sp = [("g", "governing", 600, .9), ("sch", "schedule", 400, .2), ("r1", "related", 400, .8)]
    return "sch" in [x[0] for x in L.budget(s, sp, 1100)]


FIX = Check("budgeter keeps the schedule span", 1, "evidence", "platform", keeps_schedule,
            components=("budgeter",))
FIX_F4 = (FIX.run(L.System()).passed, FIX.run(L.system_with("F4")).passed)
runs_per_commit = 1 + flake["clean"][0] + flake["clean"][0] ** 2
smoke_per_commit = runs_per_commit * SMOKE_USD

# ---- the inventory before Chapter 40 (the ice-cream cone) ----
BEFORE_L1 = ["Chapter 2 smoke tests (ten, run as a block)", "budget never exceeded", "edition in force on date of loss",
             "cache key names the watermark", "direct-book span visible to every tenant"]
BEFORE = [BY[n] for n in BEFORE_L1] + [SMOKE, BY["end-to-end: wrong-clause on 1,000"],
                                       BY["production: 50 determinations reviewed by adjusters"]]
BEFORE_RUNGS = [("every commit", (1,), 0), ("release replay", (4,), 72),
                ("production sample", (5,), 72 + 168)]
LADDER_RUNGS = [("every commit", (1,), 0), ("merge", (2,), 2),
                ("release candidate, nightly", (3, 4), 14),
                ("shadow week", (5,), 14 + 168)]


def climb(pool, rungs, s, seed=0):
    spent = []
    for name, layers, hours in rungs:
        for c in [c for c in pool if c.layer in layers]:
            spent.append(c)
            if not c.run(s, seed).passed:
                k = cost(spent)
                return {"rung": name, "check": c.name, "hours": hours,
                        "minutes": round(k["seconds"] / 60, 1) if c.layer < 5 else None,
                        "dollars": round(k["dollars"], 2)}
    return {"rung": "never", "check": None, "hours": None, "minutes": None,
            "dollars": round(cost(spent)["dollars"], 2)}


def catch_rate(c, s):
    if c.deterministic:
        return 0.0 if c.run(s, 0).passed else 1.0
    return sum(1 for k in SEEDS if not c.run(s, k).passed) / len(SEEDS)


MATRIX = {f: {n: max(catch_rate(c, L.system_with(f)) for c in L.suite.layer(n))
              for n in LAYERS} for f in L.FAULTS}
FALSE_ALARM = {c.name: catch_rate(c, L.System()) for c in CHECKS if not c.deterministic}
PROD_ORDER = sorted(L.suite.layer(5), key=lambda c: c.seconds)       # the hourly monitor first


def first(f):
    s = L.system_with(f)
    if f == "F12":                       # a production event: no candidate to gate
        return (climb(BEFORE, [("production sample", (5,), 168)], s),
                climb(PROD_ORDER, [("production monitor", (5,), 1)], s))
    runs = [climb(BEFORE, BEFORE_RUNGS, s, k) for k in SEEDS]
    b = max(runs, key=lambda r: sum(x["rung"] == r["rung"] for x in runs))
    b["share"] = sum(x["rung"] == b["rung"] for x in runs) / len(runs)
    return b, climb(CHECKS, LADDER_RUNGS, s)


FIRST = {f: first(f) for f in L.FAULTS}
PAIR = L.System(frozenset({"F1", "F6"}))
PAIR_FAILS = [(c.layer, c.name) for c in CHECKS if not c.run(PAIR, 0).passed]
LAYERCOST = by_layer(CHECKS)

# a two-week release: 150 commits, 25 merges, 3 candidate nights, one shadow week
REL = {"every commit": (150, LAYERCOST[1]["dollars"]), "merge": (25, LAYERCOST[2]["dollars"]),
       "release candidate": (3, LAYERCOST[3]["dollars"] + LAYERCOST[4]["dollars"]),
       "shadow week": (1, LAYERCOST[5]["dollars"])}
REL_USD = {k: n * d for k, (n, d) in REL.items()}
BEFORE_REL = {"every commit": 150 * smoke_per_commit, "release replay": 2 * L.DET_USD * 1000,
              "production sample": 0.0}
HUMAN_H = sum(L.HUMAN_MINUTES.values()) / 60
CLEAN = ladder(L.suite, L.System(), [r[1] for r in LADDER_RUNGS])
F9RUN = ladder(L.suite, L.system_with("F9"), [r[1] for r in LADDER_RUNGS])
LINT_NOW = lint(CHECKS)
LINT_BEFORE = lint(BEFORE)
SELECT = [c.name for c in L.suite.select({"budgeter"}) if c.layer == 2]


def main():
    section("1. Inventory before: checks per layer")
    for n in LAYERS:
        cs = [c for c in BEFORE if c.layer == n]
        print(f"L{n} {LAYERS[n]:22s} {len(cs):>2}  " + "; ".join(c.name for c in cs))
    print("lint:", LINT_BEFORE)

    section("2. The pyramid now: checks, time and spend per run of each layer")
    for n, k in LAYERCOST.items():
        t = f"{k['seconds'] / 60:.1f} min" if n < 5 else "one week"
        print(f"L{n} {LAYERS[n]:22s} {k['checks']:>2} checks  {t:>9}  US${k['dollars']:.2f}")
    print(f"L5 human review: {HUMAN_H:.0f} adjuster hours a week")
    print("lint:", LINT_NOW or "clean")

    section("3. Laura's question: the 2025 failure pair (F1 + F6) on today's pipeline")
    for n, name in PAIR_FAILS:
        print(f"L{n}  {name}")
    print("first red on the ladder:", FIRST["F1"][1]["check"], "| also:", BY["endorsement on the declarations survives dedupe"].name)

    section("4. A judge used as a unit test: 20 determinations, pass at 17")
    print(f"cost per run US${SMOKE_USD:.3f}, about {runs_per_commit:.2f} runs per commit, "
          f"US${smoke_per_commit:.2f} per commit")
    for k, (once, retried) in flake.items():
        print(f"{k:5s} fails once {once:6.1%}   fails after three tries {retried:6.1%}")
    print("lint on the before suite flags:", [r for n, r in LINT_BEFORE if n == SMOKE.name])
    print(f"the fix, '{FIX.name}': clean passes {FIX_F4[0]}, F4 passes {FIX_F4[1]}, "
          f"{FIX.seconds:.0f} s, US${FIX.dollars:.2f}")

    section("5. Fault catch rate by layer (deterministic 1/0; else share of 40 seeds)")
    print("fault " + " ".join(f"   L{n}" for n in LAYERS) + "  description")
    for f, row in MATRIX.items():
        print(f"{f:5s} " + " ".join(f"{row[n]:5.2f}" for n in LAYERS) + f"  {L.FAULTS[f]}")
    for name, r in FALSE_ALARM.items():
        print(f"false alarm on a clean candidate: {name}: {r:.1%}")

    section("6. Streams by layers: number of checks")
    mx = matrix(CHECKS)
    print("stream      " + " ".join(f"L{n}" for n in LAYERS))
    for s in STREAMS:
        print(f"{s:11s} " + " ".join(f"{mx[(s, n)]:>2}" for n in LAYERS))

    section("7. Cost of one two-week release")
    for k, (n, d) in REL.items():
        print(f"{k:18s} {n:>4} runs x US${d:6.2f} = US${REL_USD[k]:7.2f}")
    print(f"total US${sum(REL_USD.values()):.2f} plus {2 * HUMAN_H:.0f} adjuster hours")
    print("before: " + ", ".join(f"{k} US${v:.2f}" for k, v in BEFORE_REL.items())
          + f", total US${sum(BEFORE_REL.values()):.2f}")

    section("8. The gate ladder")
    print("clean candidate: passed all", len(LADDER_RUNGS), "rungs,",
          f"spent US${CLEAN['dollars']:.2f}")
    print("F9 candidate: stops at rung", F9RUN["rung"], F9RUN["failed"],
          f"spent US${F9RUN['dollars']:.2f}")
    print("merge-time selection for a budgeter change:", SELECT)

    section("9. Time to first red: before and on the ladder")
    for f, (b, a) in FIRST.items():
        bt = "never" if b["hours"] is None else f"{b['rung']} (+{b['hours']} h)"
        if b.get("share", 1) < 1:
            bt += f" (smoke red {1 - b['share']:.0%})"
        at = "never" if a["hours"] is None else f"{a['rung']} (+{a['hours']} h)"
        am = f"{a['minutes']} min" if a["minutes"] is not None else "-"
        print(f"{f:4s} before: {bt:32s} ladder: {at:34s} {am:>9} US${a['dollars']:.2f}")

    json.dump({"matrix": MATRIX, "flake": flake, "layercost": LAYERCOST,
               "rel": REL_USD, "before_rel": BEFORE_REL,
               "first": {f: [b, a] for f, (b, a) in FIRST.items()},
               "streams": {f"{s}|{n}": v for (s, n), v in matrix(CHECKS).items()}},
              open(OUT / "lab40.json", "w"), indent=1)


if __name__ == "__main__":
    main()
