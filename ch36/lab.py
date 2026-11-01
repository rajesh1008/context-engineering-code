"""Chapter 36 lab: optimise one module, then keep the result honest.

Run from 04-code:  python3 ch36/lab.py   (standard library only, about 10 s)
"""
import json
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

from contextlab.optimise import (Gate, Manifest, Playbook, Prompt, bootstrap_fewshot,  # noqa: E402
                                 diff, digest, evaluate, evolve, paired_ci, reflect,
                                 transitions)
from lakeshore36 import (HAND_WRITTEN, METRIC_VERSION, POOL, as_demo, faithfulness,  # noqa: E402
                         is_narrow, make, needs_human, program, propose, rule_text,
                         tokens, wrong_clause)

DEV_SEED, OUT = 26, os.path.join(HERE, "out")
TRAIN, HELD, FULL = make("T", 200, 1), make("H", 400, 3), make("F", 1000, 4, misses=4)
BASE = Prompt(*HAND_WRITTEN)
R = {}


def E(p, s):
    return evaluate(program, p, s, faithfulness)


def pct(x):
    return f"{100 * x:.2f}%"


def strip(p):
    return replace(p, rules=tuple(r for r in p.rules if not is_narrow(r)))


def ids(items):
    return digest([i.id for i in items])


def optimise(dev, rounds=12):
    b, hb = bootstrap_fewshot(program, faithfulness, BASE, TRAIN, dev, as_demo=as_demo)
    e, he = evolve(program, faithfulness, b, POOL, dev)
    r, hr = reflect(program, faithfulness, e, dev, propose, rounds=rounds)
    return b, e, r, hb + he + hr


def main():
    dev = make("D", 100, DEV_SEED)
    print("1. Sets and the hand-written module")
    base_dev, base_held, base_full = E(BASE, dev), E(BASE, HELD), E(BASE, FULL)
    print(f"   dev {len(dev)}  held-out {len(HELD)}  full mix {len(FULL)}  train {len(TRAIN)}")
    print(f"   faithfulness dev {pct(base_dev.mean)}  held-out {pct(base_held.mean)}  full {pct(base_full.mean)}")
    print(f"   wrong-clause full mix {wrong_clause(BASE, FULL)}/1,000   prompt tokens {tokens(BASE)}")
    R["base"] = dict(dev=base_dev.mean, held=base_held.mean, full=base_full.mean,
                     wrong=wrong_clause(BASE, FULL), tokens=tokens(BASE))

    print("2. Metric as code: what the optimiser reads back")
    for k, f in list(base_dev.feedback.items())[:3]:
        print(f"   {k}: score 0  {f['msg']} ({f['state']}, {f['form']})")
    print(f"   {len(base_dev.feedback)} failing dev items; metric {METRIC_VERSION}")

    print("3-5. Three optimisers on the dev set")
    b, e, r, hist = optimise(dev)
    for name, p in (("bootstrapped few-shot", b), ("+ evolutionary rules", e), ("+ reflective", r)):
        print(f"   {name:24s} dev {pct(E(p, dev).mean)}  held-out {pct(E(p, HELD).mean)}"
              f"  rules {len(p.rules)}  demos {len(p.demos)}  tokens {tokens(p)}")
    print(f"   candidates evaluated {len(hist)}; demos chosen {list(r.demos)}")
    print(f"   rules added {diff(BASE, r)['rules_added']}")
    R["stages"] = {n: dict(dev=E(p, dev).mean, held=E(p, HELD).mean, tokens=tokens(p))
                   for n, p in (("bootstrap", b), ("evolve", e), ("reflect", r))}
    R["history"] = [(round(s, 4), round(E(p, HELD).mean, 4)) for p, s in hist]

    print("6. Failure injection: the eval set was the optimisation set")
    narrow = [x for x in r.rules if is_narrow(x)]
    c = strip(r)
    dr, hr_, hc = E(r, dev), E(r, HELD), E(c, HELD)
    print(f"   reported on dev   {pct(base_dev.mean)} -> {pct(dr.mean)}  (+{100 * (dr.mean - base_dev.mean):.1f} points)")
    print(f"   held-out, as is   {pct(base_held.mean)} -> {pct(hr_.mean)}  ({100 * (hr_.mean - base_held.mean):+.2f})")
    print(f"   held-out, item-specific rules removed {pct(hc.mean)}  ({100 * (hc.mean - base_held.mean):+.2f})")
    print(f"   item-specific rules {len(narrow)}: {[rule_text(x) for x in narrow]}")
    ci = paired_ci(base_held, hc)
    t = transitions(base_held, hc)
    print(f"   paired 95% interval on held-out gain: {100 * ci[1]:+.2f} to {100 * ci[2]:+.2f} points")
    print(f"   item transitions held-out: {t}")
    R["inject"] = dict(dev_before=base_dev.mean, dev_after=dr.mean, held_raw=hr_.mean,
                       held_corrected=hc.mean, ci=ci, transitions=t, narrow=narrow)

    print("   38 dev draws (seeds 2-39), same optimiser:")
    scan = []
    for sd in range(2, 40):
        d = make("D", 100, sd)
        _, _, rr, _ = optimise(d)
        scan.append((sd, E(rr, d).mean - E(BASE, d).mean, E(rr, HELD).mean - base_held.mean,
                     E(strip(rr), HELD).mean - base_held.mean))
    m = [sum(x[i] for x in scan) / len(scan) for i in (1, 2, 3)]
    print(f"   mean gain dev {100 * m[0]:+.2f}  held-out as is {100 * m[1]:+.2f}  corrected {100 * m[2]:+.2f}")
    print(f"   draws where held-out fell below the hand-written prompt: {sum(1 for x in scan if x[2] < 0)} of {len(scan)}")
    R["scan"] = scan

    print("7. An optimiser allowed to edit the pinned block")
    eu, _ = evolve(program, faithfulness, BASE, POOL, dev, may_edit_pinned=True,
                   pinned_variants=[("P1", "P1r")], seed=7, generations=8)
    print(f"   winner pinned {eu.pinned} rules {eu.rules}")
    print(f"   winner held-out {pct(E(eu, HELD).mean)} vs hand-written {pct(base_held.mean)}")
    print(f"   dev {pct(E(eu, dev).mean)}  renewal-mismatch claims in dev {sum(i.renewal_mismatch for i in dev)}"
          f"  in full mix {sum(i.renewal_mismatch for i in FULL)}")
    print(f"   full-mix wrong-clause {wrong_clause(BASE, FULL)} -> {wrong_clause(eu, FULL)} per 1,000")
    forced = replace(BASE, pinned=("P1r", "P2"))
    print(f"   forced P1r: dev {pct(E(forced, dev).mean)} vs {pct(base_dev.mean)};"
          f" wrong-clause {wrong_clause(forced, FULL)}/1,000")
    R["pinned"] = dict(winner=list(eu.pinned), wrong=wrong_clause(eu, FULL),
                       forced_dev=E(forced, dev).mean, forced_wrong=wrong_clause(forced, FULL))

    print("8. A self-evolving playbook over a stream of 600 determinations")
    stream = make("S", 600, 5)
    R["ace"] = {}
    modes = {"delta, curator, human stop": (True, True, False),
             "delta, no curator": (False, True, False),
             "monolithic rewrite (240 chars)": (True, True, True),
             "delta, curator, no human stop": (True, False, False)}
    for mode, (curate, stop, rewrite) in modes.items():
        pb = Playbook({x: [1, 0] for x in BASE.rules})
        line = []
        for w in range(12):
            window = stream[50 * w:50 * (w + 1)]
            p = Prompt(BASE.pinned, pb.rules())
            fails = E(p, window).feedback
            props = []
            for k in range(min(4, len(fails))):
                cand = propose(p, fails, k)
                if cand:
                    new = [x for x in cand.rules if x not in p.rules]
                    props += [(x, True) for x in new]
            pb.delta(props, needs_human if stop else (lambda x: False),
                     keep=(lambda x: not x.startswith("K:")) if curate else (lambda x: True))
            for x in pb.bullets:
                if x.startswith("K:"):
                    pb.bullets[x][1] += 1
            if rewrite:
                pb.rewrite(240, "N4", lambda x: len(rule_text(x)))
            p = Prompt(BASE.pinned, pb.rules())
            line.append((w + 1, E(p, HELD).mean, len(pb.bullets), sum(len(rule_text(x)) for x in pb.bullets)))
        print(f"   {mode:31s} held-out after windows 1/6/12: "
              f"{pct(line[0][1])} {pct(line[5][1])} {pct(line[-1][1])}  bullets {line[-1][2]}"
              f"  queued for a human {len(pb.queued)}  final {sorted(pb.bullets)}")
        R["ace"][mode] = dict(line=line, queued=pb.queued, final=sorted(pb.bullets))

    print("9. When the hand-written prompt wins")
    small, small_dev = [], []
    for sd in range(2, 40):
        d = make("D", 20, sd)
        _, _, rr, _ = optimise(d, rounds=6)
        small.append(E(rr, HELD).mean - base_held.mean)
        small_dev.append(E(rr, d).mean - E(BASE, d).mean)
    big = [x[2] for x in scan]
    for name, xs in (("20-item dev", small), ("100-item dev", big)):
        print(f"   {name}, 38 draws: held-out change mean {100 * sum(xs) / len(xs):+.2f}"
              f"  range {100 * min(xs):+.2f} to {100 * max(xs):+.2f}  worse in {sum(1 for x in xs if x < 0)}")
    print(f"   reported dev gain, 20-item dev: mean {100 * sum(small_dev) / len(small_dev):+.2f}"
          f"  range {100 * min(small_dev):+.2f} to {100 * max(small_dev):+.2f}")
    n2 = BASE.with_rule("N2")
    print(f"   adding a generic 'Be concise.': held-out {pct(base_held.mean)} -> {pct(E(n2, HELD).mean)}")
    R["small"], R["small_dev"], R["n2"] = small, small_dev, E(n2, HELD).mean

    print("10. Gate, manifest and diff")
    gate = Gate(BASE.pinned_hash)
    for name, cand, held_s in (("raw optimised", r, hr_), ("item rules removed", c, hc),
                               ("pinned edited (P1r)", forced, E(forced, HELD))):
        ok, why = gate.check(cand, paired_ci(base_held, held_s), wrong_clause(BASE, FULL),
                             wrong_clause(cand, FULL), [x for x in cand.rules if is_narrow(x)])
        print(f"   {name:20s} adopt={ok}  {why}")
    man = Manifest(c, "bootstrap(k=3,12)+evolve(6x6)+reflect(12)", 36, "stand-in-36",
                   METRIC_VERSION, ids(TRAIN), ids(dev), ids(HELD), E(c, dev).mean, hc.mean)
    rec = man.record()
    print("   " + json.dumps(rec, sort_keys=True)[:300])
    print(f"   diff vs hand-written: {diff(BASE, c)}")
    R["manifest"] = rec
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=list)
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
