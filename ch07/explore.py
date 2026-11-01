"""Chapter 7 explorations: parts A-L of output.txt (run lab.py first; this appends).

Deterministic; no API key. Run: python3 lab.py && python3 explore.py
"""
from __future__ import annotations

from collections import Counter

from common import HERE, MODELS, ROOF, SEEDS, WATER, adh, corpus, pipe, traces
from contextlab.instructions import (HISTORY_7, ROOF_V32, SKILLS, V22, V30, V30_WATER, V31,
                                     V31_WATER, V32, Skill, adherence_probes, clashes)
from standin import cost, faithfulness, hit_at_5, mean_tokens, rate, unsupported, wrong

out: list[str] = []
p = out.append


# ------------------------------------------------------------ Listing 7.11
def resolve(ins):
    decisions, kept = [], []
    for sk in ins.skills:
        lost = [r for r in sk.negates if r in ins.rules]
        kept.append(None if lost else sk)
        for r in lost:
            decisions.append(f"rule {r} (tier 0) beats skill {sk.name} (tier 1): skill withheld "
                             f"for topics {', '.join(sk.topics)}; owner must rewrite it")
    if any(k is None for k in kept):
        fallback = {s.name: s for s in V22.skills}
        skills = tuple(k if k is not None else fallback[ins.skills[i].name]
                       for i, k in enumerate(kept))
        return ins.bump(ins.version + "+resolved", skills=skills), decisions
    return ins, decisions


# ------------------------------------------------------------ Listings 7.14 and 7.15
def release_gate_v2(ins, baseline, models=MODELS, tolerance=0.05):
    from contextlab.instructions import adherence_probes, lint
    lines, ok = [], True
    found = clashes(ins)
    lines.append(f"clashes: {len(found)}")
    ok &= not found                      # any clash fails, hierarchy rule or not
    problems = lint(ins, corpus)
    lines.append(f"example lint problems: {len(problems)}")
    ok &= not problems
    changed = {s.name for s in ins.skills} - {s.name for s in baseline.skills if s in ins.skills}
    for rule in ("precedence", "applicability"):
        probes = adherence_probes(corpus, rule)
        scoped = [d for d in probes if any(d.topic in s.topics for s in ins.skills
                                           if s.name in changed)] or probes
        for m in models:
            new, old = adh(pipe(ins, m), scoped), adh(pipe(baseline, m), scoped)
            drop = (old - new) / len(scoped)
            flag = "" if drop <= tolerance else "  REGRESSION"
            ok &= drop <= tolerance
            lines.append(f"{rule:13} {m:10} scoped n={len(scoped):3}  production {old:3}  "
                         f"candidate {new:3}  drop {drop:+.1%}{flag}")
    return ok, lines


def release_gate_v1(ins, baseline, models=MODELS, floor=0.95):
    """The first design (Listing 7.13): an absolute floor. Both the full probe set and the set
    scoped to the changed skills must reach the floor on every model."""
    from contextlab.instructions import lint
    lines, ok = [], True
    found = clashes(ins)
    unguarded = bool(found) and "hierarchy" not in ins.rules
    lines.append(f"clashes: {len(found)}{' (UNGUARDED)' if unguarded else ''}")
    ok &= not unguarded
    problems = lint(ins, corpus)
    lines.append(f"example lint problems: {len(problems)}")
    ok &= not problems
    changed = {s.name for s in ins.skills} - {s.name for s in baseline.skills if s in ins.skills}
    for rule in ("precedence", "applicability"):
        probes = adherence_probes(corpus, rule)
        scoped = [d for d in probes if any(d.topic in s.topics for s in ins.skills
                                           if s.name in changed)] or probes
        for m in models:
            pl = pipe(ins, m)
            a, s = adh(pl, probes) / len(probes), adh(pl, scoped) / len(scoped)
            flag = "" if min(a, s) >= floor else "  BELOW FLOOR"
            ok &= min(a, s) >= floor
            lines.append(f"{rule:13} {m:10} all {a:6.1%} (n={len(probes)})  scoped {s:6.1%} "
                         f"(n={len(scoped)}){flag}")
    return ok, lines


def hot_edit(ins):
    """The deployed copy of v3.2 after a six-word 'softening' of the roof skill (Listing 7.9)."""
    text = ROOF_V32.text.replace("the bulletin controls (rule 1)", "the bulletin is considered")
    assert text != ROOF_V32.text
    roof = Skill(ROOF_V32.name, ROOF_V32.topics, text, prefers=ROOF_V32.prefers)
    return ins.bump(ins.version, skills=(roof,) + ins.skills[1:])


def main() -> None:
    probes = adherence_probes(corpus, "precedence")
    roof = [d for d in probes if d.topic in ROOF]
    water_dets = [d for d in corpus.determinations if d.topic in WATER]

    p("")
    p(f"A. v3.0 roof adherence ({len(roof)} probes) over seeds {SEEDS[0]}-{SEEDS[-1]}")
    for m in MODELS:
        for ins in (V30, V31, V32):
            r = [adh(pipe(ins, m, s), roof) for s in SEEDS]
            p(f"  {ins.version} {m:<10} correct of {len(roof)}: {r}  mean {sum(r) / len(r):.1f}  "
              f"min {min(r)} max {max(r)}")
    p("")

    p("B. Leak: water skill countermands quote_only")
    p(f"  clash check: {clashes(V30_WATER)}")
    n_eval = sum(d in water_dets for d in corpus.eval)
    p(f"  water determinations: claims 1-500 {len(water_dets)}, eval set {n_eval}")
    for m in MODELS:
        for ins in (V22, V30_WATER, V31_WATER):
            wa = unsupported(traces(ins, m, water_dets))
            ev = traces(ins, m, corpus.eval)
            p(f"  {ins.version:<11} {m:<10} water additions {wa:>3}/{len(water_dets)}  "
              f"eval additions {unsupported(ev):>2}/{len(ev)}  eval wrong {rate(ev):.1f}%  "
              f"faithfulness {faithfulness(ev):.3f}")
    p("")

    p("C. Version history (sim-large)")
    p("  version  fingerprint  rules skills  ex clash eval wrong  probes  roof  holdout")
    hard = corpus.hard()
    for ins in HISTORY_7:
        pl = pipe(ins, "sim-large")
        ev = [pl.run(d) for d in corpus.eval]
        hw = [pl.run(d) for d in hard]
        p(f"  {ins.version:<9}{ins.fingerprint()}{len(ins.rules):>6}{len(ins.skills):>7}"
          f"{len(ins.examples):>4}{len(clashes(ins)):>6}{rate(ev):>10.1f}%"
          f"{f'{adh(pl, probes)}/{len(probes)}':>8}{f'{adh(pl, roof)}/{len(roof)}':>6}"
          f"{100 * wrong(hw) / len(hw):>8.1f}%")
    p(f"  holdout n={len(hard)}")
    p("")

    p("D. Hot edit: deployed text differs from the registered version")
    deployed = hot_edit(V32)
    p(f"  registered {V32.version} fingerprint {V32.fingerprint()}")
    p(f"  deployed   {deployed.version} fingerprint {deployed.fingerprint()}")
    p(f"  version labels equal: {V32.version == deployed.version}; "
      f"fingerprints equal: {V32.fingerprint() == deployed.fingerprint()}")
    for ln in V32.diff(deployed):
        p(f"    {ln}")
    p("")

    p("E. What each test set sees for v2.2 -> v3.0 (sim-large and sim-mid)")
    n_roof_eval = sum(d.topic in ROOF for d in corpus.eval)
    for m in ("sim-large", "sim-mid"):
        for label, dets in (("eval", corpus.eval), ("probes", probes)):
            b = [t.wrong for t in traces(V22, m, dets)]
            a = [t.wrong for t in traces(V30, m, dets)]
            nw = sum(x and not y for x, y in zip(a, b))
            nr = sum(y and not x for x, y in zip(a, b))
            tail = f"; roof-topic dets in eval {n_roof_eval}" if label == "eval" else ""
            p(f"  {m}: {label} wrong {sum(b)} -> {sum(a)} (newly wrong {nw}, newly right {nr}){tail}")
    p("")

    n = len(water_dets) * len(SEEDS)
    p(f"F. Leak at scale: water claims 1-500 ({len(water_dets)}) x seeds {SEEDS[0]}-{SEEDS[-1]} "
      f"= {n} answers per cell")
    top = Counter()
    for m in MODELS:
        for ins in (V22, V30_WATER, V31_WATER):
            k = 0
            for s in SEEDS:
                for t in traces(ins, m, water_dets, s):
                    if t.answer.addition:
                        k += 1
                        if ins is V30_WATER and m == "sim-small":
                            top[t.answer.addition] += 1
            p(f"  {ins.version:<11} {m:<10} answers with an unsupported addition {k:>3}/{n} ({k / n:.1%})")
    p("  top additions under v3.0-water, sim-small:")
    for text, c in top.most_common(2):
        p(f'    "{text[:40]}" {c}')
    p("")

    p("G. One leaked answer, side by side (sim-small, seed 2026)")
    before = {t.det.determination_id: t for t in traces(V22, "sim-small", water_dets)}
    for t in traces(V30_WATER, "sim-small", water_dets):
        b = before[t.det.determination_id]
        if t.answer.addition and not b.answer.addition and not t.wrong:
            d = t.det
            p(f"  {d.determination_id} {d.name}, {d.state}, {d.topic}; gold {t.gold.doc_label}")
            p(f"  v2.2:       {b.answer_text}")
            p(f"  v3.0-water: {t.answer_text}")
            p(f"  rules in force under v3.0-water: {sorted(set(V30_WATER.rules) - set(t.answer.dropped))}")
            break
    p("")

    p("H. Release gate (floor 95% on the probes a change touches) and cost")
    for ins in (V22, V30, V31, V32):
        ok, lines = release_gate_v1(ins, V22)
        p(f"  {ins.version}: {'PASS' if ok else 'FAIL'}")
        out.extend(f"    {ln}" for ln in lines)
    p("  prompt tokens and cost per determination, sim-large, eval set:")
    for ins in (V22, V31, V32):
        ev = traces(ins, "sim-large", corpus.eval)
        p(f"    {ins.version}  tokens {mean_tokens(ev):,.0f}  cost ${cost(ev):.5f}")
    p("")

    p("I. Release gate v2: per model, against production v2.2, on the skills a change touches")
    for ins in (V30, V31, V32, V30_WATER, V31_WATER):
        ok, lines = release_gate_v2(ins, V22)
        p(f"  {ins.version}: {'PASS' if ok else 'FAIL'}")
        out.extend(f"    {ln}" for ln in lines)
    p("")

    p("J. resolve(): conflicts settled in code before assembly (fall back to last clean skill)")
    for ins in (V30, V30_WATER):
        fixed, decisions = resolve(ins)
        p(f"  {ins.version} -> {fixed.version}; clashes after: {len(clashes(fixed))}; "
          f"fingerprint {fixed.fingerprint()} (v2.2 is {V22.fingerprint()})")
        out.extend(f"    {d}" for d in decisions)
        ev = traces(fixed, "sim-large", corpus.eval)
        p(f"    roof probes {adh(pipe(fixed, 'sim-large'), roof)}/{len(roof)}; eval wrong {rate(ev):.1f}%")
    p("")

    p("K. Probe inventory (claims 1-500; example sources excluded)")
    for rule in ("precedence", "applicability"):
        pr = adherence_probes(corpus, rule)
        p(f"  {rule:13} all {len(pr):3}  roof topics {sum(d.topic in ROOF for d in pr):3}  "
          f"water topics {sum(d.topic in WATER for d in pr):3}")
    p("")

    p("L. Scoreboard, sim-large, v2.2 and v3.2 (Chapter 6 format)")
    for ins in (V22, V32):
        for label, dets in ((f"eval {len(corpus.eval)}", corpus.eval),
                            (f"holdout {len(corpus.holdout)}", corpus.holdout)):
            ts = traces(ins, "sim-large", dets)
            p(f"  {ins.version} {label:<13}wrong-clause {rate(ts):.1f}%  hit@5 {hit_at_5(ts):.2f}  "
              f"faithfulness {faithfulness(ts):.3f}  tokens {mean_tokens(ts):,.0f}  cost ${cost(ts):.5f}")

    path = HERE / "output.txt"
    head = path.read_text().split("\nA. ")[0].rstrip("\n") + "\n"
    path.write_text(head + "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
    print("\n".join(out))
