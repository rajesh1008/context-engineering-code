"""Chapter 30 lab: the ten failure scripts, read, diagnosed and turned into tests.
Run from 04-code: python3 ch30/lab.py

Note: deterministic stand-in reader with seeded sampling (lakeshore30.py); no model runs.
"""
from __future__ import annotations
import sys, pathlib, json, time
from collections import Counter
here = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent)]
from contextlab.repro import reproduce, flips, seeds_needed, Regression, run_suite, rng_for  # noqa
from contextlab.diagnostics import diagnose                                              # noqa
import lakeshore30 as L                                                                   # noqa
from scripts30 import SCRIPTS, BY_CLASS, distraction, clash, fragmentation, _gold         # noqa
from signals30 import SIGNALS, TESTS                                                      # noqa
from contextlab.repro import Script                                                       # noqa

SEEDS = 20
CASES = L.base_cases()
NAIVE, CUR = L.runner(L.naive), L.runner(L.current)
RES = {}


def rate(script, run, seeds=SEEDS):
    return reproduce(script, CASES, run, seeds)


def s1_control():
    print("== 1. Fifty sound Lakeshore packets, 20 seeds each (1,000 reads)")
    tok = sum(c.tokens for c in CASES) / len(CASES)
    out = {}
    for name, run in (("naive", NAIVE), ("current", CUR)):
        w = sum(run(c, s) == "wrong" for c in CASES for s in range(SEEDS))
        out[name] = w
        print(f"  {name:8s} wrong {w}/1,000")
    print(f"  mean packet {tok:.0f} tokens")
    RES["control"] = out; RES["mean_tokens"] = round(tok)


def s2_scripts():
    print("== 2. Ten scripts: wrong per 1,000 reads, naive builder and current pipeline")
    rows = {}
    for sc in SCRIPTS:
        a, b = rate(sc, NAIVE), rate(sc, CUR)
        sh = a.shape(SEEDS)
        n = seeds_needed(a.rate) if a.rate < 1 else 1
        rows[sc.cls] = dict(naive=a.wrong, current=b.wrong, handoff=b.handoff,
                            control=a.control_wrong, shape=sh, seeds95=n)
        print(f"  {sc.cls:16s} naive {a.wrong:4d}  current {b.wrong:3d} (+{b.handoff} hand-offs)"
              f"  control {a.control_wrong}  cases every/some/no seed "
              f"{sh['every seed']}/{sh['some seeds']}/{sh['no seed']}  reruns for 95% {n}")
    RES["scripts"] = rows
    for p in (0.25, 0.086, 0.013, 0.004, 0.001):
        print(f"  reruns to see a {p:.1%} failure once with 95% confidence: {seeds_needed(p)}")
    RES["seeds_needed"] = {str(p): seeds_needed(p) for p in (0.25, 0.086, 0.013, 0.004, 0.001)}


def s3_signals():
    print("== 3. Reading the signals: share of injected packets on which each signal fires")
    mat = {}
    for sc in SCRIPTS:
        bad = [sc.inject(c, rng_for(sc.cls, c.cid)) for c in CASES]
        mat[sc.cls] = {k: sum(f(b) for b in bad) for k, f in SIGNALS.items()}
        fired = ", ".join(f"{k} {v}" for k, v in mat[sc.cls].items() if v)
        print(f"  {sc.cls:16s} {fired}")
    RES["signals"] = mat
    print("   run-level diagnosis of every wrong naive read (fix applied, same seed):")
    fid = {}
    for sc in SCRIPTS:
        cnt = Counter()
        for c in CASES:
            bad = sc.inject(c, rng_for(sc.cls, c.cid))
            for s in range(SEEDS):
                if L.run(bad, s) == "wrong":
                    ok = lambda x, s=s: L.run(x, s) == "right"
                    cnt[diagnose(bad, ok, TESTS).primary] += 1
        fid[sc.cls] = dict(cnt)
        print(f"  {sc.cls:16s} {dict(cnt)}")
    RES["diagnosis"] = fid


def _labels(c):
    from dataclasses import replace
    return L.with_spans(c, [replace(s, labelled=True) if not s.applies else s for s in c.spans])


def s4_side_by_side():
    print("== 4. Distraction and confusion side by side: wrong per 1,000 reads")
    srt = lambda c: L.with_spans(c, sorted(c.spans, key=lambda s: -s.sim))
    pipes = {"neither": srt, "labels": lambda c: srt(_labels(c)),
             "dedup (keyed)": lambda c: L.dedup(srt(c), keyed=True),
             "both": lambda c: L.dedup(srt(_labels(c)), keyed=True)}
    out = {}
    for cls in ("confusion", "distraction"):
        out[cls] = {k: rate(BY_CLASS[cls], L.runner(p)).wrong for k, p in pipes.items()}
        print(f"  {cls:12s} {out[cls]}")
    RES["side_by_side"] = out


def s5_dose():
    print("== 5. Distraction dose: notes added, on-topic and off-topic (naive = current)")
    out = {"on": {}, "off": {}}
    for k in range(0, 9):
        for tag, on in (("on", True), ("off", False)):
            sc = Script("distraction", lambda c, r, k=k, on=on: distraction(c, r, k, on))
            out[tag][k] = rate(sc, CUR).wrong
        print(f"  k={k}  on-topic {out['on'][k]:4d}  off-topic {out['off'][k]:4d}")
    RES["dose"] = out


def s6_clash():
    print("== 6. Clash: which side is newer, and a stated precedence")
    from dataclasses import replace

    def variant(newer_side):
        def inj(c, r):
            b = clash(c, r)
            return L.with_spans(b, [replace(s, newer=(s.kind == newer_side)) for s in b.spans])
        return Script("clash", inj)
    out = {}
    for side, label in (("rival", "skill text newer"), ("gold", "bulletin newer"),
                        ("none", "same date")):
        sc = variant(side)
        out[label] = {"naive": rate(sc, NAIVE).wrong, "current": rate(sc, CUR).wrong}
        print(f"  {label:17s} naive {out[label]['naive']:4d}  with precedence {out[label]['current']:3d}")
    RES["clash"] = out


def s7_fragmentation():
    print("== 7. Fragmentation: where the second half is, and fixed-size chunking")
    def apart(c, r):
        b = fragmentation(c, r)
        s = list(b.spans)
        return L.with_spans(b, [s[0]] + s[2:] + [s[1]])
    def half(c, r):
        b = fragmentation(c, r)
        return L.with_spans(b, [b.spans[0]] + list(b.spans[2:]))
    out = {}
    loose = L.runner(lambda c: L.defend(c, strict=False))
    for label, inj in (("halves adjacent", fragmentation), ("halves apart", apart),
                       ("second half not retrieved", half)):
        sc = Script("fragmentation", inj)
        lo, cu = rate(sc, loose), rate(sc, CUR)
        out[label] = {"naive": rate(sc, NAIVE).wrong, "before": lo.wrong,
                      "current": cu.wrong, "handoff": cu.handoff}
        print(f"  {label:26s} naive {out[label]['naive']:4d}  without completeness check "
              f"{lo.wrong:4d}  with it {cu.wrong:4d} wrong, {cu.handoff} hand-offs")
    cut = {}
    rules = [_gold(c).tokens for c in CASES]
    for size in (128, 256, 512, 1024, 2048):
        n = sum((rng_for("off", size, i, j).randrange(size) + L_) > size
                for i, L_ in enumerate(rules) for j in range(20))
        cut[size] = round(n / (len(rules) * 20), 3)
    print(f"  share of governing rules cut by a fixed-size chunker: {cut}")
    RES["fragmentation"] = out; RES["chunk_cut"] = cut


def _reembed(c):
    """A new embedding model: every similarity moves a little, nothing else changes."""
    from dataclasses import replace
    return L.with_spans(c, [replace(s, sim=round(s.sim + rng_for("reembed", s.sid)
                                                  .uniform(-0.03, 0.03), 3)) for s in c.spans])


def s8_flips():
    print("== 8. Flips under common seeds: the aggregate hides movement")
    out = {}
    allbad = [sc.inject(c, rng_for(sc.cls, c.cid)) for sc in SCRIPTS for c in CASES]
    out["re-embedded, naive, all scripts"] = flips(allbad, _reembed, NAIVE, SEEDS)
    for cls in ("poisoning", "confusion", "clash", "distraction", "rot"):
        sub = [sc.inject(c, rng_for(sc.cls, c.cid)) for sc in SCRIPTS if sc.cls == cls
               for c in CASES]
        out[f"  of which {cls}"] = flips(sub, _reembed, NAIVE, SEEDS)
    b2 = sum(CUR(x, s) == "wrong" for x in allbad for s in range(SEEDS))
    k = L.runner(L.current_dedup_keyed)
    a2 = sum(k(x, s) == "wrong" for x in allbad for s in range(SEEDS))
    rw2 = sum(CUR(x, s) != "wrong" and k(x, s) == "wrong" for x in allbad for s in range(SEEDS))
    wr2 = sum(CUR(x, s) == "wrong" and k(x, s) != "wrong" for x in allbad for s in range(SEEDS))
    out["current -> keyed dedup, all scripts"] = (b2, a2, rw2, wr2)
    for name, v in out.items():
        print(f"  {name:40s} wrong {v[0]} -> {v[1]}; right->wrong {v[2]}, wrong->right {v[3]}")
    RES["flips"] = out


def suite(distraction_guard=0.30):
    out = []
    for sc in SCRIPTS:
        naive = RES["scripts"][sc.cls]["naive"] / 1000
        canary = max(0.005, int(naive * 0.5 / 0.005) * 0.005)
        guard = distraction_guard if sc.cls == "distraction" else 0.01
        out.append(Regression(sc, round(canary, 3), guard, SEEDS))
    return out


def s9_regression():
    print("== 9. The regression suite: canary on the naive builder, guard on the candidate")
    res = {}
    for name, cand, dg in (("current", L.current, 0.30),
                           ("current + dedup", L.current_dedup, 0.30),
                           ("current + keyed dedup", L.current_dedup_keyed, 0.30),
                           ("keyed dedup, guard tightened", L.current_dedup_keyed, 0.15)):
        t0 = time.time()
        rows = run_suite(suite(dg), CASES, NAIVE, L.runner(cand))
        bad = [r["cls"] for r in rows if not (r["canary_ok"] and r["guard_ok"])]
        res[name] = {r["cls"]: (round(r["reference"], 3), round(r["candidate"], 3),
                                r["canary_ok"], r["guard_ok"]) for r in rows}
        print(f"  {name:29s} failing: {bad or 'none'}  ({time.time() - t0:.1f} s)")
        for r in rows:
            if r["cls"] in ("confusion", "distraction") or not r["guard_ok"]:
                print(f"    {r['cls']:14s} reference {r['reference']:.3f}  candidate "
                      f"{r['candidate']:.3f}  canary {'ok' if r['canary_ok'] else 'BROKEN'}  "
                      f"guard {'ok' if r['guard_ok'] else 'FAIL'}")
    for t in suite(0.15):
        print(f"  pinned {t.script.cls:16s} canary >= {t.canary:.3f}  guard <= {t.guard:.3f}")
    RES["suite"] = res
    RES["pins"] = {t.script.cls: (t.canary, t.guard) for t in suite(0.15)}
    reads = 2 * len(SCRIPTS) * len(CASES) * SEEDS * 2
    print(f"  reads per suite run: {reads:,} (injected and control, both pipelines)")
    RES["reads_per_suite"] = reads


if __name__ == "__main__":
    t0 = time.time()
    for f in (s1_control, s2_scripts, s3_signals, s4_side_by_side, s5_dose, s6_clash,
              s7_fragmentation, s8_flips, s9_regression):
        f()
    (here / "out").mkdir(exist_ok=True)
    (here / "out" / "results.json").write_text(json.dumps(RES, indent=1, default=str))
    print(f"done in {time.time() - t0:.0f} s")
