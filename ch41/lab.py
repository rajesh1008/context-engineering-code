"""Chapter 41 lab: error analysis on Lakeshore's overrides, and an eval set that drifted.
Run from 04-code: python3 ch41/lab.py

Note: deterministic stand-in world (lakeshore41.py, notes41.py); no model runs.
"""
from __future__ import annotations
import sys, pathlib, json, random
from collections import Counter, defaultdict
here = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent)]
from contextlab.erroranalysis import (OpenCode, prevalence, pareto, cohen_kappa,  # noqa
                                      saturation, tvd, unsupported, reweight, wilson_upper)
import lakeshore41 as L                                                            # noqa
from notes41 import note_for, CODEBOOK_V1, CODEBOOK_V2, code_pair                  # noqa

PROD = L.production()
GOLD = L.gold_set()
EXTRA_MULTI = 104    # gold-v1.2: multi-peril items at the October share
RES = {}


def acc(rows, version):
    return sum(L.right(d, version) for d in rows) / len(rows) if rows else float("nan")


def s1_scores():
    print("== 1. The score and the overrides, July to October")
    g = {v: round(100 * acc(GOLD, v), 2) for v in ("before", "released")}
    weeks = []
    for w in range(17):
        rows = [d for d in PROD if d["week"] == w]
        v = L.released(w)
        over = sum(L.outcome(d, v)[1] for d in rows) / len(rows)
        weeks.append(dict(week=w, month=L.week_month(w), version=v,
                          gold=g[v], override=round(100 * over, 2),
                          right=round(100 * acc(rows, v), 2)))
    for x in weeks[::2]:
        print(f"  week {x['week']:2d} {x['month']} {x['version']:8s} gold {x['gold']:.2f}%"
              f"  production right {x['right']:.2f}%  overrides {x['override']:.2f}%")
    pre = [x["override"] for x in weeks if x["week"] in (5, 6, 7, 8)]
    post = [x["override"] for x in weeks if x["week"] >= 13]
    print(f"  gold set {g['before']:.2f}% -> {g['released']:.2f}%;"
          f" overrides {sum(pre)/4:.2f}% (Aug) -> {sum(post)/4:.2f}% (Oct)")
    RES["gold"] = g; RES["weeks"] = weeks
    RES["override_aug"] = round(sum(pre) / 4, 2); RES["override_oct"] = round(sum(post) / 4, 2)


def overrides(weeks, version=None, n=120, seed=41):
    pool = [d for d in PROD if d["week"] in weeks
            and L.outcome(d, version or L.released(d["week"]))[1]]
    return random.Random(seed).sample(pool, n), len(pool)


def s2_open_coding():
    print("== 2. Open coding: 120 overridden determinations from October")
    sample, pool = overrides(range(13, 17))
    codes = []
    for d in sample:
        cat = L.outcome(d, "released")[2]
        codes.append((OpenCode(d["id"], "adjuster", cat == "not_a_failure", note_for(cat, d),
                               d["segment"]), cat))
    for oc, _ in codes[:4]:
        print(f"  {oc.trace_id} {oc.segment:11s} acceptable={oc.acceptable!s:5s} {oc.note}")
    acceptable = sum(oc.acceptable for oc, _ in codes)
    print(f"  pool {pool:,} overrides in four weeks; sample 120; acceptable on reading {acceptable}")
    RES["open"] = dict(pool=pool, n=120, acceptable=acceptable,
                       examples=[oc.note for oc, _ in codes[:4]])
    RES["_codes"] = codes


def s3_axial():
    print("== 3. Axial coding: codebook v1 (Chapter 29 lenses) against v2")
    codes = RES["_codes"]
    v1 = [CODEBOOK_V1.apply(oc.note) for oc, _ in codes]
    v2 = [CODEBOOK_V2.apply(oc.note) for oc, _ in codes]
    truth = [t for _, t in codes]
    c1, c2 = Counter(v1), Counter(v2)
    print("  v1:", dict(c1.most_common()))
    print("  v2:", dict(c2.most_common()))
    split = Counter((a, b) for a, b in zip(v1, v2) if a != b)
    for (a, b), n in split.most_common():
        print(f"  v1 {a!s:18s} -> v2 {b:18s} {n}")
    agree = sum(a == t for a, t in zip(v2, truth))
    print(f"  v2 matches the reader's intended category on {agree}/120")
    RES["axial"] = dict(v1={str(k): v for k, v in c1.items()}, v2=dict(c2),
                        moves={f"{a}->{b}": n for (a, b), n in split.items()}, v2_match=agree)
    RES["_v2"] = v2


def s4_pareto():
    print("== 4. Counts and severity-weighted counts (v2)")
    rows = pareto(RES["_v2"], CODEBOOK_V2)
    for k, n, w in rows:
        print(f"  {k:22s} {n:3d}  weighted {w:3d}")
    RES["pareto"] = [list(r) for r in rows]


def s5_agreement():
    print("== 5. Two coders on the same 60 traces, before and after the alignment session")
    truth = [t for _, t in RES["_codes"][:60]]
    out = {}
    for phase in ("before", "after"):
        a, b = code_pair(truth, phase)
        k = cohen_kappa(a, b)
        per = {}
        for cat in sorted(set(truth)):
            per[cat] = round(cohen_kappa([x == cat for x in a], [y == cat for y in b]), 2)
        pa = sum(x == y for x, y in zip(a, b)) / 60
        out[phase] = dict(kappa=round(k, 2), agree=round(pa, 3), per=per)
        print(f"  {phase:6s} raw agreement {pa:.1%}  kappa {k:.2f}")
    print("  per category (before -> after):",
          {c: (out['before']['per'][c], out['after']['per'][c]) for c in out['after']['per']})
    RES["agreement"] = out


def s6_saturation():
    print("== 6. Saturation: distinct categories against traces read (stop after 30 with nothing new)")
    srcs = {}
    gold_fail = [d for d in GOLD if not L.right(d, "released")]
    labs = [L.CAUSE[d["segment"]][0][0] for d in gold_fail]
    srcs["gold-set failures"] = labs
    jul, _ = overrides(range(0, 4), n=120, seed=3)
    srcs["July overrides"] = [L.outcome(d, "before")[2] for d in jul]
    srcs["October overrides"] = [t for _, t in RES["_codes"]]
    out = {}
    for name, labels in srcs.items():
        curve, stop = saturation(labels, run=30)
        sp = labels.index("second_peril") + 1 if "second_peril" in labels else None
        later = []
        if stop:
            seen = set(labels[:stop])
            for i, l in enumerate(labels[stop:], stop + 1):
                if l not in seen:
                    seen.add(l); later.append((i, l))
        out[name] = dict(n=len(labels), curve=curve, stop=stop, final=curve[-1], second_peril_at=sp,
                         after_stop=later)
        print(f"  {name:18s} read {len(labels):3d}  categories {curve[-1]}  stop at {stop}"
              f"  second peril first seen at {sp}  found after the stop {later}")
    RES["saturation"] = out


def s7_drift():
    print("== 7. Failure injection: gold-v1.1 (mix close to April) against the production mix")
    n = sum(L.GOLD.values())
    gmix = {s: L.GOLD.get(s, 0) / n for s in L.SEGMENTS}
    drift = {}
    for m in L.MONTHS:
        uns = unsupported(L.GOLD, L.MIX[m])
        drift[m] = dict(tvd=round(tvd(gmix, L.MIX[m]), 3), unsupported=round(sum(uns.values()), 3))
        print(f"  {m}: total variation {drift[m]['tvd']:.3f}  production share with no gold items"
              f" {drift[m]['unsupported']:.0%}")
    seg = {v: {s: acc([d for d in GOLD if d['segment'] == s], v) for s in L.GOLD} for v in ("before", "released")}
    oct_rows = [d for d in PROD if d["month"] == "Oct"]
    prod = {v: acc(oct_rows, v) for v in ("before", "released")}
    rw = {v: reweight(seg[v], L.MIX["Oct"]) for v in ("before", "released")}
    pseg = {v: {s: acc([d for d in oct_rows if d["segment"] == s], v) for s in L.SEGMENTS}
            for v in ("before", "released")}
    d_raw = 100 * (acc(GOLD, "released") - acc(GOLD, "before"))
    d_rw = 100 * (rw["released"][0] - rw["before"][0])
    d_prod = 100 * (prod["released"] - prod["before"])
    print(f"  release effect: gold set {d_raw:+.2f} pts; gold re-weighted to October {d_rw:+.2f} pts"
          f" (covers {1 - rw['released'][1]:.0%}); October production {d_prod:+.2f} pts")
    for s in L.SEGMENTS:
        print(f"    {s:11s} production {100*pseg['before'][s]:6.2f} -> {100*pseg['released'][s]:6.2f}")
    RES["drift"] = dict(months=drift, d_raw=round(d_raw, 2), d_rw=round(d_rw, 2), d_prod=round(d_prod, 2),
                        prod={k: round(100 * v, 2) for k, v in prod.items()},
                        seg_prod={v: {s: round(100 * x, 2) for s, x in pseg[v].items()} for v in pseg},
                        gold_seg={v: {s: round(100 * x, 2) for s, x in seg[v].items()} for v in seg})


def s8_fix():
    print("== 8. From category to fix: one question per peril, replayed on October")
    oct_rows = [d for d in PROD if d["month"] == "Oct"]
    multi = [d for d in oct_rows if d["segment"] == "multi_peril"]
    out = {}
    for v in ("released", "per_peril"):
        k = sum(not L.right(d, v) for d in multi)
        out[v] = dict(all=round(100 * acc(oct_rows, v), 2), multi_n=len(multi), multi_wrong=k,
                      multi_upper=round(100 * wilson_upper(k, len(multi)), 2))
        print(f"  {v:9s} October right {out[v]['all']:.2f}%  multi-peril wrong {k}/{len(multi)}"
              f" (upper 95% {out[v]['multi_upper']:.2f}%)")
    worse = [s for s in L.SEGMENTS if s != "multi_peril" and
             sum(L.right(d, "per_peril") for d in oct_rows if d["segment"] == s) <
             sum(L.right(d, "released") for d in oct_rows if d["segment"] == s)]
    print(f"  segments where any claim got worse: {worse or 'none'}")
    rng = random.Random(9)
    extra = [dict(id=f"G-multi-{i:03d}", segment="multi_peril", u=rng.random()) for i in range(EXTRA_MULTI)]
    ext = GOLD + extra
    eg = {v: round(100 * acc(ext, v), 2) for v in ("before", "released", "per_peril")}
    print(f"  extended gold set ({len(ext)}): before {eg['before']:.2f}  released {eg['released']:.2f}"
          f"  per-peril {eg['per_peril']:.2f}")
    passes = [d for d in oct_rows if not L.outcome(d, "released")[1]]
    samp = random.Random(12).sample(passes, 200)
    silent = Counter(L.outcome(d, "released")[2] for d in samp if not L.right(d, "released"))
    print(f"  200 un-overridden October determinations read: wrong {sum(silent.values())} {dict(silent)}")
    RES["fix"] = dict(out, worse=worse, extended=eg, silent=dict(silent))


def s9_ship():
    print("== 9. The exit rule, written before the replay")
    oct_rows = [d for d in PROD if d["month"] == "Oct"]
    sev = [c.name for c in CODEBOOK_V2.categories if c.severity == 3 and c.name != "missing_evidence"]
    rows = {}
    for v in ("released", "per_peril"):
        cnt = Counter(L.outcome(d, v)[2] for d in oct_rows if not L.right(d, v))
        rows[v] = {c: dict(k=cnt.get(c, 0), upper=round(100 * wilson_upper(cnt.get(c, 0), len(oct_rows)), 3))
                   for c in sev}
    for c in sev:
        a, b = rows["released"][c], rows["per_peril"][c]
        print(f"  {c:22s} released {a['k']:4d} (upper {a['upper']:.3f}%)  per-peril {b['k']:4d} (upper {b['upper']:.3f}%)")
    first = next(m for m in L.MONTHS if RES["drift"]["months"][m]["tvd"] >= .05
                 or RES["drift"]["months"][m]["unsupported"] >= .05)
    print(f"  drift alarm (TVD >= 0.05 or unsupported share >= 5%) first fires: {first}")
    ok = (RES["fix"]["per_peril"]["multi_upper"] <= 10 and not RES["fix"]["worse"])
    print(f"  exit rule (multi-peril upper bound <= 10%, no segment worse): {'PASS' if ok else 'FAIL'}")
    RES["ship"] = dict(rows=rows, alarm_month=first, exit_pass=ok)


SECTIONS = [s1_scores, s2_open_coding, s3_axial, s4_pareto, s5_agreement, s6_saturation,
            s7_drift, s8_fix, s9_ship]

if __name__ == "__main__":
    for f in SECTIONS:
        f()
    out = here / "out"; out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps({k: v for k, v in RES.items() if not k.startswith("_")},
                                                 indent=1))
