"""Chapter 29 lab: the taxonomy as code. Run from 04-code: python3 ch29/lab.py

Note: deterministic stand-in reader (standin29.py); no model runs.
"""
from __future__ import annotations
import sys, pathlib, json
from collections import Counter, defaultdict
here = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent)]
from contextlab.diagnostics import Diagnostics, CLASSES, NOT_CONTEXT   # noqa: E402
from standin29 import run, with_spans                               # noqa: E402
from tests29 import TESTS                                           # noqa: E402
import ledger29 as L                                                # noqa: E402
import fullmix                                                      # noqa: E402

diag = lambda c: Diagnostics.diagnose(c, run, TESTS)
RES = {}


def s1_probes():
    print("== 1. Ten probe families, 40 cases each: wrong before and after the class fix")
    fam = L.probes()
    conf = defaultdict(Counter)
    rows = {}
    for cls, cases in fam.items():
        test = next(t for t in TESTS if t.cls == cls)
        before = sum(not run(c) for c in cases)
        after = sum(not run(test.fix(c)) for c in cases)
        for c in cases:
            if not run(c):
                conf[cls][diag(c).primary] += 1
        rows[cls] = (before, after)
        print(f"  {cls:17s} wrong {before:2d}/40 -> {after:2d}/40 after: {test.remedy}")
    print("  diagnosed primary for the wrong cases (injected -> found):")
    for cls in CLASSES:
        print(f"    {cls:17s} {dict(conf[cls])}")
    RES["probes"] = rows
    RES["confusion_matrix"] = {k: dict(v) for k, v in conf.items()}


def s1b_mixed():
    print("== 1b. Mixed packets: one causal defect plus harmless signals (100 cases)")
    cases = L.mixed()
    sig_ok = cf_ok = still_right = 0
    for cls, c in cases:
        fired = [t.cls for t in TESTS if t.signal(c)]
        sig_ok += bool(fired) and fired[0] == cls
        d = diag(c)
        cf_ok += d.primary == cls
        still_right += run(c)
    print(f"  signals per case (mean) {sum(len([t for t in TESTS if t.signal(c)]) for _, c in cases) / len(cases):.1f}")
    print(f"  first signal = injected class: {sig_ok}/100; counterfactual primary: {cf_ok}/100;"
          f" right anyway {still_right}")
    RES["mixed"] = (sig_ok, cf_ok, still_right)


def s1c_fix_matrix():
    print("== 1c. Which fix flips which family (share of the family's wrong cases)")
    fam = L.probes()
    M = {}
    print("  " + " " * 17 + " ".join(f"{t.cls[:5]:>5s}" for t in TESTS))
    for cls, cases in fam.items():
        bad = [c for c in cases if not run(c)]
        row = [sum(run(t.fix(c)) for c in bad) / len(bad) for t in TESTS]
        M[cls] = row
        print(f"  {cls:17s}" + " ".join(f"{v:5.2f}" for v in row))
    RES["fix_matrix"] = M


def s2_ledger():
    print("== 2. Error analysis: the 82 wrong under Chapter 8's configuration")
    rows, dets = L.ledger()
    codes = Counter(r[2] for r in rows)
    print(f"  determinations {len(dets)}, wrong {len(rows)}; open codes {len(codes)}")
    for code, n in codes.most_common():
        print(f"    {n:3d}  {code}")
    agree, by_cls, sec, joint, by_fix = 0, Counter(), Counter(), 0, Counter()
    for det_id, stratum, code, hand, case, fix in rows:
        d = diag(case)
        agree += d.primary == hand
        by_cls[d.primary] += 1
        joint += d.joint
        for s in d.secondary:
            sec[(d.primary, s)] += 1
        by_fix[(d.primary, fix)] += 1
    sym = sum(L.SYMPTOM[r[2]] == r[3] for r in rows)
    sym_nc = sum(L.SYMPTOM[r[2]] == "not_context" for r in rows)
    print(f"  symptom-first label agrees with the axial code: {sym}/{len(rows)};"
          f" 'model error' written for {sym_nc}")
    print(f"  flowchart primary agrees with hand axial code: {agree}/{len(rows)}")
    RES["symptom"] = (sym, sym_nc)
    print("  axial classes:", dict(by_cls.most_common()))
    print("  secondary classes:", {f"{a}+{b}": n for (a, b), n in sec.items()},
          f"joint-only fixes {joint}")
    print("  class x fixing chapter:")
    for (c, f), n in sorted(by_fix.items(), key=lambda x: -x[1]):
        print(f"    {n:3d}  {c:17s} {f}")
    RES["ledger"] = {"classes": dict(by_cls), "codes": dict(codes), "agree": agree,
                     "joint": joint, "by_fix": {f"{c}|{f}": n for (c, f), n in by_fix.items()}}
    return rows, dets


def s3_incidents():
    print("== 3. Incidents to date, reclassified")
    out = []
    for iid, ch, story, root, case in L.incidents():
        d = diag(case)
        sec = ",".join(d.secondary) or "-"
        print(f"  {iid} {ch:8s} {d.primary:17s} sec {sec:28s} joint {d.joint!s:5s} root: {root}"
              f" | {story}")
        out.append((iid, ch, d.primary, d.secondary, d.joint, root, story))
    RES["incidents"] = out


def s4_injection():
    print("== 4. Failure injection: one case, two classes")
    cases = L.injection()
    c = cases[0]
    d = diag(c)
    print(f"  {c.cid}: tokens {c.tokens}, primary {d.primary}, secondary {d.secondary},"
          f" joint {d.joint}")
    naive = next(t.cls for t in reversed(TESTS) if t.signal(c))
    print(f"  last-signal-wins label (what a symptom-first triage writes): {naive}")
    conf = next(t for t in TESTS if t.cls == "confusion")
    litm = next(t for t in TESTS if t.cls == "lost_in_middle")
    base = sum(not run(x) for x in cases)
    reorder = sum(not run(litm.fix(x)) if litm.signal(x) else not run(x) for x in cases)
    admit = sum(not run(conf.fix(x)) for x in cases)
    print(f"  batch of {len(cases)}: wrong {base}; after reorder only {reorder};"
          f" after admission only {admit}")
    RES["injection"] = {"primary": d.primary, "secondary": d.secondary, "naive": naive,
                        "base": base, "reorder": reorder, "admit": admit}


def s5_blast(dets):
    print("== 5. Blast radius on the full mix (exposed, wrong under Chapter 8)")
    w = lambda d: fullmix.outcome(d, "ch08") == "wrong"
    rows = {
        "pointer edition (confusion)": lambda d: d.pointer_wrong,
        "unindexed scans (missing evidence)": lambda d: d.stratum == "scanned",
        "multi-state claims (22 confusion, 2 missing)": lambda d: d.stratum == "multi-state",
        "third-party header (3 poisoning, 1 drift)": lambda d: d.stratum == "third-party",
    }
    out = {}
    for name, f in rows.items():
        e, n = Diagnostics.blast_radius(dets, f, w)
        out[name] = (e, n)
        print(f"  {name:40s} exposed {e:3d}  wrong {n:2d}  ({n / e:.0%})")
    RES["blast"] = out


def s6_ch04_api():
    print("== 6. Chapter 4's API is unchanged")
    print("  ", [n for n in ("rot_curve", "knee", "clash", "confusion", "distraction",
                            "rot", "summary", "diagnose", "blast_radius")
                 if hasattr(Diagnostics, n)])


if __name__ == "__main__":
    s1_probes()
    s1b_mixed()
    s1c_fix_matrix()
    rows, dets = s2_ledger()
    s3_incidents()
    s4_injection()
    s5_blast(dets)
    s6_ch04_api()
    (here / "results.json").write_text(json.dumps(RES, indent=1, default=str))
