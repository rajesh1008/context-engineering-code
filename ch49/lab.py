"""Chapter 49 lab: build the packet-review exercise, inject a vocabulary rubric, fix it.
Run from 04-code: python3 ch49/lab.py

Note: deterministic stand-in world (lakeshore49.py); no model and no real candidate.
"""
import pathlib
import random
import sys

here = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent)]

from contextlab.hiring import (VERSION, cohen_kappa, score_anchored, score_vocab,
                               spearman, top_k_overlap)
from lakeshore49 import DEFECTS, PACKET, pool

RES = {}
P = pool()
REV = [p[0] for p in P]
DRILL = [p[3] for p in P]
FLU = [p[2] for p in P]


def s1_exercise():
    print(f"== 1. The exercise (contextlab v{VERSION}): {len(PACKET)} spans, {len(DEFECTS)} seeded defects")
    no_page = [s.sid for s in PACKET if s.page is None]
    print("   spans missing a page:", no_page)
    RES["exercise"] = {"spans": len(PACKET), "defects": len(DEFECTS), "no_page": no_page}
    found = {d.did: sum(r.findings[d.did][0] for r in REV) for d in DEFECTS}
    print("   candidates finding each defect (of 40):", found)
    RES["found"] = found
    RES["skill_fluency"] = spearman([p[1] for p in P], FLU)
    print("   pool check, Spearman(skill, name fluency):", RES["skill_fluency"])


def s2_inject():
    print("== 2. Failure injection: a rubric that rewards vendor names")
    v = [score_vocab(r) for r in REV]
    RES["vocab"] = {"rho_drill": spearman(v, DRILL), "rho_fluency": spearman(v, FLU),
                    "top5": top_k_overlap(v, DRILL)}
    best = max(range(40), key=lambda i: (v[i], -i))
    med = sorted(DRILL)[20]
    top5 = sorted(range(40), key=lambda i: (-v[i], i))[:5]
    RES["vocab_top5_below_median"] = sum(DRILL[i] < med for i in top5)
    print("   of its top five, below the median drill score:", RES["vocab_top5_below_median"])
    RES["vocab_top"] = (REV[best].name, v[best], round(DRILL[best], 2), P[best][1])
    print("   Spearman with held-out drill:", RES["vocab"]["rho_drill"],
          "| with name fluency:", RES["vocab"]["rho_fluency"], "| top-5 overlap:", RES["vocab"]["top5"])
    print("   rubric's first choice:", RES["vocab_top"])


def s3_fix():
    a = [score_anchored(r, DEFECTS) for r in REV]
    RES["anch"] = {"rho_drill": spearman(a, DRILL), "rho_fluency": spearman(a, FLU),
                   "top5": top_k_overlap(a, DRILL)}
    best = max(range(40), key=lambda i: (a[i], -i))
    RES["anch_top"] = (REV[best].name, a[best], round(DRILL[best], 2), P[best][1])
    print("== 3. Fix: anchored per-defect rubric, names score nothing")
    print("   Spearman with held-out drill:", RES["anch"]["rho_drill"],
          "| with name fluency:", RES["anch"]["rho_fluency"], "| top-5 overlap:", RES["anch"]["top5"])
    print("   rubric's first choice:", RES["anch_top"])
    RES["scores"] = {"vocab": [score_vocab(r) for r in REV], "anch": a, "drill": DRILL}


def _band(x, cuts):
    return sum(x > c for c in cuts) + 1


def s4_raters():
    rng = random.Random(4949)
    v = [score_vocab(r) for r in REV]
    mv = sorted(v)
    hol = []
    for _ in range(2):
        hol.append([_band(x + rng.gauss(0, 2.5), [mv[8], mv[16], mv[24], mv[32]]) for x in v])
    anc = []
    for _ in range(2):
        row = []
        for r in REV:
            s = score_anchored(r, DEFECTS) + sum(rng.random() < .08 for _ in DEFECTS) * rng.choice([-1, 1])
            row.append(_band(s, [3, 7, 11, 15]))
        anc.append(row)
    RES["kappa"] = {"impression": cohen_kappa(*hol), "anchored": cohen_kappa(*anc)}
    print("== 4. Two raters, five bands: kappa impression", RES["kappa"]["impression"],
          "| anchored", RES["kappa"]["anchored"])


SECTIONS = [s1_exercise, s2_inject, s3_fix, s4_raters]

if __name__ == "__main__":
    for f in SECTIONS:
        f()
    import json, pathlib
    pathlib.Path(__file__).with_name("out").joinpath("results.json").write_text(json.dumps(RES, indent=1))
