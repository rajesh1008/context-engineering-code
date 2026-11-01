"""Chapter 43 lab: a candidate in shadow, an exit rule written first, evidence packs.

Run from 04-code: python3 ch43/lab.py  (a few seconds; writes ch43/output.txt and ch43/out/*.json)
"""
from __future__ import annotations

import copy
import json
import pathlib
import random
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from contextlab.shadow import (EditInPlace, ExitRule, cell_stats, explain, funnel,  # noqa: E402
                               pooled_rule, redact, sha, wilson)
import lakeshore43 as L  # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
R: dict = {}
LINES: list = []
KEY = b"lakeshore-evidence-salt-key-demo"


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LINES.append(s)


RULE = ExitRule("shadow-exit v1 (frozen 2026-09-18)")

# 1. the run ----------------------------------------------------------------------
say("== 1. Shadow run (harness v9.4): candidate sim-large@2026-10 on layer v9.2, 21 Sep - 2 Oct 2026 ==")
say("exit rule digest", RULE.digest(), "written before the first mirrored claim")
arrived, eligible, pairs = L.shadow_run()
F = funnel(arrived, eligible, pairs)
for k, v in F:
    say(f"  {k:<22}{v:>7,}")
R["funnel"] = F

# 2. agreement --------------------------------------------------------------------
say("\n== 2. Agreement with the adjuster's final determination ==")
S = cell_stats(pairs)
n = sum(s["n"] for s in S.values())
ca, pa = sum(s["cand"] for s in S.values()), sum(s["prod"] for s in S.values())
say(f"overall: candidate {ca}/{n} = {ca/n:.2%}  production {pa}/{n} = {pa/n:.2%}")
say(f"adjudicated wrong (all causes): candidate {sum(s['cand_wrong'] for s in S.values())}, "
    f"production {sum(s['prod_wrong'] for s in S.values())}")
cells = sorted(S, key=lambda c: (list(L.LINES).index(c[0]), list(L.STATES).index(c[1])))
for c in cells:
    s = S[c]
    lo, hi = wilson(s["cand"], s["n"])
    say(f"  {c[0]:<4} {c[1]}  n {s['n']:>4}  cand {s['cand']/s['n']:.1%} [{lo:.1%},{hi:.1%}]  "
        f"prod {s['prod']/s['n']:.1%}  b {s['b']:>2}  c {s['c']:>2}")
R["cells"] = {f"{c[0]}-{c[1]}": S[c] for c in cells}
mn = [p for p in pairs if p.cell == ("HO", "MN") and p.comparable()]
say("MN homeowners: late-notice determinations whose candidate draft denied under the base "
    "condition:", sum(1 for p in mn if p.cand and p.cand["clause"].endswith("#duties")
                      and p.cand["clause"].startswith("HO-3")),
    "; production:", sum(1 for p in mn if p.prod["clause"].startswith("HO-3")
                         and p.prod["clause"].endswith("#duties")))

# 3. exit rule ----------------------------------------------------------------------
say("\n== 3. The exit rule, as written, and the rule proposed the night before ==")
E = RULE.evaluate(pairs)
g = E["gates"]
say(f"run gates: parse {g['parse']:.2%} ok={g['parse_ok']}; p95 latency x{g['latency_ratio']:.2f} "
    f"ok={g['latency_ok']}; cost x{g['cost_ratio']:.3f} ok={g['cost_ok']}; packs ok={g['packs_ok']}")
cnt = Counter(E["cells"].values())
say("cells:", dict(cnt))
for c in cells:
    if E["cells"][c] != "release":
        say(f"  {c[0]}-{c[1]}: {E['cells'][c]} (n {S[c]['n']})")
share = sum(S[c]["n"] for c in cells if E["cells"][c] == "release") / n
say(f"share of comparable volume in released cells: {share:.1%}")
PR = pooled_rule(pairs)
say(f"pooled rule (overall >=95%, cell floor 90%): overall {PR['overall']:.2%}; "
    f"releases {sum(v == 'release' for v in PR['cells'].values())} cells, "
    f"including HO-MN: {PR['cells'][('HO', 'MN')]}")
mn_s = S[("HO", "MN")]
daily = 3200 * L.LINES["HO"] * L.STATES["MN"]
excess = (mn_s["b"] - mn_s["c"]) / mn_s["n"]
say(f"if HO-MN had been released: about {daily:.0f} drafts a day, {excess:.1%} excess wrong, "
    f"so about {daily*excess:.0f} wrong drafts a day put in front of adjusters")
R["exit"] = {"digest": E["digest"], "gates": g, "cells": {f"{a}-{b}": v for (a, b), v in E["cells"].items()},
             "pooled": {f"{a}-{b}": v for (a, b), v in PR["cells"].items()}, "pooled_overall": PR["overall"]}

# 4. fix and re-shadow ---------------------------------------------------------------
say("\n== 4. Pin the amendment with the clause it amends; re-shadow HO-MN ==")
_, _, fixed_all = L.shadow_run(pin=True)
E2 = RULE.evaluate(fixed_all)
changed = [c for c in cells if E2["cells"][c] != E["cells"][c]]
say("replay of the same 2,000 with the pin: cells whose status changed:",
    [f"{a}-{b}:{E['cells'][(a, b)]}->{E2['cells'][(a, b)]}" for a, b in changed])
_, _, fresh = L.shadow_run(n=160, seed=4304, start=L.date(2026, 10, 7), days=6, pin=True, cell=("HO", "MN"))
S3 = cell_stats(fresh)[("HO", "MN")]
say(f"fresh week, HO-MN only (7-14 Oct): n {S3['n']}, cand {S3['cand']/S3['n']:.1%}, "
    f"prod {S3['prod']/S3['n']:.1%}, b {S3['b']} c {S3['c']} -> {RULE.judge_cell(S3)}")
small = [c for c in cells if E["cells"][c] == "shadow"]
say("cells left in shadow: business days to 60 comparable if mirrored at 100% instead of 6.6%:")
for c in small:
    per_day = 3200 * L.LINES[c[0]] * L.STATES[c[1]] * (1 - L.P["ineligible"]) * (1 - L.P["still_open"])
    need = max(0, RULE.min_n - S[c]["n"])
    say(f"  {c[0]}-{c[1]}: have {S[c]['n']}, ~{per_day:.0f} a day, {-(-need // per_day):.0f} more days")
R["reshadow"] = {"changed": [f"{a}-{b}" for a, b in changed], "fresh": S3}

# 5. holdback ------------------------------------------------------------------------
say("\n== 5. After release: visible draft versus 2% blind holdback (10 days) ==")
vis, blind, accepted = L.holdback()
say(f"visible: {vis[0]}/{vis[1]} = {vis[0]/vis[1]:.2%}; blind: {blind[0]}/{blind[1]} = {blind[0]/blind[1]:.2%}"
    f" (assumption: a wrong visible draft accepted {L.P['accept_wrong_draft']:.0%} of the time)")
say(f"wrong visible drafts accepted by the adjuster (in the visible count as agreement): {accepted}")
lo, hi = wilson(*blind)
say(f"blind interval [{lo:.2%}, {hi:.2%}]")
R["holdback"] = {"visible": vis, "blind": blind, "accepted_wrong": accepted}

# 6. the examination -----------------------------------------------------------------
say("\n== 6. Examination: 20 Wisconsin homeowners denials at random, 7-23 Oct ==")
ds = L.documents("mutable")
pool, edited = L.wi_pool(ds, KEY)
say(f"pool {len(pool):,} determinations; endorsement replaced in place: {edited}")
frame = [r for r in pool if r["line"] == "HO"]
say(f"sampling frame (homeowners denials and closed without payment): {len(frame):,}")
sample = random.Random(2710).sample(frame, 20)


def rebuild(rec, how):
    p = copy.deepcopy(rec["pack"])
    doc = next(it for it in p.items if it.kind == "document")
    if how == "by_id":
        text = ds.current[rec["doc"]]
    else:
        text = ds.by_hash(rec["doc_hash"])
        if text is None:
            p.notes.append("document bytes for the recorded hash are not recoverable")
            return p
        if sha(text) != sha(ds.current[rec["doc"]]):
            p.notes.append(f"{rec['doc'][0]}@{rec['doc'][1]} was replaced in place after this "
                           f"determination; shown as recorded (hash {rec['doc_hash'][:12]}), "
                           "recovered from the nightly snapshot")
    doc.segments[0]["text"] = sha(text)[:16] + " " + text
    return p


aff = [r for r in pool if r["doc"][0] == L.WI_AMEND and r["day"] < "2026-10-16"]
say(f"pool determinations citing {L.WI_AMEND} before the edit: {len(aff)}; after: "
    f"{sum(1 for r in pool if r['doc'][0] == L.WI_AMEND and r['day'] >= '2026-10-16')}")
for how in ("by_id", "by_hash"):
    packs = [rebuild(r, how) for r in sample]
    ok = sum(p.verify(KEY) for p in packs)
    flagged = sum(bool(p.notes) for p in packs)
    say(f"  rebuild {how:<8} verify {ok}/20; notes raised {flagged}")
    R[f"exam_{how}"] = {"verify": ok, "notes": flagged}
hit = [r for r in sample if r in aff]
say("sampled determinations affected by the edit:", [r["det_id"] for r in hit])
silent = sum(1 for r in sample if r in aff)
say(f"v9.2 behaviour (no seal, document by id): {silent} packs would show text the model never saw, "
    "with nothing to say so")
ds2 = L.documents("write_once")
pool2, edited2 = L.wi_pool(ds2, KEY)
say(f"write-once store: edit refused -> {edited2}; editions of {L.WI_AMEND}: "
    f"{sorted(e for d, e in ds2.current if d == L.WI_AMEND)}")
try:
    ds2.put(L.WI_AMEND, "2025-03", "changed", "2026-10-27T09:00")
except EditInPlace as e:
    say("  overwrite attempt:", e)
R["exam"] = {"pool": len(pool), "affected": len(aff), "sampled_affected": [r["det_id"] for r in hit]}

# 7. redaction -----------------------------------------------------------------------
say("\n== 7. Redaction for the examiner's copy: third-party and payment details ==")
for naive in (True, False):
    packs = [rebuild(r, "by_hash") for r in sample]
    res = [redact(p, {"third_party", "payment"}, naive=naive, key=KEY) for p in packs]
    ok = sum(p.verify(KEY) for p in packs)
    say(f"  {'edit and rehash' if naive else 'keep commitments':<17} redacted {sum(r['redacted'] for r in res)}"
        f"  refused (quoted evidence) {sum(len(r['refused']) for r in res)}  verify {ok}/20")
    R[f"redact_{'naive' if naive else 'commit'}"] = {"verify": ok, "redacted": sum(r["redacted"] for r in res),
                                                    "refused": sum(len(r["refused"]) for r in res)}
    if not naive:
        keep = packs

# 8. explain -------------------------------------------------------------------------
say("\n== 8. explain() for one sampled determination ==")
pick = next((p for p in keep if p.notes), keep[0])
say(explain(pick, KEY))
same_twice = explain(pick, KEY) == explain(copy.deepcopy(pick), KEY)
say("deterministic:", same_twice)

(OUT / "results.json").write_text(json.dumps(R, indent=1, default=str))
(HERE / "output.txt").write_text("\n".join(LINES) + "\n")
