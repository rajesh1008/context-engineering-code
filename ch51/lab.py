"""Chapter 51 lab: the last twenty percent.

Run from 04-code/ch51:  python3 lab.py > output.txt   (writes out/results.json)
Deterministic stand-ins in lakeshore51.py; no network, standard library only.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import lakeshore51 as L                                                   # noqa: E402
from contextlab.finish import (VERSION, prioritise, run_checklist, drill,  # noqa: E402
                               compare, targets)

OUT = {"version": VERSION}


def section(t):
    print(f"\n== {t}")


def s1_targets():
    section("1. Margaret's targets, before the last twenty percent")
    sb = {"lookup_minutes": 1.29, "wrong_clause_pct": 0.3, "traceable_pct": 100 * 29 / 30}
    res = targets(sb, L.GOALS)
    for k, v in res.items():
        print(f"{k:18s} baseline {L.BASELINE[k]:>6}  now {v['value']:>7.2f}  target {v['target']:>7}  met {v['met']}")
    OUT["targets_before"] = res
    return res


def s2_backlog():
    section("2. Edge-case backlog: frequency order vs escaped harm per day")
    by_f = prioritise(L.BACKLOG, "frequency")
    by_p = prioritise(L.BACKLOG, "priority")
    print("rank  by frequency            by priority (escaped harm / day)")
    for i, (a, b) in enumerate(zip(by_f, by_p), 1):
        print(f"{i:>4}  {a.key} {a.per_10k:>5.1f}/10k        {b.key} harm {b.escaped_harm():>5.2f} pri {b.priority():.3f}")
    top5f = {c.key for c in by_f[:5]}
    top5p = {c.key for c in by_p[:5]}
    total = sum(c.escaped_harm() for c in L.BACKLOG)
    top5_share = sum(c.escaped_harm() for c in by_p[:5]) / total
    top5_days = sum(c.days for c in by_p[:5])
    f5_share = sum(c.escaped_harm() for c in by_f[:5]) / total
    f5_days = sum(c.days for c in by_f[:5])
    print(f"top five shared by both orders: {sorted(top5f & top5p)}")
    print(f"priority top five: {top5_share:.0%} of escaped harm for {top5_days:g} days")
    print(f"frequency top five: {f5_share:.0%} of escaped harm for {f5_days:g} days")
    print(f"backlog: {len(L.BACKLOG)} items, {sum(c.days for c in L.BACKLOG):g} days")
    OUT["backlog"] = {"priority": [c.key for c in by_p], "frequency": [c.key for c in by_f],
                      "top5_share": round(top5_share, 4), "top5_days": top5_days,
                      "f5_share": round(f5_share, 4), "f5_days": f5_days,
                      "harm": {c.key: round(c.escaped_harm(), 2) for c in L.BACKLOG}}


def s3_checklist():
    section("3. Hardening checklist (24 checks)")
    for name, cfg in (("before", L.CONFIG_BEFORE), ("after", L.CONFIG_AFTER)):
        res, failed = run_checklist(L.CHECKS, cfg)
        print(f"{name}: {sum(res.values())}/{len(res)} pass; failed: {[c.key for c in failed]}")
        for c in failed:
            print(f"   {c.key} [{c.area}] {c.text}")
        OUT[f"checklist_{name}"] = [c.key for c in failed]
    broken = dict(L.CONFIG_AFTER)
    del broken["behaviour_probes"]
    _, failed = run_checklist(L.CHECKS, broken)
    print(f"evidence missing for H24 -> fails closed: {[c.key for c in failed]}")


def s4_runbooks():
    section("4. Runbooks, timed in a tabletop drill (minutes to contained)")
    rows = {}
    for rb in L.RUNBOOKS:
        d = L.DISCOVERY[rb.key]
        w, wo = drill(rb, d, True), drill(rb, d, False)
        rows[rb.key] = (w, wo)
        print(f"{rb.key} {rb.trigger[:52]:52s} owner {rb.owner:30s} with {w:>4}  without {wo:>4}")
    OUT["runbooks"] = rows
    print(f"total with {sum(w for w, _ in rows.values())}, without {sum(wo for _, wo in rows.values())}")


def s5_exam():
    section("5. Examination drill: 30 determinations, 1 Oct to 27 Nov 2026")
    s = L.exam_sample()
    ok = [x for x in s if x["verifies"] and x["reason_code"]]
    print(f"packs verify by hash: {sum(x['verifies'] for x in s)}/30")
    print(f"complete (verifies and carries model or manual reason): {len(ok)}/30")
    miss = [x["id"] for x in s if not x["reason_code"]]
    print(f"incomplete: {miss} (drafted by hand during a gateway outage; no reason code)")
    print(f"minutes to produce, total {sum(x['minutes'] for x in s):.1f}, max {max(x['minutes'] for x in s)}")
    OUT["exam"] = {"verify": 30, "complete": len(ok), "missing": miss,
                   "minutes": round(sum(x["minutes"] for x in s), 1)}


def s6_next_model():
    section("6. A model two generations newer, on the same diagnostics")
    gold = L.gold_items()
    prod, cand = L.run_model(gold, "production"), L.run_model(gold, "candidate")
    for label, adapter in (("as delivered", False), ("after output-format adapter", True)):
        rows = compare(L.diagnostics(adapter), prod, cand)
        print(f"-- candidate {label}")
        for r in rows:
            print(f"   {r['diagnostic']:34s} production {r['production']:>9.2f}  candidate {r['candidate']:>9.2f}  alarm {r['alarm']}")
        OUT[f"next_{label.split()[0]}"] = rows
    probes = L.probe_items()
    pw = sum(r["wrong"] for r in L.run_model(probes, "production"))
    cw = sum(r["wrong"] for r in L.run_model(probes, "candidate"))
    print(f"gold items exercising 'older bulletin narrows newer endorsement': {sum(i['precedence_probe'] for i in gold)}")
    print(f"behaviour probes ({L.PROBES}): production wrong {pw}, candidate wrong {cw}")
    OUT["probes"] = {"n": L.PROBES, "production": pw, "candidate": cw}


def s7_targets_after():
    section("7. Margaret's targets, after")
    sb = {"lookup_minutes": 1.29, "wrong_clause_pct": 0.3, "traceable_pct": 100.0}
    res = targets(sb, L.GOALS)
    for k, v in res.items():
        print(f"{k:18s} now {v['value']:>7.2f}  target {v['target']:>7}  met {v['met']}")
    OUT["targets_after"] = res


if __name__ == "__main__":
    print(f"contextlab {VERSION}; Chapter 51 lab; deterministic stand-ins")
    s1_targets(); s2_backlog(); s3_checklist(); s4_runbooks(); s5_exam(); s6_next_model(); s7_targets_after()
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "results.json").write_text(json.dumps(OUT, indent=1, default=str))
