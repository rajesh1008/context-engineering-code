"""Chapter 44 lab: govern the copies, not just the documents.

Run from 04-code/ch44:  python3 lab.py > output.txt   (writes out/results.json)
Deterministic stand-ins in lakeshore44.py; no network, standard library only.
"""
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import lakeshore44 as L                                                  # noqa: E402
from contextlab.policy import Policy                                     # noqa: E402
from contextlab.governance import (VERSION, Register, TermAwarePolicy,   # noqa: E402
                                   reconcile, classify, impact, fan_out, exit_drill,
                                   coverage, calendar)

OUT = {"version": VERSION}


def section(t):
    print(f"\n== {t}")


def treatments():
    end = L.SPANS[2]
    return {
        "no record": Policy(L.SPANS),
        "end date": Policy([replace(end, effective_to=L.RETIRE) if s is end else s
                            for s in L.SPANS]),
        "term-aware": TermAwarePolicy(L.SPANS, {L.END: L.RETIRE}),
    }


def replay(pol, losses):
    rows = []
    for loss, ts in losses:
        res = pol.resolve([s.sid for s in L.SPANS], L.request(loss, ts))
        rows.append((loss, ts, res.governing, res.governing == L.truth(ts)))
    return rows


# ------------------------------------------------------------------ 1
section("1. The incident: Michigan loss-payment determinations, 1 Sep to 27 Oct 2026")
losses = L.michigan_losses()
to_date = [x for x in losses if x[0] <= date(2026, 10, 27)]
rows = replay(treatments()["no record"], to_date)
new_terms = [r for r in rows if r[1] >= L.RETIRE]
wrong = [r for r in rows if not r[3]]
print(f"determinations: {len(rows)}; on terms that began on or after 1 Sep: {len(new_terms)}")
print(f"cited the retired endorsement where the new edition governs: {len(wrong)}")
print("each of those drafts omitted the 15-day written notice the new edition requires")
OUT["incident"] = {"dets": len(rows), "new_terms": len(new_terms), "wrong": len(wrong)}

# ------------------------------------------------------------------ 2
section("2. Three ways to record a retirement, Sep 2026 to Aug 2027")
series = {}
print(f"{'treatment':<12}{'dets':>6}{'wrong':>7}{'per 1,000':>11}")
for name, pol in treatments().items():
    r = replay(pol, losses)
    w = [x for x in r if not x[3]]
    by = Counter(x[0].strftime("%Y-%m") for x in w)
    months = sorted({x[0].strftime("%Y-%m") for x in r})
    series[name] = [by.get(m, 0) for m in months]
    print(f"{name:<12}{len(r):>6}{len(w):>7}{1000 * len(w) / len(r):>11.0f}")
    sample = next((x for x in w), None)
    if sample:
        print(f"   e.g. loss {sample[0]}, term from {sample[1]}: rules chose {sample[2]}")
OUT["treatments"] = {"months": months, "series": series,
                     "totals": {k: sum(v) for k, v in series.items()}, "dets": len(losses)}

# ------------------------------------------------------------------ 3
section("3. Reconciliation against the catalogue")
docs, spans_of = L.build_catalogue()
copies = L.build_copies(docs, spans_of)
clean = reconcile(docs, copies, L.LIVE_MODEL)
print(f"catalogue: {len(docs)} document versions; copies: {len(copies)} rows "
      f"in {len({c.store for c in copies})} stores")
print(f"clean run: {len(clean)} findings")
docs2, copies2, log = L.inject(docs, copies)
found = reconcile(docs2, copies2, L.LIVE_MODEL)
cls = Counter(f[2] for f in found)
by_store = defaultdict(Counter)
for f in found:
    by_store[f[0]][f[2]] += 1
print(f"after the year's eight defects: {len(found)} findings")
for k, v in sorted(cls.items()):
    print(f"  {k:<16}{v:>4}  " + ", ".join(f"{s} {n}" for s, n in sorted(
        (s, c[k]) for s, c in by_store.items() if c[k])))
expected = set(log.values())
seen = set(cls)
print(f"defects found: {len(expected & seen)} of {len(expected)}; "
      f"finding classes with no injected cause: {len(seen - expected)}")
caught = [d for d, c in log.items() if c in L.EVENT_CAUGHT]
print(f"defects the event-driven checks caught: {len(caught)} of {len(log)}")
for d, c in log.items():
    print(f"  {d:<42}{c:<16}{L.EVENT_CAUGHT.get(c, 'nothing fired')}")
avg = sum(spans_of.values()) / len(spans_of)
print(f"if lifecycle state were inside the embedded text, the retirement would re-embed "
      f"{spans_of[L.END]} spans and the quarter's 31 status changes about {round(31 * avg)}; "
      f"kept beside the vector, they re-embed none")
OUT["reconcile"] = {"docs": len(docs), "copies": len(copies), "clean": len(clean),
                    "findings": len(found), "classes": dict(cls),
                    "by_store": {s: dict(c) for s, c in by_store.items()},
                    "defects": len(log), "event_caught": len(caught)}

# ------------------------------------------------------------------ 4
section("4. Embedding refresh: change classes")
rows4 = []
print(f"{'change':<9}{'touches':<10}{'rows':>7}  {'class':<10}{'re-embed':<9}gates")
for c in L.CHANGES:
    k, re, g = classify(c)
    rows4.append((c.cid, c.touches, c.spans, k, re))
    print(f"{c.cid:<9}{c.touches:<10}{c.spans:>7}  {k:<10}{'yes' if re else 'no':<9}{len(g)}")
OUT["changes"] = rows4
rng = random.Random(1006)
old = [f"END-{i:03d}@v#{j}" for i in range(146) for j in range(6)]
mapping = {}
for o in old:
    x = rng.random()
    mapping[o] = None if x < 0.02 else o.replace("#", "#c")
dependants = {"claim memory": rng.sample(old, 310), "result cache": rng.sample(old, 120),
              "gold set": rng.sample(old, 236), "evidence packs": rng.sample(old, 540)}
plan = impact(mapping, dependants)
print("restatement plan for CH-1006 (span ids change):")
for s, p in plan.items():
    print(f"  {s:<16} refs {p['refs']:>4}  remap {p['remap']:>4}  unmappable {p['unmappable']:>3}")
OUT["plan"] = plan

# ------------------------------------------------------------------ 5
section("5. The asset register")
for v in ("v0.9", "v1.0"):
    reg = Register(L.register(v))
    g = reg.gaps()
    print(f"register {v}: {len(reg.assets)} assets, {len(g)} gaps")
    for a, why in g:
        print(f"  - {a}: {why}")
    OUT[f"gaps_{v}"] = len(g)
reg = Register(L.register("v1.0"))
print("downstream of evidence packs:", ", ".join(reg.downstream("evidence packs")))
print("downstream of forms catalogue:", ", ".join(reg.downstream("forms catalogue")))

# ------------------------------------------------------------------ 6
section("6. Erasure ER-2026-0131 across the register")
for v in ("v0.9", "v1.0"):
    plan6, un = fan_out(Register(L.register(v)), L.ERASURE_HOLDINGS)
    held = {k: x for k, x in plan6.items() if x[1]}
    print(f"register {v}: {len(held)} assets hold rows; no answer for {len(un)}: {', '.join(un)}")
    if v == "v1.0":
        for k, (ans, n) in held.items():
            print(f"  {k:<28}{n:>3}  {ans}")
        OUT["erasure"] = {k: [a, n] for k, (a, n) in held.items()}
    OUT[f"erasure_unanswered_{v}"] = len(un)

# ------------------------------------------------------------------ 7
section("7. Partner exit drill: Northwoods Savings")
for v in ("v0.9", "v1.0"):
    d = exit_drill(Register(L.register(v)), L.EXIT_HOLDINGS)
    bad = [k for k, x in d.items() if x[0].startswith("cannot")]
    print(f"register {v}: {len(d)} assets hold partner rows "
          f"({sum(x[1] for x in d.values())} rows); cannot select by tenant: {bad or 'none'}")
    OUT[f"exit_{v}"] = {k: list(x) for k, x in d.items()}

# ------------------------------------------------------------------ 8
section("8. Policy to evidence")
after = {p: {k: (v or "added in ch44") for k, v in links.items()}
         for p, links in L.CONTROLS_BEFORE.items()}
for name, ctl in (("before", L.CONTROLS_BEFORE), ("after", after)):
    n, rows8 = coverage(ctl)
    print(f"{name}: {n} of {len(ctl)} controls complete")
    if name == "before":
        for p, miss in rows8.items():
            print(f"  {p:<40}missing: {', '.join(miss) or '-'}")
        OUT["coverage"] = {p: len(m) for p, m in rows8.items()}

# ------------------------------------------------------------------ 9
section("9. The governance calendar for 2027")
ev = calendar(2027, {"Fox Valley Bank": date(2027, 4, 1)})
c9 = Counter(e[1] for e in ev)
print(f"{len(ev)} events: " + ", ".join(f"{k} {v}" for k, v in sorted(c9.items())))
for e in ev:
    if e[1] not in ("weekly", "monthly"):
        print(f"  {e[0]}  {e[1]:<12}{e[2]}")
OUT["calendar"] = [(str(d), k, t) for d, k, t in ev]

(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "results.json").write_text(json.dumps(OUT, indent=1, default=str))
print("\nwrote out/results.json")
