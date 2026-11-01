"""Chapter 18 lab: agentic retrieval on Lakeshore's hard claims.

Run from 04-code/ch18:  PYTHONPATH=.. python3 lab.py > output.txt
"""
import json
import statistics as st
from collections import Counter

import corpus18 as C
import model18 as M
from contextlab.agentic import Budget, cost_usd, route

HARD = C.claims(M.CORPUS, 160, True, 181)
EASY = C.claims(M.CORPUS, 240, False, 182)
MIX_HARD = C.claims(M.CORPUS, 65, True, 1000)      # the 65 graph-routed claims inside the 1,000 mix (Ch 17)
OUT = {}


def say(*a):
    print(*a)


def summary(rows):
    n = len(rows)
    sent = sum(r[2]["sentences"] for r in rows)
    return {"n": n, "wrong": sum(r[2]["wrong"] for r in rows),
            "faith": round(sum(r[2]["supported"] for r in rows) / sent, 3),
            "tokens": round(st.mean(r[1].tokens for r in rows)),
            "calls": round(st.mean(r[1].calls for r in rows), 2),
            "rounds": round(st.mean(r[1].rounds for r in rows), 2),
            "cost": round(st.mean(cost_usd(r[1]) for r in rows), 5),
            "model_calls": round(st.mean(r[1].model_calls for r in rows), 2)}


def line(name, s):
    say(f"{name:<34} wrong {s['wrong']:>3}/{s['n']:<3} faithfulness {s['faith']:.3f} "
        f"tokens {s['tokens']:>6,} tool calls {s['calls']:>5.2f} US${s['cost']:.4f}")


print("corpus:", Counter(s.kind for s in M.CORPUS.values()), "| hard claims", len(HARD),
      "| easy claims", len(EASY), "| governing spans per hard claim, mean",
      round(st.mean(len(c.required) for c in HARD), 2))

# == 1. One claim, three ways ===============================================
say("\n== 1. One hard claim, three ways")
ex = next(c for c in HARD if c.cid == "H181-075")
say("claim:", ex.question)
say("policy record:", ex.policy.schedule)
say("governing spans:", sorted(ex.required))
for way in ("fixed", "graph", "agentic"):
    c, tr, sc = M.evaluate([ex], way)[0]
    say(f"  {way:<8} packet {len(tr.packet):>2} spans, missing {sorted(sc['missing']) or 'none'}, "
        f"wrong={sc['wrong']}, tokens {tr.tokens:,}")
tr = M.agentic(ex)
for rnd, tool, arg, hits in tr.log:
    a = arg if isinstance(arg, str) else arg[0]
    say(f"    round {rnd} {tool:<6} {a[:58]:<58} -> {', '.join(hits[:3])}{' ...' if len(hits) > 3 else ''}")
say(f"    stop: {tr.stop} after {tr.rounds} rounds, {tr.calls} tool calls")
OUT["example"] = {"log": [(r, t, a if isinstance(a, str) else a[0], h) for r, t, a, h in tr.log]}

say("\nhard claims (160) and easy claims (240), three ways:")
OUT["three"] = {}
for nm, S in (("hard", HARD), ("easy", EASY)):
    for way in ("fixed", "graph", "agentic"):
        s = summary(M.evaluate(S, way)); OUT["three"][f"{nm}-{way}"] = s
        line(f"  {nm:<5} {way}", s)

# == 2. Query transformations ===============================================
say("\n== 2. Query transformations, one shot, no authority filter (hard claims)")
def hyde(c):
    words = set(M.rewrite(c.question).lower().split())
    f = max(C.FAM, key=lambda x: len(words & set(C.FAM[x][3].split())))
    current = C.EDITIONS[f][-1]                 # what the model remembers
    return M.CORPUS[f"LK-{f}@{current}"].text.replace(f"LK-{f} ", "")

def decompose(c):
    perils = [M.rewrite(C.PHRASE[f]) + " endorsement" for f in c.entries]
    bulletins = [f"{C.STATES[c.policy.state]} bulletin " + C.FAM[f][0]
                 for f in c.entries]
    return perils + bulletins
OUT["qt"] = {}
for name, qf in (("raw question", lambda c: [c.question]), ("rewritten", lambda c: [M.rewrite(c.question)]),
                 ("decomposed", decompose), ("HyDE", lambda c: [hyde(c)])):
    rec, top1_ok, top1_n = [], 0, 0
    for c in HARD:
        got = set()
        for q in qf(c):
            got |= set(M.search(q, c, 10, authority=False))
        ends = {r for r in c.required if r.startswith("LK-")}
        rec.append(len(ends & got) / len(ends))
        first = next((h for h in M.search(qf(c)[0], c, 10, kind="endorsement", authority=False)), None)
        if first and M.CORPUS[first].family == c.entries[0]:
            top1_n += 1; top1_ok += c.policy.schedule[c.entries[0]] == M.CORPUS[first].edition
    OUT["qt"][name] = {"recall": round(st.mean(rec), 3), "edition_ok": round(top1_ok / max(top1_n, 1), 3)}
    say(f"  {name:<14} governing endorsements in top 10: {st.mean(rec):.3f}   "
        f"top endorsement on the policy's edition: {top1_ok}/{top1_n}")
old = [c for c in HARD if c.policy.schedule[c.entries[0]] != C.EDITIONS[c.entries[0]][-1]]
say(f"  hard claims whose policy carries an older edition of the first endorsement: {len(old)}/{len(HARD)}")

# == 3. The first move ======================================================
say("\n== 3. The first move (slot check, default budget)")
OUT["first"] = {}
for nm, S in (("hard", HARD), ("easy", EASY)):
    for first in ("decompose", "fixed"):
        s = summary(M.evaluate(S, "agentic", first=first)); OUT["first"][f"{nm}-{first}"] = s
        line(f"  {nm:<5} first move = {first}", s)

# == 4. Sufficiency checks ==================================================
say("\n== 4. Sufficiency checks (hard claims)")
def first_complete(c, tr):
    return next((i + 1 for i, p in enumerate(tr.snapshots) if c.required <= set(p)), None)
OUT["judges"] = {}
for judge in ("always", "self", "slots"):
    rows = M.evaluate(HARD, "agentic", judge=judge)
    s = summary(rows)
    early = sum(1 for c, tr, sc in rows if sc["missing"])
    over = sum(tr.rounds - first_complete(c, tr) for c, tr, sc in rows if first_complete(c, tr))
    stops = Counter(tr.stop for _, tr, _ in rows)
    s.update(early=early, over=over, stops=dict(stops)); OUT["judges"][judge] = s
    line(f"  judge = {judge}", s)
    say(f"      stopped with a governing span missing: {early}; rounds after the packet was complete: {over}; stops {dict(stops)}")
slots_rows = M.evaluate(HARD, "agentic")
opened = Counter(n.split(":")[0] for c in HARD for n in
                 [x.name for x in M.slot_check(c)(M.State(c.question, {}, packet=M.fixed(c)))[1]])
say("  slots open after the first move, by kind:", dict(opened))
OUT["slots_open"] = dict(opened)

# == 5. Failure injection: a judge that is always satisfied ===================
say("\n== 5. Failure injection: the always-satisfied judge")
rows = M.evaluate(HARD, "agentic", judge="always")
conf = sum(1 for _, _, sc in rows if sc["wrong"])
say(f"  hard claims answered after one round: {len(rows)}; wrong {conf}; "
    f"every one stopped as 'sufficient': {all(tr.stop == 'sufficient' for _, tr, _ in rows)}")
miss = Counter(M.CORPUS[m].kind for _, _, sc in rows for m in sc["missing"])
say("  governing spans missing, by kind:", dict(miss))
OUT["always_missing"] = dict(miss)

# == 6. Round cap ===========================================================
say("\n== 6. Round cap (hard claims, slot check)")
OUT["cap"] = []
for cap in range(1, 7):
    s = summary(M.evaluate(HARD, "agentic", budget=Budget(max_rounds=cap)))
    OUT["cap"].append({"cap": cap, **s}); line(f"  max rounds {cap}", s)
dist = Counter(tr.rounds for _, tr, _ in slots_rows)
say("  rounds used with cap 6:", dict(sorted(dist.items())))
OUT["rounds_dist"] = dict(dist)

# == 7. Cost ================================================================
say("\n== 7. Cost per determination (hard claims)")
OUT["cost"] = {}
for way, kw in (("fixed", {}), ("graph", {}), ("agentic", {})):
    rows = M.evaluate(HARD, way, **kw)
    parts = {k: round(st.mean(tr.tokens_in[k] for _, tr, _ in rows)) for k in ("plan", "read", "judge", "answer")}
    parts["output"] = round(st.mean(tr.tokens_out for _, tr, _ in rows))
    lat = sorted(1.4 * tr.model_calls + 0.25 * tr.calls for _, tr, _ in rows)
    p50, p95 = lat[len(lat) // 2], lat[int(len(lat) * 0.95)]
    OUT["cost"][way] = {**parts, "p50": round(p50, 1), "p95": round(p95, 1),
                        "usd": round(st.mean(cost_usd(tr) for _, tr, _ in rows), 5)}
    say(f"  {way:<8} input by part {parts}  latency p50 {p50:.1f}s p95 {p95:.1f}s  "
        f"US${OUT['cost'][way]['usd']:.4f}")

# == 8. Isolation ===========================================================
say("\n== 8. Search sub-agent: lead context with and without isolation (hard claims)")
OUT["iso"] = {}
for iso in (True, False):
    rows = M.evaluate(HARD, "agentic", isolate=iso)
    peak = st.mean(max(tr.lead) for _, tr, _ in rows)
    s = summary(rows); s["lead_peak"] = round(peak); OUT["iso"][str(iso)] = s
    say(f"  isolate={iso!s:<5} peak lead context {peak:>6,.0f} tokens; total tokens {s['tokens']:,}; "
        f"model calls {s['model_calls']}")
deep = max(HARD, key=lambda c: M.agentic(c).rounds * 100 + len(c.required))
OUT["iso_example"] = {str(i): M.agentic(deep, isolate=i).lead for i in (True, False)}
say(f"  deepest claim {deep.cid}: lead context by round, isolated {OUT['iso_example']['True']}, "
    f"not isolated {OUT['iso_example']['False']}")

# == 9. When a fixed pipeline wins: routing =================================
say("\n== 9. Routing (graph walk unless the router sends the claim to the loop)")
def feats(c, edges=False):
    onward = sum(1 for f in c.cause_codes for t, _ in C.REFS[f] if t in c.policy.schedule)
    f = {"perils": len(c.cause_codes), "onward_refs": onward}
    if edges:      # does the claim's chain cross an edge the graph team has quarantined?
        f["untrusted_edge"] = any((a, t) in M.QUARANTINED for a in c.cause_codes
                                  for t, _ in C.REFS[a] if t in c.policy.schedule)
    return f
OUT["route"] = {}
for ver, edges in (("v1", False), ("v2", True)):
    for nm, S in (("hard", HARD), ("easy", EASY)):
        routes = [route(feats(c, edges)).name for c in S]
        rows = [M.evaluate([c], "agentic" if r == "agentic" else "graph")[0] for c, r in zip(S, routes)]
        s = summary(rows); s["to_loop"] = routes.count("agentic"); OUT["route"][f"{ver}-{nm}"] = s
        line(f"  router {ver} {nm:<5} ({s['to_loop']:>3} to loop)", s)

# == 10. The loop that never stopped =========================================
say("\n== 10. A check that can never be satisfied (demands a bulletin for every endorsement)")
OUT["never"] = {}
for label, b, iso in (("budget 6 rounds / 20k tokens", Budget(), True),
                      ("cap 30 rounds only, isolated", Budget(30, 10**9, 10**9), True),
                      ("cap 30 rounds only, not isolated", Budget(30, 10**9, 10**9), False)):
    R = [M.agentic(c, judge="never", budget=b, isolate=iso) for c in HARD]
    s = {"tokens": round(st.mean(r.tokens for r in R)), "max": max(r.tokens for r in R),
         "rounds": round(st.mean(r.rounds for r in R), 1),
         "usd": round(st.mean(cost_usd(r) for r in R), 4), "stops": dict(Counter(r.stop for r in R))}
    OUT["never"][label] = s
    say(f"  {label:<34} mean tokens {s['tokens']:>7,} max {s['max']:>8,} rounds {s['rounds']:>5} "
        f"US${s['usd']:.4f} stops {s['stops']}")

# == 11. Scoreboard =========================================================
say("\n== 11. Scoreboard inputs: the 65 graph-routed claims inside the 1,000-determination mix")
OUT["mix"] = {}
for way in ("graph", "agentic"):
    s = summary(M.evaluate(MIX_HARD, way)); OUT["mix"][way] = s
    line(f"  {way}", s)
routes = [route(feats(c, True)).name for c in MIX_HARD]
rows = [M.evaluate([c], "agentic" if r == "agentic" else "graph")[0] for c, r in zip(MIX_HARD, routes)]
s = summary(rows); s["to_loop"] = routes.count("agentic"); OUT["mix"]["routed"] = s
line(f"  routed v2 ({s['to_loop']} to the loop)", s)
json.dump(OUT, open("out/results.json", "w"), indent=1, default=str)
