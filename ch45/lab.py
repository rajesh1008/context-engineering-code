"""Chapter 45 lab: a simulated year of operating Lakeshore's context layer.

Run from 04-code:  python3 ch45/lab.py > ch45/output.txt
No model is called. The year (November 2026 to October 2027) is the stand-in in
lakeshore45.py; every number after November 2026 is illustrative. The tools under
test are contextlab.operations v9.8.
"""
import json
import os
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import lakeshore45 as L                                                    # noqa: E402
from contextlab.operations import (VERSION, Artefact, Calendar, Inventory, Job,   # noqa: E402
                                   TunedParam, headroom, mix_adjusted, replay_compare,
                                   stale_params)

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
RES = {"version": VERSION}
POP = 20000          # simulated determinations per month: the "true" rate
REVIEW = 1000        # the monthly reviewed sample Lakeshore actually sees


def section(t):
    print(f"\n== {t}")


def run_month(m, reg, n, tag):
    wins = L.windows(reg)
    return [L.decide(c, reg, wins) for c in L.claims(m, n, tag)]


def rate(rs, outcome):
    return sum(r["outcome"] == outcome for r in rs) / len(rs)


# ------------------------------------------------------------------ 1
section("1. The calendar: every scheduled job is an unattended loop")
found = Calendar([
    Job("nightly ingest and re-index", "daily", "ingest pipeline", stop="queue empty",
        state="ingest ledger", verifier="publication gate", budget="", human=""),
    Job("freshness check (Ch24)", "daily", "freshness monitor", "change log reconciled",
        "freshness ledger", "change-log diff", "alert after 2 h", "on-call engineer"),
    Job("fingerprint probe (Ch14)", "daily", "probe", "probe set scored", "registry",
        "tolerance 0.999", "probe set fixed", "on-call engineer"),
    Job("memory consolidation (Ch22)", "daily", "consolidation job", "batch written",
        "memory store", "consolidation job", "batch cap", "memory owner"),
    Job("earlier-claim re-validation (Ch23)", "daily", "re-validator", "due list empty",
        "memory store", "policy record", "", ""),
    Job("weekly sample report (Ch39)", "weekly", "evaluator", "sample scored",
        "report archive", "adjuster labels", "sample size fixed", "Margaret Hale"),
    Job("red-team suite (Ch37)", "weekly", "red team runner", "240 attacks run",
        "results store", "harm checker", "suite fixed", "security lead"),
    Job("quarterly re-baseline", "quarterly", "evaluator", "gold set scored",
        "", "adjuster labels", "gold set fixed", "Laura Jensen"),
    Job("retire-by review (Ch14)", "quarterly", "registry", "every entry reviewed",
        "registry", "platform lead", "one meeting", "Laura Jensen"),
])
audit = found.audit()
print(f"As found, November 2026: {len(found.jobs)} jobs, {found.runs_per_year()} runs a year, "
      f"{len(audit)} with gaps")
for name, gaps in audit.items():
    print(f"  {name:38s} missing: {', '.join(gaps)}")
fixed = Calendar([Job(j.name, j.cadence, j.assembler, j.stop or "queue empty or budget spent",
                      j.state or "baseline archive (never overwritten)",
                      "adjuster sample" if j.verifier == j.assembler else j.verifier,
                      j.budget or "pages per night capped; bulletins first",
                      j.human or "context platform on-call") for j in found.jobs] + [
    Job("re-tune on growth", "event", "tuning harness", "headroom back under alarm",
        "tuned-at manifest", "shadow on production index", "one week of shadow", "Laura Jensen"),
    Job("upgrade replay", "event", "replay harness", "every slice within tolerance",
        "replay archive", "per-slice comparison", "2,000 determinations", "Laura Jensen"),
    Job("pending-bulletin flag", "event", "freshness monitor", "bulletin published",
        "flag register", "compliance", "until publication", "Margaret Hale"),
])
print(f"As operated: {len(fixed.jobs)} jobs ({len(fixed.jobs) - len(found.jobs)} event-driven added), "
      f"{fixed.runs_per_year()} scheduled runs a year, {len(fixed.audit())} with gaps")
RES["calendar"] = {"found_jobs": len(found.jobs), "found_gaps": {k: v for k, v in audit.items()},
                   "fixed_jobs": len(fixed.jobs), "runs_per_year": fixed.runs_per_year()}

# ------------------------------------------------------------------ 2
section("2. Corpus growth: what doubled, and which signal saw it first")
months = []
for m in range(12):
    mid = L.FIRST[m] + timedelta(days=14)
    months.append(L.pages(mid))
print("Corpus pages, mid-month:", ", ".join(f"{L.SHORT[m]} {p:,}" for m, p in enumerate(months)))
print(f"Growth over the year: {months[0]:,} -> {months[-1]:,} ({months[-1] / months[0]:.2f}x); "
      f"notes migrated on {L.NOTES_INGEST}: {L.corpus(L.NOTES_INGEST)['notes']:,} pages")
k_param = TunedParam("reranker pool K", 40, {"corpus_pages": L.pages(date(2026, 11, 1)),
                                            "embed_model": "open-base@1.3"})
first_stale = None
d = date(2026, 11, 1)
while d <= date(2027, 10, 31):
    if stale_params([k_param], {"corpus_pages": L.pages(d), "embed_model": "open-base@1.3"}):
        first_stale = d
        break
    d += timedelta(days=1)
print(f"Tuned-at manifest: K tuned at {k_param.tuned_at['corpus_pages']:,} pages; "
      f"growth limit 25% first exceeded on {first_stale}: "
      f"{stale_params([k_param], {'corpus_pages': L.pages(first_stale)})[0][1]}")

base_weeks, alarm_week, weekly = [], None, []
wk = date(2026, 11, 2)
consec = 0
while wk < date(2027, 10, 25):
    import random as _r
    rng = _r.Random(f"wk{wk}")
    cl = L.claims(L.month_of(wk), 22400, tag=f"wk{wk}")
    for c in cl:
        c.day = wk + timedelta(days=rng.randrange(7))
    ranks = [1 + int(L.n_eff(c.day, L.AS_RUN.partitioned(c.day)) * c.q) for c in cl]
    h = headroom(ranks, L.K)
    weekly.append((str(wk), h))
    if wk < date(2027, 1, 1):
        base_weeks.append(h)
    else:
        b = sum(base_weeks) / len(base_weeks)
        consec = consec + 1 if h > 2 * b else 0
        if consec == 2 and alarm_week is None:
            alarm_week = wk
    wk += timedelta(days=7)
hb = sum(base_weeks) / len(base_weeks)
print(f"Headroom (cited governing span in ranks 31-40 of 40), weekly, label-free: baseline "
      f"{hb:.2%}; alarm (over 2x baseline two weeks running) in the week of {alarm_week}")
RES["growth"] = {"pages": months, "first_stale": str(first_stale), "headroom_base": hb,
                 "alarm_week": str(alarm_week), "weekly_headroom": weekly}

pop = {reg.name: [run_month(m, reg, POP, "pop") for m in range(12)] for reg in (L.AS_RUN, L.AS_OPERATED)}
rev = {reg.name: [run_month(m, reg, REVIEW, "review") for m in range(12)] for reg in (L.AS_RUN, L.AS_OPERATED)}
gold_frozen = sum(r["hit"] for r in pop["as run"][0]) / POP
apr = pop["as run"][5]
print(f"April re-baseline, as run: gold hit@40 on the frozen November snapshot {gold_frozen:.2%}; "
      f"the same measure on the production-built index {sum(r['hit'] for r in apr) / POP:.2%}")
for label, m in (("Month one (Nov 2026)", 0), ("Month nine (Jul 2027)", 8)):
    rs, rv = pop["as run"][m], rev["as run"][m]
    print(f"{label}, as run: wrong-clause {rate(rs, 'wrong') * 1000:.1f}/1,000 (reviewed sample "
          f"{sum(r['outcome'] == 'wrong' for r in rv)}/1,000), hit@40 {sum(r['hit'] for r in rs) / POP:.2%}, "
          f"headroom {headroom([r['rank'] for r in rs], L.K):.2%}")

section("2b. Re-tune options on the May 2027 corpus (as run had none of them)")
may = L.claims(6, POP, "pop")
opts = [("do nothing (K 40, notes in pool)", None, 40), ("widen pool to K 80", None, 80),
        ("partition notes (K 40)", date(2027, 3, 26), 40), ("partition notes and K 80", date(2027, 3, 26), 80)]
table = []
for name, part, k in opts:
    reg = L.Regime(name, partition_from=part, floors=[], memory_from=L.CUTOVER, k=k)
    rs = [L.decide(c, reg) for c in may]
    notes_tok = 52 if part is None else 14
    cost = L.cost_per_det(L.instruction_lines(False)[6], notes_tok, k)
    row = (name, sum(r["hit"] for r in rs) / POP, rate(rs, "wrong") * 1000, cost)
    table.append(row)
    print(f"  {name:34s} hit@K {row[1]:.2%}  wrong-clause {row[2]:.1f}/1,000  cost US${row[3]:.4f}")
RES["retune_options"] = table

# ------------------------------------------------------------------ 3
section("3. Bulletin season: effective before published")
items = L.bulletins()
print(f"{len(items)} bulletins and editions in the year, {sum(b.emergency for b in items)} emergency "
      f"(effective on issue); per month: " + ", ".join(f"{L.SHORT[m]} {n}" for m, n in enumerate(L.BULLETINS_PER_MONTH)))
bull = {}
for reg in (L.AS_RUN, L.AS_OPERATED):
    wins = L.windows(reg)
    exp = L.exposure(reg)
    pub = L.publish_dates(items, reg)
    lags = sorted((pub[b.id] - b.issued).days for b in items)
    season = sum(exp[m] for m in (1, 2, 3, 4))
    decided = 0 if reg.pending_flag else sum(exp.values())
    bull[reg.name] = {"late": len(wins), "exposure": round(sum(exp.values())), "season": round(season),
                      "march": round(exp[4]), "decided_superseded": round(decided),
                      "referred": round(sum(exp.values())) if reg.pending_flag else 0,
                      "median_lag": lags[len(lags) // 2]}
    print(f"  {reg.name:12s} published after effective: {len(wins)}; median issue-to-publish "
          f"{lags[len(lags) // 2]} days; determinations inside a window {sum(exp.values()):,.0f} "
          f"(Dec-Mar {season:,.0f}, of which March {exp[4]:,.0f}); decided under the superseded rule "
          f"{decided:,.0f}; referred by the pending flag {bull[reg.name]['referred']:,}")
RES["bulletins"] = bull

# ------------------------------------------------------------------ 4
section("4. Failure injection: the embedding model is retired mid-quarter")
print(f"Notice {L.NOTICE}; retire date {L.RETIRE}; cutover {L.CUTOVER}; next quarterly re-baseline "
      f"{L.REBASELINES[2]}")
inv = Inventory([
    Artefact("policy corpus index", "index", "open-base@1.3", "re-embed, dual index, shadow (Ch14)", order=1),
    Artefact("underwriting notes index", "index", "open-base@1.3", "re-embed with corpus", order=1),
    Artefact("claimant memory index", "index", "open-base@1.3", "re-embed memory store (Ch21)", "memory owner", 1),
    Artefact("lks-adapter-2", "adapter", "open-base@1.3", "retrain on new base -> lks-adapter-3", order=2),
    Artefact("fingerprint probe set", "probe", "open-base@1.3", "re-record probe vectors", order=3),
    Artefact("near-duplicate threshold 0.97", "threshold", "open-base@1.3", "recalibrate on replay", order=6),
    Artefact("withhold floor 0.62", "threshold", "open-base@1.3", "recalibrate on replay by quantile", order=6),
    Artefact("score-drift alarm baseline", "baseline", "open-base@1.3", "re-baseline from replay", order=7),
])
runbook = {"policy corpus index", "underwriting notes index", "lks-adapter-2", "fingerprint probe set"}
left = inv.left_behind("open-base@1.3", runbook)
print(f"Inventory: {len(inv.dependants('open-base@1.3'))} artefacts derived from open-base@1.3; "
      f"the Chapter 14 runbook covered {len(runbook)}; left behind: {', '.join(left)}")
print("Per-slice gate on gold, production-built index (as run): passed in every slice "
      "(the corpus index was rebuilt correctly)")
jun_run, jun_op = pop["as run"][7], pop["as operated"][7]
may_run = pop["as run"][6]
print(f"As run, June: withhold-and-refer {rate(may_run, 'withheld'):.1%} in May -> "
      f"{rate(jun_run, 'withheld'):.1%} in June; repeat-claimant determinations without their memory "
      f"block {sum(r['memory_missing'] for r in jun_run) / POP:.1%} of all June determinations")
mem_days = (date(2027, 8, 12) - L.CUTOVER).days
mem_prod = sum(L.PER_DAY * L.VOLUME[L.month_of(L.CUTOVER + timedelta(days=i))] * L.REPEAT_SHARE
               for i in range(mem_days))
print(f"As run: adjusters reported the referral surge in 9 days and the floor was patched by hand to 0.58 "
      f"on 2027-06-18; the empty memory block was found on 2027-08-12, after {mem_days} days and about "
      f"{mem_prod:,.0f} production determinations")

rs = L.replay_sample()
def stack(c, new, floor, memory):
    s = c.s_new if new else c.s_old
    base = c.u[3] < L.WITHHOLD_BASE[c.cause]
    return {"id": c.id, "slice": c.cause,
            "withheld": base or s < floor, "memory_present": (not c.repeat) or memory}
tol = {"withheld": 0.02, "memory_present": 0.0}
old = [stack(c, False, L.OLD_FLOOR, True) for c in rs]
asrun = replay_compare(old, [stack(c, True, L.OLD_FLOOR, False) for c in rs], "slice", tol)
print(f"Upgrade replay, {len(rs):,} May determinations, runbook-only stack: "
      f"{'passed' if asrun['passed'] else 'BLOCKED'}; {len(asrun['blocks'])} slice-metric breaches, e.g. "
      + "; ".join(f"{s} {m} {d:+.1%}" for s, m, d in asrun["blocks"][:3]))
print("Generated plan, in order:")
for step in inv.plan("open-base@1.3", "open-base@2.0"):
    print("  " + step)
fixd = replay_compare(old, [stack(c, True, L.NEW_FLOOR, True) for c in rs], "slice", tol)
print(f"Replay after the plan (floor 0.62 -> {L.NEW_FLOOR} by quantile, memory re-embedded): "
      f"{'passed' if fixd['passed'] else 'BLOCKED'}; largest withhold change in any slice "
      f"{max(abs(v['withheld']) for v in fixd['deltas'].values()):.1%}")
print(f"As operated, June: withhold-and-refer {rate(jun_op, 'withheld'):.1%}; memory block missing in "
      f"{sum(r['memory_missing'] for r in jun_op)} determinations")
RES["deprecation"] = {"left_behind": left, "replay_blocks_runbook": len(asrun["blocks"]),
                      "replay_examples": asrun["blocks"][:6], "replay_fixed_passed": fixd["passed"],
                      "new_floor": L.NEW_FLOOR, "withheld_may": rate(may_run, "withheld"),
                      "withheld_jun_run": rate(jun_run, "withheld"), "withheld_jun_op": rate(jun_op, "withheld"),
                      "memory_missing_prod": round(mem_prod), "memory_days": mem_days}

# ------------------------------------------------------------------ 5
section("5. Drift and the quarterly re-baseline")
ref = {c: sum(L.MIX[c][m] for m in (0, 1)) for c in L.CAUSES}
raw, adj = [], []
for m in range(12):
    rs_m = pop["as run"][m]
    by = {}
    for c in L.CAUSES:
        sub = [r for r in rs_m if r["cause"] == c]
        by[c] = sum(r["outcome"] == "withheld" for r in sub) / max(1, len(sub))
    raw.append(rate(rs_m, "withheld"))
    adj.append(mix_adjusted(by, ref))
b_raw, b_adj = sum(raw[:2]) / 2, sum(adj[:2]) / 2
fires_raw = [L.SHORT[m] for m in range(2, 12) if raw[m] > b_raw + 0.01]
fires_adj = [L.SHORT[m] for m in range(2, 12) if adj[m] > b_adj + 0.01]
print("Withhold-and-refer, as run, raw:          " + " ".join(f"{L.SHORT[m]} {raw[m]:.1%}" for m in range(12)))
print("Withhold-and-refer, as run, mix-adjusted: " + " ".join(f"{L.SHORT[m]} {adj[m]:.1%}" for m in range(12)))
print(f"Alarm at baseline + 1 point: raw fires in {', '.join(fires_raw)}; mix-adjusted fires in "
      f"{', '.join(fires_adj)}")
aged = L.gold_aged_share()
true_jul = rate(pop["as operated"][8], "wrong")
print(f"Gold set ageing: items whose governing edition was superseded: " +
      ", ".join(f"{L.SHORT[m]} {aged[m]:.1%}" for m in (0, 2, 5, 8, 11)))
print(f"Gold scored on the production index in July without relabelling: wrong-clause "
      f"{(true_jul + aged[8] * (1 - true_jul)) * 100:.1f}% (a correct answer citing the new edition is "
      f"marked wrong); after relabelling the {round(aged[8] * 1040)} aged items: {true_jul * 1000:.1f}/1,000")
RES["drift"] = {"raw": raw, "adjusted": adj, "fires_raw": fires_raw, "fires_adj": fires_adj,
                "gold_aged": aged, "gold_unrelabelled_jul": true_jul + aged[8] * (1 - true_jul),
                "gold_aged_items_jul": round(aged[8] * 1040)}

# ------------------------------------------------------------------ 6
section("6. Incidents that need six months to appear")
for j in (False, True):
    bl, hits = L.revalidation_backlog(j)
    peak = max(bl.values())
    when = [d for d, v in bl.items() if v == peak][0] if peak else None
    label = "jittered 150-180 days" if j else "fixed 180 days"
    print(f"  Re-validation clock {label:22s}: peak backlog {peak:,} records"
          f"{' on ' + str(when) if peak else ''}; repeat-claimant determinations that ran without "
          f"their earlier-claim record {hits:,}")
    RES[f"cohort_{'jitter' if j else 'fixed'}"] = {"peak": peak, "when": str(when), "hits": hits}
run_lines, op_lines = L.instruction_lines(False), L.instruction_lines(True)
print(f"  Determination instructions, lines at month end, as run:   " + " ".join(map(str, run_lines)))
print(f"  Determination instructions, lines at month end, with ledger: " + " ".join(map(str, op_lines)))
print(f"  Cost per determination from instruction growth alone, October: as run "
      f"US${L.cost_per_det(run_lines[-1], 14):.4f}, with ledger US${L.cost_per_det(op_lines[-1], 14):.4f} "
      f"(entry US${L.ENTRY_COST:.4f})")
RES["instructions"] = {"as_run": run_lines, "ledger": op_lines}

# ------------------------------------------------------------------ 7
section("7. Twelve months of scoreboard (illustrative; simulated year)")
board = {}
for reg in (L.AS_RUN, L.AS_OPERATED):
    lines = run_lines if reg.name == "as run" else op_lines
    rows = []
    for m in range(12):
        rs_m, rv_m = pop[reg.name][m], rev[reg.name][m]
        mid = L.FIRST[m] + timedelta(days=14)
        notes_tok = 0 if L.corpus(mid)["notes"] == 0 else (14 if reg.partitioned(mid) else 52)
        rows.append({"month": L.MONTHS[m], "pages": months[m],
                     "wrong_per_1000": round(rate(rs_m, "wrong") * 1000, 1),
                     "review_wrong": sum(r["outcome"] == "wrong" for r in rv_m),
                     "withheld": round(rate(rs_m, "withheld"), 4),
                     "headroom": round(headroom([r["rank"] for r in rs_m], L.K), 4),
                     "memory_missing": round(sum(r["memory_missing"] for r in rs_m) / POP, 4),
                     "cost": round(L.cost_per_det(lines[m], notes_tok), 4), "instr_lines": lines[m]})
    board[reg.name] = rows
    print(f"{reg.name}:")
    print("  month     pages   wrong/1k  review  withheld  headroom  cost      lines")
    for r in rows:
        print(f"  {r['month']:8s} {r['pages']:7,}  {r['wrong_per_1000']:5.1f}    {r['review_wrong']:3d}     "
              f"{r['withheld']:6.1%}   {r['headroom']:6.2%}   US${r['cost']:.4f} {r['instr_lines']:4d}")
vol = [L.PER_DAY * L.VOLUME[m] * L.DAYS[m] for m in range(12)]
for name in board:
    est = sum(board[name][m]["wrong_per_1000"] / 1000 * vol[m] for m in range(12))
    print(f"Estimated wrong-clause determinations in the year, {name}: {est:,.0f} of {sum(vol):,.0f}")
    RES[f"wrong_year_{name.replace(' ', '_')}"] = round(est)
RES["board"] = board
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(RES, f, indent=1, default=str)
print("\nNote: stand-in year; no hosted or local model was called.")
