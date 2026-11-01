"""Chapter 23 lab: a two-hour adjuster session, four ways to hold it, and the memory lifecycle.

Run from 04-code/ch23:  python3 lab.py > output.txt   (writes out/results.json)
Deterministic stand-ins: session23.py (the session) and lifecycle23.py (Chapter 22's replay).
"""
import json
import pathlib
import sys
from collections import Counter
from dataclasses import replace
from datetime import date

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import session23 as S                                   # noqa: E402
import lifecycle23 as Y                                 # noqa: E402
from contextlab.statestore import StateStore, Fact, checkpoint_id, erase, sweep, LifecyclePolicy, RetentionRule  # noqa: E402

OUT = {}


def clock(minute):
    h, m = divmod(int(round(minute)), 60)
    return f"{13 + h}:{m:02d}"


def section(t):
    print(f"\n== {t}")


# 1. Tom's afternoon ------------------------------------------------------------
section("1. Tom's afternoon, Thursday 22 October 2026, from 13:00")
tom = S.build_session(23)
claims = sorted({i.claim for i in tom.items if i.claim != "*"})
kinds = Counter(i.kind for i in tom.items if i.role == "tool")
print(f"claims {len(claims)} ({', '.join(claims)}); items {len(tom.items)}; model calls "
      f"{sum(i.role == 'assistant' for i in tom.items)}; last call minute {tom.items[-1].minute} "
      f"({clock(tom.items[-1].minute)}); tokens if nothing is ever removed "
      f"{sum(i.tokens for i in tom.items) + S.SYS_TOKENS:,}")
print("tool results:", dict(kinds), "| reserve over US$25,000:", sorted(tom.big))
for cid, (text, act, scope) in tom.rules.items():
    when = next(i.minute for i in tom.items if i.rule == cid)
    print(f"  {cid} stated at {clock(when)}: {text}")
OUT["session"] = {"claims": len(claims), "items": len(tom.items), "end_minute": tom.items[-1].minute,
                  "tokens": sum(i.tokens for i in tom.items) + S.SYS_TOKENS, "big": sorted(tom.big)}

# 2. Full history and the window ----------------------------------------------------
section("2. Full history: the failure injection is the afternoon itself")
runs = {s: S.run(tom, s) for s in S.STRATEGIES}
f = runs["full"]
at60 = max((t for m, t, o in f.trace if m < 60), default=0)
print(f"window at 14:00: {at60:,} tokens; overflow at minute {f.overflow_minute} "
      f"({clock(f.overflow_minute)}); calls refused after it {f.lost}")
for h in (0, 1):
    a, w = f.by_hour[h]
    print(f"  hour {h + 1}: answers {a}, wrong {w} ({w / a:.1%})")
OUT["full"] = {"at60": at60, "overflow": f.overflow_minute, "lost": f.lost, "by_hour": f.by_hour}

# 3. Four strategies on Tom's afternoon ------------------------------------------------
section("3. Four ways to hold the afternoon")
print(f"{'strategy':<11}{'answers':>8}{'wrong':>6}{'withheld':>9}{'violations':>11}{'refused':>8}"
      f"{'peak':>9}{'compactions':>12}{'re-fetch':>9}{'US$':>7}")
OUT["tom"] = {}
for s, r in runs.items():
    print(f"{s:<11}{r.answers:>8}{r.wrong:>6}{r.withheld:>9}{r.violations:>11}{r.lost:>8}"
          f"{r.peak:>9,}{r.compactions:>12}{r.refetches:>9}{r.cost():>7.2f}")
    OUT["tom"][s] = {k: getattr(r, k) for k in ("answers", "wrong", "withheld", "violations", "lost",
                                                "peak", "compactions", "refetches")} | {"cost": round(r.cost(), 2)}

OUT["traces"] = {s: r.trace for s, r in runs.items()}
st_run = runs["structured"]
OUT["composition"] = {"structured": st_run.window, "minute": tom.items[-1].minute}

# 4. Forty afternoons ------------------------------------------------------------------
section("4. Forty afternoons (seeds 1 to 40), same adjuster workload")
agg = {s: Counter() for s in S.STRATEGIES}
for seed in range(1, 41):
    sess = S.build_session(seed)
    for s in S.STRATEGIES:
        r = S.run(sess, s)
        a = agg[s]
        for k in ("answers", "wrong", "withheld", "violations", "lost", "refetches", "compactions",
                  "input_tokens", "cached_tokens", "output_tokens"):
            a[k] += getattr(r, k)
        a["overflowed"] += r.overflow_minute is not None
        a["cost"] += r.cost()
        a["peak"] = max(a["peak"], r.peak)
        for h in (0, 1):
            a[f"a{h}"] += r.by_hour[h][0]
            a[f"w{h}"] += r.by_hour[h][1]
calls = agg["structured"]["answers"]
print(f"model calls requested: {calls:,}")
print(f"{'strategy':<11}{'served':>7}{'wrong':>7}{'withheld':>9}{'violations':>11}{'refused':>8}"
      f"{'overflowed':>11}{'hour1 wrong':>12}{'hour2 wrong':>12}{'US$/afternoon':>14}{'cached':>8}")
OUT["forty"] = {}
for s, a in agg.items():
    served = a["answers"]
    print(f"{s:<11}{served:>7,}{a['wrong'] / served:>7.1%}{a['withheld'] / served:>9.1%}{a['violations']:>11}"
          f"{a['lost']:>8}{a['overflowed']:>8} of 40{a['w0'] / a['a0']:>12.1%}{a['w1'] / a['a1']:>12.1%}"
          f"{a['cost'] / 40:>14.2f}{a['cached_tokens'] / a['input_tokens']:>8.0%}")
    OUT["forty"][s] = dict(a) | {"cost_per": round(a["cost"] / 40, 2)}

# 5. Checkpoint and restore ---------------------------------------------------------------
section("5. Checkpoint at 14:11, crash, restore")
cut = [i for i in tom.items if i.minute <= 71]
part = S.run(replace(tom, items=cut), "structured")
st = part.store
packet = [k for k, t in part.window if k.startswith("I")]
by_ref = st.checkpoint(packet, {"HO 04 95": "2024-06", "HO-3": "2011-05"}, len(cut))
snap = st.checkpoints[by_ref]
rendered = sum(t for _, t in part.window)
ref_tokens = len(json.dumps(snap)) // 4
back = StateStore.restore(snap, "tbrennan")
again = back.checkpoint(snap["packet"], snap["editions"], snap["turn"])
print(f"active claim {part.active}; claims in state {len(st.claims)}; pinned {len(st.pinned)}; "
      f"facts {sum(len(c.facts) for c in st.claims.values())}")
print(f"checkpoint {by_ref}: by reference ~{ref_tokens:,} tokens; rendered window {rendered:,} tokens")
print(f"restored checkpoint {again}; identical: {again == by_ref}")
OUT["checkpoint"] = {"id": by_ref, "ref_tokens": ref_tokens, "rendered": rendered, "identical": again == by_ref}

# 6. The progress sheet ---------------------------------------------------------------------
section("6. The progress sheet at the end of the afternoon")
print(runs["structured"].store.sheet())

# 7. The memory store on 5 October ------------------------------------------------------------
section("7. Chapter 22's memory store under the lifecycle policy (5 October 2026)")
rep = {}
Y.lifecycle_store(Y.L.make_mix(), report=rep)
rv = rep["revalidated"]
print("before:", rep["before"])
print(f"re-validated earlier-claim records {rv['checked']}: unchanged {rv['unchanged']}, corrected {len(rv['corrected'])}"
      f" (each replaced by a pointer to the policy system, re-read at use)")
print(f"expired by retention {rep['expired']}; after: {rep['after']}")
OUT["sweep"] = {"before": rep["before"], "after": rep["after"], "checked": rv["checked"],
                "corrected": len(rv["corrected"]), "expired": rep["expired"]}

# 8. Full mix: which part of the lifecycle fixes what --------------------------------------------
section("8. Full mix, 1,000 determinations, write gate on (Chapter 22's close)")
rows = [("Chapter 22 close", False, 8, {}), ("+ re-validation", True, 8, {"retention": False}),
        ("+ read budget 6", True, 6, {"retention": False}), ("+ retention 730 days", True, 6, {})]
OUT["mix"] = {}
for name, lc, k, kw in rows:
    d, s, l, o, _ = Y.replay(lc, k, **kw)
    w = Y.R.wrong(d, o)
    reads = sum(len(r.mids) for r in l.reads)
    print(f"{name:<22} wrong {len(w)} of 1,000 ({len(w) / 10:.1f}%) memory reads {reads:,} "
          f"lookup saved {Y.lookup_saved(d, l):.1f} s per repeat-claimant determination  {sorted(w)}")
    OUT["mix"][name] = {"wrong": len(w), "ids": sorted(w), "reads": reads, "saved": round(Y.lookup_saved(d, l), 1)}

# 9. Retention period -----------------------------------------------------------------------------
section("9. Retention period for claimant memories (read budget 6, re-validation on)")
OUT["retention"] = {}
for days in (180, 365, 730, None):
    Y.POLICY.rules["claimant"] = RetentionRule("claimant", days)
    d, s, l, o, _ = Y.replay(True, 6)
    w = Y.R.wrong(d, o)
    live = sum(m.status == "live" and m.scope == "claimant" for m in s.values())
    print(f"keep {str(days) + ' days' if days else 'forever':<10} wrong {len(w)}  claimant memories live at the end {live:,}  "
          f"lookup saved {Y.lookup_saved(d, l):.1f} s")
    OUT["retention"][str(days)] = {"wrong": len(w), "live": live, "saved": round(Y.lookup_saved(d, l), 1)}
Y.POLICY.rules["claimant"] = RetentionRule("claimant", 730)

# 10. Erasure --------------------------------------------------------------------------------------
section("10. One erasure request, end to end")
d, s, l, o, _ = Y.replay(True, 6)
rep_pols = Counter(m.subject for m in s.values() if m.scope in ("policy", "claimant") and m.status != "expired")
pol = max(sorted(p for p in rep_pols if any(x.policy == p and x.repeat and x.state == "WI" for x in d)),
          key=lambda p: rep_pols[p])
mine = {m.mid for m in s.values() if m.subject == pol}
readers = len(l.readers(mine))
refs = {x.did for x in d if x.policy == pol}
ck = StateStore("S-erase", "tbrennan")
ck.record("C03", Fact("C03.amount", "US$8,400", f"{pol}/record#amount"))
ck.record("C03", Fact("C03.edition", "2024-06", "LK-HO0495@2024-06#p2"))
ck.checkpoint(["I12"], {"HO 04 95": "2024-06"}, 12)
by_status = Counter(s[m].status for m in mine)
r = erase(s, pol, refs, "ER-2026-0117", date(2026, 10, 26), Y.POLICY, ck.checkpoints)
print(f"subject {pol}: memory records {len(mine)} {dict(by_status)}; read by {readers} determinations in the replay")
print(f"deleted {len(r.deleted)}, re-derived {len(r.rederived)}, dropped {len(r.dropped_derived)}, "
      f"checkpoints redacted {r.checkpoints_redacted}; the governed span in the checkpoint stays")
print("tombstone:", r.tombstone)
d2, s2, *_ = Y.replay(True, 6)
hold = erase(s2, pol, refs, "ER-2026-0118", date(2026, 10, 26), Y.POLICY, holds={pol})
print(f"same request under a litigation hold: deleted {len(hold.deleted)}, held {len(hold.held)}")
inc = s2["M02116"]
pol_origins = [o_.ref for o_ in inc.origins if any(x.did == o_.ref for x in d2)]
print(f"incident record M02116 (status {inc.status}): origins {[o_.ref for o_ in inc.origins]}; "
      f"origins that are a policyholder's determination {len(pol_origins)}, so no policyholder's request reaches it")
OUT["erasure"] = r.tombstone | {"records": len(mine), "readers": readers, "held_under_hold": len(hold.held),
                                "incident_policyholder_origins": len(pol_origins)}

# 11. Tenancy -----------------------------------------------------------------------------------------
section("11. Tenancy: two bank-partner programmes on Lakeshore paper")
by_t = Counter(Y.tenant_of(m.subject) for m in s.values() if m.scope in ("policy", "claimant") and m.status == "live")
print("live policy and claimant memories by tenant:", dict(by_t))
fam_t = Counter(Y.tenant_of(x.policy) for x in d)
print("determinations by tenant:", dict(fam_t))
exit_ = [m.mid for m in s.values() if m.scope in ("policy", "claimant") and Y.tenant_of(m.subject) == "partner-b"]
print(f"partner-b exit: {len(exit_)} records to export and erase, by subject; team memories with a partner-b origin "
      f"{sum(1 for m in s.values() if m.scope == 'team' and any(Y.tenant_of(next((x.policy for x in d if x.did == o_.ref), '')) == 'partner-b' for o_ in m.origins if o_.kind == 'determination'))}")
OUT["tenancy"] = {"by_tenant": dict(by_t), "partner_b_exit": len(exit_)}

(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "results.json").write_text(json.dumps(OUT, indent=1, default=str))
