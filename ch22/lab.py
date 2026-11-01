"""Chapter 22 lab: trace a poisoned memory, quarantine it, and catch a second one.

Note: deterministic stand-in (see lakeshore22.py). Run: python3 lab.py > output.txt
"""
import json
import pathlib
import random
import re
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from contextlab import memguard as G
import lakeshore22 as L
import replay22 as R

RES = {}


def rate(n):
    return f"{n} of 1,000 ({n / 10:.1f}%)"


def rerun_fn(dets, store, log):
    D = {d.did: d for d in dets}
    return lambda did, mask=(): R.answer(D[did], [store[m] for m in log.read_by(did) if m not in mask])


def section(t):
    print(f"\n== {t}")


# 1. Chapter 21's two weeks replayed on the full pipeline, with the read log on
dets, store, log, out, _ = R.run(memory=False)
base = R.wrong(dets, out)
dets, store, log, out, _ = R.run()
ch21 = R.wrong(dets, out)
D = {d.did: d for d in dets}
section("Scoreboard")
print(f"Chapter 20 close (memory off): {rate(len(base))}")
cause_path = len(base) + len(L.FOLLOWED & set(ch21))
print(f"Chapter 21 close (cause path): {rate(cause_path)}")
print(f"Full pipeline, read log on:    {rate(len(ch21))}")
RES["ch20"], RES["ch21"], RES["ch21_cause_path"] = len(base), len(ch21), cause_path

# 2. Trace back: leave-one-out on every new wrong determination
section("Trace back from the new wrong determinations")
rerun = rerun_fn(dets, store, log)
groups = Counter()
sig = {}
for did in sorted(set(ch21) - set(base)):
    c = G.culprits(did, log, rerun)
    root = G.root_of(store, c[0]) if len(c) == 1 else "none (every memory)"
    kind = ("poison" if root == "M02116" else "stale" if c and store[c[0]].scope == "policy" and len(c) == 1
            else "over-remembering")
    groups[kind] += 1
    sig[did] = kind
    d = D[did]
    print(f"{did} {d.day} {d.state} {d.peril:<12} {d.adjuster} reads {len(log.read_by(did))} "
          f"culprits {len(c)} root {root} -> {kind}")
RES["causes"] = dict(groups)

# 3. Trace forward: every determination the family touched
section("Trace forward from M02116")
fam = G.family(store, "M02116")
readers = log.readers(fam)
outcome = Counter()
rows = []
for r in readers:
    d = D[r.did if hasattr(r, 'did') else r.det] if False else D[r.det]
    t = d.truth()
    o = ("wrong (family)" if sig.get(d.did) == "poison" else "wrong (other cause)" if out[d.did] != t
         else "agreed" if d.peril == "burst_pipe" else "outranked")
    outcome[o] += 1
    rows.append((d.did, str(d.day), d.peril, d.adjuster, [m for m in r.mids if m in fam], o))
print(f"family {sorted(fam)}; reads by member "
      f"{ {m: sum(m in r.mids for r in log.reads) for m in sorted(fam)} }")
print(f"determinations that read the family: {len(readers)}; outcome {dict(outcome)}")
by_day = Counter((str(D[r.det].day), "wrong" if out[r.det] != D[r.det].truth() else "right") for r in readers)
for day in L.DAYS:
    print(f"  {day}: read {by_day[(str(day), 'right')] + by_day[(str(day), 'wrong')]:>2}, wrong {by_day[(str(day), 'wrong')]}")
src = store["M02116"]
print(f"origins of M02116: {[(o.kind, o.ref) for o in src.origins]}; authority {src.authority()}")
print(f"origins of M-A01-D: {[o.ref for o in store['M-A01-D'].origins]} parents {store['M-A01-D'].parents}")
RES["readers"], RES["outcome"], RES["family"] = len(readers), dict(outcome), sorted(fam)
RES["reads_by_day"] = {str(k[0]) + ":" + k[1]: v for k, v in by_day.items()}
RES["readers_rows"] = rows

# 4. Remediation: rerun the readers with the family masked
section("Remediation rerun")
changed = [r.det for r in readers if rerun(r.det, mask=tuple(fam)) != out[r.det]]
fixed = [x for x in changed if rerun(x, mask=tuple(fam)) == D[x].truth()]
print(f"rerun {len(readers)} with family masked: changed {len(changed)}, now right {len(fixed)}, "
      f"newly wrong {len(changed) - len(fixed)}")
RES["remediation"] = {"rerun": len(readers), "changed": len(changed), "fixed": len(fixed)}

# 5. Quarantine, rollback or text search, applied on Friday 16 October
section("Three ways to remove it (applied 16 October)")
day = L.DAYS[-1]
d0 = [d for d in dets if d.day == day]
opts = {}
for name, kw in [("none", {}), ("quarantine family", {"quarantine_on": (day, "M02116")}),
                 ("rollback to 4 Oct", {"rollback_on": (day, L.POISON_NIGHT)})]:
    _, st, lg, o, ev = R.run(**kw)
    lost = sum(1 for d in d0 for m in log.read_by(d.did) if m not in lg.read_by(d.did) and m not in fam)
    w = sum(o[d.did] != d.truth() for d in d0)
    removed = ev[0][3] if ev else 0
    opts[name] = {"removed": removed, "wrong_that_day": w, "legit_reads_lost": lost}
mid_window = {m.mid for m in store.values() if L.POISON_NIGHT <= m.written < day} - fam
later = sum(1 for r in log.reads if r.day > L.POISON_NIGHT for m in r.mids if m in mid_window)
opts["rollback to 4 Oct"]["legit_destroyed"] = len(mid_window)
opts["rollback to 4 Oct"]["legit_destroyed_reads_so_far"] = later
pat = re.compile(r"freez", re.I)
hits = [m for m in store.values() if m.status == "live" and pat.search(m.text)]
legit = [m.mid for m in hits if m.mid not in fam]
opts["text search"] = {"removed": len(hits), "legit_removed": len(legit),
                       "family_missed": sorted(fam - {m.mid for m in hits})}
for k, v in opts.items():
    print(f"{k:<20} {v}")
RES["removal"] = opts

# 6. Failure injection: a second poison on 19 October
section("Failure injection: M-T0419 (Illinois hail, 12 October)")
inj = {}
for name, kw in [("no guard", {}), ("read-time monitor", {"monitor": True}), ("write gate", {"gate": True})]:
    ds, st, lg, o, ev = R.run(second=True, **kw)
    rd = lg.readers({"M-T0419"})
    w2 = [r.det for r in rd if R.final(o[r.det]) != D[r.det].truth()]
    held = sorted(m.mid for m in st.values() if m.status == "held")
    tot = len(R.wrong(ds, o))
    rev = [x for x in o if o[x].startswith("review:")]
    fa = [x for x in rev if not sig.get(x) == "poison" and x not in [r.det for r in rd]]
    inj[name] = {"reads": len(rd), "wrong": len(w2), "total_wrong": tot, "held": held,
                 "reviews": len(rev), "false_alarms": len(fa),
                 "events": [(str(e[0]), e[1], e[2]) for e in ev]}
    print(f"{name:<18} reads {len(rd):>2}, wrong {len(w2)}, full mix {rate(tot)}, "
          f"sent to review {len(rev)} (false alarms {len(fa)}), held {held}")
    for e in ev:
        print(f"   {e[0]} {e[1]} {e[2]}")
RES["injection"] = inj

# 7. Chapter 22 close: the family quarantined and the readers re-run
section("Chapter 22 close")
_, st, lg, o, _ = R.run(gate=True)
close = R.wrong(dets, o)
print(f"write gate on, poison never readable: {rate(len(close))}; "
      f"remaining causes {Counter(sig.get(x, 'residual') for x in close)}")
RES["ch22"] = len(close)

# 8. Stale facts: memories the policy record now contradicts
section("Stale facts")
truth = {d.policy: {"ho0495": d.ho0495} for d in dets}
st_ids = G.stale(store, lambda p, on: truth.get(p), L.DAYS[-1])
reads_stale = sum(1 for r in log.reads for m in r.mids if m in st_ids)
pol = [m for m in store.values() if m.scope == "policy"]
print(f"earlier-claim episodes naming an endorsement {len(pol)}; contradicted by the record {len(st_ids)}; "
      f"read in the replay {reads_stale} times; wrong {groups['stale']}")
RES["stale"] = {"snapshots": len(pol), "stale": len(st_ids), "reads": reads_stale, "wrong": groups["stale"]}

# 9. Over-remembering: memory budget per packet
section("Over-remembering: memories per packet")
sweep = {}
rep = [d for d in dets if d.repeat]
for k in (0, 2, 4, 6, 8, 12, 16):
    _, st, lg, o, _ = R.run(k_max=k)
    w = sum(o[d.did] != d.truth() for d in rep)
    reads = [lg.read_by(d.did) for d in rep]
    saved = sum((45 if any(m.startswith("M-P") for m in rs) else 0)
                + 20 * min(2, sum(m.startswith("M-C") for m in rs)) for rs in reads) / len(rep)
    p = sum(L.P_DISTRACT * max(0, len(rs) - 6) for rs in reads) / len(rep) * 1000
    sweep[k] = {"wrong": w, "saved_s": round(saved, 1), "expected_distracted_per_1000": round(p, 2)}
    print(f"k={k:>2}: repeat-claimant wrong {w:>2} of {len(rep)}, lookup time saved {saved:5.1f} s/det, "
          f"expected distraction errors {p:.2f} per 1,000")
RES["over"] = sweep

# 10. Drift: summary of a summary versus re-derived from source
section("Drift through repeated consolidation")
rng = random.Random(L.SEED + 7)
mem = [[True] * 4 for _ in range(200)]           # policy, edition, effective date, condition
drift = []
for rnd in range(1, 9):
    for q in mem:
        for i in range(4):
            if q[i] and rng.random() < 0.08:
                q[i] = False
    kept = sum(map(sum, mem)) / 800
    general = sum(not q[0] for q in mem)
    fresh = [[rng.random() >= 0.08 for _ in range(4)] for _ in range(200)]   # re-derived from source
    fk, fg = sum(map(sum, fresh)) / 800, sum(not q[0] for q in fresh)
    drift.append({"round": rnd, "kept": round(kept, 3), "generalised": general,
                  "fresh_kept": round(fk, 3), "fresh_generalised": fg})
    print(f"round {rnd}: qualifiers kept {kept:.1%}, policy qualifier lost {general:>3}; "
          f"re-derived from source {fk:.1%}, lost {fg}")
RES["drift"] = drift

# 11. Cross-tenant leakage: same address, two policies
section("Cross-tenant leakage: 24 addresses with a landlord and a tenant policy")
leak = {}
for key in ("policy", "address"):
    st, lg = {}, G.ReadLog()
    for a in range(24):
        for who in ("DP", "HO4"):
            pid = f"P-{who}-{a:02d}"
            for j in range(3):
                st[f"M-{pid}-{j}"] = G.MemoryRecord(f"M-{pid}-{j}", "claimant", pid, "note", L.DAYS[0])
    for a in range(24):
        for who in ("DP", "HO4"):
            pid = f"P-{who}-{a:02d}"
            mids = [m for m in st if (st[m].subject == pid if key == "policy"
                                      else st[m].subject.endswith(f"-{a:02d}"))]
            lg.log(f"D-{pid}", L.DAYS[1], mids, (pid,))
    v = G.scope_violations(lg, st)
    leak[key] = {"reads": sum(len(r.mids) for r in lg.reads), "violations": len(v),
                 "determinations": len({x[0] for x in v})}
    print(f"keyed by {key:<8}: memory reads {leak[key]['reads']}, cross-tenant {len(v)} "
          f"in {leak[key]['determinations']} determinations")
RES["leak"] = leak

# 12. One determination, as the read log records it
section("One determination: D0025, 5 October")
dets0, store0, log0, out0, _ = R.run()
d0 = next(d for d in dets0 if d.did == "D0025")
print(f"{d0.did} {d0.day} {d0.state} {d0.peril} {d0.policy} adjuster {d0.adjuster} "
      f"HO 04 95 on record: {d0.ho0495}")
print(f"  answer {out0[d0.did]}   truth {d0.truth()}")
for mid in log0.read_by(d0.did):
    m = store0[mid]
    print(f"  read {mid:<8} {m.scope:<8} written {m.written} authority {m.authority():<10} "
          f"origins {[o.kind + ':' + o.ref for o in m.origins] or ['-']} parents {list(m.parents) or '-'}")

pathlib.Path("out").mkdir(exist_ok=True)
json.dump(RES, open("out/results.json", "w"), indent=1, default=str)
