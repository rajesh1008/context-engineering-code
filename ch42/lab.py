"""Chapter 42 lab: instrument the pipeline, reconstruct, redact, watch and alert.

Run: python3 lab.py  (about 20 s; writes output.txt and out/*.json, out/*.csv)
"""
from __future__ import annotations

import copy
import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from contextlab.observe import (Alarm, Ewma, Ledger, Store, h, reconstruct,  # noqa: E402
                                reconstructability)
from lakeshore42 import (BAD, FIXED, PII, LEAK_PATTERNS, Config, Sinks, canary, corpus,  # noqa: E402
                         facts_text, make_claim, monitor, note_text, render, replay_set, run, tokens)
import re  # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
R: dict = {}
LINES: list = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LINES.append(s)


def wrong(ev):
    o = ev.output
    return o["status"] != "handoff" and o["clause"] != o["governing"]


def instrumented(rules, n=1000):
    docs, cfg, L, S = corpus(0), Config(), Ledger(), Sinks(rules=list(rules))
    packets = {}
    for c in replay_set(n):
        ev, p = run(c, docs, cfg, S, L)
        packets[ev.det_id] = p
    return L, S, packets


def leaks(lines):
    return sum(len(re.findall(p, l)) for l in lines for p in LEAK_PATTERNS)


# 1 ------------------------------------------------------------------------------
def s1():
    say("== 1. Instrument the replay set (1,000 determinations) ==")
    L, S, packets = instrumented(PII)
    w = [e for e in L.events if wrong(e)]
    say(f"wrong clause: {len(w)}/1000 ({len(w)/10:.1f}%)")
    for e in w:
        g = next(o for o in e.offered if o.span_id == e.output["governing"])
        note = next(o for o in e.offered if o.rank == 1)
        say(f"  {e.det_id} {e.request['state']} {e.request['cause']}: governing {g.span_id} rank {g.rank} "
            f"{g.decision} ({g.reason}); note {note.tokens} tokens; cited {e.output['clause']}")
    drops = sum(any(o.decision == "dropped" for o in e.offered) for e in L.events)
    dropped = Counter(o.reason for e in L.events for o in e.offered if o.decision == "dropped")
    gov_dropped = sum(any(o.decision == "dropped" and o.span_id == e.output["governing"] for o in e.offered)
                      for e in L.events)
    full = sum(len(p.encode()) for p in packets.values()) + L.bytes()
    refs = L.bytes() + S.evidence.bytes() + S.claimfile.bytes()
    ref_only = L.bytes() + S.evidence.bytes()
    cached = sum(e.cache["cached"] for e in L.events); inp = sum(e.cache["input"] for e in L.events)
    say(f"determinations with any span dropped for budget: {drops}; spans dropped: {dict(dropped)}; "
        f"governing span dropped: {gov_dropped}")
    say(f"bytes per determination: packet text in the record {full/1000:,.0f}; refs + stores {refs/1000:,.0f} "
        f"(trace store only, claim file is the system of record: {ref_only/1000:,.0f})")
    say(f"evidence blobs {len(S.evidence.blobs)} for {S.evidence.writes} writes; claim-file blobs "
        f"{len(S.claimfile.blobs)}")
    say(f"cached share of input tokens: {cached/inp:.1%}; mean input tokens {inp/1000:.0f}")
    R["s1"] = dict(wrong=len(w), wrong_ids=[e.det_id for e in w], drops=drops, gov_dropped=gov_dropped,
                   bytes_full=round(full / 1000), bytes_refs=round(refs / 1000),
                   bytes_trace_only=round(ref_only / 1000), evidence_blobs=len(S.evidence.blobs),
                   writes=S.evidence.writes, cached_share=round(cached / inp, 4), mean_tokens=round(inp / 1000))
    return L, S, packets


# 2 ------------------------------------------------------------------------------
def s2(L, S, packets):
    say("\n== 2. Reconstruct every determination and check the chain ==")
    res = reconstructability(L.events, S.evidence | S.claimfile if False else _both(S), render)
    say(f"exact reconstructions: {res['exact']}/{res['n']}")
    same = sum(reconstruct(e, _both(S), render)["packet"] == packets[e.det_id] for e in L.events)
    say(f"rebuilt packet identical to the one sent: {same}/1000")
    say(f"chain verifies: {L.verify() is None}")
    T = copy.deepcopy(L)
    T.events[500].output["status"] = "covered" if T.events[500].output["status"] == "denied" else "denied"
    say(f"edit event 500's status -> first bad event: {T.verify()}")
    T = copy.deepcopy(L); del T.events[700]
    say(f"delete event 700 -> first bad event: {T.verify()}")
    R["s2"] = dict(exact=res["exact"], identical=same, tamper_edit=500, tamper_delete=700)


class _Both:
    def __init__(self, S):
        self.S = S

    def get(self, k):
        return self.S.evidence.get(k) or self.S.claimfile.get(k)


def _both(S):
    return _Both(S)


# 3 ------------------------------------------------------------------------------
def s3(L, S, packets):
    say("\n== 3. Three ways to log content ==")
    meta = sum(len(json.dumps({"det": e.det_id, "model": e.versions["model"], "in": e.cache["input"],
                               "status": e.output["status"]})) for e in L.events)
    full_lines = list(packets.values())
    rows = [("no content (metadata only)", meta / 1000, 0, 0),
            ("full content on the trace", R["s1"]["bytes_full"], 1000,
             leaks(full_lines)),
            ("store by reference", R["s1"]["bytes_trace_only"], R["s2"]["exact"], leaks(S.oplog))]
    for name, b, rec, lk in rows:
        say(f"  {name:30s} bytes/det {b:7,.0f}  reconstructable {rec:4d}/1000  personal items in telemetry {lk}")
    R["s3"] = [dict(name=n, bytes=round(b), reconstructable=rec, leaks=lk) for n, b, rec, lk in rows]


# 4 ------------------------------------------------------------------------------
def s4(L):
    say("\n== 4. Sampling: is the trace there when the complaint arrives? ==")
    import random
    r = random.Random(42)
    den = [e for e in L.events if e.output["status"] == "denied"]
    out = {}
    for name, keep in [("head 10%", lambda e: r.random() < 0.10),
                       ("head 10% + every denial", lambda e: e.output["status"] == "denied" or r.random() < 0.10),
                       ("every determination", lambda e: True)]:
        kept = [e for e in L.events if keep(e)]
        have = sum(1 for e in kept if e.output["status"] == "denied")
        wr = sum(1 for e in kept if wrong(e))
        out[name] = dict(kept=len(kept), denials=have, wrong=wr)
        say(f"  {name:26s} kept {len(kept):4d}/1000; denials kept {have}/{len(den)}; wrong determinations kept {wr}/4")
    R["s4"] = out


# 5 ------------------------------------------------------------------------------
def s5():
    say("\n== 5. Redaction: the rule that removed the examiner's span ==")
    out = {}
    for name, rules in [("no redaction", []), ("scoped PII rules", PII),
                        ("PII + August rule on every sink", PII + [BAD]),
                        ("PII + August rule scoped to telemetry", PII + [FIXED])]:
        L, S, _ = instrumented(rules)
        res = reconstructability(L.events, _both(S), render)
        miss = Counter(m for e in L.events for m in reconstruct(e, _both(S), render)["missing"])
        out[name] = dict(exact=res["exact"], leaks=leaks(S.oplog),
                         missing_evidence={k: v for k, v in miss.items() if ":" not in k},
                         missing_personal=sum(v for k, v in miss.items() if ":" in k),
                         dets_missing_evidence=sum(any(":" not in m for m in reconstruct(e, _both(S), render)["missing"])
                                                   for e in L.events))
        say(f"  {name:38s} exact {res['exact']:4d}/1000  telemetry leaks {leaks(S.oplog):4d}  "
            f"missing: evidence {sum(v for k, v in miss.items() if ':' not in k)}, "
            f"personal {sum(v for k, v in miss.items() if ':' in k)} "
            f"{sorted(k for k in miss if ':' not in k)}")
        if name.startswith("PII + August rule on every"):
            ex = next(e for e in L.events if e.request["state"] == "WI" and e.request["cause"] == "wb"
                      and e.output["clause"] and e.output["clause"].startswith("HO0495"))
            rc = reconstruct(ex, _both(S), render)
            stored = S.evidence.get(next(o.content for o in ex.offered if o.span_id == ex.output["clause"]))
            say(f"  examiner's pick {ex.det_id}: cited {ex.output['clause']}; exact {rc['exact']}; "
                f"missing {rc['missing']}")
            say(f"  stored text now reads: {stored[:80]}...")
            out["examiner"] = dict(det=ex.det_id, cited=ex.output["clause"], stored=stored[:80])
            sample = L.events[:200]
            g = reconstructability(sample, _both(S), render)
            say(f"  gate on 200 sampled events with the rule as wired: {g['exact']}/200 exact -> "
                f"{'PASS' if g['exact'] == 200 else 'FAIL'}")
            out["gate_bad"] = g["exact"]
        if name.startswith("PII + August rule scoped"):
            g = reconstructability(L.events[:200], _both(S), render)
            say(f"  gate on 200 sampled events with the rule scoped: {g['exact']}/200 exact -> "
                f"{'PASS' if g['exact'] == 200 else 'FAIL'}")
            out["gate_fixed"] = g["exact"]
    R["s5"] = out


# 6 & 7 ----------------------------------------------------------------------------
DAYS, PER_DAY, CANARIES = 90, 250, 20
E1, E2, E3, E4 = 41, 50, 62, 70


def simulate(changes: bool = True):
    """Ninety days of live traffic, left unattended, with four changes nobody flagged."""
    daily, slices, complaint_pool = [], defaultdict(lambda: defaultdict(lambda: [0, 0])), []
    cache_fix = None
    L = Ledger(); S = Sinks(rules=PII)
    for d in range(DAYS):
        on = changes
        docs = corpus(d if on else 0, bulletin_broken=on and d >= E1)
        ts = on and d >= E3
        late = on and d >= E4
        cfg = Config(budget=900 if late else 301, k=12 if late else 6, ts_in_prefix=ts)
        row = Counter()
        for i in range(PER_DAY):
            c = make_claim(d, i)
            ev, _ = run(c, docs, cfg, S, L)
            req = ev.output["governing"]
            covered = any(o.span_id == req and o.decision == "admitted" for o in ev.offered)
            row["n"] += 1; row["cov"] += covered; row["wrong"] += wrong(ev)
            row["flag"] += monitor(ev, ev.det_id); row["handoff"] += ev.output["status"] == "handoff"
            row["drop"] += any(o.decision == "dropped" for o in ev.offered)
            row["tok"] += ev.cache["input"]; row["cached"] += ev.cache["cached"]
            row["top"] += max(o.score for o in ev.offered if o.tier == "evidence")
            sl = slices[(c.state, c.peril)][d]; sl[0] += 1; sl[1] += covered
            if c.upload_lag_h is not None:
                complaint_pool.append((c, ev))
            if i < CANARIES:
                cv = canary(ev)
                band = "<1000" if ev.cache["input"] < 1000 else "1000-1299" if ev.cache["input"] < 1300 else "1300+"
                row["can_n"] += 1; row["can_hit"] += monitor(cv, "can" + ev.det_id)
                row["b_n_" + band] += 1; row["b_hit_" + band] += monitor(cv, "can" + ev.det_id)
        row["day"] = d; row["corpus"] = len(docs)
        daily.append(row)
        if ts and cache_fix is None and row["cached"] / row["tok"] < 0.30:
            cache_fix = d               # the day the cache alert fires
    return daily, slices, complaint_pool, cache_fix, L


def s6(complaint_pool):
    say("\n== 6. The complaint: what the model saw, and what it did not ==")
    late = [(c, ev) for c, ev in complaint_pool if c.upload_lag_h > 0 and c.upload_lag_h <= 48]
    flips = [(c, ev) for c, ev in late if c.has_endorsement and not c.duty_met]
    say(f"water back-up claims with an upload: {len(complaint_pool)}; upload after the snapshot (<=48 h): "
        f"{len(late)}; of those, endorsement on policy and inspection said duty not kept: {len(flips)}")
    c, ev = min((x for x in flips if x[0].state == "WI"), key=lambda x: x[0].upload_lag_h)
    say(f"complaint: {ev.det_id} ({c.state}, {c.date}); status {ev.output['status']}; cited {ev.output['clause']}")
    say(f"  snapshot read at {ev.tools[0]['read_at']} (call {ev.tools[0]['call_id']}); "
        f"upload arrived {c.upload_lag_h:.1f} h later")
    say(f"  spans admitted: {[o.span_id for o in ev.offered if o.decision == 'admitted']}")
    say(f"  dropped: {[(o.span_id, o.reason) for o in ev.offered if o.decision == 'dropped']}")
    c2 = copy.copy(c); c2.duty_met = True
    ev2, _ = run(c2, corpus(c.day), Config(), Sinks(rules=PII), Ledger())
    say(f"  replay with the invoice in the claim file: {ev2.output['status']} under {ev2.output['clause']}")
    R["s6"] = dict(uploads=len(complaint_pool), late=len(late), flips=len(flips), det=ev.det_id,
                   date=str(c.date), read_at=ev.tools[0]["read_at"], lag_h=c.upload_lag_h,
                   status=ev.output["status"], cited=ev.output["clause"], replay=ev2.output["status"])


def s7(daily, slices, cache_fix):
    say("\n== 7. Ninety days on the dashboard ==")
    with open(OUT / "daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "n", "coverage", "wrong", "flag_rate", "handoff", "drop_share", "mean_tokens",
                    "cached_share", "top_score", "canary_recall", "corpus"])
        for r in daily:
            w.writerow([r["day"], r["n"], r["cov"] / r["n"], r["wrong"], r["flag"] / r["n"], r["handoff"],
                        r["drop"] / r["n"], r["tok"] / r["n"], r["cached"] / r["tok"], r["top"] / r["n"],
                        r["can_hit"] / r["can_n"], r["corpus"]])
    with open(OUT / "slices.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["state", "peril", "day", "n", "covered"])
        for (st, p), byday in sorted(slices.items()):
            for d in range(DAYS):
                n, k = byday[d]
                w.writerow([st, p, d, n, k])

    def span(a, b, key, den="n"):
        num = sum(daily[d][key] for d in range(a, b)); dd = sum(daily[d][den] for d in range(a, b))
        return num / dd
    for a, b, lab in [(0, E1, "days 0-40"), (E1, E2, "days 41-49"), (E2, E4, "days 50-69"), (E4, DAYS, "days 70-89")]:
        say(f"  {lab:11s} coverage {span(a, b, 'cov'):.2%}  wrong {span(a, b, 'wrong'):.2%}  "
            f"monitor flags {span(a, b, 'flag'):.2%}  canary recall {span(a, b, 'can_hit', 'can_n'):.1%}  "
            f"tokens {span(a, b, 'tok'):.0f}  cached {span(a, b, 'cached', 'tok'):.1%}")
    mnrf = lambda a, b: (sum(slices[('MN', 'rf')][d][1] for d in range(a, b)) /
                         max(1, sum(slices[('MN', 'rf')][d][0] for d in range(a, b))))
    say(f"  MN roof slice coverage: before day {E1} {mnrf(0, E1):.1%}; days {E1}-49 {mnrf(E1, E2):.1%}; "
        f"after {mnrf(E2, DAYS):.1%}; slice share of traffic "
        f"{sum(slices[('MN','rf')][d][0] for d in range(DAYS)) / sum(r['n'] for r in daily):.1%}")
    say(f"  cache: timestamp in prefix from day {E3}; cache alert fires on day {cache_fix} "
        f"(cached share day {E3}: {daily[E3]['cached']/daily[E3]['tok']:.1%})")
    bands = {}
    for band in ["<1000", "1000-1299", "1300+"]:
        pre = [daily[d] for d in range(DAYS)]
        n = sum(r["b_n_" + band] for r in pre); k = sum(r["b_hit_" + band] for r in pre)
        bands[band] = (k / n if n else None, n)
    say(f"  canary recall by packet length: " + "; ".join(f"{b} {v[0]:.1%} (n={v[1]})" for b, v in bands.items() if v[0] is not None))
    R["s7"] = dict(periods={lab: dict(cov=round(span(a, b, 'cov'), 4), wrong=round(span(a, b, 'wrong'), 4),
                                      flags=round(span(a, b, 'flag'), 4),
                                      canary=round(span(a, b, 'can_hit', 'can_n'), 4),
                                      tokens=round(span(a, b, 'tok')),
                                      cached=round(span(a, b, 'cached', 'tok'), 4))
                           for a, b, lab in [(0, E1, "0-40"), (E1, E2, "41-49"), (E2, E4, "50-69"), (E4, DAYS, "70-89")]},
                   mnrf=[round(mnrf(0, E1), 4), round(mnrf(E1, E2), 4), round(mnrf(E2, DAYS), 4)],
                   cache_fix=cache_fix, bands={b: [round(v[0], 4), v[1]] for b, v in bands.items() if v[0] is not None})


def alerts(daily, slices):
    base = sum(r["cov"] for r in daily[:28]) / sum(r["n"] for r in daily[:28])
    import statistics
    sd = statistics.pstdev([r["cov"] / r["n"] for r in daily[:28]])
    static = [r["day"] for r in daily if r["cov"] / r["n"] < 0.97]
    ew = Ewma(base, sd)
    for r in daily:
        ew.observe(r["day"], r["cov"] / r["n"])
    alarms = {}
    for key, byday in slices.items():
        n0 = sum(byday[d][0] for d in range(28)); k0 = sum(byday[d][1] for d in range(28))
        if n0 < 50:
            continue
        a = Alarm(f"{key[0]}-{key[1]}", k0 / n0)
        for d in range(DAYS):
            a.observe(d, byday[d][1], byday[d][0])
        alarms[key] = a
    return static, ew.fired, alarms


def s8(daily, slices, quiet):
    say("\n== 8. Alerting on retrieval drift: three rules, same ninety days ==")
    base = sum(r["cov"] for r in daily[:28]) / sum(r["n"] for r in daily[:28])
    import statistics
    sd = statistics.pstdev([r["cov"] / r["n"] for r in daily[:28]])
    static = [r["day"] for r in daily if r["cov"] / r["n"] < 0.97]
    ew = Ewma(base, sd)
    for r in daily:
        ew.observe(r["day"], r["cov"] / r["n"])
    alarms = {}
    for key, byday in slices.items():
        n0 = sum(byday[d][0] for d in range(28)); k0 = sum(byday[d][1] for d in range(28))
        if n0 < 50:
            continue
        a = Alarm(f"{key[0]}-{key[1]}", k0 / n0)
        for d in range(DAYS):
            a.observe(d, byday[d][1], byday[d][0])
        alarms[key] = a
    qs, qe, qa = alerts(*quiet)
    quiet_counts = dict(static=len(qs), ewma=len(qe), slice=sum(len(a.fired) for a in qa.values()))
    say(f"  quiet ninety days (no changes): static {len(qs)} alerts, ewma {len(qe)}, "
        f"slice alarms {quiet_counts['slice']} across {len(qa)} slices")

    def first(days, a, b=DAYS):
        x = [d for d in days if a <= d < b]
        return x[0] if x else None
    fa = lambda days: len([d for d in days if d < E1])
    slice_days = sorted(d for a in alarms.values() for d in a.fired)
    mn = alarms.get(("MN", "rf"))
    out = {"baseline": round(base, 4), "sd": round(sd, 4),
           "static": dict(first_e1=first(static, E1, E2), first_e2=first(static, E2, E4), false=fa(static)),
           "ewma": dict(first_e1=first(ew.fired, E1, E2), first_e2=first(ew.fired, E2, E4), false=fa(ew.fired)),
           "slice": dict(first_e1=first(mn.fired if mn else [], E1, E2),
                         first_e2=first([d for k, a in alarms.items() if k != ("MN", "rf") for d in a.fired], E2, E4),
                         false=fa(slice_days), slices=len(alarms),
                         fired={f"{k[0]}-{k[1]}": a.fired[:3] for k, a in alarms.items() if a.fired})}
    for k in ["static", "ewma", "slice"]:
        v = out[k]
        say(f"  {k:6s} first alert for the bulletin (day {E1}+): {v['first_e1']}; for corpus growth (day {E2}+): "
            f"{v['first_e2']}; false alerts days 0-40: {v['false']}")
    say(f"  baseline coverage {base:.2%} (sd {sd:.2%}); slices watched {len(alarms)}; fired: {out['slice']['fired']}")
    out["quiet"] = quiet_counts
    R["s8"] = out


def main():
    L, S, packets = s1()
    s2(L, S, packets)
    s3(L, S, packets)
    s4(L)
    s5()
    daily, slices, pool, fix, _ = simulate()
    s6(pool)
    s7(daily, slices, fix)
    qd, qsl, _, _, _ = simulate(changes=False)
    s8(daily, slices, (qd, qsl))
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=str))
    (HERE / "output.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
