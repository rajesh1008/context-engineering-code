"""Chapter 38 lab: run the context layer behind a simulated model gateway.

Deterministic stand-ins (seed 38): a gateway with seven-day keys, residency
routing, deprecation aliases and shared or per-caller caches. Writes
output.txt and out/*.json. Section numbers match the chapter's.
"""
import json
import math
import os
import random
from collections import Counter

from contextlab.gateway import (KeyIssuer, CredentialProvider, SimGateway,
                                ResidencyPolicy, verify_record, cache_namespace,
                                probe_check, join_traces, cosine, _h, VERSION)
from lakeshore38 import (SEED, DAY, DETS_PER_DAY, EMBED_RETIRE_AT, catalogue,
                         FEATURES, PROBES, corpus, queries, embed,
                         verifier_scores, CITE, model_sentences)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
LINES = []
PRICE_IN, PRICE_OUT, READ = 3.0, 15.0, 0.1     # US$ per million, stand-in


def say(*a):
    LINES.append(" ".join(str(x) for x in a))


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)


def pct(a, b):
    return 100.0 * a / b if b else 0.0


# 1 ------------------------------------------------------------------------
def section_capabilities():
    say("== 1. What the context layer loses behind the gateway ==")
    c = Counter((f[2]) for f in FEATURES)
    for name, ch, tr, raw in FEATURES:
        say(f"  {name:32s} ch{ch:<3d} translated={tr:9s} raw={raw}")
    say("  translated path:", dict(c))
    rng = random.Random(SEED)
    prefix_t, suffix_t, out_t, replicas = 9000, 3000, 700, 4
    groups = [f"{s}-{f}" for s in "ABCDEF" for f in ("HO3", "HO5", "HO4", "DP3")]
    res = {}
    ttl = 10 / 60          # provider cache entry lives 10 minutes after last use
    times = sorted(rng.uniform(8, 24) for _ in range(DETS_PER_DAY))
    picks = [rng.choice(groups) for _ in times]
    spread = [rng.randrange(replicas) for _ in times]
    for mode in ("routing key forwarded", "routing key dropped"):
        last, inp, cached, cost = {}, 0, 0, 0.0
        for t, g, rr in zip(times, picks, spread):
            rep = int(_h(g)[:4], 16) % replicas if mode.startswith("routing key f") else rr
            hit = (g, rep) in last and t - last[(g, rep)] <= ttl
            last[(g, rep)] = t
            c_t = prefix_t if hit else 0
            inp += prefix_t + suffix_t
            cached += c_t
            cost += ((prefix_t + suffix_t - c_t) * PRICE_IN + c_t * PRICE_IN * READ
                     + out_t * PRICE_OUT) / 1e6
        res[mode] = {"hit_rate": round(pct(cached, inp), 1),
                     "cost_per_det": round(cost / DETS_PER_DAY, 4)}
        say(f"  {mode:22s} token-weighted hit {res[mode]['hit_rate']}%  "
            f"US${res[mode]['cost_per_det']:.4f}/determination")
    a, b = res["routing key forwarded"], res["routing key dropped"]
    say(f"  cost increase when the key is dropped: {pct(b['cost_per_det'] - a['cost_per_det'], a['cost_per_det']):.1f}%")
    dump("capabilities.json", {"features": FEATURES, "cache": res})
    return res


# 2 ------------------------------------------------------------------------
def section_residency():
    say("== 2. Residency for embeddings, caches and judges ==")
    pol = ResidencyPolicy()
    inventory = [  # call site, data class, region before, region after
        ("Determination generation", "claim_file", "us", "us"),
        ("Query embedding", "claim_file", "us", "us"),
        ("Span embedding (nightly)", "public", "us", "us"),
        ("Provider prompt cache", "claim_file", "us", "us"),
        ("Citation verifier (judge)", "claim_file", "global", "us"),
        ("Policyholder letter draft", "claim_file", "global", "us"),
        ("Gateway access log", "claim_file", "n/a", "us"),
    ]
    viol_before = [r[0] for r in inventory if not pol.permits(r[1], r[2]) and r[2] != "n/a"]
    viol_after = [r[0] for r in inventory if not pol.permits(r[1], r[3])]
    say("  outside policy before:", viol_before)
    say("  outside policy after:", viol_after)
    # Outage hour: both US deployments of the generation model unavailable.
    out = {}
    calls = DETS_PER_DAY // 24
    for fb in ("any", "in_region", "none"):
        iss = KeyIssuer()
        gw = SimGateway(catalogue(), iss, fallback=fb)
        gw.down |= {"sim-large@2026-04/us-1", "sim-large@2026-04/us-2"}
        sec = iss.current(10)[1]
        recs = [gw.call("claims-assistant", "sim-large@2026-04", sec, 10.0,
                        "P" * 400, "S" * 40, 700, allowed_regions={"us"}) for _ in range(calls)]
        ok = [r for r in recs if r.status == 200]
        bad = [r for r in ok if not pol.permits("claim_file", r.region)]
        out[fb] = {"calls": calls, "served": len(ok), "outside_region": len(bad),
                   "queued_503": sum(r.status == 503 for r in recs)}
        say(f"  outage hour, fallback={fb:9s}", out[fb])
    # one US deployment down only
    iss = KeyIssuer()
    gw = SimGateway(catalogue(), iss, fallback="in_region")
    gw.down.add("sim-large@2026-04/us-1")
    r = gw.call("claims-assistant", "sim-large@2026-04", iss.current(10)[1], 10.0,
                "P", "S", 1, allowed_regions={"us"})
    say(f"  one US deployment down, in_region: served by {r.deployment}, fallback={r.fallback}")
    # Judge moves: recalibrate the verifier threshold.
    gold, old, new = verifier_scores(random.Random(SEED))

    def flags(scores, th):
        f = [i for i, s in enumerate(scores) if s < th]
        real = sum(1 for i in f if gold[i] == 0)
        return len(f), real, sum(1 for g in gold if g == 0)
    sweep = []
    for th in [x / 100 for x in range(40, 76, 5)]:
        n_o, r_o, tot = flags(old, th)
        n_n, r_n, _ = flags(new, th)
        sweep.append({"th": th, "old_prec": round(pct(r_o, n_o), 1),
                      "old_rec": round(pct(r_o, tot), 1),
                      "new_prec": round(pct(r_n, n_n), 1), "new_rec": round(pct(r_n, tot), 1),
                      "old_flags": n_o, "new_flags": n_n})
    for s in sweep:
        say("  verifier th={th:.2f} old P/R {old_prec}/{old_rec} ({old_flags} flags)  "
            "new P/R {new_prec}/{new_rec} ({new_flags} flags)".format(**s))
    base = next(s for s in sweep if s["th"] == 0.6)
    cands = [s for s in sweep if s["new_rec"] >= base["old_rec"] - 2.0]
    best = max(cands, key=lambda s: (s["new_prec"], -s["th"])) if cands else base
    say(f"  old verifier at 0.60: P {base['old_prec']}% R {base['old_rec']}%;"
        f" new verifier at 0.60: P {base['new_prec']}% R {base['new_rec']}%;"
        f" new threshold chosen {best['th']:.2f}: P {best['new_prec']}% R {best['new_rec']}%")
    dump("residency.json", {"inventory": inventory, "outage": out, "sweep": sweep,
                            "chosen": best})
    return out, sweep, best


# 3 ------------------------------------------------------------------------
def section_cache_isolation():
    say("== 3. Cross-tenant cache-hit test ==")
    res = {}
    rng = random.Random(SEED + 3)
    prefixes = ["".join(rng.choice("abcdefgh ") for _ in range(16000)) for _ in range(250)]
    for mode in ("pooled", "client_key", "identity"):
        iss = KeyIssuer()
        gw = SimGateway(catalogue(), iss, cache_mode=mode)
        sec = iss.current(5)[1]
        hits = 0
        for i, p in enumerate(prefixes):
            gw.call("claims-assistant", "sim-large@2026-04", sec, 5.0, p, "q", 10,
                    client_key="lakeshore-claims", allowed_regions={"us"})
            r = gw.call("underwriting-pilot", "sim-large@2026-04", sec, 5.0, p, "q", 10,
                        client_key="lakeshore-claims", allowed_regions={"us"})
            hits += r.cached_tokens > 0
        own = gw.call("claims-assistant", "sim-large@2026-04", sec, 5.0, prefixes[0], "q", 10,
                      client_key="lakeshore-claims", allowed_regions={"us"})
        res[mode] = {"probes": len(prefixes), "cross_tenant_hits": hits,
                     "own_repeat_cached": own.cached_tokens > 0}
        say(f"  cache_mode={mode:10s}", res[mode])
    say("  namespace for claims-assistant:", cache_namespace("claims-assistant"))
    dump("cache_isolation.json", res)
    return res


# 4 ------------------------------------------------------------------------
def section_rotation():
    say("== 4. Seven-day rotation and the embedding job that cached a key ==")
    rng = random.Random(SEED + 4)
    days = 14
    arrivals = sorted(d * DAY + rng.uniform(8, 18) for d in range(days) for _ in range(2))
    restart_at = 11 * DAY + 10      # next scheduled deploy restarts the worker
    out = {}
    for mode in ("at_start", "per_batch"):
        iss = KeyIssuer(period_h=168, overlap_h=24)
        gw = SimGateway(catalogue(), iss)
        cp = CredentialProvider(iss, mode=mode, started_at=0.0)
        last_ok, runs, indexed, fails = 0.0, [], {}, []
        for d in range(1, days + 1):
            t = d * DAY + 2
            if mode == "at_start" and t > restart_at:
                cp = CredentialProvider(iss, mode=mode, started_at=restart_at)
            pending = [a for a in arrivals if a < t and a not in indexed]
            sec = cp.secret(t)
            r = gw.call("embed-job", "open-base@1.2" if t < EMBED_RETIRE_AT else "open-base@1.3",
                        sec, t, "", "x" * 400)
            if r.status == 401:
                sec2 = cp.refresh(t)
                if sec2:
                    r = gw.call("embed-job", "open-base@1.3", sec2, t, "", "x")
            ok = r.status == 200
            runs.append({"day": d, "t": t, "status": r.status, "pending": len(pending)})
            if ok:
                last_ok = t
                for a in pending:
                    indexed[a] = t
            else:
                fails.append(d)
        lags = [indexed.get(a, days * DAY + 2 + DAY) - a for a in arrivals]
        # watermark lag sampled hourly
        wm, lo = [], 0.0
        for h in range(int(days * DAY) + 1):
            done = [r for r in runs if r["t"] <= h and r["status"] == 200]
            lo = done[-1]["t"] if done else 0.0
            wm.append(round(h - lo, 1))
        alarm = next((h for h, v in enumerate(wm) if v > 30 and h > DAY), None)
        stale_dets = 0
        for a in arrivals:
            stale_h = max(0.0, indexed.get(a, days * DAY) - a - DAY)
            stale_dets += round(DETS_PER_DAY * 0.005 * stale_h / 24)
        out[mode] = {"failed_runs": fails, "max_bulletin_lag_h": round(max(lags), 1),
                     "max_watermark_lag_h": max(wm), "alarm_hour": alarm,
                     "excess_stale_determinations": stale_dets,
                     "secret_fetches": cp.fetches, "watermark": wm}
        say(f"  mode={mode:9s} failed nights {fails} max bulletin lag {max(lags):.1f} h "
            f"max watermark lag {max(wm)} h alarm at hour {alarm} "
            f"excess stale determinations {stale_dets}")
    dump("rotation.json", out)
    return out


# 5 ------------------------------------------------------------------------
def retrieval_hits(index_model, query_model, docs, qs, k=10):
    D = [embed(index_model, d) for d in docs]
    hits = 0
    for tgt, words in qs:
        q = embed(query_model, words)
        top = sorted(range(len(D)), key=lambda i: -cosine(q, D[i]))[:k]
        hits += tgt in top
    return hits


def bm25_hits(docs, qs, k=10):
    N = len(docs)
    df = Counter(w for d in docs for w in set(d))
    hits = 0
    for tgt, words in qs:
        sc = [sum(math.log(1 + N / df[w]) for w in words if w in set(d)) for d in docs]
        top = sorted(range(N), key=lambda i: -sc[i])[:k]
        hits += tgt in top
    return hits


def section_deprecation():
    say("== 5. Deprecation under a pinned name ==")
    rng = random.Random(SEED + 5)
    docs = corpus(rng)
    qs = queries(rng, docs)
    ref = [embed("open-base@1.2", p.split()) for p in PROBES]
    res = {"queries": len(qs)}
    res["dense_matched"] = retrieval_hits("open-base@1.2", "open-base@1.2", docs, qs)
    res["dense_mismatched"] = retrieval_hits("open-base@1.2", "open-base@1.3", docs, qs)
    res["bm25"] = bm25_hits(docs, qs)
    res["dense_after_reembed"] = retrieval_hits("open-base@1.3", "open-base@1.3", docs, qs)
    for k in ("dense_matched", "dense_mismatched", "bm25", "dense_after_reembed"):
        say(f"  hit@10 {k:20s} {res[k]}/{len(qs)} ({pct(res[k], len(qs)):.1f}%)")
    pol = ResidencyPolicy()
    modes = {}
    t = EMBED_RETIRE_AT + 1
    for mode in ("refuse", "forward", "forward_masked"):
        iss = KeyIssuer()
        gw = SimGateway(catalogue(), iss, alias_mode=mode)
        sec = iss.current(t)[1]
        r = gw.call("query-embed", "open-base@1.2", sec, t, "", "q" * 40)
        served = "open-base@1.3" if r.status == 200 else None
        findings = verify_record(r, "open-base@1.2", "claim_file", pol)
        ok, m = probe_check(lambda p: embed(served, p.split()), PROBES, ref) if served \
            else (False, None)
        modes[mode] = {"status": r.status, "reported": r.served, "record_findings": findings,
                       "probe_ok": ok, "probe_min_cos": None if m is None else round(m, 3),
                       "caught_by": "404" if r.status == 404 else
                       ("record" if findings else ("probe" if not ok else "nothing"))}
        say(f"  alias_mode={mode:14s}", modes[mode])
    # probe min cosine by day, pinned name always requested, masked gateway
    series = []
    for d in range(0, 14):
        tt = d * DAY + 6
        served = "open-base@1.3" if tt >= EMBED_RETIRE_AT else "open-base@1.2"
        _, m = probe_check(lambda p: embed(served, p.split()), PROBES, ref)
        series.append({"day": d, "served": served, "min_cos": round(m, 3)})
    res["modes"], res["series"] = modes, series
    dump("deprecation.json", res)
    return res


# 6 ------------------------------------------------------------------------
def section_migration():
    say("== 6. The migration eval for a deprecated generation model ==")
    rng = random.Random(SEED + 6)
    n = 1000
    gates = {"wrong_clause_max": 4, "citation_parse_min": 99.5, "tokens_max_pct": 10.0,
             "verifier_flags_max": 0.35}
    res = {}
    wrong_old = set(rng.sample(range(n), 4))
    wrong_new = set(list(wrong_old)[:3]) | set(rng.sample(range(n), 1))
    for model, wrong, tok, flags in (("sim-large@2026-04", wrong_old, 1.0, 0.30),
                                     ("sim-large@2026-10", wrong_new, 1.062, 0.31)):
        sents = model_sentences(random.Random(SEED + len(model) + int(model[-2:])), model, n * 5)
        parsed = sum(1 for s in sents if CITE.search(s))
        res[model] = {"wrong_clause": len(wrong), "citation_parse_pct": round(pct(parsed, len(sents)), 2),
                      "input_tokens_rel": tok, "verifier_flags_per_det": flags}
        say(f"  {model}", res[model])
    new = res["sim-large@2026-10"]
    verdict = {
        "wrong_clause": new["wrong_clause"] <= gates["wrong_clause_max"],
        "citation_parse": new["citation_parse_pct"] >= gates["citation_parse_min"],
        "tokens": (new["input_tokens_rel"] - 1) * 100 <= gates["tokens_max_pct"],
        "verifier_flags": new["verifier_flags_per_det"] <= gates["verifier_flags_max"]}
    say("  gates:", gates)
    say("  verdict:", verdict)
    # cache cold start on cutover day: hit rate by 10-minute bucket, 08:00-10:00
    rng2 = random.Random(SEED + 7)
    times = sorted(rng2.uniform(8, 24) for _ in range(DETS_PER_DAY))
    last, buckets = {}, {}
    for t in times:
        g = rng2.randrange(24)
        hit = g in last and t - last[g] <= 10 / 60
        last[g] = t
        b = int((t - 8) * 6)
        hb = buckets.setdefault(b, [0, 0])
        hb[0] += hit
        hb[1] += 1
    cold = [round(pct(*buckets[b]), 1) for b in range(12)]
    steady = round(pct(sum(v[0] for k, v in buckets.items() if k >= 12),
                       sum(v[1] for k, v in buckets.items() if k >= 12)), 1)
    say("  steady-state request hit rate after 10:00:", steady)
    say("  cutover-day hit rate by 10-minute bucket from 08:00:", cold)
    res.update({"gates": gates, "verdict": verdict, "cold_start": cold,
                "wrong_overlap": len(wrong_old & wrong_new)})
    dump("migration.json", res)
    return res


# 7 ------------------------------------------------------------------------
def section_trace():
    say("== 7. Gateway logs as part of the trace ==")
    rng = random.Random(SEED + 8)
    pol = ResidencyPolicy()
    dets = list(range(DETS_PER_DAY))
    letters = set(d for d in dets if rng.random() < 0.22)
    sample = rng.sample(dets, 40)
    # Before: direct calls; letters untraced, verifier region "global".
    model_ok = sum(1 for d in sample if d not in letters)
    region_ok = 0
    say(f"  before: sample of 40, model answerable {model_ok}/40, region answerable {region_ok}/40"
        f" (letters in sample {sum(d in letters for d in sample)})")
    # After: every call through the gateway; trace keeps the gateway request id.
    iss = KeyIssuer()
    gw = SimGateway(catalogue(), iss, transforms=("redact",))
    traces, findings = [], Counter()
    trunc_at = set(rng.sample(dets, 3))
    for d in dets:
        t = 8 + d * 16 / DETS_PER_DAY
        sec = iss.current(t)[1]
        calls = [("sim-large@2026-04", "claim_file"), ("open-base@1.2", "claim_file"),
                 ("sim-verify-us@2026-06", "claim_file")]
        if d in letters:
            calls.append(("sim-letter@2026-05", "claim_file"))
        if d in trunc_at:
            gw.transforms = ("redact", "truncate")
        for m, dc in calls:
            r = gw.call("claims-assistant", m, sec, t, "POLICYHOLDER " * 300, "q" * 80, 50,
                        allowed_regions={"us"})
            traces.append({"det": d, "model": m, "request_id": r.request_id})
            for f in verify_record(r, m, dc, pol):
                findings[f] += 1
        gw.transforms = ("redact",)
    j = join_traces(traces, gw.log)
    by_det = Counter(t["det"] for t in traces)
    say("  after:", j, "calls per determination", round(len(traces) / len(dets), 2))
    say("  verify findings:", dict(findings))
    ids = {r.request_id: r for r in gw.log}
    s_after = sum(1 for d in sample if all(ids[t["request_id"]].served and ids[t["request_id"]].region
                                           for t in traces if t["det"] == d))
    say(f"  after: sample of 40 answerable on model and region {s_after}/40")
    res = {"before": {"model_ok": model_ok, "region_ok": region_ok,
                      "letters_in_sample": sum(d in letters for d in sample)},
           "after": j, "findings": dict(findings), "sample_after": s_after,
           "calls": len(traces), "dets": len(by_det)}
    dump("trace.json", res)
    return res


def main():
    say(f"contextlab.gateway v{VERSION}, seed {SEED}")
    section_capabilities()
    section_residency()
    section_cache_isolation()
    section_rotation()
    section_deprecation()
    section_migration()
    section_trace()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("\n".join(LINES))


if __name__ == "__main__":
    main()
