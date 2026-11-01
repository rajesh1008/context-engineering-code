"""Chapter 13 lab: swap the from-scratch index for a vector database.

Run from 04-code with the Chapter 13 virtual environment:
    .venv-ch13/bin/python ch13/lab.py
Writes ch13/out/output.txt and the CSV files the figures are drawn from.
"""
import collections
import csv
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import numpy as np  # noqa: E402

import corpus13 as C  # noqa: E402
from encoders import SimEncoderV2  # noqa: E402
from contextlab.retrieval.dense import SimEncoder  # noqa: E402
from contextlab.vectorstore import FlatStore, Row, Scope, TenantLeak, check  # noqa: E402
from contextlab.vectorstore_lance import LanceStore  # noqa: E402
from lancedb.index import FTS  # noqa: E402

OUT = os.path.join(HERE, "out")
DB = os.path.join(OUT, "lancedb_pq")
HN = os.path.join(OUT, "lancedb_hnsw")
RES, DELTA, INJ = {}, {}, {}
K = 10          # hit@10 and recall@10 are reported on the first ten
KP = 20         # the pipeline reranks the top 20 dense candidates (Chapter 12)
V7 = "spans-v7"  # Lakeshore's production index name (Chapter 14 migrates it to spans-v8)


def v7(**kw):
    kw.setdefault("nprobes", 100)
    kw.setdefault("refine", 20)
    return LanceStore(DB, V7, **kw)
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def pct(xs, p):
    return float(np.percentile(xs, p)) if len(xs) else 0.0


def run(store, dets, qv, k=K, scope_of=None, **kw):
    """Search every determination; return per-determination results."""
    out = []
    for d in dets:
        scope = Scope(scope_of(d) if scope_of else C.SCOPE[d.tenant])
        out.append(store.search(qv[d.qid], k, scope, **kw))
    return out


def summary(dets, res, ref=None):
    by = collections.defaultdict(lambda: collections.Counter())
    for i, (d, r) in enumerate(zip(dets, res)):
        for key in (d.tenant, "all"):
            c = by[key]
            c["n"] += 1
            c["hit"] += d.gold in r.ids[:K]
            c["ret"] += len(r.hits)
            c["incomplete"] += not r.complete
            c["empty"] += not r.hits
            if ref is not None:
                c["recall"] += len(set(r.ids[:K]) & set(ref[i].ids[:K])) / max(1, len(ref[i].ids[:K]))
    return by


def lat(res):
    ms = [r.ms for r in res]
    return pct(ms, 50), pct(ms, 99)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    t0 = time.time()
    spans, dets = C.build()
    enc = SimEncoder()
    rows = [Row(s.span_id, s.tenant, s.doc_type, s.text,
                enc.encode(s.text).astype(np.float32)) for s in spans.values()]
    qv = {d.qid: enc.encode(d.text).astype(np.float32) for d in dets}
    by_t = collections.Counter(s.tenant for s in spans.values())

    say("== 1. The corpus and the determinations")
    say(f"spans {len(spans):,}: " + ", ".join(f"{t} {n:,} ({n / len(spans):.1%})"
                                            for t, n in by_t.most_common()))
    kinds = collections.Counter(s.doc_type for s in spans.values())
    say("by type: " + ", ".join(f"{k} {v:,}" for k, v in kinds.most_common()))
    say("determinations: " + ", ".join(f"{t} {n}" for t, n in C.DETS.items()))
    say(f"embedding {len(rows):,} spans with sim-embed: {time.time() - t0:.0f} s")

    # ---------------------------------------------------------------- 2
    say("\n== 2. The swap: same interface, three engines, 1,000 determinations")
    flat = FlatStore()
    flat.upsert(rows)
    t1 = time.time()
    pq = LanceStore(DB, V7)
    pq.upsert(rows)
    pq.build_index("ivf_pq", partitions=256, sub_vectors=32)
    say(f"lancedb load + IVF_PQ (256 partitions, 32 sub-vectors, library defaults at query time): "
        f"{time.time() - t1:.0f} s, {du(DB)} MB on disk")
    t1 = time.time()
    hn = LanceStore(HN, nprobes=16, ef=80)
    hn.upsert(rows)
    hn.build_index("ivf_hnsw_sq", partitions=16, m=20, ef_construction=150)
    say(f"lancedb load + IVF_HNSW_SQ (16 partitions, m=20, ef_construction=150): "
        f"{time.time() - t1:.0f} s, {du(HN)} MB on disk")
    ref = run(flat, dets, qv, k=KP)
    engines = [("flat (from scratch)", ref),
               ("ivf_pq, defaults", run(pq, dets, qv, k=KP)),
               ("ivf_hnsw_sq, ef=80", run(hn, dets, qv, k=KP)),
               ("ivf_pq, nprobes=100, refine=20", run(v7(), dets, qv, k=KP)),
               ("spans-v7 + exact < 5,000", run(v7(exact_below=5000), dets, qv, k=KP))]
    RES.update(engines)
    say(f"{'engine':30} {'tenant':11} {'hit@10':>7} {'recall@10':>9} {'p50 ms':>7} {'p99 ms':>7}")
    rows_swap = []
    for name, res in engines:
        s = summary(dets, res, ref)
        p50, p99 = lat(res)
        for t in ("lakeshore", "foxvalley", "northwoods", "all"):
            c = s[t]
            rec = c["recall"] / c["n"]
            tail = f" {p50:7.1f} {p99:7.1f}" if t == "all" else ""
            say(f"{name:30} {t:11} {c['hit'] / c['n']:7.1%} {rec:9.3f}{tail}")
            rows_swap.append([name, t, c["hit"] / c["n"], rec, p50, p99])
        leaks = sum(any(h.tenant not in C.SCOPE[d.tenant] for h in r.hits) for d, r in zip(dets, res))
        lost = sum(d.gold in a.ids[:K] and d.gold not in b.ids[:K] for d, a, b in zip(dets, ref, res))
        gained = sum(d.gold in b.ids[:K] and d.gold not in a.ids[:K] for d, a, b in zip(dets, ref, res))
        lost20 = sum(d.gold in a.ids and d.gold not in b.ids for d, a, b in zip(dets, ref, res))
        gained20 = sum(d.gold in b.ids and d.gold not in a.ids for d, a, b in zip(dets, ref, res))
        inc = sum(not r.complete for r in res)
        say(f"{'':30} governing span @10 lost {lost}, gained {gained}; @20 lost {lost20}, gained {gained20}; "
            f"leaks {leaks}; incomplete {inc}")
        DELTA[name] = (lost20, gained20)
    write_csv("swap.csv", ["engine", "tenant", "hit", "recall", "p50", "p99"], rows_swap)

    # ---------------------------------------------------------------- 3
    say("\n== 3. Failure injection: prefilter=False on spans-v7, before small-tenant routing, to cut latency")
    post = v7(prefilter=False)
    pres = run(post, dets, qv, k=KP)
    pre = RES["ivf_pq, nprobes=100, refine=20"]
    s_pre, s_post = summary(dets, pre, ref), summary(dets, pres, ref)
    say(f"{'tenant':11} {'returned/20':>11} {'incomplete':>10} {'empty':>6} {'hit@10 pre':>10} {'post':>6}")
    rows_inj = []
    for t in ("lakeshore", "foxvalley", "northwoods", "all"):
        c, b = s_post[t], s_pre[t]
        say(f"{t:11} {c['ret'] / c['n']:11.2f} {c['incomplete'] / c['n']:10.1%} {c['empty'] / c['n']:6.1%} "
            f"{b['hit'] / b['n']:10.1%} {c['hit'] / c['n']:6.1%}")
        rows_inj.append([t, c["ret"] / c["n"], c["incomplete"] / c["n"], c["empty"] / c["n"],
                         b["hit"] / b["n"], c["hit"] / c["n"]])
    write_csv("injection.csv", ["tenant", "returned", "incomplete", "empty", "hit_pre", "hit_post"], rows_inj)
    say(f"latency: post-filter p50 {lat(pres)[0]:.1f} ms p99 {lat(pres)[1]:.1f} ms; "
        f"pre-filter p50 {lat(pre)[0]:.1f} ms p99 {lat(pre)[1]:.1f} ms")
    lost_inj = [(d, r) for d, a, r in zip(dets, pre, pres) if d.gold in a.ids and d.gold not in r.ids]
    lost10 = sum(d.gold in a.ids[:K] and d.gold not in r.ids[:K] for d, a, r in zip(dets, pre, pres))
    say(f"governing span lost from the top 10: {lost10}; from the top 20 the reranker sees: {len(lost_inj)}")
    kinds_lost = collections.Counter((d.tenant, spans[d.gold].doc_type) for d, _ in lost_inj)
    empty_new = len(lost_inj) - sum(1 for d, r in lost_inj if r.hits)
    wrong = sum(1 for d, r in lost_inj if r.hits)
    say(f"governing span lost: {len(lost_inj)} {dict(kinds_lost)}")
    say(f"newly empty packets (withheld): {empty_new}; packets without the governing span: {wrong}")
    say(f"aggregate hit@10 {s_pre['all']['hit'] / 1000:.1%} -> {s_post['all']['hit'] / 1000:.1%}")
    INJ.update(wrong=wrong, withheld=empty_new, res=pres)
    plan = v7().t.search(qv[dets[-1].qid]).distance_type("cosine").where(
        Scope(C.SCOPE["foxvalley"]).where(), prefilter=False).limit(K).explain_plan()
    say("plan, post-filter:")
    for line in plan.splitlines():
        say("   " + line.strip()[:110])
    plan = v7().t.search(qv[dets[-1].qid]).distance_type("cosine").where(
        Scope(C.SCOPE["foxvalley"]).where(), prefilter=True).limit(K).explain_plan()
    say("plan, pre-filter:")
    for line in plan.splitlines():
        say("   " + line.strip()[:110])

    # ---------------------------------------------------------------- 4
    say("\n== 4. Selectivity sweep on spans-v7: filtered recall@10 against exact, 100 queries")
    sample = dets[::10]
    sel_rows = []
    for frac in (500, 200, 100, 50, 20, 10, 5, 2):
        where = f"bucket < {frac}"
        acc = {True: [], False: []}
        got = {True: [], False: []}
        for d in sample:
            v = qv[d.qid]
            ex = (v7().t.search(v).distance_type("cosine").where(where, prefilter=True)
                  .bypass_vector_index().select(["span_id"]).limit(K).to_arrow()["span_id"].to_pylist())
            for pf in (True, False):
                ids = (v7().t.search(v).distance_type("cosine").where(where, prefilter=pf).nprobes(100).refine_factor(20)
                       .select(["span_id"]).limit(K).to_arrow()["span_id"].to_pylist())
                acc[pf].append(len(set(ids) & set(ex)) / max(1, len(ex)))
                got[pf].append(len(ids))
        sel = frac / 1000
        sel_rows.append([sel, np.mean(acc[True]), np.mean(acc[False]), np.mean(got[True]), np.mean(got[False])])
        say(f"selectivity {sel:6.1%}: pre-filter recall {np.mean(acc[True]):.3f} ({np.mean(got[True]):.1f} returned), "
            f"post-filter recall {np.mean(acc[False]):.3f} ({np.mean(got[False]):.1f} returned)")
    write_csv("selectivity.csv", ["selectivity", "recall_pre", "recall_post", "ret_pre", "ret_post"], sel_rows)

    # ---------------------------------------------------------------- 5
    say("\n== 5. Retrieval ranks by relevance, not entitlement")
    leak_rows = []
    everyone = ("lakeshore", "foxvalley", "northwoods", "public")
    for t in ("lakeshore", "foxvalley", "northwoods"):
        ds = [d for d in dets if d.tenant == t]
        ungated = run(v7(), ds, qv, scope_of=lambda d: everyone)
        exposed = sum(any(h.tenant not in C.SCOPE[t] for h in r.hits) for r in ungated)
        foreign = np.mean([sum(h.tenant not in C.SCOPE[t] for h in r.hits) for r in ungated])
        blocked = 0
        for r in run(v7(), ds, qv):
            try:
                check(r, Scope(C.SCOPE[t]))
            except TenantLeak:
                blocked += 1
        leak_rows.append([t, exposed / len(ds), foreign, blocked / len(ds)])
        say(f"{t:11} ungated: {exposed / len(ds):6.1%} of searches show another tenant's spans "
            f"({foreign:.1f} of 10 on average); scoped: {blocked / len(ds):.1%}")
    write_csv("leakage.csv", ["tenant", "exposed", "foreign_per_10", "scoped"], leak_rows)

    # ---------------------------------------------------------------- 6
    say("\n== 6. Index tuning: recall@10 against exact, 200 Lakeshore queries, pre-filter")
    tq = [d for d in dets if d.tenant == "lakeshore"][:200]
    tref = run(flat, tq, qv)
    tune_rows = []

    def sweep(store, label, settings, key):
        for val in settings:
            kw = {key: val} if key == "ef" else {}
            if key == "nprobes":
                store.nprobes = val
            res = run(store, tq, qv, **kw)
            rec = np.mean([len(set(a.ids) & set(b.ids)) / K for a, b in zip(res, tref)])
            p50, p99 = lat(res)
            tune_rows.append([label, val, rec, p50, p99])
            say(f"{label:24} {key}={val:<4} recall@10 {rec:.3f}  p50 {p50:5.1f} ms  p99 {p99:5.1f} ms")

    p50f, p99f = lat(tref)
    tune_rows.append(["flat (from scratch)", 0, 1.0, p50f, p99f])
    say(f"{'flat (from scratch)':24} exact      recall@10 1.000  p50 {p50f:5.1f} ms  p99 {p99f:5.1f} ms")
    sweep(LanceStore(DB, V7), "ivf_pq", (1, 2, 5, 10, 20, 50, 100), "nprobes")
    sweep(LanceStore(DB, V7, refine=20), "ivf_pq + refine x20", (1, 2, 5, 10, 20, 50, 100), "nprobes")
    sweep(LanceStore(HN, nprobes=16), "ivf_hnsw_sq (16 probes)", (10, 20, 40, 80, 160, 320), "ef")
    write_csv("tuning.csv", ["index", "setting", "recall", "p50", "p99"], tune_rows)

    # ---------------------------------------------------------------- 7
    say("\n== 7. Small tenants: exact search below 5,000 eligible rows")
    small = [d for d in dets if d.tenant != "lakeshore"]
    sref = run(flat, small, qv)
    route_rows = []
    for label, st in (("ivf_pq, defaults", LanceStore(DB, V7)),
                      ("ivf_pq, nprobes=100, refine=20", v7()),
                      ("exact below 5,000", v7(exact_below=5000))):
        res = run(st, small, qv)
        rec = np.mean([len(set(a.ids) & set(b.ids)) / max(1, len(b.ids)) for a, b in zip(res, sref)])
        p50, p99 = lat(res)
        inc = sum(not r.complete for r in res)
        route_rows.append([label, rec, p50, p99, inc])
        say(f"{label:20} partner recall@10 {rec:.3f}  incomplete {inc}  p50 {p50:.1f} ms  p99 {p99:.1f} ms")
    write_csv("routing.csv", ["route", "recall", "p50", "p99", "incomplete"], route_rows)

    # ---------------------------------------------------------------- 8
    say("\n== 8. Embedding lifecycle")
    life_rows = []
    # 8a: appends land unindexed until the index is optimised
    extra = [Row(f"LA-NEW-{i:05d}", "lakeshore", "manual", rows[i].text + " revised",
                 rows[i].vector) for i in range(10_000)]
    lc = v7()
    base50, base99 = lat(run(lc, tq[:200], qv))
    lc.upsert(extra)
    st_ = lc.t.index_stats("vector_idx")
    un50, un99 = lat(run(lc, tq[:200], qv))
    t3 = time.time()
    lc.t.optimize()
    opt_s = time.time() - t3
    st2 = lc.t.index_stats("vector_idx")
    op50, op99 = lat(run(lc, tq[:200], qv))
    say(f"10,000 rows appended: unindexed {st_.num_unindexed_rows:,}; p50 {base50:.1f} -> {un50:.1f} ms, "
        f"p99 {base99:.1f} -> {un99:.1f} ms")
    say(f"after optimize() ({opt_s:.0f} s): unindexed {st2.num_unindexed_rows:,}; p50 {op50:.1f} ms, p99 {op99:.1f} ms")
    life_rows += [["unindexed", "before append", base50, base99], ["unindexed", "10,000 unindexed", un50, un99],
                  ["unindexed", "after optimize", op50, op99]]
    lc.delete([r.span_id for r in extra])
    lc.t.optimize()
    # 8b: stale vectors (text edited, vector not refreshed) found by a hash audit
    edited = {d.gold for d in dets if d.tenant == "northwoods"}
    stale = {sid: spans[sid].text for sid in edited}
    say(f"edited spans: {len(edited)}; content-hash audit finds "
        f"{sum(C_hash_mismatch(lc, sid, stale[sid] + ' (2026-11 edition)') for sid in edited)} stale vectors")
    # 8c: a new embedding model, rolled out row by row into the same table
    enc2 = SimEncoderV2()
    v2 = {s.span_id: enc2.encode(s.text).astype(np.float32) for s in spans.values()}
    qv2 = {d.qid: enc2.encode(d.text).astype(np.float32) for d in dets}
    ids = list(spans)
    order = np.random.default_rng(5).permutation(len(ids))
    ev = dets[::4]
    # orthogonal Procrustes adapter, new query space -> old space, 3,000 paired spans
    train = [ids[i] for i in order[-3000:]]
    A = np.vstack([v2[i] for i in train])
    B = np.vstack([rows_by(rows)[i].vector for i in train])
    u, _, vt = np.linalg.svd(A.T @ B)
    W = u @ vt
    old = FlatStore()
    old.upsert(rows)
    mig = FlatStore()
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        cut = int(frac * len(ids))
        new = set(ids[i] for i in order[:cut])
        mig.upsert([Row(r.span_id, r.tenant, r.doc_type, r.text, v2[r.span_id] if r.span_id in new else r.vector,
                        embed_version="2" if r.span_id in new else "1") for r in rows])

        def hit(store, qmap, version=None, f=None):
            return np.mean([d.gold in store.search(f(qmap[d.qid]) if f else qmap[d.qid], K,
                                                   Scope(C.SCOPE[d.tenant]), version=version).ids for d in ev])
        naive = hit(mig, qv2)
        inplace = hit(mig, qv, "1") if frac < 1 else hit(mig, qv2, "2")
        shadow = hit(old, qv) if frac < 1 else hit(mig, qv2, "2")
        adapt = hit(old, qv2, f=lambda q: q @ W)
        life_rows.append(["migration", frac, naive, inplace, shadow, adapt])
        say(f"re-embedded {frac:4.0%}: mixed table + new queries {naive:6.1%}; in place, old queries pinned to v1 "
            f"{inplace:6.1%}; shadow table {shadow:6.1%}; adapter into the old table {adapt:6.1%}")
    write_csv("lifecycle.csv", ["exp", "x", "a", "b", "c", "d"], life_rows)

    # ---------------------------------------------------------------- 9
    say("\n== 9. Hybrid in production: full-text index plus vectors, fused by RRF")
    lance = v7()
    lance.t.create_index("text", config=FTS(), replace=True)
    idq = identifier_queries(dets, spans)
    hyb_rows = []
    for label, qs in (("lay determinations", dets[::2]), ("identifier queries", idq)):
        for mode in ("vector", "hybrid"):
            ms, hit = [], 0
            for d in qs:
                tq0 = time.perf_counter()
                where = Scope(C.SCOPE[d.tenant]).where()
                if mode == "vector":
                    q = lance.t.search(qv[d.qid] if d.qid in qv else enc.encode(d.text)).distance_type("cosine").nprobes(100).refine_factor(20)
                else:
                    v = qv[d.qid] if d.qid in qv else enc.encode(d.text)
                    q = lance.t.search(query_type="hybrid").vector(v).text(d.text)
                got = q.where(where, prefilter=True).select(["span_id"]).limit(K).to_arrow()["span_id"].to_pylist()
                ms.append((time.perf_counter() - tq0) * 1000)
                hit += d.gold in got
            hyb_rows.append([label, mode, hit / len(qs), pct(ms, 50), pct(ms, 99)])
            say(f"{label:20} {mode:7} hit@10 {hit / len(qs):6.1%}  p50 {pct(ms, 50):5.1f} ms  p99 {pct(ms, 99):5.1f} ms")
    write_csv("hybrid.csv", ["queries", "mode", "hit", "p50", "p99"], hyb_rows)
    new_row = Row("NO-BUL-2026-11", "northwoods", "bulletin", "Northwoods Savings programme notice: "
                  "ice dam claims at seasonal cabins require photographs of the attic insulation",
                  enc.encode("Northwoods Savings programme notice: ice dam claims at seasonal cabins").astype(np.float32))
    lance.upsert([new_row])
    fts_found = "NO-BUL-2026-11" in lance.t.search("attic insulation ice dam", query_type="fts").where(
        "tenant = 'northwoods'", prefilter=True).limit(5).to_arrow()["span_id"].to_pylist()
    say(f"a span added after the full-text index was built is found by full-text search: {fts_found}; "
        f"unindexed rows in the text index: {lance.t.index_stats('text_idx').num_unindexed_rows}")

    # ---------------------------------------------------------------- 10
    say("\n== 10. Observability: fourteen days, the prefilter change ships on day 9; day 11 restores it and adds routing")
    obs_rows = observe(DB, dets, qv, spans, enc)
    write_csv("observability.csv", ["day", "agg_hit", "lk_canary", "fv_canary", "nw_canary",
                                    "incomplete_lk", "incomplete_fv", "incomplete_nw", "p50", "p99"], obs_rows)

    # ---------------------------------------------------------------- 11
    say("\n== 11. Scoreboard, full mix of 1,000 (carried in: 35 wrong-clause, 11 withheld, at the Chapter 11 close)")
    say("a governing span that leaves the top 20 the reranker sees makes that determination wrong (or withheld,")
    say("if the packet is empty); a span that enters is not counted as a fix, because the carried-in errors")
    say("are not dense-recall misses")
    base = 35
    for name, res in RES.items():
        lost, gained = DELTA[name]
        w = base + lost
        p50, p99 = lat(res)
        say(f"{name:30} wrong {w:3}  rate {w / 10:.1f}%  dense p50 {p50:5.1f} ms  p99 {p99:5.1f} ms"
            f"  (to re-adjudicate: {lost + gained})")
    w = base + DELTA["ivf_pq, nprobes=100, refine=20"][0] + INJ["wrong"]
    p50, p99 = lat(INJ["res"])
    say(f"{'incident (post-filter)':30} wrong {w:3}  rate {w / 10:.1f}%  dense p50 {p50:5.1f} ms  p99 {p99:5.1f} ms"
        f"  withheld 11 + {INJ['withheld']}")
    say(f"total lab time {time.time() - t0:.0f} s")
    with open(os.path.join(OUT, "output.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


def du(path):
    return subprocess.run(["du", "-sm", path], capture_output=True, text=True).stdout.split()[0]


def s_lat(res):
    p50, p99 = lat(res)
    return f"{p50:.1f} ms, p99 {p99:.1f} ms"


_ROWS = {}


def rows_by(rows):
    if not _ROWS:
        _ROWS.update({r.span_id: r for r in rows})
    return _ROWS


def C_hash_mismatch(store, sid, new_text):
    from contextlab.vectorstore import digest
    got = store.t.search().where(f"span_id = '{sid}'").select(["content_hash"]).limit(1).to_list()
    return bool(got) and got[0]["content_hash"] != digest(new_text)


def identifier_queries(dets, spans):
    out = []
    for i, d in enumerate(dets):
        s = spans[d.gold]
        if s.doc_type == "endorsement":
            num = s.text.split(".")[0]
            out.append(C.Det(f"ID{i:04d}", d.tenant, f"does {num} apply here: {d.text}", d.gold, "identifier"))
    return out


def observe(db, dets, qv, spans, enc):
    """Daily: 200 sampled determinations plus 20 frozen canaries per tenant.

    A canary is a determination whose exact filtered top 10 was recorded at
    release; its score is today's overlap with that answer.
    """
    rng = np.random.default_rng(11)
    exact = FlatStore()
    exact.upsert([Row(s.span_id, s.tenant, s.doc_type, s.text, enc.encode(s.text).astype(np.float32))
                  for s in spans.values()])
    canaries = {t: [d for d in dets if d.tenant == t][:20] for t in C.TENANTS}
    frozen = {d.qid: set(exact.search(qv[d.qid], K, Scope(C.SCOPE[d.tenant])).ids)
              for ds in canaries.values() for d in ds}
    rows = []
    for day in range(1, 15):
        incident = 9 <= day <= 10
        st = v7(prefilter=not incident, exact_below=5000 if day >= 11 else 0)
        day_dets = [dets[i] for i in rng.choice(len(dets), 200, replace=False)]
        res = run(st, day_dets, qv)
        agg = np.mean([d.gold in r.ids for d, r in zip(day_dets, res)])
        can, inc = {}, {}
        for t, ds in canaries.items():
            rr = run(st, ds, qv)
            can[t] = np.mean([len(set(r.ids) & frozen[d.qid]) / K for d, r in zip(ds, rr)])
            inc[t] = np.mean([not r.complete for r in rr])
        p50, p99 = lat(res)
        rows.append([day, agg, can["lakeshore"], can["foxvalley"], can["northwoods"],
                     inc["lakeshore"], inc["foxvalley"], inc["northwoods"], p50, p99])
        alarm = "  ALARM" if min(can.values()) < 0.9 or max(inc.values()) > 0.05 else ""
        say(f"day {day:2} hit@10 {agg:5.1%}  canary recall lk {can['lakeshore']:.2f} fv {can['foxvalley']:.2f} "
            f"nw {can['northwoods']:.2f}  incomplete lk {inc['lakeshore']:4.0%} fv {inc['foxvalley']:4.0%} "
            f"nw {inc['northwoods']:4.0%}  p50 {p50:4.1f} ms{alarm}")
    return rows


if __name__ == "__main__":
    main()
