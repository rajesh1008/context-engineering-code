"""Chapter 20 lab: audit the baseline, rebuild Lakeshore retrieval, measure.

Run from 04-code/:  python3 ch20/lab.py > ch20/output.txt
"""
import csv, os, random, sys
from collections import Counter, defaultdict
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import ledger20 as L
import rebuild as R
from contextlab.embeddings import EmbeddingRegistry
from contextlab.retrieval.corpus import build_corpus
from contextlab.retrieval.queries import build_queries
from contextlab.retrieval.fusion import ranked, rrf

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
CORPUS = build_corpus()
QUERIES = build_queries(CORPUS)
N = len(QUERIES)
KEYS = ("hit", "gov", "gov1", "leaky", "tokens")


def say(*a):
    print(*a)


def run(cfg):
    st = R.Stack(cfg, CORPUS)
    return st, [st.packet(q) for q in QUERIES]


def summary(P):
    s = {k: sum(p[k] for p in P) for k in KEYS}
    s["tokens"] = round(s["tokens"] / len(P))
    return s


def line(name, s):
    return (f"  {name:34s} hit@5 {s['hit'] / N:.3f}  gov@5 {s['gov'] / N:.3f}  "
            f"gov@1 {s['gov1'] / N:.3f}  leaky {s['leaky']:4d}  tokens {s['tokens']:4d}")


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def group(q):
    return q.split if q.split != "new" else "new:" + q.stratum


def by(P, key):
    acc = defaultdict(lambda: Counter())
    for q, p in zip(QUERIES, P):
        acc[key(q)]["n"] += 1
        for k in ("hit", "gov", "gov1", "leaky"):
            acc[key(q)][k] += p[k]
    return acc


def section1():
    say("== 1. Audit of the baseline (Chapter 9's configuration)")
    st, P = run(R.BASELINE)
    say(f"  build {st.build_id}: {len(CORPUS)} spans in the inventory -> "
        f"{len(st.units)} fixed windows of {R.BASELINE.window} tokens")
    covered = {s for v in st.delivers.values() for s in v}
    lost = sorted(set(CORPUS) - covered)
    say(f"  spans no window holds whole (coverage < 0.8): {len(lost)} of {len(CORPUS)}")
    need = [q for q in QUERIES if q.governing in lost]
    say(f"  queries whose governing span no window holds: {len(need)}")
    s = summary(P)
    say(line("baseline", s))
    rows = []
    for g, c in sorted(by(P, group).items()):
        say(f"    {g:18s} n {c['n']:3d}  gov@5 {c['gov'] / c['n']:.3f}  gov@1 {c['gov1'] / c['n']:.3f}"
            f"  leaky {c['leaky'] / c['n']:.3f}")
        rows.append({"group": g, "n": c["n"], "gov5": round(c["gov"] / c["n"], 3),
                     "gov1": round(c["gov1"] / c["n"], 3), "config": "baseline"})
    leaks = Counter()
    for q, p in zip(QUERIES, P):
        for sid in p["spans"]:
            sp = CORPUS[sid]
            if not R.admitted(sp, q):
                leaks[("other state" if sp.state and sp.state != q.state
                       else "endorsement not on policy" if sp.doc_type == "endorsement"
                       else "other")] += 1
    say("  inadmissible spans in baseline packets, by reason:", dict(leaks))
    return P, rows, s


def section2():
    say("\n== 2. Re-chunk: fixed windows against structural spans")
    rows = []
    for chunk, size in (("fixed", 128), ("fixed", 256), ("fixed", 512), ("structural", 0)):
        for stack, base in (("keyword only", R.BASELINE), ("rebuilt stack", R.REBUILT)):
            cfg = replace(base, chunking=chunk, window=size or 256)
            st, P = run(cfg)
            s = summary(P)
            name = f"{chunk}{'-' + str(size) if size else ''}, {stack}"
            say(line(name, s) + f"  units {len(st.units)}")
            rows.append({"chunking": chunk, "window": size, "stack": stack, "units": len(st.units),
                         **{k: (s[k] / N if k != "tokens" else s[k]) for k in KEYS}})
    write("rechunk.csv", rows)
    return rows


LADDER = [("baseline (Ch 9)", R.BASELINE),
          ("+ structural spans (Ch 10)", replace(R.BASELINE, chunking="structural")),
          ("+ hybrid, dense and BM25 (Ch 12)", replace(R.BASELINE, chunking="structural",
                                                      retriever="hybrid")),
          ("+ cross-encoder re-rank (Ch 12)", replace(R.BASELINE, chunking="structural",
                                                     retriever="hybrid", rerank=True)),
          ("+ authority filter (Ch 15)", replace(R.REBUILT, precedence=False, graph=False)),
          ("+ precedence order (Ch 15)", replace(R.REBUILT, graph=False)),
          ("+ graph links (Ch 17) = rebuilt", R.REBUILT)]


def rank_of(lst, sid):
    return lst.index(sid) if sid in lst else 10 ** 6


def section3():
    say("\n== 3. Re-index and re-rank: the ladder, one stage added at a time")
    rows, packets = [], {}
    for name, cfg in LADDER:
        st, P = run(cfg)
        s = summary(P)
        packets[name] = P
        say(line(name, s))
        rows.append({"stage": name, **{k: (s[k] / N if k != "tokens" else s[k]) for k in KEYS}})
    write("ladder.csv", rows)
    say("  fusion tax: hybrid ranks the governing span below the better single retriever")
    base = replace(R.REBUILT, rerank=False, precedence=False, graph=False)
    runs = {r: R.Stack(replace(base, retriever=r), CORPUS) for r in ("keyword", "dense", "hybrid")}
    tax, tax_by, rescued = 0, Counter(), 0
    rr = R.Stack(replace(base, rerank=True), CORPUS)
    for q in QUERIES:
        rk = {r: rank_of(s.candidates(q), q.governing) for r, s in runs.items()}
        if rk["hybrid"] > min(rk["keyword"], rk["dense"]):
            tax += 1
            tax_by[q.style] += 1
            if rank_of(rr.retrieve(q), q.governing) <= min(rk["keyword"], rk["dense"]):
                rescued += 1
    say(f"    taxed queries {tax} of {N}; by style {dict(sorted(tax_by.items()))}")
    say(f"    of those, re-ranking puts the governing span back at or above the better rank: {rescued}")
    write("fusion.csv", [{"style": k, "taxed": v} for k, v in sorted(tax_by.items())])
    h, t_rrf, g_w, g_r = runs["hybrid"], 0, 0, 0
    for q in QUERIES:
        best = min(rank_of(runs[r].candidates(q), q.governing) for r in ("keyword", "dense"))
        rrf_list = [i for i in ranked(rrf(h.index.bm25(q.text), h.dense(q.text)))[:100] if h.admit(i, q)]
        t_rrf += rank_of(rrf_list, q.governing) > best
        g_w += q.governing in h.candidates(q)[:5]
        g_r += q.governing in rrf_list[:5]
    say(f"    reciprocal rank fusion instead: taxed {t_rrf}; gov@5 before re-rank {g_r / N:.3f} (weighted {g_w / N:.3f})")
    deep = sum(q.governing in rr.cross.rerank(q.text, rr.candidates(q)[:50], rr.texts)[0][:5] for q in QUERIES)
    say(f"    re-rank depth 50 instead of 30: gov@5 {deep / N:.3f}")
    return packets


def section4(packets):
    say("\n== 4. Precedence and entitlement switched on")
    off, on = packets["+ cross-encoder re-rank (Ch 12)"], packets["+ authority filter (Ch 15)"]
    prec = packets["+ precedence order (Ch 15)"]
    say(f"  packets holding an inadmissible span: {sum(p['leaky'] for p in off)} -> "
        f"{sum(p['leaky'] for p in on)}")
    say(f"  inadmissible spans in packets: {sum(p['leaks'] for p in off)} -> "
        f"{sum(p['leaks'] for p in on)}")
    fixed = [q for q, a, b in zip(QUERIES, on, prec) if b["gov1"] and not a["gov1"]]
    broke = [(q, b) for q, a, b in zip(QUERIES, on, prec) if a["gov1"] and not b["gov1"]]
    say(f"  precedence: governing span moved to first place {len(fixed)}, moved out {len(broke)}")
    say("    fixed, by governing type:", dict(Counter(CORPUS[q.governing].doc_type for q in fixed)))
    say("    moved out, by type now first:",
        dict(Counter(CORPUS[b['units'][0]].doc_type for q, b in broke)))
    say("  interaction, graph links off: governing span first (gov@1)")
    rows = []
    for auth in (False, True):
        for pr in (False, True):
            cfg = replace(R.REBUILT, authority=auth, precedence=pr, graph=False)
            s = summary(run(cfg)[1])
            say(f"    authority {'on ' if auth else 'off'}  precedence {'on ' if pr else 'off'}"
                f"  gov@1 {s['gov1'] / N:.3f}  leaky {s['leaky']}")
            rows.append({"authority": auth, "precedence": pr, "gov1": s["gov1"] / N,
                         "leaky": s["leaky"]})
    write("interaction.csv", rows)
    return fixed, broke


ABLATIONS = [("rebuilt (all stages)", R.REBUILT),
             ("- structural spans (fixed-256)", replace(R.REBUILT, chunking="fixed")),
             ("- dense (BM25 only)", replace(R.REBUILT, retriever="keyword")),
             ("- BM25 (dense only)", replace(R.REBUILT, retriever="dense")),
             ("- re-rank", replace(R.REBUILT, rerank=False)),
             ("- authority filter", replace(R.REBUILT, authority=False)),
             ("- precedence order", replace(R.REBUILT, precedence=False)),
             ("- graph links", replace(R.REBUILT, graph=False))]


def bootstrap(a, b, key, n=2000, seed=20):
    rng = random.Random(seed)
    d = [y[key] - x[key] for x, y in zip(a, b)]
    stats = sorted(sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return sum(d) / N, stats[int(0.025 * n)], stats[int(0.975 * n)]


def section5(base_P, base_rows):
    say("\n== 5. Before and after on the full set of 1,000")
    _, P = run(R.REBUILT)
    s0, s1 = summary(base_P), summary(P)
    say(line("before: baseline", s0))
    say(line("after: rebuilt", s1))
    for key in ("gov", "gov1"):
        m, lo, hi = bootstrap(base_P, P, key)
        say(f"  paired bootstrap, {key}: +{m:.3f} (95% interval {lo:+.3f} to {hi:+.3f})")
    rows = list(base_rows)
    b0, b1 = by(base_P, group), by(P, group)
    for g in sorted(b1):
        c0, c1 = b0[g], b1[g]
        say(f"    {g:18s} n {c1['n']:3d}  gov@5 {c0['gov'] / c0['n']:.3f} -> {c1['gov'] / c1['n']:.3f}"
            f"  gov@1 {c0['gov1'] / c0['n']:.3f} -> {c1['gov1'] / c1['n']:.3f}")
        rows.append({"group": g, "n": c1["n"], "gov5": round(c1["gov"] / c1["n"], 3),
                     "gov1": round(c1["gov1"] / c1["n"], 3), "config": "rebuilt"})
    write("strata.csv", rows)
    st0, st1 = by(base_P, lambda q: q.style), by(P, lambda q: q.style)
    srows = []
    for g in sorted(st1):
        say(f"    style {g:11s} n {st1[g]['n']:3d}  gov@5 {st0[g]['gov'] / st0[g]['n']:.3f} -> "
            f"{st1[g]['gov'] / st1[g]['n']:.3f}")
        srows.append({"style": g, "n": st1[g]["n"], "before": round(st0[g]["gov"] / st0[g]["n"], 3),
                      "after": round(st1[g]["gov"] / st1[g]["n"], 3)})
    write("styles.csv", srows)
    say("  ablation: remove one stage from the rebuilt stack")
    rows = []
    for name, cfg in ABLATIONS:
        _, Pa = run(cfg)
        s = summary(Pa)
        say(line(name, s))
        rows.append({"config": name, **{k: (s[k] / N if k != "tokens" else s[k]) for k in KEYS}})
    write("ablation.csv", rows)
    misses = [(q, p) for q, p in zip(QUERIES, P) if not p["gov"]]
    say(f"  rebuilt stack, governing span not in the top 5: {len(misses)}; by style",
        dict(Counter(q.style for q, _ in misses)))
    say("    by governing type", dict(Counter(CORPUS[q.governing].doc_type for q, _ in misses)))
    return P


INJECT = [("matched: index and queries on @1", {}),
          ("upgrade, no re-index", {"query_model": "sim-embed@2"}),
          ("upgrade, half re-indexed", {"query_model": "sim-embed@2", "reindexed": 0.5}),
          ("upgrade, fully re-indexed", {"query_model": "sim-embed@2", "reindexed": 1.0})]


def section6():
    say("\n== 6. Failure injection: an embedding model upgrade without re-indexing")
    rows = []
    for stack, base in (("dense only", replace(R.REBUILT, retriever="dense", rerank=False)),
                        ("rebuilt stack", R.REBUILT)):
        for name, kw in INJECT:
            st, P = run(replace(base, **kw))
            s = summary(P)
            top = sum(max(st.dense(q.text).values()) for q in QUERIES) / N
            say(line(f"{stack}: {name}", s) + f"  mean top cosine {top:.3f}")
            rows.append({"stack": stack, "case": name, "top_cosine": round(top, 3),
                         **{k: (s[k] / N if k != "tokens" else s[k]) for k in KEYS}})
    write("injection.csv", rows)
    say("  the guard: Chapter 14's registry, probes re-embedded before any query")
    reg = EmbeddingRegistry(R.PROBES)
    reg.register(R.Pinned("sim-embed@1"), "self-hosted")
    reg.bind_index("lakeshore-spans", "sim-embed@1")
    for name, model in (("same model", R.Pinned("sim-embed@1")),
                        ("new model, new name", R.Pinned("sim-embed@2")),
                        ("new model, old name", R.Pinned("sim-embed@1", R.SimEncoderV2()))):
        rep = reg.check(model, "sim-embed@1")
        say(f"    {name:22s} min probe cosine {rep.min_cosine:+.3f}  "
            f"may_query {reg.may_query('lakeshore-spans', model)}")
    return rows


def section7():
    say("\n== 7. The determination ledger: full mix of 1,000")
    for ch, wrong, what, note in L.HISTORY:
        say(f"  {ch:30s} {wrong:3d} wrong ({wrong / 10:.1f}%)  {what} {note}".rstrip())
    left = []
    for r in L.RESIDUAL:
        before = L.fixed_by_rebuild(r, L.GRAPH_BUILD_BEFORE)
        after = L.fixed_by_rebuild(r, L.GRAPH_BUILD_REBUILT)
        say(f"    {r.det} {r.stratum:13s} {r.layer:8s} {r.cause:55s} "
            f"{'FIXED' if after and not before else 'still wrong'}")
        if not after:
            left.append(r)
    say(f"  Ch 20 rebuild: {len(left)} wrong ({len(left) / 10:.1f}%)")
    say("  still wrong by layer:", dict(Counter(r.layer for r in left)))
    write("ledger.csv", [{"det": r.det, "stratum": r.stratum, "layer": r.layer, "cause": r.cause,
                          "after_rebuild": "still wrong" if r in left else "fixed",
                          "next": r.next_owner} for r in L.RESIDUAL])
    write("history.csv", [{"chapter": c, "wrong": w, "what": t} for c, w, t, _ in L.HISTORY]
          + [{"chapter": "Ch 20 rebuild", "wrong": len(left), "what": "clean rebuild"}])
    return left


def main():
    base_P, base_rows, _ = section1()
    section2()
    packets = section3()
    section4(packets)
    section5(base_P, base_rows)
    section6()
    section7()


if __name__ == "__main__":
    main()
