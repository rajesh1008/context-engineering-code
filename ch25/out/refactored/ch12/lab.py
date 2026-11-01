"""Chapter 12 lab: the retrieval ladder, rung by rung, on Lakeshore's full mix.

Deterministic stand-ins: sim-embed (bi-encoder) and sim-rerank (cross-encoder)
are rules, not trained models; see contextlab/retrieval/dense.py and rerank.py.
Run: python3 lab.py > out/output.txt   (CSV files for the figures go to out/)
"""
import csv
import os
import sys
import time
from collections import defaultdict
from statistics import mean

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contextlab.retrieval import (HNSW, InvertedIndex, SimCrossEncoder, SimEncoder,  # noqa
                                  DenseIndex, hit_at_k, mrr, ndcg_at_k, recall_at_k)
from contextlab.retrieval.corpus import Query, admitted  # noqa
from contextlab.retrieval.fusion import ranked, reciprocal_rank_fusion, weighted  # noqa
from contextlab.retrieval.pipeline import RUNGS, Ladder  # noqa
from contextlab.retrieval.queries import TWINS, build_queries  # noqa
import codebase  # noqa

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
STYLES = ["kw", "lay", "identifier", "numeric", "negation", "multihop"]


def section(t):
    print(f"\n== {t}")


def dump(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def evaluate(rank_fn, Q):
    m = defaultdict(list)
    for q in Q:
        r = rank_fn(q)
        m["hit1"].append(hit_at_k(r, q.gold, 1))
        m["hit5"].append(hit_at_k(r, q.gold, 5))
        m["rec5"].append(recall_at_k(r, q.gold, 5))
        m["mrr"].append(mrr(r, q.gold))
        m["ndcg10"].append(ndcg_at_k(r, q.gold, 10))
        m["gov5"].append(q.governing in r[:5])
        m["gov1"].append(q.governing in r[:1])
        m["st_" + q.style].append(hit_at_k(r, q.gold, 5))
        m["st1_" + q.style].append(q.governing in r[:1])
        m["split_" + q.split].append(hit_at_k(r, q.gold, 5))
        for k in range(1, 21):
            m[f"rec@{k}"].append(recall_at_k(r, q.gold, k))
    return {k: mean(v) for k, v in m.items()}


def setup(L, Q):
    section("1. Corpus and queries")
    kinds = defaultdict(int)
    for s in L.corpus.values():
        kinds[s.doc_type] += 1
    print(f"spans {len(L.corpus)}: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    for field in ("split", "style", "stratum"):
        c = defaultdict(int)
        for q in Q:
            c[getattr(q, field)] += 1
        print(f"queries by {field}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    print(f"mean admitted spans per query: "
          f"{mean(sum(admitted(s, q) for s in L.corpus.values()) for q in Q[:200]):.1f}")


def bm25_from_scratch(L, Q):
    section("2. BM25 from scratch: saturation, length and the TF-IDF trap")
    k1 = 1.2
    rows = [(tf, tf, round(tf * (k1 + 1) / (tf + k1), 3)) for tf in range(0, 11)]
    print("tf   raw   bm25 (k1=1.2, average length)")
    for r in rows:
        print(f"{r[0]:>2} {r[1]:>5} {r[2]:>6}")
    dump("saturation.csv", ["tf", "raw", "bm25"], rows)
    lrows = []
    for b in (0.0, 0.25, 0.5, 0.75, 1.0):
        for ratio in (0.25, 0.5, 1, 2, 4, 8):
            lrows.append((b, ratio, round(3 * (k1 + 1) / (3 + k1 * (1 - b + b * ratio)), 3)))
    dump("length.csv", ["b", "len_ratio", "score_tf3"], lrows)
    tf = evaluate(lambda q: L._admit(ranked(L.index.tfidf(q.text)), q), Q)
    bm = evaluate(L.keyword, Q)
    print(f"tf-idf (raw tf, no length norm): Hit@1 {tf['hit1']:.3f} Hit@5 {tf['hit5']:.3f} MRR {tf['mrr']:.3f}")
    print(f"bm25 (k1 1.2, b 0.75):           Hit@1 {bm['hit1']:.3f} Hit@5 {bm['hit5']:.3f} MRR {bm['mrr']:.3f}")
    defs = sum(1 for q in Q if L._admit(ranked(L.index.tfidf(q.text)), q)[:1]
               and "definitions" in L._admit(ranked(L.index.tfidf(q.text)), q)[0])
    defs_b = sum(1 for q in Q if L.keyword(q)[:1] and "definitions" in L.keyword(q)[0])
    print(f"queries whose rank-1 span is a Definitions section: tf-idf {defs}, bm25 {defs_b}")
    q = next(q for q in Q if q.style == "kw" and q.split == "eval")
    print(f"worked query {q.qid}: {q.text!r}")
    for t in sorted(set(L.index.analyser(q.text))):
        print(f"  term {t:14} df {L.index.df(t):>3}  idf {L.index.idf(t):.3f}")
    s = L.index.bm25(q.text)
    for d in L.keyword(q)[:3]:
        print(f"  {d:34} len {L.index.length[d]:>3}  bm25 {s[d]:.3f}  gold {q.gold.get(d, 0)}")


def sweep_k1_b(L, Q):
    section("3. BM25 parameters on the holdout split (MRR)")
    hold = [q for q in Q if q.split == "holdout"]
    rows = []
    for k1 in (0.5, 0.9, 1.2, 1.5, 2.0, 3.0):
        for b in (0.0, 0.25, 0.5, 0.75, 1.0):
            r = evaluate(lambda q: L._admit(ranked(L.index.bm25(q.text, k1, b)), q), hold)
            rows.append((k1, b, round(r["mrr"], 4), round(r["hit1"], 4)))
    best = max(rows, key=lambda r: r[2])
    print(f"default k1 1.2 b 0.75: MRR {[r for r in rows if r[0] == 1.2 and r[1] == 0.75][0][2]}")
    print(f"best k1 {best[0]} b {best[1]}: MRR {best[2]}   worst MRR {min(r[2] for r in rows)}")
    dump("k1b.csv", ["k1", "b", "mrr", "hit1"], rows)


def dense_vs_keyword(L, Q, res):
    section("4. Keyword against dense, by query style (Hit@5, governing at rank 1)")
    print(f"{'style':11}{'n':>5}{'kw Hit@5':>10}{'dense':>8}{'kw @1':>8}{'dense @1':>9}")
    for st in STYLES:
        n = sum(q.style == st for q in Q)
        k, d = res["keyword"], res["dense"]
        print(f"{st:11}{n:>5}{k['st_' + st]:>10.3f}{d['st_' + st]:>8.3f}"
              f"{k['st1_' + st]:>8.3f}{d['st1_' + st]:>9.3f}")


def fusion(L, Q):
    section("5. Fusion: raw sum, RRF and weighted (alpha tuned on eval, scored on the rest)")
    ev, rest = [q for q in Q if q.split == "eval"], [q for q in Q if q.split != "eval"]

    def run(f, qs):
        return evaluate(lambda q: L._admit(ranked(f(q))[:100], q), qs)
    rows = []
    for a in [i / 10 for i in range(11)]:
        f = lambda q, a=a: weighted(L.index.bm25(q.text), L.dense.search(q.text), a)
        e, r = run(f, ev), run(f, rest)
        rows.append((a, round(e["hit5"], 4), round(r["hit5"], 4), round(r["mrr"], 4)))
        print(f"alpha {a:.1f}: eval Hit@5 {e['hit5']:.3f} | rest Hit@5 {r['hit5']:.3f} MRR {r['mrr']:.3f}")
    raw = run(lambda q: weighted(L.index.bm25(q.text), L.dense.search(q.text), 0.5,
                                 normalise=False), rest)
    rr = run(lambda q: reciprocal_rank_fusion(L.index.bm25(q.text), L.dense.search(q.text)), rest)
    best = max(rows, key=lambda r: (r[1], -abs(r[0] - 0.5)))
    print(f"raw sum (no normalisation, 0.5/0.5): rest Hit@5 {raw['hit5']:.3f} MRR {raw['mrr']:.3f}")
    print(f"RRF (k=60): rest Hit@5 {rr['hit5']:.3f} MRR {rr['mrr']:.3f}")
    print(f"alpha chosen on eval: {best[0]} (rest Hit@5 {best[2]})")
    dump("fusion.csv", ["alpha", "eval_hit5", "rest_hit5", "rest_mrr"], rows)
    dump("fusion_ref.csv", ["method", "rest_hit5", "rest_mrr"],
         [("raw", raw["hit5"], raw["mrr"]), ("rrf", rr["hit5"], rr["mrr"])])
    q = next(q for q in Q if q.style == "lay" and q.split == "new")
    lex, den = L.index.bm25(q.text), L.dense.search(q.text)
    lr, dr = ranked(lex), ranked(den)              # ranks over the whole index
    print(f"score scales on {q.qid}: bm25 top {max(lex.values()):.2f}, dense top {max(den.values()):.3f}")
    dump("scales.csv", ["kind", "score"], [("bm25", v) for v in lex.values()] +
         [("dense", v) for v in den.values()])
    print(f"RRF worked example {q.qid}: {q.text!r} gold {q.governing}")
    fused = reciprocal_rank_fusion(lex, den)
    for d in L._admit(ranked(fused), q)[:4]:
        a = lr.index(d) + 1 if d in lr else None
        b = dr.index(d) + 1 if d in dr else None
        print(f"  {d:34} bm25 rank {a}  dense rank {b}  rrf {fused[d]:.5f}")

    def first(lst, q):
        return next((i + 1 for i, d in enumerate(lst) if d in q.gold), 999)
    worse = better = lost = 0
    for q in Q:                                    # the fusion tax, per query
        best_single = min(first(L.keyword(q), q), first(L.dense_rung(q), q))
        h = first(L.hybrid(q), q)
        worse += h > best_single
        better += h < best_single
        lost += h > 5 >= best_single
    print(f"fusion tax (alpha {L.alpha}): hybrid ranks the first gold span below the better "
          f"single retriever on {worse} of {len(Q)}, above both on {better}; "
          f"drops it out of the top 5 on {lost}")


def rerank_depth(L, Q):
    section("6. Rerank depth: quality against pairs scored (sim-rerank, 6 ms a pair)")
    rows = []
    hy = evaluate(L.hybrid, Q)
    print(f"no rerank: Hit@1 {hy['hit1']:.3f} MRR {hy['mrr']:.3f}")
    for depth in (5, 10, 20, 30, 50, 100):
        L.cross.pairs_scored = 0
        r = evaluate(lambda q: L.reranked(q, depth), Q)
        pairs = L.cross.pairs_scored / len(Q)
        rows.append((depth, round(r["hit1"], 4), round(r["mrr"], 4), round(pairs, 1), round(pairs * 6, 1)))
        print(f"depth {depth:>3}: Hit@1 {r['hit1']:.3f} MRR {r['mrr']:.3f} "
              f"pairs/query {pairs:.1f} est. ms {pairs * 6:.0f}")
    dump("rerank_depth.csv", ["depth", "hit1", "mrr", "pairs", "ms"], rows)
    moves = []
    for q in Q:
        before, after = L.hybrid(q), L.reranked(q)
        b = before.index(q.governing) + 1 if q.governing in before else 101
        a = after.index(q.governing) + 1 if q.governing in after else 101
        moves.append((q.qid, q.style, b, a))
    dump("rank_moves.csv", ["qid", "style", "before", "after"], moves)
    up = sum(a < b for _, _, b, a in moves)
    down = sum(a > b for _, _, b, a in moves)
    print(f"governing span moved up {up}, down {down}, unchanged {len(moves) - up - down}")
    for st in STYLES:
        mv = [m for m in moves if m[1] == st]
        print(f"  {st:11} up {sum(a < b for *_, b, a in mv):>3} down {sum(a > b for *_, b, a in mv):>3}")


def ladder(L, Q, res):
    section("7. The ladder on the full mix of 1,000")
    print(f"{'rung':10}{'Hit@1':>7}{'Hit@5':>7}{'R@5':>7}{'MRR':>7}{'nDCG10':>8}{'gov@5':>7}{'eval H@5':>9}")
    rows, curves, heat = [], [], []
    for r in RUNGS:
        m = res[r]
        print(f"{r:10}{m['hit1']:>7.3f}{m['hit5']:>7.3f}{m['rec5']:>7.3f}{m['mrr']:>7.3f}"
              f"{m['ndcg10']:>8.3f}{m['gov5']:>7.3f}{m['split_eval']:>9.3f}")
        rows.append((r, m["hit1"], m["hit5"], m["rec5"], m["mrr"], m["ndcg10"], m["gov5"], m["split_eval"]))
        curves += [(r, k, m[f"rec@{k}"]) for k in range(1, 21)]
        heat += [(r, st, m["st_" + st], m["st1_" + st]) for st in STYLES]
    dump("ladder.csv", ["rung", "hit1", "hit5", "rec5", "mrr", "ndcg10", "gov5", "eval_hit5"], rows)
    dump("recall_curves.csv", ["rung", "k", "recall"], curves)
    dump("heat.csv", ["rung", "style", "hit5", "gov1"], heat)
    misses = {r: sum(1 - h for h in [hit_at_k(L.run(r, q), q.gold, 5) for q in Q]) for r in ("keyword", "graph")}
    print(f"packets missing every gold span: keyword {misses['keyword']:.0f}, graph {misses['graph']:.0f} of {len(Q)}")
    for split in ("eval", "holdout", "new"):
        print(f"  {split:8} Hit@5 " + "  ".join(f"{r} {res[r]['split_' + split]:.3f}" for r in RUNGS))


def timing(L, Q):
    section("8. Wall-clock per query on this box (mean of 200 queries, ms)")
    sub = Q[:200]
    for r in ("keyword", "dense", "hybrid"):
        t = time.perf_counter()
        for q in sub:
            L.run(r, q)
        print(f"{r:8} {(time.perf_counter() - t) * 1000 / len(sub):.2f} ms (measured)")
    print("reranked adds about 15 admitted pairs x 6 ms = about 90 ms (cost model, not measured)")


def negation(L):
    section("9. Failure injection: negated near-twins (mould, LM-HO 2311 attached)")
    enc = L.dense.encoder
    (tn, gn), (tp, gp) = TWINS["neg"], TWINS["pos"]
    print(f"cosine(neg query, pos query) = {float(enc.encode(tn) @ enc.encode(tp)):.3f}")
    rows = []
    for tag, text, gold in (("neg", tn, gn), ("pos", tp, gp)):
        q = Query("TW-" + tag, "x", "x", "negation", text, "HO-3", "WI", "2024-06",
                  ("LM-HO 2311",), "mould", {gold: 2}, gold)
        print(f"{tag}: {text!r} -> correct span {gold}")
        for r in ("keyword", "dense", "hybrid", "reranked"):
            top = L.run(r, q)[:2]
            ok = top[0] == gold
            rows.append((tag, r, top[0], int(ok)))
            print(f"  {r:9} rank 1 {top[0]:22} {'right' if ok else 'WRONG'}   rank 2 {top[1]}")
    dump("twins.csv", ["query", "rung", "rank1", "correct"], rows)
    lex = L.index.bm25(tn)
    den = L.dense.search(tn)
    cross = SimCrossEncoder(L.index, enc)
    for d in (gn, gp):
        print(f"  neg query vs {d:20} bm25 {lex.get(d, 0):.2f} dense {den[d]:.3f} "
              f"rerank {cross.score(tn, L.texts[d]):.3f}")


def hnsw(seed=3):
    section("10. HNSW: recall@10 against distance computations (5,000 clustered vectors, dim 32)")
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((50, 32))
    X = centres[rng.integers(0, 50, 5000)] + 0.35 * rng.standard_normal((5000, 32))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Qv = centres[rng.integers(0, 50, 200)] + 0.35 * rng.standard_normal((200, 32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    truth = [set(np.argsort(-(X @ q))[:10]) for q in Qv]
    rows = []
    for M in (4, 8, 16):
        h = HNSW(32, M=M, ef_construction=80)
        for v in X:
            h.add(v)
        for ef in (10, 20, 40, 80, 160):
            h.distances = 0
            rec = mean(len(set(h.search(q, 10, ef)) & t) / 10 for q, t in zip(Qv, truth))
            d = h.distances / len(Qv)
            rows.append((M, ef, round(rec, 4), round(d, 1)))
            print(f"M {M:>2} ef {ef:>3}: recall@10 {rec:.3f}  distances/query {d:.0f} (flat 5000)")
    dump("hnsw.csv", ["M", "ef", "recall10", "distances"], rows)


def code_setting():
    section("11. Second setting: a SaaS billing codebase with vendored buggy twins (Austin)")
    docs, queries, calls, reachable = codebase.build()
    idx, den = InvertedIndex(), DenseIndex(SimEncoder())
    for d, t in docs.items():
        idx.add(d, t)
        den.add(d, t)
    def j(scores):
        return ranked({d: s + codebase.jitter(d) for d, s in scores.items()})

    def hy(q):
        return j(weighted(idx.bm25(q), den.search(q), 0.5))
    rungs = {
        "keyword": lambda q: j(idx.bm25(q)),
        "dense": lambda q: j(den.search(q)),
        "hybrid": hy,
        "+reachable": lambda q: [d for d in hy(q) if d in reachable],
        "+exec check": lambda q: [d for d in hy(q) if codebase.passes_test(d, docs)],
    }
    rows = []
    print(f"{'rung':12}{'style':11}{'Hit@1':>7}{'Hit@10':>8}{'twin@1':>8}")
    for name, f in rungs.items():
        for st in ("nl", "identifier", "error"):
            qs = [q for q in queries if q[0] == st]
            h1 = h10 = tw = 0
            for _, text, gold in qs:
                r = f(text)
                h1 += r[0] == gold
                h10 += gold in r[:10]
                tw += r[0].split("::")[1] == gold.split("::")[1] and r[0] != gold
            rows.append((name, st, h1 / len(qs), h10 / len(qs), tw / len(qs)))
            print(f"{name:12}{st:11}{h1 / len(qs):>7.3f}{h10 / len(qs):>8.3f}{tw / len(qs):>8.3f}")
    dump("code.csv", ["rung", "style", "hit1", "hit10", "twin1"], rows)


def rare_terms(L, Q):
    section("12. Rare terms: questions with a term in three or fewer spans (Hit@5)")
    from contextlab.retrieval.text import analyse
    rare = [q for q in Q if any(0 < L.index.df(t) <= 3 for t in analyse(q.text))]
    rest = [q for q in Q if q not in rare]
    print(f"questions with a rare term {len(rare)}, without {len(rest)}")
    for r in RUNGS:
        h = [mean(hit_at_k(L.run(r, q), q.gold, 5) for q in g) for g in (rare, rest)]
        print(f"  {r:10} rare {h[0]:.3f}  rest {h[1]:.3f}")


def rewriting(L, Q):
    section("13. Query rewriting: adjuster shorthand against the glossary (Hit@5)")
    from contextlab.retrieval.text import GLOSSARY, rewrite
    short = {v: k for k, v in GLOSSARY.items() if " endorsement" not in v}
    def abbreviate(t):
        for full, s in sorted(short.items(), key=lambda kv: -len(kv[0])):
            t = t.replace(full, s)
        return t
    base = [q for q in Q if abbreviate(q.text) != q.text]
    print(f"questions containing a glossary phrase: {len(base)}")
    for r in ("keyword", "dense", "hybrid"):
        h = []
        for f in (lambda t: t, abbreviate, lambda t: rewrite(abbreviate(t))):
            h.append(mean(hit_at_k(L.run(r, Query(**{**q.__dict__, "text": f(q.text)})), q.gold, 5)
                          for q in base))
        print(f"  {r:8} as written {h[0]:.3f}  shorthand {h[1]:.3f}  rewritten {h[2]:.3f}")


def traces(L, Q):
    section("14. Three questions, rung by rung (rank of the governing span; - = not in top 10)")
    lay = next(q for q in Q if q.style == "lay" and q.governing not in L.run("keyword", q)[:5])
    picks = [lay] + [next(q for q in Q if q.style == s and q.split == "new") for s in ("identifier", "multihop")]
    for q in picks:
        ranks = []
        for r in RUNGS:
            top = L.run(r, q)[:10]
            ranks.append(str(top.index(q.governing) + 1) if q.governing in top else "-")
        print(f"{q.qid} {q.style:10} gov {q.governing}")
        print(f"   text: {q.text}")
        print("   " + "  ".join(f"{r} {k}" for r, k in zip(RUNGS, ranks)))


def pooling(L, Q, depth=10):
    section("15. Pooled labelling: what a judge sees if only one rung builds the pool")
    for builder in ("keyword", "dense"):
        seen = [set(L.run(builder, q)[:depth]) for q in Q]
        found = mean(q.governing in s for q, s in zip(Q, seen))
        print(f"pool from {builder:8} top {depth}: governing span in pool for {found:.3f} of questions")
        for r in RUNGS:
            true = mean(hit_at_k(L.run(r, q), q.gold, 5) for q in Q)
            judged = mean(hit_at_k(L.run(r, q), {g: v for g, v in q.gold.items() if g in s}, 5)
                          for q, s in zip(Q, seen))
            print(f"   {r:10} true Hit@5 {true:.3f}  as judged {judged:.3f}")
    union = [set().union(*(L.run(r, q)[:depth] for r in RUNGS)) for q in Q]
    print(f"pool from all five rungs: governing span in pool for "
          f"{mean(q.governing in s for q, s in zip(Q, union)):.3f}; mean pool size "
          f"{mean(len(s) for s in union):.1f}")


def strata(L, Q):
    section("16. The ladder by stratum (stand-ins): Hit@5 / governing span at rank 1")
    print("stratum          n  " + "  ".join(f"{r:>13}" for r in RUNGS))
    for st in ("docstore", "policy-admin", "scanned", "multi-state", "guidance", "third-party"):
        g = [q for q in Q if q.stratum == st]
        cells = []
        for r in RUNGS:
            runs = [L.run(r, q) for q in g]
            h = mean(hit_at_k(x, q.gold, 5) for x, q in zip(runs, g))
            one = mean(x[:1] == [q.governing] for x, q in zip(runs, g))
            cells.append(f"{h:.3f} / {one:.3f}")
        print(f"{st:14} {len(g):4}  " + "  ".join(f"{c:>13}" for c in cells))


def main():
    L = Ladder()
    Q = build_queries(L.corpus)
    setup(L, Q)
    bm25_from_scratch(L, Q)
    sweep_k1_b(L, Q)
    res = {r: evaluate(lambda q, r=r: L.run(r, q), Q) for r in RUNGS}
    dense_vs_keyword(L, Q, res)
    fusion(L, Q)
    rerank_depth(L, Q)
    ladder(L, Q, res)
    timing(L, Q)
    negation(L)
    hnsw()
    code_setting()
    rare_terms(L, Q)
    rewriting(L, Q)
    traces(L, Q)
    pooling(L, Q)
    strata(L, Q)


if __name__ == "__main__":
    main()
