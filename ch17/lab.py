"""Chapter 17 lab: build Lakeshore's policy graph and answer multi-hop questions.

Run from 04-code/:  python3 ch17/lab.py   (writes ch17/out/)
Stand-ins: sim-extract proposes edges; sim-reader answers correctly whenever
every span the answer needs is in its packet. Prices are illustrative, in US$.
"""
from __future__ import annotations

import csv, json, os, random, sys
from collections import Counter, defaultdict
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "ch15"), HERE]

from contextlab.graph import Candidate, Edge, Graph, compose, validate, SPINE
from contextlab.policy import Policy
from contextlab.retrieval.index import InvertedIndex, top
from corpus17 import GOLD_EDGES, control, multi_hop, sim_extract, spans

OUT = os.path.join(HERE, "out")
CH16_FULL_WRONG = 16          # ASSUMED Chapter 16 close on the full mix [VERIFY against ch15.md/ch16.md]
GOV = ("form", "endorsement", "bulletin")


def setup():
    sp = spans()
    pol = Policy(sp)
    pol.unknown_blocks = True
    idx = InvertedIndex()
    for s in sp:
        idx.add(s.sid, s.text)
    return sp, pol, idx


def gold_edges(sp):
    g = set(GOLD_EDGES)
    g |= {(s.doc_id, "HAS_CLAUSE", f"clause:{s.topic}") for s in sp if s.kind == "form"}
    g |= {(s.doc_id, "CONSTRAINS", f"clause:{s.topic}") for s in sp if s.kind == "bulletin"}
    return g


def review(edges, gold, minutes=3.0):
    """A person decides every quarantined edge that is not a conflict."""
    out, spent = [], 0.0
    for e in edges:
        if e.status == "quarantined" and not e.reason.startswith("conflicts"):
            spent += minutes
            key = (e.c.src, e.c.rel, e.c.dst)
            e = replace(e, status="published" if key in gold else "rejected",
                        reason="approved by reviewer" if key in gold else "rejected by reviewer")
        out.append(e)
    return out, spent


def build(pol, sp, derive=True, conflicts="edition", reviewed=True, extra=()):
    cands = sim_extract(sp) + list(extra)
    edges = validate(cands, pol, derive_validity=derive, conflict_scope=conflicts)
    mins = 0.0
    if reviewed:
        edges, mins = review(edges, gold_edges(sp))
        for e in edges:                          # approved edges get derived validity
            if e.reason == "approved by reviewer":
                src = pol.spans[e.c.source_sid]
                end = pol.end_of(src)[0] if derive else src.effective_to
                e.valid_from, e.valid_to = str(src.effective_from), (str(end) if end else None)
    return cands, edges, Graph(edges), mins


def pr(keys, gold):
    keys = set(keys)
    tp = len(keys & gold)
    return (tp / len(keys) if keys else 0.0, tp / len(gold))


def tokens(sp_by, sids):
    return sum(int(len(sp_by[x].text.split()) * 1.33) + 12 for x in sids)


class Runner:
    def __init__(self, sp, pol, idx, graph):
        self.sp, self.pol, self.idx, self.g = sp, pol, idx, graph
        self.by = {s.sid: s for s in sp}

    def vector(self, q, k):
        allowed = {s.sid for s in self.sp if self.pol.admit(s, q.req).ok}
        return top(self.idx.bm25(q.question, allowed=allowed), k)

    def judge(self, q, chain):
        if chain.status != "ok":
            return "withheld"
        limits = {r[1] for r in chain.relations if r[1].startswith("LIMIT")}
        ok = (set(chain.evidence) == set(q.gold) and set(chain.inapplicable) == set(q.inapplicable)
              and limits == ({q.relation} if q.relation else limits & set()))
        return "correct" if ok else "wrong"

    def ch16(self, q):
        packet = self.vector(q, 4)
        res = self.pol.resolve(packet, q.req)
        if res.status != "ok":
            return "withheld", packet
        ok = (res.governing,) == tuple(q.gold) and not q.inapplicable
        return ("correct" if ok else "wrong"), packet

    def reader(self, q, k):
        packet = self.vector(q, k)
        return ("correct" if set(q.gold) <= set(packet) else "wrong"), packet

    def graph_only(self, q):
        ch = compose(self.g, self.pol, q.req, q.req.cause)
        return self.judge(q, ch), ch.evidence

    def topic_of(self, q):
        """The clause the question is about: summed BM25 by topic over governing spans."""
        allowed = {s.sid for s in self.sp if s.kind in GOV and s.edition_family != "PAP"
                   and self.pol.admit(s, q.req).ok}
        by = defaultdict(float)
        for sid, v in self.idx.bm25(q.question, allowed=allowed).items():
            by[self.by[sid].topic] += v
        return max(by, key=by.get) if by else q.req.cause

    def hybrid(self, q, hops=2, scoped=True):
        hits = self.vector(q, 4)
        gov = [h for h in hits if self.by[h].kind in GOV]
        ch = compose(self.g, self.pol, q.req, self.topic_of(q),
                     seeds={self.by[h].doc_id for h in gov}, hops=hops, scoped=scoped)
        ctx = [h for h in hits if self.by[h].kind == "guideline"][:1]
        return self.judge(q, ch), ch.evidence + ctx, ch

    def naive(self, q, hops):
        """Neighbourhood expansion: every admitted span within h hops of the hits."""
        hits = self.vector(q, 4)
        docs = {self.by[h].doc_id for h in hits}
        reached = self.g.walk(docs, q.req.date_of_loss, hops,
                              rels=tuple(SPINE)) if hops else {d: 0 for d in docs}
        nodes = set(reached) | docs
        topics = {n.split(":")[1] for n in nodes if n.startswith("clause:")} | {self.by[h].topic for h in hits}
        extra = [s.sid for s in self.sp if s.doc_id in nodes and s.kind in GOV
                 and s.topic in topics and self.pol.admit(s, q.req).ok]
        packet = list(dict.fromkeys(hits + sorted(extra)))
        return ("correct" if set(q.gold) <= set(packet) else "wrong"), packet


def score(runner, qs, fn):
    rows = []
    for q in qs:
        out = fn(q)
        rows.append((q, out[0], tokens(runner.by, out[1])))
    c = Counter(r[1] for r in rows)
    return {"correct": c["correct"], "withheld": c["withheld"], "wrong": c["wrong"],
            "n": len(qs), "tokens": round(sum(r[2] for r in rows) / len(rows))}, rows


# ---- the maintenance ledger (illustrative prices, US$) ----
LEDGER = dict(docs_changed=[55, 48, 61, 52, 57, 50, 160, 70, 54, 49, 58, 53],
              edges_per_doc=4, in_tok=7_500, out_tok=900, in_price=3.0, out_price=15.0,
              gate_minutes_per_edge=1.0, quarantine_minutes=3.0, reviewer_rate=70.0,
              steward_fte=0.25, steward_loaded=165_000, store_month=400.0,
              determinations_month=3_200 * 22)


def ledger(quarantine_rate, mean_degree, graph_share, avoided_per_1000):
    L, rows = LEDGER, []
    for m, n in enumerate(L["docs_changed"], 1):
        docs = n * (1 + mean_degree)                     # each change re-reads its neighbours
        extract = docs * (L["in_tok"] * L["in_price"] + L["out_tok"] * L["out_price"]) / 1e6
        edges = n * L["edges_per_doc"]
        gate = edges * L["gate_minutes_per_edge"] / 60 * L["reviewer_rate"]
        quar = edges * quarantine_rate * L["quarantine_minutes"] / 60 * L["reviewer_rate"]
        steward = L["steward_fte"] * L["steward_loaded"] / 12
        total = extract + gate + quar + steward + L["store_month"]
        rows.append(dict(month=m, docs_changed=n, docs_reextracted=round(docs), extract=round(extract, 2),
                         gate=round(gate, 2), quarantine=round(quar, 2), steward=round(steward, 2),
                         store=L["store_month"], total=round(total, 2)))
    total = sum(r["total"] for r in rows) / 12
    dets = L["determinations_month"]
    avoided = dets * avoided_per_1000 / 1000
    return rows, dict(month_mean=round(total, 2), per_determination=round(total / dets, 4),
                      per_graph_routed=round(total / (dets * graph_share), 3),
                      avoided_month=round(avoided), break_even_per_avoided=round(total / avoided, 2))


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    from corpus15 import full_mix
    os.makedirs(OUT, exist_ok=True)
    sp, pol, idx = setup()
    hard, ctrl = multi_hop(), control()
    log, res = [], {}
    say = lambda *a: log.append(" ".join(str(x) for x in a))

    # 1. extraction, validation, review
    gold = gold_edges(sp)
    cands, raw_edges, _, _ = build(pol, sp, reviewed=False)
    _, edges, g, minutes = build(pol, sp)
    stages = {"proposed": [(c.src, c.rel, c.dst) for c in cands],
              "validated": [(e.c.src, e.c.rel, e.c.dst) for e in raw_edges if e.status == "published"],
              "reviewed": [(e.c.src, e.c.rel, e.c.dst) for e in edges if e.status == "published"]}
    ext = []
    for rel in sorted({k[1] for k in gold}):
        g_rel = {k for k in gold if k[1] == rel}
        row = {"relation": rel, "gold": len(g_rel)}
        for st, keys in stages.items():
            p, r = pr([k for k in keys if k[1] == rel], g_rel)
            row[f"{st}_p"], row[f"{st}_r"] = round(p, 3), round(r, 3)
        ext.append(row)
    write_csv("extraction.csv", ext)
    status = Counter(e.status for e in raw_edges)
    res["extraction"] = {st: dict(zip(("precision", "recall"), map(lambda x: round(x, 3), pr(k, gold))))
                         for st, k in stages.items()}
    res["extraction"].update(candidates=len(cands), gold=len(gold), status=dict(status),
                             review_minutes=minutes, false_positives_published=sorted(
                                 "|".join(k) for k in set(stages["reviewed"]) - gold))
    say("extraction", json.dumps(res["extraction"]))
    for e in raw_edges:
        if e.status != "published":
            say("  ", e.status, e.c.src, e.c.rel, e.c.dst, round(e.c.confidence or 0, 2), e.reason)

    # 2. four configurations, hard and control
    R = Runner(sp, pol, idx, g)
    configs = {"ch16 (lexical top-4 + resolver)": R.ch16,
               "vector reader k=4": lambda q: R.reader(q, 4),
               "vector reader k=8": lambda q: R.reader(q, 8),
               "graph only (coded cause)": R.graph_only,
               "hybrid (vector entry + graph chain)": lambda q: R.hybrid(q)[:2]}
    table, per_t = [], []
    for name, fn in configs.items():
        for label, qs in (("hard", hard), ("control", ctrl)):
            s, rows = score(R, qs, fn)
            table.append({"config": name, "set": label, **s})
            if label == "hard":
                for t, c in sorted(Counter((r[0].template, r[1]) for r in rows).items()):
                    per_t.append({"config": name, "template": t[0], "outcome": t[1], "n": c})
    write_csv("configs.csv", table); write_csv("by_template.csv", per_t)
    res["configs"] = table
    for r in table:
        say(f"{r['config']:38s} {r['set']:8s} correct {r['correct']:2d} withheld {r['withheld']:2d} "
            f"wrong {r['wrong']:2d} tokens {r['tokens']}")

    # 3. before review: quarantined edges keep their neighbourhood withheld
    R0 = Runner(sp, pol, idx, build(pol, sp, reviewed=False)[2])
    s, rows = score(R0, hard, lambda q: R0.hybrid(q)[:2])
    res["before_review"] = s
    say("hybrid before review", s, Counter(r[0].template for r in rows if r[1] == "withheld"))

    # 4. failure injection: the edge from the superseded 2022 edition
    inj = []
    for name, conflicts, scoped in (("family edges, no checks", None, False),
                                    ("family edges + conflict check", "family", False),
                                    ("edition-scoped edges", "edition", True)):
        Ri = Runner(sp, pol, idx, build(pol, sp, conflicts=conflicts)[2])
        s, rows = score(Ri, hard, lambda q: Ri.hybrid(q, scoped=scoped)[:2])
        lim = Counter(r[1] for r in rows if r[0].template in ("shared limit", "additional limit"))
        inj.append({"loader": name, **s, "limit_correct": lim["correct"],
                    "limit_withheld": lim["withheld"], "limit_wrong": lim["wrong"]})
        say("injection", inj[-1])
    write_csv("injection.csv", inj); res["injection"] = inj

    # 5. hops: neighbourhood expansion against the composed chain
    hops = []
    for h in range(4):
        s, _ = score(R, hard, lambda q: R.naive(q, h))
        sc, _ = score(R, ctrl, lambda q: R.naive(q, h))
        hops.append({"hops": h, "complete": s["correct"], "n": s["n"], "tokens": s["tokens"],
                     "control_tokens": sc["tokens"]})
        say("expansion", hops[-1])
    write_csv("hops.csv", hops); res["hops"] = hops

    # 6. the full mix: who reaches the graph, and Chapter 15's six
    mix, via, six = full_mix(), Counter(), []
    for d in mix:
        gold_d = set(d.gold) if isinstance(d.gold, tuple) else {d.gold}
        two = bool(d.carried and "two governing" in d.carried)
        ch = compose(g, pol, d.req, d.req.cause, cite_constrained=two)
        via[ch.via] += 1
        if ch.via == "graph" and not two:
            via["graph_contains_reviewer_span"] += gold_d <= set(ch.evidence)
        if two:
            six.append({"did": d.did, "composed": "|".join(ch.evidence), "matches": set(ch.evidence) == gold_d})
    res["full_mix"] = {"n": len(mix), **via, "six": six}
    say("full mix", dict(via), "six matched", sum(x["matches"] for x in six))

    # 7. the ledger
    share = (via["graph"]) / len(mix)
    nb = defaultdict(set)                         # documents that share an edge
    for e in edges:
        if e.status == "published" and not e.c.dst.startswith("clause:"):
            nb[e.c.src].add(e.c.dst); nb[e.c.dst].add(e.c.src)
    docs = {s.doc_id for s in sp if s.kind in GOV}
    mean_degree = round(sum(len(nb[d]) for d in docs) / len(docs), 2)
    q_rate = status["quarantined"] / max(1, len(cands))
    rows, summary = ledger(q_rate, mean_degree, share, avoided_per_1000=6)
    write_csv("ledger.csv", rows)
    res["ledger"] = {**summary, "graph_share": round(share, 4), "mean_degree": mean_degree,
                     "quarantine_rate": round(q_rate, 3), "assumed_ch16_full_wrong": CH16_FULL_WRONG,
                     "projected_ch17_full_wrong": CH16_FULL_WRONG - sum(x["matches"] for x in six)}
    say("ledger", json.dumps(res["ledger"]))
    res["graph"] = {"nodes": len({n for e in edges if e.status == "published" for n in (e.c.src, e.c.dst)}),
                    "edges": sum(e.status == "published" for e in edges), "spans": len(sp)}
    say("graph", res["graph"])
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, "output.txt"), "w").write("\n".join(log) + "\n")
    print("\n".join(log))
    return res


if __name__ == "__main__":
    main()
