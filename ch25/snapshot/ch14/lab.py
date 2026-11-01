"""Chapter 14 lab: choose, adapt and migrate an embedding model.

Note: deterministic stand-ins (corpus14.py, contextlab/embeddings.py). No
provider is called; every number below is produced by this script.
Run: python3 lab.py > output.txt   (writes out/results.json for the figures)
"""
import json
import math
import os
from collections import Counter

import numpy as np

import corpus14 as C
from contextlab.embeddings import (EmbeddingRegistry, QueryAdapter, n_tokens,
                                   quantise, search, storage_bytes, words)

K = 10
SLICES = ["forms", "endorsements", "bulletins", "marine"]
R = {}


def section(t):
    print(f"\n== {t}")


def hits(top, ids, golds):
    return np.array([any(ids[j] == g or ids[j].startswith(g + "#p") for j in row)
                     for row, g in zip(top, golds)])


def by_slice(h, sl):
    return {s: round(float(h[sl == s].mean()), 3) for s in SLICES}


def bm25(docs, queries, k=K, k1=1.2, b=0.75):
    toks = [words(d) for d in docs]
    df = Counter(t for d in toks for t in set(d))
    avg, N = sum(map(len, toks)) / len(toks), len(toks)
    tf = [Counter(d) for d in toks]
    out = []
    for q in queries:
        sc = np.zeros(N)
        for t in set(words(q)):
            if t in df:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                for i in range(N):
                    f = tf[i][t]
                    if f:
                        sc[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(toks[i]) / avg))
        out.append(np.argsort(-sc, kind="stable")[:k])
    return np.array(out)


spans = C.build_spans()
ids = [s.sid for s in spans]
texts = [s.text for s in spans]
gold = C.gold_set(spans)
gq, golds = [q for q, _, _ in gold], [g for _, g, _ in gold]
sl = np.array([x for *_, x in gold])
span_of = {s.sid: s for s in spans}
M = C.models()

section(f"0. Corpus: {len(spans)} spans "
        f"({Counter(s.kind for s in spans)}), {len(gold)} gold questions "
        f"({dict(Counter(map(str, sl)))})")

# 1 ---------------------------------------------------------------- leaderboard vs gold
section("1. Public-style leaderboard versus Lakeshore's gold set (dense only, hit@10)")
ldocs, lq = C.leaderboard_set()
D, Q, R["models"] = {}, {}, {}
print(f"{'model':11}{'dims':>6}{'leaderboard':>13}{'gold':>7}  " + "".join(f"{s:>13}" for s in SLICES))
for n, m in M.items():
    D[n] = m.encode(texts, truncate=False)
    Q[n] = m.encode(gq)
    lb = float(np.mean([i in row for row, (_, i) in
                        zip(search(m.encode([q for q, _ in lq]), m.encode(ldocs), K), lq)]))
    h = hits(search(Q[n], D[n], K), ids, golds)
    R["models"][n] = {"dims": m.dim, "leaderboard": round(lb, 3),
                      "gold": round(float(h.mean()), 3), "slices": by_slice(h, sl)}
    print(f"{n:11}{m.dim:>6}{lb:>13.3f}{h.mean():>7.3f}  "
          + "".join(f"{v:>13.3f}" for v in by_slice(h, sl).values()))

# 2 ---------------------------------------------------------------- vocabulary
section("2. Where general embeddings fail: adjuster vocabulary")
example = next(q for q, g, s in gold if s == "bulletins" and "ale" in q.split())
gid = next(g for q, g, s in gold if q == example)
print(f"question: '{example}'  governing span: {gid}")
for n in M:
    top = search(M[n].encode([example]), D[n], 3)[0]
    print(f"  {n:10} top 3: " + ", ".join(ids[j] for j in top))
R["jargon"] = {}
for n, m in M.items():
    h = hits(search(Q[n], D[n], K), ids, golds)
    jq = np.array([any(t in C.JARGON_TOKENS for t in q.split()) for q in gq])
    miss = ~h
    unmapped = sum(1 for q, mi in zip(gq, miss) if mi and any(
        t in C.JARGON_TOKENS and m._direction(t)[1] < 1 for t in q.split()))
    R["jargon"][n] = {"with_jargon": round(float(h[jq].mean()), 3),
                      "plain_only": round(float(h[~jq].mean()), 3),
                      "misses": int(miss.sum()), "misses_unmapped_jargon": unmapped}
    print(f"  {n:10} hit@10 jargon questions {h[jq].mean():.3f}  plain-only "
          f"{h[~jq].mean():.3f}  misses {miss.sum():3d}, of which an unmapped "
          f"jargon term: {unmapped}")

# 3 ---------------------------------------------------------------- adapter
section("3. A query-side adapter on open-base, trained on synthetic questions")
pos = {s.sid: i for i, s in enumerate(spans)}
base = M["open-base"]


def fit_adapter(families=("BOP",), name="lks-adapter-1"):
    pairs = C.synthetic_pairs(spans, families=families)
    Qt = base.encode([q for q, _ in pairs])
    return QueryAdapter(1.0, name).fit(Qt, D["open-base"][[pos[g] for _, g in pairs]]), len(pairs)


A, n_pairs = fit_adapter()
h0 = hits(search(Q["open-base"], D["open-base"], K), ids, golds)
h1 = hits(search(A(Q["open-base"]), D["open-base"], K), ids, golds)
st = np.array([str(span_of[g].state) for g in golds])
fam = np.array([span_of[g].family for g in golds])
trained = np.isin(st, ["wisconsin", "minnesota", "illinois"])
held = np.isin(st, ["iowa", "michigan", "indiana"])
print(f"synthetic training pairs: {n_pairs} (bulletins WI, MN, IL + BOP forms; "
      f"no gold question used)")
print(f"{'':24}{'before':>8}{'after':>8}")
rows = [("all gold", h0.mean(), h1.mean())] + \
       [(s, h0[sl == s].mean(), h1[sl == s].mean()) for s in SLICES] + \
       [("bulletins, trained states", h0[trained].mean(), h1[trained].mean()),
        ("bulletins, held-out states", h0[held].mean(), h1[held].mean())]
for r in rows:
    print(f"{r[0]:26}{r[1]:>6.3f}{r[2]:>8.3f}")
gained = [(q, g) for q, g, a, b in zip(gq, golds, h0, h1) if b and not a]
lost = [(q, g) for q, g, a, b in zip(gq, golds, h0, h1) if a and not b]
print(f"questions gained {len(gained)}, lost {len(lost)}")
for q, g in lost:
    top = search(A(base.encode([q])), D["open-base"], 3)[0]
    print(f"  lost: '{q}' ({g}) -> now top 3: " + ", ".join(ids[j] for j in top))
pap = (fam == "HO-3") & (sl != "bulletins")
A2, n2 = fit_adapter(("BOP", "HO-3"), "lks-adapter-2")
h2 = hits(search(A2(Q["open-base"]), D["open-base"], K), ids, golds)
print(f"homeowners form questions: before {h0[pap].mean():.3f}, adapter-1 {h1[pap].mean():.3f}, "
      f"adapter-2 (HO-3 forms added, {n2} pairs) {h2[pap].mean():.3f}")
print("adapter-2 by slice: " + ", ".join(f"{k} {v:.3f}" for k, v in by_slice(h2, sl).items())
      + f"; all {h2.mean():.3f}; held-out-state bulletins {h2[held].mean():.3f}")
ex2 = next(q for q, g in gained if span_of[g].kind == "bulletin"
           and span_of[g].state in ("iowa", "michigan", "indiana"))
print(f"a held-out-state question the adapter fixed: '{ex2}' ({dict((q, g) for q, g in gained)[ex2]})")
for n_, A_ in [("open-base", None), ("adapter-1", A), ("adapter-2", A2)]:
    qv = base.encode([ex2])
    top = search(qv if A_ is None else A_(qv), D["open-base"], 3)[0]
    print(f"  {n_:10} top 3: " + ", ".join(ids[j] for j in top))
R["adapter"] = {"before": by_slice(h0, sl), "after": by_slice(h1, sl),
                "after2": by_slice(h2, sl), "all": [round(float(h0.mean()), 3),
                                                    round(float(h1.mean()), 3),
                                                    round(float(h2.mean()), 3)],
                "trained": [round(float(h0[trained].mean()), 3), round(float(h1[trained].mean()), 3)],
                "held": [round(float(h0[held].mean()), 3), round(float(h1[held].mean()), 3), round(float(h2[held].mean()), 3)],
                "ho3": [round(float(h0[pap].mean()), 3), round(float(h1[pap].mean()), 3),
                        round(float(h2[pap].mean()), 3)],
                "gained": len(gained), "lost": len(lost)}
ADAPTER = A2

# 4 ---------------------------------------------------------------- dims and quantisation
section("4. Dimensions and quantisation (hit@10 on all gold)")
SPANS_PROD = 118_000
R["quant"] = {}
for n in ["gen-large", "open-base"]:
    full = M[n].dim
    grid = [d for d in [1024, 768, 512, 256, 128, 64] if d <= full]
    R["quant"][n] = {}
    print(f"{n} ({'Matryoshka-ordered' if M[n].matryoshka else 'flat'})")
    print(f"  {'dims':>5}" + "".join(f"{t:>10}" for t in ["float32", "int8", "binary"]))
    for d in grid:
        row = []
        for t in ["float32", "int8", "binary"]:
            h = hits(search(quantise(Q[n], d, "float32"), quantise(D[n], d, t), K), ids, golds)
            row.append(round(float(h.mean()), 3))
        R["quant"][n][d] = row
        print(f"  {d:>5}" + "".join(f"{v:>10.3f}" for v in row))
print(f"storage for {SPANS_PROD:,} spans: " + "; ".join(
    f"{d} x {t} = {storage_bytes(SPANS_PROD, d, t) / 1e6:,.1f} MB"
    for d, t in [(1024, "float32"), (768, "float32"), (768, "int8"), (256, "int8"),
                 (1024, "binary")]))

# 5 ---------------------------------------------------------------- migration
section("5. Migration: sim-embed@1 -> open-base@1.2 + lks-adapter-2")
reg = EmbeddingRegistry(C.PROBES)
reg.register(M["sim-embed"], "self-hosted")
reg.bind_index("spans-v7", "sim-embed@1")
e_new = reg.register(base, "self-hosted", adapter=ADAPTER, on_overlong="truncate")
corpus_tokens = sum(n_tokens(t) for t in texts)
print(f"lab corpus {corpus_tokens:,} tokens to re-embed; production corpus about "
      f"20,000,000 tokens")
# dual write + backfill: the production embedding service enforces its limit
D_prod = base.encode(texts, truncate=True)
reg.bind_index("spans-v8", e_new.model_id)
log = C.query_log(spans)
lq_txt, lq_g = [q for q, _, _ in log], [g for _, g, _ in log]
lsl = np.array([s for *_, s in log])
old_top = search(M["sim-embed"].encode(lq_txt), D["sim-embed"], K)
new_top = search(ADAPTER(base.encode(lq_txt)), D_prod, K)
ov = np.array([len(set(a) & set(b)) / K for a, b in zip(old_top, new_top)])
print(f"shadow replay: {len(log)} questions; mean overlap@10 old vs new {ov.mean():.3f}")
print("  by slice: " + ", ".join(f"{s} {ov[lsl == s].mean():.3f} (n={int((lsl == s).sum())})"
                                 for s in SLICES))
gate_lab = hits(search(ADAPTER(Q["open-base"]), D["open-base"], K), ids, golds)
gate_prod = hits(search(ADAPTER(Q["open-base"]), D_prod, K), ids, golds)
h_old = hits(search(Q["sim-embed"], D["sim-embed"], K), ids, golds)
print(f"cut-over gate as run (lab-built index, aggregate): new {gate_lab.mean():.3f} "
      f"vs live {h_old.mean():.3f} -> PASS")
print(f"Sunday probe on the production index, by slice: "
      + ", ".join(f"{s} {v:.3f}" for s, v in by_slice(gate_prod, sl).items()))
long_spans = [s for s in spans if n_tokens(s.text) > base.max_input_tokens]
print(f"spans over the {base.max_input_tokens}-token limit: {len(long_spans)} "
      f"({Counter(s.family for s in long_spans)}); embedded as prefixes -> ROLLBACK to spans-v7")
# fix: refuse overlong input, re-span, rebuild, gate per slice on the production index
e_new.on_overlong = "refuse"
refused = [s.sid for s in spans if not reg.admit_span(e_new.model_id, s.text)]
spans2 = C.respan(spans, 480)
ids2 = [s.sid for s in spans2]
D2 = base.encode([s.text for s in spans2], truncate=True)
assert all(reg.admit_span(e_new.model_id, s.text) for s in spans2)
gate2 = hits(search(ADAPTER(Q["open-base"]), D2, K), ids2, golds)
old_s = by_slice(h_old, sl)
new_s = by_slice(gate2, sl)
ok = all(new_s[s] >= old_s[s] for s in SLICES)
print(f"refused at ingest: {len(refused)}; re-spanned corpus {len(spans2)} spans")
print("per-slice gate on the rebuilt production index (live -> new): "
      + ", ".join(f"{s} {old_s[s]:.3f}->{new_s[s]:.3f}" for s in SLICES)
      + f"; all {h_old.mean():.3f}->{gate2.mean():.3f} -> {'PASS' if ok else 'FAIL'}")
R["migration"] = {"overlap": round(float(ov.mean()), 3),
                  "overlap_slices": {s: round(float(ov[lsl == s].mean()), 3) for s in SLICES},
                  "live": old_s, "lab": by_slice(gate_lab, sl), "prod": by_slice(gate_prod, sl),
                  "fixed": new_s, "all": [round(float(h_old.mean()), 3),
                                          round(float(gate_lab.mean()), 3),
                                          round(float(gate_prod.mean()), 3),
                                          round(float(gate2.mean()), 3)]}

# 6 ---------------------------------------------------------------- silent provider change
section("6. Failure injection: gen-large changes behind the same name")
reg.register(M["gen-large"], "api-vendor")
reg.bind_index("bakeoff-gen-large", "gen-large@2026-03")
upd = C.models(provider_update=True)["gen-large"]
print(f"model name before {M['gen-large'].model_id}, after {upd.model_id}")
S_b = Q["gen-large"] @ D["gen-large"].T
Qu = upd.encode(gq)
S_a = Qu @ D["gen-large"].T
hb = hits(np.argsort(-S_b, axis=1)[:, :K], ids, golds)
ha = hits(np.argsort(-S_a, axis=1)[:, :K], ids, golds)
print(f"gold hit@10 {hb.mean():.3f} -> {ha.mean():.3f}; mean top-1 score "
      f"{S_b.max(1).mean():.3f} -> {S_a.max(1).mean():.3f}; no error raised")
rep = reg.check(upd, "gen-large@2026-03")
print(f"registry probe check: min cosine {rep.min_cosine:.3f}, ok={rep.ok}; "
      f"may_query(bakeoff-gen-large) = {reg.may_query('bakeoff-gen-large', upd)}")
jit = C.models()["gen-large"]
jit.jitter = 2e-4
rj = reg.check(jit, "gen-large@2026-03")
print(f"same weights, different kernels (jitter): min cosine {rj.min_cosine:.5f}, ok={rj.ok}")
hb25 = hits(bm25(texts, gq), ids, golds)
print(f"fail closed to BM25 while re-embedding: hit@10 {hb25.mean():.3f}")
R["silent"] = {"before": round(float(hb.mean()), 3), "after": round(float(ha.mean()), 3),
               "top1_before": round(float(S_b.max(1).mean()), 3),
               "top1_after": round(float(S_a.max(1).mean()), 3),
               "probe_min": round(rep.min_cosine, 3), "jitter_min": round(rj.min_cosine, 5),
               "bm25": round(float(hb25.mean()), 3),
               "top1_b": S_b.max(1).round(3).tolist(), "top1_a": S_a.max(1).round(3).tolist()}

section("7. Registry")
for e in reg.entries.values():
    print(f"  {e.model_id:20} {e.provider:12} dims {e.dims:5} max_in {e.max_input_tokens:5} "
          f"overlong={e.on_overlong:8} adapter={e.adapter} fp={e.fingerprint} indexes={e.indexes}")

os.makedirs("out", exist_ok=True)
json.dump(R, open("out/results.json", "w"), indent=1)
