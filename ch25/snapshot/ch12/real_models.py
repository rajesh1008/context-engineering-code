"""The same ladder with two small real models in place of the stand-ins.

Dense: BAAI/bge-small-en-v1.5 (384 dimensions). Rerank: ms-marco-MiniLM-L-6-v2
cross-encoder. Both run on CPU through fastembed (ONNX). Needs the venv:
  python3 -m venv --system-site-packages 04-code/.venv-ch12
  04-code/.venv-ch12/bin/pip install fastembed
Run: 04-code/.venv-ch12/bin/python real_models.py > out/real_models.txt
"""
import os
import sys
import time

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from contextlab.retrieval.corpus import Query  # noqa: E402
from contextlab.retrieval.pipeline import RUNGS, Ladder  # noqa: E402
from contextlab.retrieval.queries import TWINS, build_queries  # noqa: E402
from lab import STYLES, dump, evaluate  # noqa: E402


class RealDense:
    """Same interface as DenseIndex: search(text) -> {span_id: cosine}."""
    def __init__(self, texts, name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(name)
        self.ids = list(texts)
        self.matrix = np.array(list(self.model.embed([texts[i] for i in self.ids])))
        self.cache = {}

    def search(self, text, allowed=None):
        if text not in self.cache:
            q = next(iter(self.model.query_embed(text)))
            self.cache[text] = dict(zip(self.ids, (self.matrix @ q).tolist()))
        return self.cache[text]


class RealCross:
    def __init__(self, name="Xenova/ms-marco-MiniLM-L-6-v2"):
        self.model, self.pairs_scored, self.seconds = TextCrossEncoder(name), 0, 0.0

    def rerank(self, query, candidates, texts):
        t = time.perf_counter()
        s = list(self.model.rerank(query, [texts[d] for d in candidates])) if candidates else []
        self.seconds += time.perf_counter() - t
        self.pairs_scored += len(candidates)
        scores = dict(zip(candidates, s))
        return sorted(candidates, key=lambda d: (-scores[d], d)), scores


def main():
    L = Ladder()
    Q = build_queries(L.corpus)
    L.dense, L.cross = RealDense(L.texts), RealCross()
    print("== Real-model ladder on the full mix of 1,000 (bge-small-en-v1.5, MiniLM-L6 cross-encoder)")
    res = {r: evaluate(lambda q, r=r: L.run(r, q), Q) for r in RUNGS}
    print(f"{'rung':10}{'Hit@1':>7}{'Hit@5':>7}{'R@5':>7}{'MRR':>7}{'eval H@5':>9}")
    rows = []
    for r in RUNGS:
        m = res[r]
        print(f"{r:10}{m['hit1']:>7.3f}{m['hit5']:>7.3f}{m['rec5']:>7.3f}{m['mrr']:>7.3f}{m['split_eval']:>9.3f}")
        rows.append((r, m["hit1"], m["hit5"], m["rec5"], m["mrr"], m["split_eval"]))
    dump("real_ladder.csv", ["rung", "hit1", "hit5", "rec5", "mrr", "eval_hit5"], rows)
    dump("real_curves.csv", ["rung", "k", "recall"],
         [(r, k, res[r][f"rec@{k}"]) for r in RUNGS for k in range(1, 21)])
    print("\nby style: Hit@5 / governing at rank 1")
    for st in STYLES:
        print(f"{st:11}" + "  ".join(f"{r} {res[r]['st_' + st]:.3f}/{res[r]['st1_' + st]:.3f}"
                                     for r in ("keyword", "dense", "hybrid", "reranked")))
    ms = 1000 * L.cross.seconds / max(L.cross.pairs_scored, 1)
    print(f"\ncross-encoder: {L.cross.pairs_scored} pairs, {ms:.2f} ms a pair on this box (measured)")
    print("\n== Negated near-twins with real models")
    for tag in ("neg", "pos"):
        text, gold = TWINS[tag]
        q = Query("TW", "x", "x", "negation", text, "HO-3", "WI", "2024-06", ("LM-HO 2311",),
                  "mould", {gold: 2}, gold)
        for r in ("keyword", "dense", "hybrid", "reranked"):
            top = L.run(r, q)[:2]
            print(f"  {tag} {r:9} rank 1 {top[0]:22} {'right' if top[0] == gold else 'WRONG'}")
    e = L.dense.model
    a, b = (next(iter(e.query_embed(TWINS[t][0]))) for t in ("neg", "pos"))
    print(f"  cosine(neg, pos) with bge-small: {float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)):.3f}")


if __name__ == "__main__":
    main()
