"""Chapter 2 retrieval modes: BM25 (the baseline), TF-IDF, dense and hybrid, each run through the
baseline pipeline on the 200-determination evaluation set.

All four modes rank the same candidate pool per question (the spans the BM25 fixture scores).
TF-IDF is the BM25 score with BM25's length normalisation taken out (k1 = 1.2, b = 0.75 to b = 0,
span lengths from contextlab's inverted index), so long spans are no longer discounted. The dense
mode is the cosine of the hashing stand-in embedder (contextlab.hashembed), not a trained model.
Hybrid is the weighted fusion of the BM25 and dense lists (alpha 0.5, min-max normalised). Each
mode's scores are rescaled to the BM25 range for that question, so the claim-facts bonus (edition,
form number, state) weighs the same in every mode.
"""
from __future__ import annotations

import numpy as np

import lab
from contextlab import scoreboard as sb
from contextlab.hashembed import HashingEmbedder
from contextlab.retrieval.fusion import weighted
from contextlab.retrieval.index import InvertedIndex

corpus = lab.corpus
NOTES = {"bm25": "Keyword, length-normalised", "tfidf": "Keyword, simpler weighting",
         "dense": "**Hashing stand-in embedder**, not a trained model", "hybrid": "Fusion of the two lists"}
NAMES = {"bm25": "BM25 (baseline)", "tfidf": "TF-IDF", "dense": "Dense", "hybrid": "Hybrid (BM25 plus dense)"}


def rescale(raw: dict[str, float], like: dict[str, float]) -> dict[str, float]:
    """Map raw scores onto [0, max of the BM25 scores for this question]."""
    hi, top = max(like.values()), max(raw.values(), default=0.0) or 1.0
    return {sid: round(hi * max(x, 0.0) / top, 2) for sid, x in raw.items()}


K1, B = 1.2, 0.75


def unnormalise(pool: dict[str, float], length: dict[str, int], avgdl: float) -> dict[str, float]:
    """BM25 with b = 0: undo the length normalisation of each span's score (TF-IDF's weighting)."""
    return {sid: x * (1 + K1 * (1 - B + B * length[sid] / avgdl)) / (K1 + 1) for sid, x in pool.items()}


def build_modes() -> None:
    retriever = lab.base.retriever
    bm25 = retriever.tables["bm25"]
    spans = sorted({sid for t in bm25.values() for sid in t})
    index = InvertedIndex()
    for sid in spans:
        index.add(sid, corpus.span(sid).text)
    emb = HashingEmbedder()
    vecs = emb.embed([corpus.span(sid).text for sid in spans])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    row = {sid: i for i, sid in enumerate(spans)}
    tfidf, dense, hybrid = {}, {}, {}
    for q, pool in bm25.items():
        tfidf[q] = rescale(unnormalise(pool, index.length, index.avgdl), pool)
        qv = emb.embed([q])[0]
        qv /= np.linalg.norm(qv) + 1e-12
        dense[q] = rescale({sid: float(vecs[row[sid]] @ qv) for sid in pool}, pool)
        hybrid[q] = rescale(weighted(pool, dense[q]), pool)
    for mode, table in (("tfidf", tfidf), ("dense", dense), ("hybrid", hybrid)):
        retriever.add_mode(mode, table)


def main() -> None:
    build_modes()
    dets = corpus.eval_set(200)
    print("| Retrieval mode | Wrong of 200 | hit@1 | hit@5 | Note |")
    print("| --- | --- | --- | --- | --- |")
    for mode in ("bm25", "tfidf", "dense", "hybrid"):
        s = sb.score(lab.run_all(lab.base.with_config(mode=mode), dets))
        print(f"| {NAMES[mode]} | {s.wrong} | {s.hit1:.2f} | {s.hit5:.2f} | {NOTES[mode]} |")


if __name__ == "__main__":
    main()
