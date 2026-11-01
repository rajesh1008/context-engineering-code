"""Listing 3.4: two editions of HO 04 95, embedded for retrieval with the offline HashingEmbedder."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from common import tee
import standin as S
from contextlab.hashembed import HashingEmbedder

IDS = ["HO 04 95 (2019-03)|Schedule", "HO 04 95 (2019-03)|Coverage",
       "HO 04 95 (2024-06)|Schedule", "HO 04 95 (2024-06)|Coverage"]


def cosines():
    spans = [S.CORPUS.by_id[i] for i in IDS]
    q = "Does the policy cover water that backs up through a floor drain, and up to what limit?"
    emb = HashingEmbedder()
    vecs = emb.embed([q] + [s.text for s in spans])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return emb, spans, vecs


def main():
    emb, spans, vecs = cosines()
    print(f"embedder {emb.name}-{emb.version}, dim {emb.dim}")
    print("cosine with the question:")
    for s, v in zip(spans, vecs[1:]):
        print(f"  {s.label:<40}{v @ vecs[0]:.3f}")
    print("cosine between editions:")
    for (i, a), (j, b) in combinations(enumerate(spans, 1), 2):
        print(f"  {a.label} vs {b.label}: {vecs[i] @ vecs[j]:.3f}")
    print("characters: " + ", ".join(f"{s.label} {len(s.text)}" for s in spans))


if __name__ == "__main__":
    tee("3.4_embeddings", main)
