"""contextlab.hashembed (Chapter 3): an offline feature-hashing embedder.

HashingEmbedder maps lower-cased words to signed buckets of a fixed dimension (384 by
default), hashed with a keyed BLAKE2b (the key is fixed by `seed`). It needs no model download and is fully deterministic. It
sees wording only: two editions of a document that share most of their words embed
close together whatever their effective dates, which is the point Chapter 3 makes.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_WORD = re.compile(r"[a-z0-9$]+(?:[.,][0-9]+)*")


class HashingEmbedder:
    name, version = "hash-embed", "v1"

    def __init__(self, dim: int = 384, seed: int = 114):
        self.dim, self._key = dim, f"hash-embed-v{seed}".encode()

    def _features(self, text: str) -> list[str]:
        return _WORD.findall(text.lower())

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for f in self._features(t):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8, key=self._key).digest(), "big")
                out[i, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return out
