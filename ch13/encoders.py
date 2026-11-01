"""sim-embed v2: the same recipe with new concept directions (a new model)."""
import hashlib

import numpy as np

from contextlab.retrieval import dense as D


def _dir2(name: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(("v2|" + name).encode()).digest()[:8], "big")
    v = np.random.default_rng(seed).standard_normal(D.DIM)
    return v / np.linalg.norm(v)


class SimEncoderV2(D.SimEncoder):
    name = "sim-embed-2"

    def piece(self, tok):
        orig = D._direction
        D._direction = _dir2
        try:
            return super().piece(tok)
        finally:
            D._direction = orig
