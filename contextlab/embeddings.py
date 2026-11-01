"""contextlab v3.4: embedding models as versioned, registered dependencies.

Chapter 14. Four things live here:
  * SimEmbedder: a deterministic stand-in for an embedding model. It learns
    nothing; its vocabulary coverage, dimension and input limit are the
    whole model, stated so you can argue with them.
  * QueryAdapter: a query-side linear adapter fitted by ridge regression
    toward the identity (the cheap end of "fine-tuning").
  * quantise / storage_bytes: dimension truncation and int8 or binary codes.
  * EmbeddingRegistry: one entry per (model, version) with its fingerprint
    on a fixed probe set, its input limit and truncation policy, and the
    indexes built with it. Queries are refused against an index whose
    embedder fingerprint does not match.
Note: all models here are stand-ins; no provider is called.
"""
from __future__ import annotations

import hashlib
import math
from datetime import date

import numpy as np
from pydantic import BaseModel, Field

W_KNOWN, W_UNKNOWN = 1.0, 0.35
STOP = set("a an and are as at be by for from in into is it its of on or the "
           "to was will with we you your our does do did has have can if when "
           "what which who any all this that these than under per".split())


def words(text: str) -> list[str]:
    out = []
    for raw in text.lower().replace("/", " ").split():
        w = raw.strip(".,;:()[]?\"'")
        if w and w not in STOP:
            out.append(w)
    return out


def n_tokens(text: str) -> int:
    """Token estimate used throughout the lab: 1.3 tokens per word."""
    return math.ceil(len(text.split()) * 1.3)


def _unit(seed: str, dim: int) -> np.ndarray:
    h = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    v = np.random.default_rng(h).standard_normal(dim)
    return v / np.linalg.norm(v)


def _covered(model: str, token: str, share: float) -> bool:
    h = int.from_bytes(hashlib.sha256(f"{model}|{token}".encode()).digest()[:4], "big")
    return (h / 2**32) < share


class SimEmbedder:
    """A stand-in bi-encoder.

    lexicon: token -> concept for every token the *world* knows. A model maps
    a plain token to its concept with probability `plain`, and a jargon token
    with probability `jargon`; anything else gets a private hashed direction
    at weight 0.35. `matryoshka` front-loads variance so truncation degrades
    gracefully. `drift` (radians) turns every direction partway towards a new
    `space`: a provider retraining the weights behind an unchanged name.
    """

    def __init__(self, name, version, dim, plain, jargon, lexicon, jargon_tokens,
                 max_input_tokens=8192, matryoshka=False, space=None, jitter=0.0,
                 drift=0.0):
        self.name, self.version, self.dim = name, version, dim
        self.plain, self.jargon = plain, jargon
        self.lexicon, self.jargon_tokens = lexicon, jargon_tokens
        self.max_input_tokens, self.matryoshka = max_input_tokens, matryoshka
        self.space = space or f"{name}@{version}"
        self.jitter, self.drift = jitter, drift
        self._profile = (1.0 / np.sqrt(1.0 + np.arange(dim) / 48.0)
                         if matryoshka else np.ones(dim))
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model_id(self) -> str:
        return f"{self.name}@{self.version}"

    def _direction(self, token: str) -> tuple[np.ndarray, float]:
        concept = self.lexicon.get(token)
        share = self.jargon if token in self.jargon_tokens else self.plain
        if concept and (token == concept or _covered(self.name, token, share)):
            key, w = f"c:{concept}", W_KNOWN
        else:
            key, w = f"t:{token}", W_UNKNOWN
        if key not in self._cache:
            v = _unit(f"{self.name}@{self.version}|{key}", self.dim)
            if self.drift:  # retrained weights: some directions move a lot, some little
                u = int.from_bytes(hashlib.sha256(f"{self.space}|a|{key}".encode())
                                   .digest()[:4], "big") / 2**32
                a = self.drift * 2 * u
                v = math.cos(a) * v + math.sin(a) * _unit(f"{self.space}|{key}", self.dim)
            self._cache[key] = v * self._profile
        return self._cache[key], w

    def encode_one(self, text: str, truncate: bool = True) -> np.ndarray:
        if truncate and n_tokens(text) > self.max_input_tokens:
            text = " ".join(text.split()[: int(self.max_input_tokens / 1.3)])
        toks = words(text)
        v = np.zeros(self.dim)
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():  # sublinear: repetition saturates
            d, w = self._direction(t)
            v += w * (1.0 + math.log(c)) * d
        if self.jitter:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
            v += np.random.default_rng(seed).standard_normal(self.dim) * self.jitter
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, truncate: bool = True) -> np.ndarray:
        return np.vstack([self.encode_one(t, truncate) for t in texts])


class QueryAdapter:
    """q' = q W, with W fitted by ridge regression toward the identity."""

    def __init__(self, lam: float = 1.0, name: str = "adapter"):
        self.lam, self.name, self.W = lam, name, None

    def fit(self, Q: np.ndarray, D: np.ndarray) -> "QueryAdapter":
        d = Q.shape[1]
        A = Q.T @ Q + self.lam * np.eye(d)
        self.W = np.linalg.solve(A, Q.T @ D + self.lam * np.eye(d))
        return self

    def __call__(self, Q: np.ndarray) -> np.ndarray:
        out = Q @ self.W
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def quantise(X: np.ndarray, dims: int | None = None, dtype: str = "float32"):
    """Truncate to `dims`, renormalise, then store as float32, int8 or binary."""
    X = X[:, :dims] if dims else X
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    if dtype == "float32":
        return X.astype(np.float32)
    if dtype == "int8":
        scale = np.abs(X).max() / 127.0
        return np.round(X / scale).astype(np.int8)
    if dtype == "binary":
        return (X > 0).astype(np.int8) * 2 - 1
    raise ValueError(dtype)


def storage_bytes(n_vectors: int, dims: int, dtype: str) -> int:
    per = {"float32": 4 * dims, "int8": dims, "binary": math.ceil(dims / 8)}[dtype]
    return n_vectors * per


def search(Qv: np.ndarray, Dv: np.ndarray, k: int = 10) -> np.ndarray:
    """Top-k indices by dot product (cosine for unit vectors)."""
    S = Qv.astype(np.float64) @ Dv.astype(np.float64).T
    return np.argsort(-S, axis=1, kind="stable")[:, :k]


# ---------------------------------------------------------------- registry

class RegistryEntry(BaseModel):
    model_id: str                      # name@version, never the name alone
    provider: str                      # "self-hosted" or the API vendor
    dims: int
    dtype: str = "float32"
    max_input_tokens: int
    on_overlong: str = "refuse"        # "refuse" or "truncate"; refuse is the default
    adapter: str | None = None
    fingerprint: str                   # hash of the probe vectors, 4 decimals
    probe_vectors: list[list[float]] = Field(repr=False)
    status: str = "candidate"          # candidate, shadow, live, retired
    indexes: list[str] = []
    approved_on: date | None = None
    retire_by: date | None = None


def fingerprint(vectors: np.ndarray) -> str:
    return hashlib.sha256(np.round(vectors, 4).tobytes()).hexdigest()[:12]


class DriftReport(BaseModel):
    model_id: str
    min_cosine: float
    mean_cosine: float
    ok: bool


class EmbeddingRegistry:
    """Pins every index to the exact embedder that built it."""

    def __init__(self, probes: list[str], tolerance: float = 0.999):
        self.probes, self.tolerance = probes, tolerance
        self.entries: dict[str, RegistryEntry] = {}
        self.index_owner: dict[str, str] = {}

    def register(self, model, provider: str, adapter=None, on_overlong="refuse",
                 dtype="float32") -> RegistryEntry:
        P = model.encode(self.probes)
        e = RegistryEntry(model_id=model.model_id, provider=provider, dims=model.dim,
                          dtype=dtype, max_input_tokens=model.max_input_tokens,
                          on_overlong=on_overlong,
                          adapter=getattr(adapter, "name", None),
                          fingerprint=fingerprint(P), probe_vectors=P.tolist())
        self.entries[e.model_id] = e
        return e

    def bind_index(self, index_id: str, model_id: str) -> None:
        self.entries[model_id].indexes.append(index_id)
        self.index_owner[index_id] = model_id

    def check(self, model, model_id: str | None = None) -> DriftReport:
        """Re-embed the probes with whatever answers to this name today."""
        e = self.entries[model_id or model.model_id]
        stored = np.array(e.probe_vectors)
        now = model.encode(self.probes)
        cos = np.sum(stored * now, axis=1)
        return DriftReport(model_id=e.model_id, min_cosine=float(cos.min()),
                           mean_cosine=float(cos.mean()),
                           ok=bool(cos.min() >= self.tolerance))

    def admit_span(self, model_id: str, text: str) -> bool:
        """Fail closed on overlong input instead of embedding a prefix."""
        e = self.entries[model_id]
        return e.on_overlong == "truncate" or n_tokens(text) <= e.max_input_tokens

    def may_query(self, index_id: str, model) -> bool:
        owner = self.index_owner.get(index_id)
        return owner is not None and self.check(model, owner).ok
