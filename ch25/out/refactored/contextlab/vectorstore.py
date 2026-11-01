"""Chapter 13: one store interface, two engines.

The pipeline asks a store for the k nearest spans *within a scope* (the
tenants a claim may see). FlatStore is the from-scratch exact engine from
Chapter 12, rewritten over a matrix. LanceStore is an adapter for an
embedded vector database (LanceDB). Both return a Result that says how many
rows were eligible and whether the store returned as many as it should, so
a caller can tell "nothing matched" from "the engine dropped them".
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

import numpy as np

_SAFE = re.compile(r"^[a-z0-9_-]{1,40}$")


class TenantLeak(RuntimeError):
    """A hit came back from a tenant outside the scope."""


class IncompleteResult(RuntimeError):
    """Fewer hits than eligible rows allow: the engine dropped some."""


@dataclass(frozen=True)
class Scope:
    tenants: tuple[str, ...]

    def __post_init__(self):
        if not self.tenants:
            raise ValueError("a scope must name at least one tenant")
        for t in self.tenants:
            if not _SAFE.match(t):
                raise ValueError(f"tenant {t!r} is not a valid tenant id")

    def where(self) -> str:
        return "tenant IN (" + ", ".join(f"'{t}'" for t in self.tenants) + ")"


@dataclass
class Row:
    span_id: str
    tenant: str
    doc_type: str
    text: str
    vector: np.ndarray
    embed_model: str = "sim-embed"
    embed_version: str = "1"
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = digest(self.text)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class Hit:
    span_id: str
    score: float
    tenant: str


@dataclass
class Result:
    hits: list[Hit]
    k: int
    eligible: int
    engine: str
    ms: float = 0.0
    versions: tuple = field(default_factory=tuple)

    @property
    def complete(self) -> bool:
        return len(self.hits) >= min(self.k, self.eligible)

    @property
    def ids(self) -> list[str]:
        return [h.span_id for h in self.hits]


def check(result: Result, scope: Scope, strict: bool = False) -> Result:
    """The two contract checks every engine must pass."""
    bad = [h for h in result.hits if h.tenant not in scope.tenants]
    if bad:
        raise TenantLeak(f"{len(bad)} hits outside {scope.tenants}: {bad[0].span_id}")
    if strict and not result.complete:
        raise IncompleteResult(
            f"{result.engine}: {len(result.hits)} hits, {result.eligible} eligible, k={result.k}")
    return result


class FlatStore:
    """Exact search: score every eligible row. The filter runs first."""

    name = "flat (from scratch)"

    def __init__(self):
        self.rows: dict[str, Row] = {}
        self._m = None

    def upsert(self, rows: list[Row]) -> None:
        for r in rows:
            self.rows[r.span_id] = r
        self._m = None

    def delete(self, ids) -> None:
        for i in ids:
            self.rows.pop(i, None)
        self._m = None

    def _matrix(self):
        if self._m is None:
            rs = list(self.rows.values())
            self._ids = [r.span_id for r in rs]
            self._ten = np.array([r.tenant for r in rs])
            self._ver = np.array([r.embed_version for r in rs])
            self._m = np.vstack([r.vector for r in rs]).astype(np.float32)
            self._masks = {}
        return self._m

    def search(self, vector, k: int, scope: Scope, version: str | None = None) -> Result:
        t0 = time.perf_counter()
        m = self._matrix()
        key = (scope.tenants, version)
        if key not in self._masks:
            mask = np.isin(self._ten, scope.tenants)
            if version is not None:
                mask &= self._ver == version
            self._masks[key] = np.flatnonzero(mask)
        idx = self._masks[key]
        sims = m[idx] @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-sims, kind="stable")[:k]
        hits = [Hit(self._ids[idx[j]], float(sims[j]), str(self._ten[idx[j]])) for j in order]
        vers = tuple(sorted(set(self._ver[idx]))) if len(idx) else ()
        return Result(hits, k, int(len(idx)), self.name,
                      (time.perf_counter() - t0) * 1000, vers)
