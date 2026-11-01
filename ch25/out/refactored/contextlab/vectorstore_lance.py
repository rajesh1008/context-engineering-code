"""LanceDB adapter for the Chapter 13 store interface (needs `lancedb`)."""
from __future__ import annotations

import time
import zlib

import lancedb
import lancedb.index as li
import numpy as np
import pyarrow as pa

from .vectorstore import Hit, Result, Row, Scope


class LanceStore:
    def __init__(self, path: str, table: str = "spans", *, prefilter: bool = True,
                 nprobes: int | None = None, refine: int | None = None,
                 exact_below: int = 0, overfetch: int = 1, ef: int | None = None):
        self.db = lancedb.connect(path)
        self.table_name = table
        self.t = None
        self.prefilter, self.nprobes, self.refine = prefilter, nprobes, refine
        self.exact_below = exact_below          # eligible rows below this: exact
        self.overfetch = overfetch              # post-filter only: fetch k * overfetch
        self.ef = ef
        self._eligible: dict[tuple, int] = {}
        if table in self.db.table_names():
            self.t = self.db.open_table(table)

    @property
    def name(self) -> str:
        mode = "pre" if self.prefilter else "post"
        return f"lancedb ({mode}-filter, nprobes={self.nprobes}, refine={self.refine})"

    @staticmethod
    def _arrow(rows: list[Row]) -> pa.Table:
        return pa.table({
            "span_id": [r.span_id for r in rows], "tenant": [r.tenant for r in rows],
            "doc_type": [r.doc_type for r in rows], "text": [r.text for r in rows],
            "embed_model": [r.embed_model for r in rows],
            "embed_version": [r.embed_version for r in rows],
            "content_hash": [r.content_hash for r in rows],
            # uniform 0-999, for selectivity experiments only
            "bucket": [zlib.crc32(r.span_id.encode()) % 1000 for r in rows],
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(np.vstack([r.vector for r in rows]).astype(np.float32).ravel()),
                len(rows[0].vector))})

    def upsert(self, rows: list[Row]) -> None:
        data = self._arrow(rows)
        if self.t is None:
            self.t = self.db.create_table(self.table_name, data=data, mode="overwrite")
        else:
            (self.t.merge_insert("span_id").when_matched_update_all()
             .when_not_matched_insert_all().execute(data))
        self._eligible.clear()

    def delete(self, ids) -> None:
        quoted = ", ".join(f"'{i}'" for i in ids)
        self.t.delete(f"span_id IN ({quoted})")
        self._eligible.clear()

    def build_index(self, kind: str = "ivf_pq", partitions: int = 256,
                    sub_vectors: int = 32, m: int = 20, ef_construction: int = 150) -> None:
        if kind == "ivf_pq":
            cfg = li.IvfPq(distance_type="cosine", num_partitions=partitions,
                           num_sub_vectors=sub_vectors)
        elif kind == "ivf_hnsw_sq":
            cfg = li.IvfHnswSq(distance_type="cosine", num_partitions=partitions,
                               m=m, ef_construction=ef_construction)
        else:
            raise ValueError(kind)
        self.t.create_index("vector", config=cfg, replace=True)
        self.t.create_index("tenant", config=li.Bitmap(), replace=True)

    def eligible(self, scope: Scope, version: str | None = None) -> int:
        key = (scope.tenants, version)
        if key not in self._eligible:
            self._eligible[key] = self.t.count_rows(self._where(scope, version))
        return self._eligible[key]

    @staticmethod
    def _where(scope: Scope, version: str | None) -> str:
        w = scope.where()
        return w if version is None else f"{w} AND embed_version = '{version}'"

    def search(self, vector, k: int, scope: Scope, version: str | None = None,
               ef: int | None = None) -> Result:
        ef = ef or self.ef
        t0 = time.perf_counter()
        n = self.eligible(scope, version)
        q = (self.t.search(np.asarray(vector, dtype=np.float32), vector_column_name="vector")
             .distance_type("cosine")
             .where(self._where(scope, version), prefilter=self.prefilter)
             .select(["span_id", "tenant", "embed_version"])
             .limit(k if self.prefilter else k * self.overfetch))
        if n < self.exact_below:
            q = q.bypass_vector_index()
        else:
            if self.nprobes:
                q = q.nprobes(self.nprobes)
            if self.refine:
                q = q.refine_factor(self.refine)
            if ef:
                q = q.ef(max(ef, k * self.overfetch))
        rows = q.to_arrow().to_pylist()[:k]
        hits = [Hit(r["span_id"], 1.0 - float(r["_distance"]), r["tenant"]) for r in rows]
        vers = tuple(sorted({r["embed_version"] for r in rows}))
        return Result(hits, k, n, self.name, (time.perf_counter() - t0) * 1000, vers)
