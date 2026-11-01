"""Hierarchical navigable small world graphs, small enough to read.

Vectors are unit length, so distance is 1 - dot product. Every distance
computed is counted, which is the honest cost unit for a graph index: wall
time in Python mostly measures Python.
"""
import heapq
import math
import random

import numpy as np


class HNSW:
    def __init__(self, dim: int, M: int = 12, ef_construction: int = 80, seed: int = 7):
        self.dim, self.M, self.M0 = dim, M, 2 * M
        self.efc = ef_construction
        self.ml = 1 / math.log(M)
        self.rng = random.Random(seed)
        self.vecs: list[np.ndarray] = []
        self.layers: list[dict[int, list[int]]] = []   # layer -> node -> nbrs
        self.entry: int | None = None
        self.distances = 0

    def _dist(self, q: np.ndarray, nodes: list[int]) -> np.ndarray:
        self.distances += len(nodes)
        return 1.0 - np.stack([self.vecs[n] for n in nodes]) @ q

    def _search_layer(self, q, entries: list[int], ef: int, layer: int):
        d0 = self._dist(q, entries)
        visited = set(entries)
        cand = [(d, n) for d, n in zip(d0, entries)]          # min-heap
        best = [(-d, n) for d, n in zip(d0, entries)]          # max-heap
        heapq.heapify(cand)
        heapq.heapify(best)
        while cand:
            d, n = heapq.heappop(cand)
            if d > -best[0][0] and len(best) >= ef:
                break
            new = [m for m in self.layers[layer].get(n, []) if m not in visited]
            if not new:
                continue
            visited.update(new)
            for dm, m in zip(self._dist(q, new), new):
                if len(best) < ef or dm < -best[0][0]:
                    heapq.heappush(cand, (dm, m))
                    heapq.heappush(best, (-dm, m))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, n) for d, n in best)

    def add(self, v: np.ndarray) -> int:
        node = len(self.vecs)
        self.vecs.append(v)
        level = int(-math.log(1 - self.rng.random()) * self.ml)
        while len(self.layers) <= level:
            self.layers.append({})
        if self.entry is None:
            for l in range(level + 1):
                self.layers[l][node] = []
            self.entry = node
            return node
        top = max(l for l in range(len(self.layers)) if self.entry in self.layers[l])
        ep = [self.entry]
        for l in range(top, level, -1):                # greedy descent
            ep = [self._search_layer(v, ep, 1, l)[0][1]]
        for l in range(min(level, top), -1, -1):
            found = self._search_layer(v, ep, self.efc, l)
            cap = self.M0 if l == 0 else self.M
            nbrs = [n for _, n in found[:self.M]]
            self.layers[l][node] = nbrs
            for n in nbrs:                              # link back, then prune
                lst = self.layers[l].setdefault(n, [])
                lst.append(node)
                if len(lst) > cap:
                    d = self._dist(self.vecs[n], lst)
                    self.layers[l][n] = [lst[i] for i in np.argsort(d)[:cap]]
            ep = [n for _, n in found]
        for l in range(top + 1, level + 1):
            self.layers[l][node] = []
        if level > top:
            self.entry = node
        return node

    def search(self, q: np.ndarray, k: int = 10, ef: int = 50) -> list[int]:
        top = max(l for l in range(len(self.layers)) if self.entry in self.layers[l])
        ep = [self.entry]
        for l in range(top, 0, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]
        return [n for _, n in self._search_layer(q, ep, max(ef, k), 0)[:k]]
