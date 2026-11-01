"""contextlab.shadow: shadow runs, exit rules and evidence packs (Chapter 43).

contextlab v9.4. A candidate context layer runs on mirrored claims and its
drafts go to a shadow ledger that no adjuster sees. Agreement is counted per
cell (line x state), never only overall. The exit rule is a frozen object with
a digest, written before the first mirrored claim. The evidence pack for one
determination is rebuilt from records (Chapter 42's event, the document store,
the review record) and checked against a Merkle root sealed at decision time.
Redaction keeps each segment's salted commitment, so the root still verifies.
`explain` renders a pack as text without any model call.

Stdlib only; every hash is SHA-256 over canonical JSON.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict

VERSION = "9.4"


def sha(obj) -> str:
    raw = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - w), min(1.0, c + w)


# ---- shadow comparison ----------------------------------------------------------

@dataclass
class Pair:
    """One mirrored determination: production draft, candidate draft, outcome."""
    det_id: str
    cell: tuple                 # (line, state)
    prod: dict                  # {"verdict", "clause"}
    cand: dict | None           # None: candidate produced nothing in time
    final: dict | None          # adjuster's final determination; None: still open
    truth: dict | None = None   # set by adjudication for disagreements
    parsed: bool = True         # candidate citations parsed by Chapter 28's parser
    prod_ms: float = 0.0
    cand_ms: float = 0.0
    prod_cost: float = 0.0
    cand_cost: float = 0.0
    pack_ok: bool = True

    def comparable(self) -> bool:
        return self.cand is not None and self.parsed and self.final is not None


def same(a: dict | None, b: dict | None) -> bool:
    return a is not None and b is not None and \
        (a["verdict"], a["clause"]) == (b["verdict"], b["clause"])


def cell_stats(pairs: list) -> dict:
    """Per cell: n, agreement of each arm with the final, and adjudicated discordance."""
    out = defaultdict(lambda: {"n": 0, "cand": 0, "prod": 0, "b": 0, "c": 0,
                               "cand_wrong": 0, "prod_wrong": 0})
    for p in pairs:
        if not p.comparable():
            continue
        s = out[p.cell]
        s["n"] += 1
        s["cand"] += same(p.cand, p.final)
        s["prod"] += same(p.prod, p.final)
        ref = p.truth or p.final          # adjudicated truth where it exists
        cw, pw = not same(p.cand, ref), not same(p.prod, ref)
        s["cand_wrong"] += cw
        s["prod_wrong"] += pw
        s["b"] += cw and not pw           # candidate wrong where production right
        s["c"] += pw and not cw           # the reverse
    return dict(out)


def funnel(arrived: int, eligible: int, pairs: list) -> list:
    ran = [p for p in pairs if p.cand is not None]
    parsed = [p for p in ran if p.parsed]
    comp = [p for p in parsed if p.final is not None]
    agree = [p for p in comp if same(p.cand, p.final)]
    dis = [p for p in comp if not same(p.cand, p.final)]
    adj = defaultdict(int)
    for p in dis:
        ref = p.truth or p.final
        key = ("candidate right" if same(p.cand, ref) else
               "production right" if same(p.prod, ref) else "both wrong")
        adj[key] += 1
    return [("arrived", arrived), ("eligible", eligible), ("mirrored", len(pairs)),
            ("candidate answered", len(ran)), ("citations parsed", len(parsed)),
            ("final recorded", len(comp)), ("agree", len(agree)),
            ("disagree", len(dis))] + sorted(adj.items())


@dataclass(frozen=True)
class ExitRule:
    """Written first, frozen, and identified by its digest."""
    name: str
    min_n: int = 60              # below this a cell stays in shadow; never pooled
    min_agree: float = 0.90      # floor on agreement with the final, per cell
    max_excess: int = 2          # adjudicated b - c, per cell
    parse_min: float = 0.995
    latency_ratio_max: float = 1.25
    cost_ratio_max: float = 1.10
    packs_required: bool = True

    def digest(self) -> str:
        return sha(asdict(self))[:16]

    def judge_cell(self, s: dict) -> str:
        if s["n"] < self.min_n:
            return "shadow"
        if s["cand"] / s["n"] < self.min_agree:
            return "hold: agreement"
        if s["b"] - s["c"] > self.max_excess:
            return "hold: excess wrong"
        return "release"

    def run_gates(self, pairs: list) -> dict:
        ran = [p for p in pairs if p.cand is not None]
        parse = sum(p.parsed for p in ran) / max(len(ran), 1)
        lat = p95([p.cand_ms for p in ran]) / max(p95([p.prod_ms for p in ran]), 1e-9)
        cost = sum(p.cand_cost for p in ran) / max(sum(p.prod_cost for p in ran), 1e-9)
        packs = all(p.pack_ok for p in ran)
        return {"parse": parse, "parse_ok": parse >= self.parse_min,
                "latency_ratio": lat, "latency_ok": lat <= self.latency_ratio_max,
                "cost_ratio": cost, "cost_ok": cost <= self.cost_ratio_max,
                "packs_ok": packs or not self.packs_required}

    def evaluate(self, pairs: list) -> dict:
        stats = cell_stats(pairs)
        gates = self.run_gates(pairs)
        whole = all(v for k, v in gates.items() if k.endswith("_ok"))
        cells = {c: (self.judge_cell(s) if whole else "hold: run gate") for c, s in stats.items()}
        return {"rule": self.name, "digest": self.digest(), "gates": gates, "cells": cells,
                "stats": stats}


def pooled_rule(pairs: list, overall: float = 0.95, floor: float = 0.90, min_n: int = 60) -> dict:
    """The rule proposed the night before: overall agreement plus a per-cell floor."""
    stats = cell_stats(pairs)
    n = sum(s["n"] for s in stats.values())
    agree = sum(s["cand"] for s in stats.values()) / max(n, 1)
    cells = {}
    for c, s in stats.items():
        ok = agree >= overall and (s["n"] < min_n or s["cand"] / s["n"] >= floor)
        cells[c] = "release" if ok else "hold"
    return {"overall": agree, "cells": cells}


def p95(xs: list) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(math.ceil(0.95 * len(ys))) - 1)]


# ---- documents -------------------------------------------------------------------

class EditInPlace(Exception):
    pass


class DocStore:
    """Policy documents by (doc_id, edition). 'mutable' lets a file be replaced
    under the same edition; 'write_once' refuses and requires a new edition."""

    def __init__(self, mode: str = "mutable"):
        self.mode = mode
        self.current: dict[tuple, str] = {}
        self.blobs: dict[str, str] = {}          # every version ever written, by hash
        self.log: list = []                      # (at, doc_id, edition, hash)
        self.snapshots: dict[str, set] = {}      # day -> hashes present that night

    def put(self, doc_id: str, edition: str, text: str, at: str) -> str:
        key, hx = (doc_id, edition), sha(text)
        if key in self.current and sha(self.current[key]) != hx and self.mode == "write_once":
            raise EditInPlace(f"{doc_id}@{edition} exists; publish a new edition")
        self.current[key] = text
        if self.mode == "write_once":
            self.blobs[hx] = text
        self.log.append((at, doc_id, edition, hx))
        return hx

    def snapshot(self, day: str) -> None:
        for t in self.current.values():
            self.blobs.setdefault(sha(t), t)     # nightly backup keeps old bytes
        self.snapshots[day] = {sha(t) for t in self.current.values()}

    def by_hash(self, hx: str, keep_days: list | None = None) -> str | None:
        if keep_days is not None and not any(hx in self.snapshots.get(d, ()) for d in keep_days):
            return None
        return self.blobs.get(hx)


# ---- evidence pack ---------------------------------------------------------------

def salt(key: bytes, *parts) -> str:
    return hmac.new(key, "|".join(map(str, parts)).encode(), hashlib.sha256).hexdigest()[:32]


def commit(s: str, text: str) -> str:
    return sha(s + text)


@dataclass
class Item:
    """One thing the pack asserts, split into segments that can be redacted alone."""
    kind: str                   # request | version | span | document | citation | tool | output | review
    name: str
    segments: list              # [{"text", "cls", "quoted"}]
    commits: list = field(default_factory=list)
    redacted: list = field(default_factory=list)   # indices shown as [REDACTED:cls]

    def leaf(self) -> str:
        return sha([self.kind, self.name, self.commits])

    def text(self) -> str:
        return "".join(f"[REDACTED:{s['cls']}]" if i in self.redacted else s["text"]
                       for i, s in enumerate(self.segments))


def merkle(leaves: list) -> str:
    level = list(leaves) or [sha("")]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


@dataclass
class Pack:
    det_id: str
    items: list
    sealed_root: str = ""
    notes: list = field(default_factory=list)

    def seal(self, key: bytes) -> str:
        for it in self.items:
            it.commits = [commit(salt(key, self.det_id, it.kind, it.name, i), s["text"])
                          for i, s in enumerate(it.segments)]
        self.sealed_root = merkle([it.leaf() for it in self.items])
        return self.sealed_root

    def verify(self, key: bytes) -> bool:
        """Recompute every unredacted segment's commitment; keep redacted ones."""
        leaves = []
        for it in self.items:
            cs = [it.commits[i] if i in it.redacted else
                  commit(salt(key, self.det_id, it.kind, it.name, i), s["text"])
                  for i, s in enumerate(it.segments)]
            leaves.append(sha([it.kind, it.name, cs]))
        return merkle(leaves) == self.sealed_root


def redact(pack: Pack, classes: set, naive: bool = False, key: bytes = b"") -> dict:
    """Redact segments by class. Quoted (cited) segments are refused, not hidden.
    naive=True overwrites the text and recomputes the commitment, as an edit would."""
    done, refused = 0, []
    for it in pack.items:
        for i, s in enumerate(it.segments):
            if s["cls"] not in classes:
                continue
            if s["quoted"]:
                refused.append((it.name, s["cls"]))
                continue
            if naive:
                s["text"] = f"[REDACTED:{s['cls']}]"
                it.commits[i] = commit(salt(key, pack.det_id, it.kind, it.name, i), s["text"])
            else:
                it.redacted.append(i)
            done += 1
    return {"redacted": done, "refused": refused}


def explain(pack: Pack, key: bytes) -> str:
    """Deterministic rendering for an examiner. No model call anywhere."""
    out = [f"Determination {pack.det_id}", f"sealed root {pack.sealed_root[:16]}",
           f"verifies: {'yes' if pack.verify(key) else 'NO'}"]
    order = ["request", "output", "version", "span", "document", "citation", "tool", "review"]
    for kind in order:
        rows = [it for it in pack.items if it.kind == kind]
        if rows:
            out.append(f"-- {kind} ({len(rows)})")
            out += [f"   {it.name}: {it.text()[:96]}" for it in rows]
    out += [f"!! {n}" for n in pack.notes] or ["no gaps recorded"]
    return "\n".join(out)
