# Copyright 2026 Lakeshore Mutual. Internal use only.
# SPDX-License-Identifier: LicenseRef-Lakeshore-Internal
# Generated header; do not edit by hand.

"""contextlab.graph: a small typed policy graph in SQLite (Chapter 17, v2.9).

Nodes are document families (forms, endorsements, bulletins) and the clauses
they speak to. Edges are relationships a reader, or a model, found in the
text: one endorsement modifies a clause, requires another endorsement, shares
its limit, or replaces it. Every edge keeps the span that asserted it. An edge
counts for a determination only when that span is admitted for it, and its
dates come from the span's place in the supersession chain. Anything the spine
does not allow is rejected; anything uncertain or contradictory is quarantined
for a person, and a quarantined edge that could apply withholds the chain.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date

# The minimal ontology spine: relation -> (source types, target types).
SPINE = {
    "HAS_CLAUSE": ({"form"}, {"clause"}),
    "MODIFIES": ({"endorsement"}, {"clause", "endorsement"}),
    "REQUIRES": ({"endorsement"}, {"endorsement"}),
    "REPLACES": ({"endorsement"}, {"endorsement"}),
    "LIMIT_WITHIN": ({"endorsement"}, {"endorsement"}),
    "LIMIT_IN_ADDITION": ({"endorsement"}, {"endorsement"}),
    "CONSTRAINS": ({"bulletin"}, {"clause"}),
    "SUPERSEDES": ({"form", "endorsement", "bulletin"}, {"form", "endorsement", "bulletin"}),
}
EXCLUSIVE = [{"LIMIT_WITHIN", "LIMIT_IN_ADDITION"}]   # at most one per pair on any date
CHAIN_RELS = ("MODIFIES", "REQUIRES", "REPLACES", "LIMIT_WITHIN",
              "LIMIT_IN_ADDITION", "CONSTRAINS")
END = "9999-12-31"


@dataclass(frozen=True)
class Candidate:
    """One relationship proposed by an extractor, with the words it read."""
    src: str
    rel: str
    dst: str
    source_sid: str
    confidence: float | None
    quote: str
    method: str = "sim-extract"


@dataclass
class Edge:
    c: Candidate
    status: str                      # published, quarantined, rejected
    reason: str
    valid_from: str | None = None
    valid_to: str | None = None


def node_type(node: str, spans_by_doc: dict) -> str | None:
    if node.startswith("clause:"):
        return "clause"
    s = spans_by_doc.get(node)
    return s.kind if s else None


def validate(cands, policy, threshold=0.7, derive_validity=True, conflict_scope="edition"):
    """Spine check, provenance check, confidence gate, validity, conflicts."""
    by_doc = {}
    for s in policy.spans.values():
        by_doc.setdefault(s.doc_id, s)
    out = []
    for c in cands:
        if c.rel not in SPINE:
            out.append(Edge(c, "rejected", "relation not in spine")); continue
        st, dt = node_type(c.src, by_doc), node_type(c.dst, by_doc)
        if st not in SPINE[c.rel][0] or dt not in SPINE[c.rel][1]:
            out.append(Edge(c, "rejected", f"{st} -{c.rel}-> {dt} not allowed")); continue
        src = policy.spans.get(c.source_sid)
        if src is None or src.version is None:
            out.append(Edge(c, "quarantined", "no source version")); continue
        if c.confidence is None or c.confidence < threshold:
            out.append(Edge(c, "quarantined", "low or missing confidence")); continue
        end = policy.end_of(src)[0] if derive_validity else src.effective_to
        out.append(Edge(c, "published", "valid", str(src.effective_from),
                        str(end) if end else None))
    if conflict_scope:
        _quarantine_conflicts(out, conflict_scope)
    return out


def _overlap(a: Edge, b: Edge) -> bool:
    return a.valid_from < (b.valid_to or END) and b.valid_from < (a.valid_to or END)


def _quarantine_conflicts(edges, scope):
    """Contradictory relations: within one edition, or across a family by date."""
    live = [e for e in edges if e.status == "published"]
    same = (lambda a, b: a.c.source_sid == b.c.source_sid) if scope == "edition" else _overlap
    for group in EXCLUSIVE:
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if ({a.c.rel, b.c.rel} == group and (a.c.src, a.c.dst) == (b.c.src, b.c.dst)
                        and same(a, b)):
                    for e in (a, b):
                        e.status, e.reason = "quarantined", f"conflicts with {b.c.source_sid if e is a else a.c.source_sid}"


class Graph:
    """The published edges in SQLite, walked with a recursive query."""

    def __init__(self, edges, path=":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE edge (src TEXT, rel TEXT, dst TEXT, source_sid TEXT,"
                        " valid_from TEXT, valid_to TEXT, confidence REAL, quote TEXT)")
        self.db.executemany("INSERT INTO edge VALUES (?,?,?,?,?,?,?,?)", [
            (e.c.src, e.c.rel, e.c.dst, e.c.source_sid, e.valid_from, e.valid_to or END,
             e.c.confidence, e.c.quote) for e in edges if e.status == "published"])
        self.held = [e for e in edges if e.status == "quarantined"]

    WALK = """
    WITH RECURSIVE live(a, b, rel) AS (
        SELECT src, dst, rel FROM edge WHERE valid_from <= :d AND :d < valid_to
        UNION ALL SELECT dst, src, rel FROM edge WHERE valid_from <= :d AND :d < valid_to),
    walk(node, depth) AS (
        SELECT value, 0 FROM json_each(:seeds)
        UNION SELECT live.b, walk.depth + 1 FROM walk JOIN live ON live.a = walk.node
        WHERE walk.depth < :hops AND live.rel IN (SELECT value FROM json_each(:rels)))
    SELECT node, MIN(depth) FROM walk GROUP BY node"""

    def walk(self, seeds, as_of: date, hops=2, rels=CHAIN_RELS) -> dict:
        rows = self.db.execute(self.WALK, {"seeds": json.dumps(sorted(seeds)), "d": str(as_of),
                                           "hops": hops, "rels": json.dumps(list(rels))})
        return dict(rows.fetchall())

    def edges(self, nodes, as_of: date):
        q = ("SELECT src, rel, dst, source_sid FROM edge WHERE valid_from <= ? AND ? < valid_to")
        ns = set(nodes)
        return [r for r in self.db.execute(q, (str(as_of), str(as_of)))
                if r[0] in ns and r[2] in ns]

    def degree(self) -> dict:
        return dict(self.db.execute("SELECT dst, COUNT(*) FROM edge GROUP BY dst").fetchall())


@dataclass
class Chain:
    evidence: list                   # sids, base clause first
    inapplicable: dict = field(default_factory=dict)   # sid -> reason
    relations: set = field(default_factory=set)        # (src, rel, dst) used
    status: str = "ok"
    reason: str = ""
    via: str = "graph"


def compose(graph: Graph, policy, req, topic: str, seeds=(), hops=2, scoped=True,
            cite_constrained=False) -> Chain:
    """Evidence for one determination: the governing spans plus every span a
    relationship says changes the answer. With no relationship on the policy,
    Chapter 15's single-span resolution stands."""
    on, clause = req.date_of_loss, f"clause:{topic}"
    attached = {e.split("@")[0] for e in req.endorsements} | set(req.editions)
    reached = graph.walk({clause, *attached, *seeds}, on, hops)
    adm = {}
    for s in policy.spans.values():
        if (s.doc_id in reached and s.kind in ("form", "endorsement", "bulletin")
                and (s.topic == topic or s.kind == "endorsement") and policy.admit(s, req).ok):
            adm.setdefault(s.doc_id, s)
    touching = {d for d, s in adm.items() if s.topic == topic}
    links = [r for r in graph.edges(set(adm) | set(reached), on)
             if r[1] in CHAIN_RELS and r[1] != "CONSTRAINS" and r[0] in adm
             and (not scoped or policy.admit(policy.spans[r[3]], req).ok)]   # the edition that said it
    def live(r, keep):                           # modifiers count both ways, others outward
        return r[0] in keep or (r[1] == "MODIFIES" and r[2] in keep)
    if not any(live(r, touching) for r in links):
        res = policy.resolve(sorted(adm[d].sid for d in touching), req)
        if res.status != "ok":
            return Chain([], status="withhold", reason=res.reason, via="rules")
        gov = policy.spans[res.governing]
        if cite_constrained and gov.kind == "bulletin" and any(
                r[0] == gov.doc_id and r[1] == "CONSTRAINS" and r[2] == clause
                for r in graph.edges({gov.doc_id, clause}, on)):   # the clause the bulletin bends
            form = [adm[d].sid for d in touching if adm[d].kind == "form"]
            return Chain(form + [gov.sid], via="graph")
        return Chain([res.governing], via="rules")
    chain = Chain([])
    keep = {d for d in touching if adm[d].kind != "form"}
    grew = True
    while grew:                                  # follow relationships to a fixed point
        grew = False
        for src, rel, dst, sid in links:
            if not live((src, rel, dst), keep) or (dst.startswith("clause:") and dst != clause):
                continue                                 # not linked, or about another clause
            chain.relations.add((src, rel, dst))
            if rel == "MODIFIES" and dst == clause:
                new = {d for d in touching if adm[d].kind == "form"}
            elif rel in ("REQUIRES", "MODIFIES") and dst not in adm:
                chain.inapplicable[adm[src].sid] = f"{rel.lower()} {dst}, not on this policy"
                new = set()
            else:
                new = {x for x in (src, dst) if x in adm}
            if new - keep:
                keep |= new; grew = True
    held = [e for e in graph.held if (e.c.src in keep or e.c.dst in keep)
            and policy.admit(policy.spans[e.c.source_sid], req).ok]   # only if it could apply here
    if held:                                     # an open question on the graph: fail closed
        return Chain([], status="withhold", reason=f"quarantined: {held[0].reason}")
    order = {"form": 0, "endorsement": 1, "bulletin": 2}
    chain.evidence = [adm[d].sid for d in sorted(keep, key=lambda d: (order[adm[d].kind], d))]
    return chain
