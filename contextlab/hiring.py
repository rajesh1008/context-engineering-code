"""contextlab.hiring: a packet-review work sample and two ways to score it (Chapter 49).

contextlab v11.0. A hiring exercise built from the same objects the context layer
uses: a packet of spans, each with provenance, into which known defects are seeded.
A candidate's review is a list of findings. Two scorers are provided:

* ``score_vocab``: the failure injection. It awards a point per finding claimed,
  with or without evidence, and two points per tool or vendor name mentioned.
* ``score_anchored``: per seeded defect, 1 for finding it, +1 for citing the span
  and the rule it breaks, +1 for a fix that would work; -1 per false finding;
  tool and vendor names score nothing.

Helpers measure what a scorer is worth: Spearman correlation with a held-out
work sample, overlap of the top k, and Cohen's kappa between two raters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

VERSION = "11.0"


@dataclass(frozen=True)
class Span:
    sid: str
    role: str            # instructions | facts | governing | modifier | context | question
    text: str
    doc: str             # provenance: document id
    version: str         # provenance: edition or effective date
    page: str | None     # provenance: page; None is itself a defect


@dataclass(frozen=True)
class Defect:
    did: str
    sid: str
    rule: str            # the rule the span breaks, in the chapter's words
    fix: str


@dataclass
class Review:
    name: str
    findings: dict = field(default_factory=dict)   # did -> (found, evidence, fix)
    false_findings: int = 0
    tool_names: int = 0
    claimed: int = 0                               # findings asserted, right or wrong


def score_vocab(r: Review) -> int:
    return r.claimed + 2 * r.tool_names


def score_anchored(r: Review, defects) -> int:
    s = 0
    for d in defects:
        found, evidence, fix = r.findings.get(d.did, (False, False, False))
        s += found + (found and evidence) + (found and evidence and fix)
    return s - r.false_findings


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> float:
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** .5
    vb = sum((y - mb) ** 2 for y in rb) ** .5
    return round(cov / (va * vb), 3)


def top_k_overlap(score, truth, k=5) -> int:
    top = lambda xs: set(sorted(range(len(xs)), key=lambda i: (-xs[i], i))[:k])
    return len(top(score) & top(truth))


def cohen_kappa(a, b) -> float:
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return round((po - pe) / (1 - pe), 2) if pe < 1 else 1.0
