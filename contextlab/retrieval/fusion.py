"""Combining ranked lists: reciprocal rank fusion and weighted scores."""
from collections import defaultdict


def ranked(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def rrf(*runs: dict[str, float], k: int = 60, depth: int = 100) -> dict[str, float]:
    """Reciprocal rank fusion: only ranks matter, so scales cannot clash."""
    fused: dict[str, float] = defaultdict(float)
    for run in runs:
        for rank, doc in enumerate(ranked(run)[:depth], start=1):
            fused[doc] += 1.0 / (k + rank)
    return dict(fused)


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    span = (hi - lo) or 1.0
    return {d: (s - lo) / span for d, s in scores.items()}


def weighted(lexical: dict[str, float], dense: dict[str, float],
             alpha: float = 0.5, depth: int = 100,
             normalise: bool = True) -> dict[str, float]:
    """alpha * dense + (1 - alpha) * lexical over the union of both top lists."""
    lex = dict((d, lexical[d]) for d in ranked(lexical)[:depth])
    den = dict((d, dense[d]) for d in ranked(dense)[:depth])
    if normalise:
        lex, den = minmax(lex), minmax(den)
    union = set(lex) | set(den)
    return {d: alpha * den.get(d, 0.0) + (1 - alpha) * lex.get(d, 0.0)
            for d in union}
