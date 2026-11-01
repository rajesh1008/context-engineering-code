"""Ranking metrics. gold maps span id -> grade (2 governing, 1 supporting)."""
import math


def hit_at_k(ranking: list[str], gold: dict[str, int], k: int) -> float:
    return float(any(d in gold for d in ranking[:k]))


def recall_at_k(ranking: list[str], gold: dict[str, int], k: int) -> float:
    return sum(1 for d in ranking[:k] if d in gold) / len(gold)


def mrr(ranking: list[str], gold: dict[str, int], k: int = 100) -> float:
    for i, d in enumerate(ranking[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: list[str], gold: dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** gold.get(d, 0) - 1) / math.log2(i + 1)
              for i, d in enumerate(ranking[:k], start=1))
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1)
               for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0
