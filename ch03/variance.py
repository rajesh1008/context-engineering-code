"""Listing 3.8: how many determinations is enough. Start and middle at 2,000 and 15,000 tokens on
all 200, the start-minus-middle gap over 2,000 random subsets of each size, and the gap by topic."""
from __future__ import annotations

import random
from collections import Counter

from common import tee
import standin as S
from experiments import dets, padded_packet, render


def hits(ds, size, depth, model):
    return [int(model.generate(render(padded_packet(d, size, depth), d), key=d.determination_id)
                .cited_span_id == d.gold_span_id) for d in ds]


def main():
    ds, model = dets(), S.SimulatedModel("sim-large")
    hit = {(size, depth): hits(ds, size, depth, model) for size in (2_000, 15_000) for depth in (0.0, 0.5)}
    for (size, depth), h in hit.items():
        print(f"{size:>6,} tokens, depth {depth:>4.0%}: {sum(h) / len(h):>6.1%} of {len(h)} cite the governing clause")
    print()
    print("start-minus-middle gap, points, 5th to 95th percentile of 2,000 random subsets")
    lines = {}
    rng = random.Random(2026)
    for size in (2_000, 15_000):
        gaps_all = [a - b for a, b in zip(hit[size, 0.0], hit[size, 0.5])]
        for n in (10, 20, 30, 60, 120):
            g = sorted(100 * sum(rng.sample(gaps_all, n)) / n for _ in range(2_000))
            lines.setdefault(size, [100 * sum(gaps_all) / len(gaps_all)]).append(f"  n={n:<3} [{g[100]:5.1f}, {g[1899]:5.1f}]")
    for size, (gap, *cells) in lines.items():
        print(f"{size:>8,} tokens (all {len(ds)}: {gap:4.1f}):" + "".join(cells))
    print()
    print("15,000 tokens by topic, topics with 8 or more determinations (n, start, middle):")
    n = Counter(d.topic for d in ds)
    for t in sorted((t for t in n if n[t] >= 8), key=lambda t: -n[t]):
        idx = [i for i, d in enumerate(ds) if d.topic == t]
        s = sum(hit[15_000, 0.0][i] for i in idx) / n[t]
        m = sum(hit[15_000, 0.5][i] for i in idx) / n[t]
        print(f"  {t:<22}{n[t]:>4}{s:>8.0%}{m:>8.0%}")


if __name__ == "__main__":
    tee("3.8_variance", main)
