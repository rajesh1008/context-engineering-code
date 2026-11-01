"""Listing 3.15: the order-flip probe (rank against reversed order) as an error detector."""
from __future__ import annotations

from collections import Counter

from common import tee
from experiments import dets as load
from order import citations


def main():
    dets = load()
    _, cited = citations(dets, True)
    rows = [(d, {o: cited[o][d.determination_id] for o in cited}) for d in dets]
    flag = [(d, c) for d, c in rows if c["rank"] != c["reversed"]]
    wrong_rank = [d for d, c in rows if c["rank"] != d.gold_span_id]
    flag_wrong = [d for d, c in flag if c["rank"] != d.gold_span_id]
    print(f"order-flip probe (rank vs reversed), fixed pipeline, sim-large, {len(dets)} determinations")
    print(f"  flagged: {len(flag)}  wrong in rank order: {len(wrong_rank)}")
    print(f"  flagged and wrong: {len(flag_wrong)}  precision {len(flag_wrong) / len(flag):.0%}  "
          f"recall {len(flag_wrong) / len(wrong_rank):.0%}")
    print(f"  extra model calls: {len(dets)} (one per determination), extra input tokens = 1x packet")
    fixed = sum(c["rank"] != d.gold_span_id and c["precedence"] == d.gold_span_id for d, c in rows)
    broke = sum(c["rank"] == d.gold_span_id and c["precedence"] != d.gold_span_id for d, c in rows)
    print(f"precedence vs rank order, paired: fixed {fixed}, broke {broke}, net {fixed - broke}")
    topics = Counter(d.topic for d, _ in flag)
    print("flagged by topic: " + ", ".join(f"{t} {k}" for t, k in topics.most_common()))


if __name__ == "__main__":
    tee("3.15_flip_probe", main)
