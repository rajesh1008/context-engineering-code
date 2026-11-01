"""Listing 3.14: the same spans in five orders, 200 determinations, with and without the
applicability filter, and the DET-0006 flip."""
from __future__ import annotations

from common import tee
import standin as S
from standin import ORDERS
from experiments import dets as load, render, retrieved


def citations(dets, policy_filter: bool):
    model = S.SimulatedModel("sim-large")
    packets = {d.determination_id: retrieved(d, policy_filter) for d in dets}
    cited = {o: {} for o in ORDERS}
    for o in ORDERS:
        for d in dets:
            r = render(packets[d.determination_id], d, o)
            cited[o][d.determination_id] = model.generate(r, key=d.determination_id, topic=d.topic).cited_span_id
    return packets, cited


def main():
    dets = load()
    n = len(dets)
    for title, filt in (("baseline (no filter)", False), ("fixed (applicability filter)", True)):
        packets, cited = citations(dets, filt)
        spans = sum(len(p.evidence) for p in packets.values()) / n
        print(f"{title}: same {spans:.1f} spans per packet, five orders")
        depends = set()
        for o in ORDERS:
            w = sum(cited[o][d.determination_id] != d.gold_span_id for d in dets)
            diff = [d.determination_id for d in dets if cited[o][d.determination_id] != cited["rank"][d.determination_id]]
            depends |= set(diff)
            print(f"  {o:<12}wrong {w:>3}/{n} ({w / n:>5.1%})  cited span differs from rank order: {len(diff):>3}")
        print(f"  determinations whose citation depends on order: {len(depends)}/{n}")
        print()
    d = next(x for x in dets if x.determination_id == "DET-0006")
    pk = packets[d.determination_id]
    gold = S.CORPUS.by_id[d.gold_span_id]
    rank = [e.span_id for e in pk.evidence].index(d.gold_span_id) + 1
    print(f"{d.determination_id} ({d.holder}, {d.state}, {d.topic}): gold {gold.provenance.label}, "
          f"rank {rank} of {len(pk.evidence)}")
    for o, tag in (("rank", "rank order    ->"), ("reversed", "reversed order->")):
        r = render(pk, d, o)
        sid = cited[o][d.determination_id]
        s_no = next(k for k, v in r.labels.items() if v == sid)
        print(f"  {tag} [{s_no}] {S.CORPUS.by_id[sid].label}")


if __name__ == "__main__":
    tee("3.14_order", main)
