"""Listing 3.7: ablating the fix (padding, applicability filter, edges order) on the lab's 60
and on all 200 determinations."""
from __future__ import annotations

from common import tee
import standin as S
from experiments import dets, padded_packet, render

ROWS = [("60k padded, clause at 50%", ("pad", 0.5)), ("60k padded, clause first", ("pad", 0.0)),
        ("60k padded, clause last", ("pad", 1.0)),
        ("unpadded, rank order", dict(order="rank")),
        ("unpadded, filter, rank order", dict(policy_filter=True, order="rank")),
        ("unpadded, edges order", dict(order="edges")),
        ("unpadded, filter, edges order", dict(policy_filter=True, order="edges"))]


def main():
    model, all_dets = S.SimulatedModel("sim-large"), dets()
    for ds in (all_dets[:60], all_dets):
        print(f"--- {len(ds)} determinations, sim-large: share citing the governing clause, mean tokens in")
        for label, how in ROWS:
            if isinstance(how, tuple):
                ok = sum(model.generate(render(padded_packet(d, 60_000, how[1]), d), key=d.determination_id)
                         .cited_span_id == d.gold_span_id for d in ds)
                print(f"  {label:<34}{ok / len(ds):>6.1%}")
            else:
                ts = S.Pipeline(S.PipelineConfig(**how)).run_many(ds)
                ok = sum(not t.wrong for t in ts)
                tok = round(sum(t.tokens_in for t in ts) / len(ts))
                print(f"  {label:<34}{ok / len(ds):>6.1%}{tok:>9,} tokens")
    print()
    d = all_dets[0]
    try:
        S.SimulatedModel("sim-small").generate(render(padded_packet(d, 60_000, 0.5), d), key=d.determination_id)
    except ValueError as e:
        print(f"sim-small, 60,000-token packet: ValueError: {e}")


if __name__ == "__main__":
    tee("3.7_ablation", main)
