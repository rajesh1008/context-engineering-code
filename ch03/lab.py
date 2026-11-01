"""Listings 3.6 and the failure and fix in Section 3.3: the position sweep on 60 determinations,
the failure injection, and the fix (applicability filter, edges order).

    python lab.py     # about a minute; writes out/3.6_sweep.txt"""
from __future__ import annotations

from common import tee
import standin as S
from experiments import DEPTHS, SIZES, padded_packet, render, sweep_set

ROWS = {"sim-large": SIZES, "sim-small": (2_000, 15_000)}


def cites(model: S.SimulatedModel, d: S.Determination, size: int, depth: float):
    r = render(padded_packet(d, size, depth), d)
    return r, model.generate(r, key=d.determination_id, topic=d.topic)


def sweep():
    ds = sweep_set()
    print(f"Position sweep: share of {len(ds)} determinations citing the governing clause")
    print()
    for name, sizes in ROWS.items():
        model = S.SimulatedModel(name)
        print(f"{name}  depth:" + "".join(f"{x:>6.0%}" for x in DEPTHS))
        for size in sizes:
            row = []
            for depth in DEPTHS:
                hit = sum(cites(model, d, size, depth)[1].cited_span_id == d.gold_span_id for d in ds)
                row.append(hit / len(ds))
            print(f"{size:>9,} tokens" + "".join(f"{x:>6.0%}" for x in row))


def failure():
    ds, model = sweep_set(), S.SimulatedModel("sim-large")
    out = [(d, *cites(model, d, 60_000, 0.5)) for d in ds]
    hit = sum(a.cited_span_id == d.gold_span_id for d, _, a in out)
    print(f"Failure injection: governing clause at 50% depth of a 60,000-token packet (sim-large): "
          f"{hit / len(ds):.0%} cite it")
    d, r, a = next((d, r, a) for d, r, a in out if a.cited_span_id != d.gold_span_id)
    g = next(s for s in r.evidence if s.span_id == d.gold_span_id)
    gold = S.CORPUS.by_id[d.gold_span_id]
    print(f"  {d.determination_id} ({d.holder}): gold {gold.provenance.label} at token {g.start:,} of "
          f"{r.tokens:,}; cited {S.CORPUS.by_id[a.cited_span_id].label}")


def fix():
    ds = sweep_set()
    ts = S.Pipeline(S.PipelineConfig(policy_filter=True, order="edges")).run_many(ds)
    hit = sum(not t.wrong for t in ts) / len(ts)
    print(f"Fix: applicable spans only, edges order: {hit:.0%} cite it, "
          f"{round(sum(t.tokens_in for t in ts) / len(ts)):,} tokens per packet")


if __name__ == "__main__":
    tee("3.6_sweep", sweep)
    tee("3.6_failure", failure)
    tee("3.6_fix", fix)
