"""Listing 5.15: route the marine family to whole-document context; the scoreboard rows.

Run: python 04-code/ch05/scoreboard_ch05.py   (writes out/scoreboard.txt)

STAND-IN: SimulatedModel, synthetic corpus; lookup minutes are Chapter 2's proxy
and faithfulness is the stand-in's per-determination score (see standin.py).
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from standin import CORPUS as corpus, Determination, Pipeline, Trace   # noqa: E402
import standin as S                                                   # noqa: E402

retrieve = Pipeline(corpus=corpus).with_config(policy_filter=True)
hybrid = retrieve.with_config(stuff="documents", order="cache")
WHOLE_DOCUMENT_FAMILIES = frozenset({"BOAT"})   # the decision table, October 31, 2026

def route(det: Determination) -> Trace:
    """One determination through the assembly its family is assigned to."""
    pipe = hybrid if det.family in WHOLE_DOCUMENT_FAMILIES else retrieve
    return pipe.run(det)


def row(label: str, traces: list[Trace]) -> str:
    k, n = S.wrong_count(traces), len(traces)
    look = statistics.median(S.lookup_minutes(t) for t in traces)
    faith = statistics.fmean(t.det.faith for t in traces)
    tok = statistics.fmean(t.answer.tokens_in + t.answer.tokens_out for t in traces)
    cost = statistics.fmean(S.trace_cost(t) for t in traces)
    return (f"{label:<34} wrong {k / n:>5.1%} ({k}/{n})  lookup {look:.1f} min  faith {faith:.3f}  "
            f"tokens {tok:,.0f}  ${cost:.5f}")


def main(out=print):
    ev, hold = corpus.eval, corpus.holdout
    pilot = [d for d in ev if not d.endorsements]
    out(row("Ch4 row: retrieve (eval 200)", [retrieve.run(d) for d in ev]))
    out(row("Ch5 shipped: BOAT whole (eval)", [route(d) for d in ev]))
    out(row("Ch5 shipped (holdout 300)", [route(d) for d in hold]))
    out(row("candidate: hybrid (eval 200)", [hybrid.run(d) for d in ev]))
    out(row("candidate: hybrid (holdout 300)", [hybrid.run(d) for d in hold]))
    out(row(f"pilot slice, retrieve (n={len(pilot)})", [retrieve.run(d) for d in pilot]))
    out(row(f"pilot slice, hybrid (n={len(pilot)})", [hybrid.run(d) for d in pilot]))


if __name__ == "__main__":
    lines = []
    main(lines.append)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "scoreboard.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
