"""Listing 5.14: per-family errors, cost and latency on sim-large, with the prompt cache.

Each family is its own run: its determinations only, in order, through a fresh
simulated cache, as a family-by-family rollout would see them.
Run: python 04-code/ch05/families.py   (writes out/families.txt)

STAND-IN: SimulatedModel, synthetic corpus, illustrative prices and latency (see standin.py).
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import standin as S                     # noqa: E402

ORDER = ["BOAT", "BOP", "PAP", "HO-3"]
RUNS = [("retrieve", "retrieve"), ("family", "stuff family"), ("hybrid", "hybrid")]


def main(out=print):
    for fam in ORDER:
        dets = [d for d in S.CORPUS.eval if d.family == fam]
        out(f"{fam}: {len(dets)} determinations, {sum(bool(d.endorsements) for d in dets)} "
            "on policies with endorsements")
        for name, strat in RUNS:
            ts = S.run_all(S.pipeline(strat), dets)
            cached = S.replay_cache(ts)
            lst = statistics.fmean(S.trace_cost(t) for t in ts)
            cch = statistics.fmean(S.trace_cost(t, c) for t, c in zip(ts, cached))
            p50 = statistics.median(S.trace_latency(t, c) for t, c in zip(ts, cached))
            tok = statistics.fmean(t.answer.tokens_in for t in ts)
            bad = ", ".join(t.det.determination_id for t in ts if not t.right) or "-"
            out(f"  {name:<8} wrong {S.wrong_count(ts):>2}  list ${lst:.5f}  cached ${cch:.5f}  "
                f"p50 {p50:.2f}s  tokens {tok:>6,.0f}  {bad}")


if __name__ == "__main__":
    lines = []
    main(lines.append)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "families.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
