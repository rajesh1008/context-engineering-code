"""Listing 5.4: why each strategy's wrong citations were wrong, sim-large.

Run: python 04-code/ch05/why_wrong.py   (writes out/why_wrong.txt)

STAND-IN: the model is a SimulatedModel and the corpus is synthetic (see standin.py).
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import standin as S                     # noqa: E402

corpus = S.CORPUS
dets = corpus.eval
base = S.pipeline("retrieve")           # its policy check is the one Chapter 2 built


def explain(pipe):
    why, wrong = Counter(), []
    for d in dets:
        t = pipe.run(d)
        if t.answer.cited_span_id == d.gold_span_id:
            continue
        span = corpus.span(t.answer.cited_span_id)
        verdict = base.policy.applies(span, d)
        reason = "applies (wrong clause)" if verdict.ok else re.sub(r"[\d()\-]+", "#", verdict.reason)
        why[reason] += 1
        wrong.append(t)
    return why, wrong


def main(out=print):
    for strat in ("stuff all", "stuff family", "hybrid", "retrieve"):
        why, wrong = explain(S.pipeline(strat))
        out(f"{strat}: {len(wrong)} wrong")
        for reason, n in why.most_common():
            out(f"{n:>5}  {reason}")
        present = [t for t in wrong if t.gold_in_packet]
        middle = sum(0.25 <= t.gold_position() <= 0.75 for t in present)
        out(f"  gold span present in {len(present)} of the wrong packets; in the middle half for {middle}")


if __name__ == "__main__":
    lines = []
    main(lines.append)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "why_wrong.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
