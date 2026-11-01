"""Chapter 5: change rate (part 1, Listing 5.12) and corpus size (part 2, Listing 5.11).

Run: python 04-code/ch05/change_rate.py   (writes out/change_rate.txt)

STAND-IN: SimulatedModel prompts over a synthetic corpus, the simulated PromptCache
(1,024-token blocks) and illustrative sim-large prices (see standin.py).
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import standin as S                                 # noqa: E402
from contextlab.cache import PromptCache            # noqa: E402
from contextlab.longctx import DOC_KINDS, Kind      # noqa: E402
from contextlab.tokens import WINDOWS, cost_usd     # noqa: E402

PER_DAY = 3_200                                     # Lakeshore's determinations a day
OUT_TOKENS = 80                                     # a determination's reply, for the scaling table
MULTIPLES = [1, 2, 4, 8, 16, 64, 600]


def replay(traces, every: int | None, edit_kind: Kind) -> tuple[float, float]:
    cache, hits, total, cost = PromptCache(), 0, 0, []
    for i, t in enumerate(traces):
        rev = i // every if every else 0
        pieces, edited = [], False
        for seg in t.rendered.segments:
            text = seg.text
            if not edited and seg.span is not None and seg.span.kind == edit_kind:
                text, edited = f"{text} [rev {rev}]", True     # the first such document changes
            pieces.append((text, seg.end - seg.start))
        cached = cache.lookup(pieces)
        hits += cached
        total += t.answer.tokens_in
        cost.append(cost_usd("sim-large", t.answer.tokens_in, t.answer.tokens_out, cached))
    return hits / total, statistics.fmean(cost)


def part1(out, dets):
    out("1. Change rate: stuffed windows on sim-large, simulated prompt cache")
    out(f"{'strategy':<13} {'edited':<13} {'changes':>14} {'cache hit':>9} {'cached $':>9} {'$ per day':>9}")
    for strat in ("stuff all", "stuff family"):
        traces = S.run_all(S.pipeline(strat), dets)
        for kind in (Kind.FORM, Kind.BULLETIN):
            for every in (None, 50, 10, 1):
                hit, cost = replay(traces, every, kind)
                changes = f"every {every} calls" if every else "never"
                out(f"{strat:<13} {kind.value:<13} {changes:>14} {hit:>9.0%} {cost:>9.5f} {cost * PER_DAY:>9.2f}")
    cost = statistics.fmean(S.trace_cost(t) for t in S.run_all(S.pipeline("retrieve"), dets))
    out(f"{'retrieve':<13} {'(no cache)':<13} {'':>16} {0:>9.0%} {cost:>9.5f} {cost * PER_DAY:>9.2f}")


def part2(out, dets):
    out("2. Corpus size: does the stuffed corpus still fit?")
    binding = sum(s.tokens for s in S.CORPUS.spans if s.kind in DOC_KINDS)
    prompt = round(statistics.median(t.answer.tokens_in for t in S.run_all(S.pipeline("stuff all"), dets)))
    out(f"  synthetic corpus: {binding:,} tokens of binding text; stuffed prompt about {prompt:,} tokens")
    out(f"{'multiple':>8} {'prompt tok':>11} " + " ".join(f"{m + ' fits':>15}" for m in S.MODELS)
        + f" {'list $/det':>10} {'list $/day':>10}")
    for k in MULTIPLES:
        tok = prompt * k
        fits = " ".join(f"{'yes' if tok <= WINDOWS[m] else 'no':>15}" for m in S.MODELS)
        c = cost_usd("sim-large", tok, OUT_TOKENS)
        out(f"{str(k) + 'x':>8} {tok:>11,} {fits} {c:>10.4f} {c * PER_DAY:>10,.0f}")


def main(out=print):
    dets = S.CORPUS.eval
    part1(out, dets)
    out("")
    part2(out, dets)


if __name__ == "__main__":
    lines = []
    main(lines.append)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "change_rate.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
