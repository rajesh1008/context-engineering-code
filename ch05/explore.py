"""Chapter 5 explorations: slices, paired comparison, holdout, the stale pointer,
the evidence budget and one determination under every strategy (parts A-G).

Run: python 04-code/ch05/explore.py   (writes out/explore.txt)

STAND-IN: the models are SimulatedModels and the corpus is synthetic (see standin.py).
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import standin as S                               # noqa: E402
from contextlab.evaluator import wilson           # noqa: E402
from contextlab.longctx import Kind               # noqa: E402

BUDGET = 3_000                                    # Chapter 4's evidence budget


def p95(xs):
    s = sorted(xs)
    return s[math.ceil(0.95 * len(s)) - 1]


def row(label, ts):
    k, n = S.wrong_count(ts), len(ts)
    lo, hi = wilson(k, n)
    tok = [t.answer.tokens_in for t in ts]
    return (f"{label:<44}{k:>4}/{n} {k / n:>6.1%} [{lo:>5.1%}, {hi:>5.1%}]  tok mean "
            f"{statistics.fmean(tok):>7,.0f} p95 {p95(tok):>7,} max {max(tok):>7,}")


def main(out=print):
    dets, hold = S.CORPUS.eval, S.CORPUS.holdout
    retrieve, hybrid = S.pipeline("retrieve"), S.pipeline("hybrid")
    rt, ht = S.run_all(retrieve, dets), S.run_all(hybrid, dets)

    out("A. eval200, sim-large")
    out(row("retrieve", rt))
    out(row("hybrid", ht))
    out("")
    out("B. slices, sim-large")
    slices = [("no endorsements", lambda d: not d.endorsements),
              ("with endorsements", lambda d: bool(d.endorsements)),
              ("form-governed", lambda d: d.gold_kind == Kind.FORM.value),
              ("endorse/bulletin", lambda d: d.gold_kind != Kind.FORM.value)]
    for name, f in slices:
        for strat, ts in (("retrieve", rt), ("hybrid", ht)):
            out(row(f"{strat} {name}", [t for t in ts if f(t.det)]))
    out("")
    out("C. paired retrieve vs hybrid, sim-large")
    fixes = sum(not r.right and h.right for r, h in zip(rt, ht))
    breaks = sum(r.right and not h.right for r, h in zip(rt, ht))
    both = sum(not r.right and not h.right for r, h in zip(rt, ht))
    out(f"  hybrid fixes {fixes}, breaks {breaks}, wrong in both {both}")
    for name, ts in (("retrieve", rt), ("hybrid", ht)):
        bad = [t for t in ts if not t.right]
        out(f"  {name} errors with gold in packet: {sum(t.gold_in_packet for t in bad)} of {len(bad)}")
    out("")
    out("D. holdout 300 (determinations 201-500), sim-large")
    out(row("retrieve holdout", S.run_all(retrieve, hold)))
    out(row("hybrid holdout", S.run_all(hybrid, hold)))
    out("")
    out("E. stale form pointer (Ch2 injection), sim-large")
    out(row("retrieve, stale", S.run_all(retrieve, dets, S.stale_pointer)))
    out(row("hybrid, stale", S.run_all(hybrid, dets, S.stale_pointer)))
    out(row("hybrid, stale + reconcile", S.run_all(hybrid.with_config(reconcile=True), dets, S.stale_pointer)))
    out("")
    out(f"F. hybrid under the Ch4 evidence budget {BUDGET:,} (does the budget bite?)")
    ev = [t.evidence_tokens for t in ht]
    out(f"  hybrid packets with evidence over {BUDGET:,} tokens: {sum(e > BUDGET for e in ev)} of {len(ev)}")
    out(f"  hybrid evidence tokens: median {statistics.median(ev):,.0f}, max {max(ev):,}")
    for m in ("sim-mid", "sim-small"):
        out(row(f"retrieve {m}", S.run_all(S.pipeline("retrieve", m), dets)))
        out(row(f"hybrid {m}", S.run_all(S.pipeline("hybrid", m), dets)))
    out("")
    for t in ht:
        if not t.right:
            cited = S.CORPUS.span(t.answer.cited_span_id).provenance.citation
            gold = S.CORPUS.span(t.det.gold_span_id).provenance.label
            out(f"   hybrid wrong {t.det.determination_id} {t.det.family} {t.det.gold_kind} gold {gold} cited {cited}")
    ndocs = [len({(s.provenance.doc_id, s.provenance.version) for s in t.packet}) for t in ht]
    out(f"  hybrid documents per packet: median {statistics.median(ndocs):.1f}, max {max(ndocs)}; "
        f"spans median {statistics.median(len(t.packet) for t in ht):.1f}")
    out("")
    d = S.CORPUS.det("DET-0001")
    out(f"G. Carter {d.determination_id} under each strategy, sim-large")
    for strat in S.STRATEGIES:
        t = S.pipeline(strat).run(d)
        c = S.CORPUS.span(t.answer.cited_span_id).provenance.citation
        out(f"  {strat:<12} {'right' if t.right else 'WRONG'} cited {c}; {t.answer.tokens_in:,} tok, "
            f"{len(t.packet)} spans")
    gold = S.CORPUS.span(d.gold_span_id).provenance.label
    out(f"  gold {gold}; claimant {d.holder} form {d.form} end {list(d.endorsements)}")


if __name__ == "__main__":
    lines = []
    main(lines.append)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "explore.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
