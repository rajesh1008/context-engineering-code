"""Chapter 4 lab: rot curves, knees, distraction, task-type knees and the
related-documents failure injection. Run: python 04-code/ch04/lab.py
Writes output.txt and the CSVs the figures are drawn from, next to this file.

STAND-IN: every model is a deterministic SimulatedModel (see standin.py); the
corpus and determinations are synthetic; prices and lookup minutes are illustrative.
"""
from __future__ import annotations

import csv, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import standin as S                                     # noqa: E402
from contextlab.diagnostics import Diagnostics, knee     # noqa: E402

MODELS = ["sim-large", "sim-mid", "sim-small"]
LENGTHS = [0] + S.NEUTRAL_TARGETS
TASK_LENGTHS = [0, 4_000, 8_000, 12_000, 16_000, 24_000, 32_000, 48_000]
INJECTED = 16_000
CAP = 3_000
dets = S.DETS_ALL[:100]                 # the first 100 of the 200-determination set
all_dets = S.DETS_ALL
current = S.Pipeline(corpus=S.CORPUS, model="sim-large")
neutral, lookalike = S.neutral, S.lookalike


def run(pipe, ds):
    """One trace per determination; None where the prompt overflowed the window."""
    out = []
    for d in ds:
        try:
            out.append(pipe.run(d))
        except ValueError:
            out.append(None)
    return out


def wrong(ts) -> float:
    return sum(t is None or t.answer.cited_span_id != t.det.gold_span_id for t in ts) / len(ts)


def lookup_min(ts, ds) -> float:
    return statistics.median(S.lookup_minutes(t, d) for t, d in zip(ts, ds))


def row_stats(ts, ds):
    ok = [t for t in ts if t is not None]
    toks = sum(t.billed_tokens for t in ok) / len(ok)
    cost = sum(S.cost(t) for t in ok) / len(ok)
    return wrong(ts), toks, cost, lookup_min(ts, ds)


def _csv(name, header, rows):
    with open(HERE / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(out=print):
    # 1. rot curves ------------------------------------------------------------
    curves = {m: Diagnostics.rot_curve(current.with_config(model=m), dets, LENGTHS, pad_fn=neutral) for m in MODELS}
    out("1. Rot curves (neutral claim-file padding; mean prompt tokens, wrong-clause rate)")
    out("  padding          sim-large            sim-mid          sim-small")
    out("              tokens   wrong     tokens   wrong     tokens   wrong")
    csv_rows = []
    for j, t in enumerate(LENGTHS):
        cells = []
        for m in MODELS:
            _, tok, w = curves[m][j]
            over = tok == t and t > S.PROFILES[m]["window"] - 2_000
            cells.append(f"{tok:>9,.0f} {w * 100:5.1f}%{'*' if over else ' '}")
            csv_rows.append([m, t, round(tok), w, int(over)])
        out(f"  {t:>7,}  " + "  ".join(cells))
    out("  * the prompt exceeds the window: the call is rejected (overflow), counted wrong")
    _csv("rot_curve.csv", ["model", "padding", "mean_prompt_tokens", "wrong_rate", "overflow"], csv_rows)

    out("")
    out("knee (largest padding within 2 points of the unpadded rate):")
    knees = []
    for m in MODELS:
        rows = curves[m]
        k = knee(rows)
        j = LENGTHS.index(k)
        win = S.PROFILES[m]["window"]
        out(f"  {m:<10} window {win:>7,}  knee {k:>7,} padded tokens (prompt {rows[j][1]:,.0f} tokens; "
            f"{rows[0][2]:.1%} -> {rows[j + 1][2]:.1%} at {LENGTHS[j + 1]:,})  knee prompt/window {rows[j][1] / win:.0%}")
        knees.append(["model", m, 100, k, round(rows[j][1])])
    lg = curves["sim-large"][LENGTHS.index(64_000)]
    out(f"  sim-large at 64,000 padded tokens ({lg[1]:,.0f} in the prompt, {lg[1] / 200_000:.0%} of the window): "
        f"{lg[2]:.1%} wrong vs {curves['sim-large'][0][2]:.1%} unpadded")

    # 2. distraction ----------------------------------------------------------
    out("")
    out("2. Same length, different padding (wrong-clause rate)")
    out("model       padding  neutral  look-alike  look-alike tokens   diff")
    drows = []
    for m in MODELS:
        for t in (4_000, 8_000, 16_000):
            wn = curves[m][LENGTHS.index(t)][2]
            ts = run(current.with_config(model=m, pad_fn=lambda d, t=t: lookalike(d, t)), dets)
            wl = wrong(ts)
            la_tok = statistics.mean(sum(s.tokens for s in lookalike(d, t) if s.kind != "page") for d in dets)
            out(f"{m:<10} {t:>8,}   {wn:5.1%}      {wl:5.1%}  {la_tok:>17,.0f} {(wl - wn) * 100:+5.1f}")
            drows.append([m, t, wn, wl, round(la_tok)])
    pools = [sum(s.tokens for s in S.pool(d)) for d in dets]
    out(f"  look-alike pool per claim: median {statistics.median(pools):,.0f} tokens, max {max(pools):,}; "
        "the rest of each padded length is claim-file pages")
    _csv("distraction.csv", ["model", "padding", "neutral", "lookalike", "lookalike_tokens"], drows)

    mid = current.with_config(model="sim-mid")
    ts_la = run(mid.with_config(pad_fn=lambda d: lookalike(d, 16_000)), dets)
    ts_n = run(mid.with_config(pad_fn=lambda d: neutral(d, 16_000)), dets)
    bad = [t for t in ts_la if t.answer.kind != "right"]
    n_la = sum(t.answer.kind == "lookalike" for t in bad)
    out(f"  sim-mid at 16,000: {len(bad)} wrong citations under look-alike padding -> an inapplicable look-alike "
        f"{n_la}, an applicable span, wrong clause {len(bad) - n_la}")
    for label, ts in (("neutral   ", ts_n), ("look-alike", ts_la)):
        sm = Diagnostics.summary(ts, S.applies)
        out(f"  diagnostics on sim-mid 16,000 {label}: " + ", ".join(f"{k} {v:,}" for k, v in sm.items()))
    small40 = run(current.with_config(model="sim-small", pad_fn=lambda d: neutral(d, 40_000)), dets)
    out(f"  overflow: sim-small with 40,000 padded tokens rejects {sum(t is None for t in small40)} of 100 calls "
        "(window 32,000)")

    # 3. task types -------------------------------------------------------------
    out("")
    out("3. Where rot starts by task type (sim-mid, neutral padding, 200 determinations)")
    out("task type                   n" + "".join(f"{t // 1000:>6}k" for t in TASK_LENGTHS) + "     knee")
    groups = [(tt, [d for d in all_dets if d.task_type == tt]) for tt in ("form clause", "endorsement or bulletin")]
    groups += [(f"family {f}", [d for d in all_dets if d.family == f]) for f in ("HO-3", "PAP", "BOAT", "BOP")]
    trows = []
    for name, ds in groups:
        rows = Diagnostics.rot_curve(mid, ds, TASK_LENGTHS, pad_fn=neutral)
        k = knee(rows)
        out(f"{name:<24}{len(ds):>4}" + "".join(f"{r[2] * 100:6.1f}%" for r in rows) + f"{k:>9,}")
        trows += [[name, len(ds), r[0], r[2]] for r in rows]
        knees.append(["task", name, len(ds), k, ""])
    _csv("task_curves.csv", ["group", "n", "padding", "wrong_rate"], trows)
    _csv("knees.csv", ["kind", "name", "n", "knee_padded", "knee_prompt_tokens"], knees)

    # 4. failure injection ------------------------------------------------------
    out("")
    out(f"4. Failure injection: look-alike padding to {INJECTED:,} tokens; fix: packet cap {CAP:,} tokens")
    out("  (the cap counts evidence tokens; tokens/det is prompt plus output, as billed)")
    out("model      pipeline                      wrong  tokens/det  cost/det lookup min")
    crows, keep = [], {}
    for m in MODELS:
        pipe = current.with_config(model=m)
        variants = [("current, clean", pipe),
                    ("injected (look-alikes)", pipe.with_config(
                        pad_fn=lambda d: lookalike(d, INJECTED))),
                    (f"injected + cap {CAP:,}", pipe.with_config(
                        pad_fn=lambda d: lookalike(d, INJECTED), budget=CAP)),
                    (f"clean + cap {CAP:,}", pipe.with_config(budget=CAP))]
        for label, p in variants:
            ts = run(p, dets)
            w, tok, c, lm = row_stats(ts, dets)
            keep[(m, label)] = (ts, w, tok)
            out(f"{m:<10} {label:<26} {w:8.1%} {tok:>11,.0f} $ {c:7.4f} {lm:10.1f}")
            crows.append([m, label, w, round(tok), round(c, 6), round(lm, 2)])
    _csv("cap.csv", ["model", "pipeline", "wrong_rate", "tokens_per_det", "cost_per_det", "lookup_min"], crows)
    out("  clean + cap drops nothing (packets are under 3,000) but the packer reorders spans by its own score, "
        "so an answer can change")
    _, wi, ti = keep[("sim-mid", "injected (look-alikes)")]
    tsc, wc, tc = keep[("sim-mid", f"injected + cap {CAP:,}")]
    out("")
    out(f"  sim-mid: tokens per determination {ti:,.0f} -> {tc:,.0f} ({1 - tc / ti:.0%} fewer); "
        f"wrong-clause {wi:.1%} -> {wc:.1%}")
    tsi = keep[("sim-mid", "injected (look-alikes)")][0]
    for a, b in zip(tsi, tsc):
        if a.answer.kind == "lookalike" and b.answer.kind == "right":
            d = a.det
            out("")
            out(f"example {d.determination_id} ({d.holder}, {d.state}, {d.form} ({d.edition}), topic {d.topic})")
            out(f"  gold:     {d.gold_label}")
            sa, sb = S.BY_ID[a.answer.cited_span_id], S.BY_ID[b.answer.cited_span_id]
            why = "endorsement not on policy" if sa.kind == "endorsement" else "not applicable to policy"
            out(f"  injected: cited {sa.label} ({a.rendered.tokens:,} tokens); policy check: {why}")
            out(f"  capped:   cited {sb.label} ({b.rendered.tokens:,} tokens)")
            break


if __name__ == "__main__":
    lines = []
    main(lambda s: (print(s), lines.append(s)))
    (HERE / "output.txt").write_text("\n".join(lines) + "\n")
