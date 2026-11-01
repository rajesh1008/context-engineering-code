"""Chapter 4 checks: A, the scoreboard model on all 200 determinations with and
without the cap; B, the From-the-Field question (same facts, 8,000 vs 80,000
padded tokens). Run: python 04-code/ch04/verify_ch04.py   (stand-in models)"""
import lab

CAP = lab.CAP


def section_a(out=print):
    out("A. Scoreboard model sim-large on the 200-determination evaluation set")
    pipe = lab.current.with_config(model="sim-large")
    for label, p in (("current, clean", pipe), (f"clean + cap {CAP:,}", pipe.with_config(budget=CAP))):
        ts = lab.run(p, lab.all_dets)
        w, tok, c, lm = lab.row_stats(ts, lab.all_dets)
        n = round(w * len(ts))
        out(f"  {label:<20} wrong {n}/{len(ts)}  {w:5.1%}  tokens/det {tok:,.0f}  cost/det ${c:.5f}  lookup {lm:.1f} min")


def section_b(out=print):
    out("B. Same facts, 8,000 vs 80,000 padded tokens (100 determinations)")
    for m in lab.MODELS:
        rows = []
        for target in (8_000, 80_000):
            for kind, fn in (("neutral", lab.neutral), ("look-alike", lab.lookalike)):
                pipe = lab.current.with_config(model=m, pad_fn=lambda d, t=target, f=fn: f(d, t))
                ts = lab.run(pipe, lab.dets)
                over = sum(t is None for t in ts)
                rows.append(f"{target // 1000}k {kind} {lab.wrong(ts):5.1%}"
                            + (f" ({over} overflow)" if over else ""))
        out(f"  {m:<10} " + " | ".join(rows))


if __name__ == "__main__":
    section_a()
    print()
    section_b()
