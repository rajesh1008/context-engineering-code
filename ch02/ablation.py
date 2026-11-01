"""Chapter 2 ablations over the 200-determination set: k, format and order (query enrichment is in stage_variants.py).
Run as a script; it prints as it goes (no main(), so the k sweep reads as Listing 2.8)."""
from __future__ import annotations

import lab
from contextlab import scoreboard as sb

dets = lab.corpus.eval_set(200)


def row(name, pipe, dets, view=None):
    print(sb.run_row(name, sb.score(lab.run_all(pipe, dets, view))))


# ---------------------------------------------------------------- the script
print("== k sweep")
print(lab.HEADER)
for k in (3, 5, 10, 20):
    row(f"baseline, k={k}", lab.base.with_config(k=k), dets)
print()
for k in (3, 5, 10, 20):
    s = sb.score(lab.run_all(lab.fixed.with_config(k=k), dets, lab.fix_view))
    print(f"fixed k={k:>2} {s.wrong}/{s.n} hit5 {s.hit5:.2f} tok {s.tokens:,.0f} ${s.cost:.5f}")
print()
print("== format")
for fmt in ("markdown", "xml", "json", "plain"):
    print(sb.row(f"baseline fmt={fmt}", sb.score(lab.run_all(lab.base.with_config(fmt=fmt), dets))))
print()
print("== order")
for order in ("rank", "edges", "precedence"):
    print(sb.row(f"fixed order={order}", sb.score(lab.run_all(lab.fixed.with_config(order=order), dets,
                                                                 lab.fix_view))))

