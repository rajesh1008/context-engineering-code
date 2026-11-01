"""Chapter 7 lab: an instruction clash injected, detected, measured and repaired.

Writes output.txt sections 1-4 (explore.py appends parts A-L). Deterministic; no API key.
Run: python3 lab.py
"""
from __future__ import annotations

from common import HERE, MODELS, ROOF, corpus, pipe, traces
from contextlab.instructions import V22, V30, V31, V32, adherence_probes, clashes
from standin import mean_tokens, rate

out: list[str] = []
p = out.append


def main() -> None:
    p("1. Failure injection: the roof skill is updated (v2.2 -> v3.0)")
    p("  diff:")
    for ln in V22.diff(V30):
        p(f"    {ln}")
    p(f"  numbered rules changed: {'yes' if V22.rules != V30.rules else 'no'}")
    p("")

    p("2. Detection")
    for ins in (V22, V30, V31, V32):
        found = clashes(ins)
        p(f"  clash check {ins.version}: {'; '.join(found) if found else 'no clashes'}")
    p("")

    probes = adherence_probes(corpus, "precedence")
    roof = [d for d in probes if d.topic in ROOF]
    p(f"3. Clash and repairs, seed 2026: roof probes ({len(roof)}) and all precedence probes "
      f"({len(probes)}) answered correctly")
    for ins in (V22, V30, V31, V32):
        for m in MODELS:
            pl = pipe(ins, m)
            r = sum(not pl.run(d).wrong for d in roof)
            a = sum(not pl.run(d).wrong for d in probes)
            ev = traces(ins, m, corpus.eval)
            p(f"  {ins.version:<5} {m:<10} clashes {len(clashes(ins))}  roof {r:>2}/{len(roof)} ({r / len(roof):6.1%})  "
              f"precedence {a:>3}/{len(probes)} ({a / len(probes):5.1%})  eval wrong {rate(ev):.1f}%  "
              f"prompt tokens {mean_tokens(ev):,.0f}")
    p("")

    p("4. The incident (sim-large, eval set)")
    before = {t.det.determination_id: t for t in traces(V22, "sim-large", corpus.eval)}
    after = traces(V30, "sim-large", corpus.eval)
    n = 0
    for t in after:
        b = before[t.det.determination_id]
        if t.wrong and not b.wrong:
            d = t.det
            n += 1
            p(f"  {d.determination_id} {d.name}, {d.state}, {d.topic}, loss {d.loss}; gold {t.gold.doc_label}")
            p(f"    v2.2 cited {b.answer.citation}")
            p(f"    v3.0 cited {t.answer.citation}")
            p(f"    skills loaded: {', '.join(t.packet.skills())}; rule the model dropped: "
              f"{', '.join(t.answer.dropped) or 'none'}")
    p(f"  claims newly wrong under v3.0: {n}")
    (HERE / "output.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
    print("\n".join(out))
