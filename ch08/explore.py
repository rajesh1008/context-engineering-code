"""Chapter 8 exploration: parts A (one rule, one claim), B and C (probes per rule and
the evaluation errors), D (two readings of R8), G (Example A and the Illinois
bulletin) and H (two underwriters' readings of R1).

    python3 explore.py      # writes out/explore.txt
"""
from __future__ import annotations

from datetime import date

from lab import HERE, R09, R10, R1_A, R1_B, R8_NO_START, evaluate  # noqa: F401
from standin import CORPUS, PipelineConfig, run_set
from contextlab.instructions import V22, V22_NO_A
from contextlab.rulebook import Rulebook, check, probes


def conflict(c) -> str:
    d = c.det
    return f"{d.determination_id} ({d.state}, {d.loss}): rule says {c.rule_says}, documents say {c.documents_say}"


def main() -> None:
    out: list[str] = []
    p = out.append

    p("A. One rule, one claim: the table resolves the facts")
    d = CORPUS.by_id["DET-0026"]
    rule = R10.rule_for(d.topic)
    p(f"  {d.determination_id} {d.state} loss {d.loss} topic {d.topic} -> {rule.resolve(d)} "
      f"(gold {CORPUS.governing(d).doc_label})")
    p("  rendered rule span:")
    for line in R10.span(d).text.splitlines():
        p(f"    {line}")
    p("")

    p("B. Per rule: probes (exception selected) and eval claims, sim-large")
    none, r10 = evaluate("sim-large", None), evaluate("sim-large", R10)
    pr0 = {t.det.determination_id: t for t in none["probe_traces"]}
    pr1 = {t.det.determination_id: t for t in r10["probe_traces"]}
    ev0 = {t.det.determination_id: t for t in none["traces"]}
    ev1 = {t.det.determination_id: t for t in r10["traces"]}
    for r in R10.rules:
        ids = [k for k, t in pr0.items() if t.det.topic == r.topic]
        eids = [k for k, t in ev0.items() if t.det.topic == r.topic]
        a0 = sum(not pr0[k].wrong for k in ids)
        a1 = sum(not pr1[k].wrong for k in ids)
        w0 = sum(ev0[k].wrong for k in eids)
        w1 = [k for k in eids if ev1[k].wrong]
        p(f"  {r.rule_id:<4} {r.topic:<21} probes {len(ids):>3}  adherence {a0:>3} -> {a1:>3}  "
          f"eval {len(eids):>3}  wrong {w0} -> {len(w1)}  {', '.join(w1) or '-'}")
    p("")
    p("C. Eval errors, sim-large: without the rulebook and with r1.0")
    topics = set(R10.topics())
    for name, e in (("none", none), ("r1.0", r10)):
        cov = [t.det.determination_id for t in e["traces"] if t.wrong and t.det.topic in topics]
        unc = [t.det.determination_id for t in e["traces"] if t.wrong and t.det.topic not in topics]
        p(f"  {name:<5} covered {cov}  uncovered {unc}")
    p("")

    p("D. Two readings of R8 tree debris that differ only in a date")
    for label, rule in (("No start date", R8_NO_START), ("From 2024-08-15", R10.rule_for("tree_debris"))):
        rc = check(Rulebook("r1.0", (rule,)), CORPUS)[0]
        p(f"  {label}: {rc.agree}/{rc.checked} agree with the documents")
        for c in rc.conflicts:
            p(f"    {conflict(c)}")
    p("")

    p("G. Example A and the Illinois bulletin claim (sim-large, no rulebook)")
    d61 = CORPUS.by_id["DET-0061"]
    for label, ins in (("v2.2", V22), ("v2.2 without Example A", V22_NO_A)):
        t = run_set(PipelineConfig(True, "sim-large", ins, None), [d61])[0]
        e = evaluate("sim-large", None, instructions=ins)
        p(f"  {label:<24} DET-0061 cites {t.answer.citation}; eval wrong {100 * e['wrong'] / e['n']:.1f}%")
    p("")

    p("H. The exclusion two underwriters read differently (R1 water backup)")
    backup19 = [d for d in CORPUS.determinations if d.topic == "water_backup" and d.carries("HO 04 95 (2019-03)")]
    for label, rule in (("Underwriter A", R1_A), ("Underwriter B", R1_B)):
        book = R10.with_rule(rule, "r1.0", "")
        rc = next(c for c in check(book, CORPUS) if c.rule.rule_id == "R1")
        tr = run_set(PipelineConfig(True, "sim-large", V22, book), backup19)
        e = evaluate("sim-large", book)
        p(f"  {label}: {rc.agree}/{rc.checked} agree with the documents; sim-large wrong on 2019-edition "
          f"backup claims {sum(t.wrong for t in tr)}/{len(tr)}; eval {100 * e['wrong'] / e['n']:.1f}%")
        for c in rc.conflicts[:2]:
            p(f"    {conflict(c)}")

    text = "\n".join(out) + "\n"
    (HERE / "out" / "explore.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
