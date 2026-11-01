"""Chapter 8 lab: the Rulebook. Elicited rules as decision tables, checked against the
documents, rendered as instruction spans, probed, and gated like code.

    python3 lab.py      # writes output.txt and review.csv, injection.csv, adherence.csv, probes.csv
"""
from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
from standin import CORPUS, PipelineConfig, run_set  # noqa: E402  (sets sys.path)
from contextlab.instructions import V22  # noqa: E402
from contextlab.rulebook import Row, Rule, Rulebook, check, load_yaml_rule, probes  # noqa: E402

D = date.fromisoformat
MODELS = ("sim-large", "sim-mid", "sim-small")

# ---------------------------------------------------------------- the twelve rules (r0.9)
RULES = (
    Rule("R1", "water_backup", "Water backup through a sewer or drain: if the policy carries HO 04 95, the endorsement decides it, any edition; otherwise the HO-3 water exclusion.",
         (Row("HO 04 95", endorsement="HO 04 95"),), "HO-3"),
    Rule("R2", "food_spoilage", "Food lost in a power outage: if the policy carries HO 04 98, the endorsement pays it back; otherwise the HO-3 power failure exclusion.",
         (Row("HO 04 98", endorsement="HO 04 98"),), "HO-3"),
    Rule("R3", "earth_movement", "Earth movement: HO 04 54 if carried; otherwise the HO-3 exclusion.",
         (Row("HO 04 54", endorsement="HO 04 54"),), "HO-3"),
    Rule("R4", "personal_property_rc", "Contents at replacement cost: HO 04 90 if carried; otherwise HO-3 loss settlement.",
         (Row("HO 04 90", endorsement="HO 04 90"),), "HO-3"),
    Rule("R5", "roof_settlement", "Roof settlement: if the policy carries LM HO 17, the endorsement settles the roof at actual cash value; otherwise the HO-3 condition.",
         (Row("LM HO 17", endorsement="LM HO 17"),), "HO-3"),
    Rule("R6", "roof_matching", "Matching undamaged roofing or siding: in Wisconsin, the 2023 bulletin until 15 March 2025 and the 2025 one after; otherwise the HO-3 loss settlement condition.",
         (Row("WI-OCI-2023-07", state="WI", on_or_after=D("2023-07-01"), before=D("2025-03-15")),
          Row("WI-OCI-2025-03", state="WI", on_or_after=D("2025-03-15"))), "HO-3"),
    Rule("R7", "frozen_pipes", "Frozen pipes: in Illinois, for losses on or after 1 May 2025, the utility interruption bulletin decides it; otherwise the HO-3 freezing peril.",
         (Row("IL-DOI-2025-05", state="IL", on_or_after=D("2025-05-01")),), "HO-3"),
    Rule("R8", "tree_debris", "Tree debris after a windstorm: in Iowa since the August 2024 bulletin, "
         "debris removal is owed even if the tree hit nothing; otherwise the HO-3 debris removal provision.",
         (Row("IA-ID-2024-08", state="IA", on_or_after=D("2024-08-15")),), "HO-3"),
    Rule("R9", "rental_reimbursement", "A rental car during repairs: if the policy carries LM PP 21, the endorsement sets the daily limit; otherwise the PAP transportation expenses.",
         (Row("LM PP 21", endorsement="LM PP 21"),), "PAP"),
    Rule("R10", "rideshare", "Driving for a rideshare company: if the policy carries LM PP 34, the endorsement covers it; otherwise the PAP exclusion for hire.",
         (Row("LM PP 34", endorsement="LM PP 34"),), "PAP"),
    Rule("R11", "deer_swerve", "Swerving to miss a deer is a collision under the PAP, period. "
         "No contact with the animal, no comprehensive.", (), "PAP"),
    Rule("R12", "spoilage", "Spoiled business stock: LM BP 12 if carried; otherwise the BOP exclusion.",
         (Row("LM BP 12", endorsement="LM BP 12"),), "BOP"),
)
R11_FIXED = load_yaml_rule(HERE / "R11.yaml")
R09 = Rulebook("r0.9", RULES, "twelve rules from the elicitation session")
R10 = R09.with_rule(R11_FIXED, "r1.0", "R11 gains the Wisconsin bulletin row; the bulletin wins the "
                    "disagreement (reviewed with Tom Brennan)")
R6_TIDY = Rule("R6", "roof_matching", "Matching undamaged roofing or siding: the OCI bulletin decides "
               "matching in Wisconsin; otherwise the HO-3 loss settlement condition.",
               (Row("WI-OCI-2023-07", state="WI", on_or_after=D("2023-07-01")),), "HO-3")
# Part H: two underwriters' readings of R1; part D: R8 without its start date
R1_A = Rule("R1", "water_backup", RULES[0].statement, (Row("HO 04 95", endorsement="HO 04 95"),), "HO-3",
            "Underwriter A")
R1_B = Rule("R1", "water_backup", "Water backup: only the 2024 edition of HO 04 95 adds it back; "
            "otherwise the HO-3 water exclusion.", (Row("HO 04 95", endorsement="HO 04 95 (2024-06)"),),
            "HO-3", "Underwriter B")
R8_NO_START = Rule("R8", "tree_debris", RULES[7].statement, (Row("IA-ID-2024-08", state="IA"),), "HO-3")
R11_BOOK = R10.with_rule(R6_TIDY, "r1.1", "R6 tidied: the two Wisconsin rows merged into one")


def pct(k, n):
    return 100.0 * k / n if n else 0.0


def evaluate(model: str, book: Rulebook | None, instructions=V22, corpus=CORPUS) -> dict:
    """Wrong-clause rate on the evaluation set, split by rule coverage, probe adherence
    and prompt tokens, for one model and one Rulebook (or none)."""
    cfg = PipelineConfig(True, model, instructions, book)
    covered_topics = set((book or R10).topics())
    ev = run_set(cfg, corpus.eval)
    pr = run_set(cfg, probes(book or R10, corpus))
    cov = [t for t in ev if t.det.topic in covered_topics]
    unc = [t for t in ev if t.det.topic not in covered_topics]
    return {
        "traces": ev, "probe_traces": pr,
        "wrong": sum(t.wrong for t in ev), "n": len(ev),
        "covered_wrong": sum(t.wrong for t in cov), "covered": len(cov),
        "uncovered_wrong": sum(t.wrong for t in unc), "uncovered": len(unc),
        "probes_ok": sum(not t.wrong for t in pr), "probes": len(pr),
        "prompt_tokens": sum(t.rendered.tokens for t in ev) / len(ev),
        "wrong_ids": [t.det.determination_id for t in ev if t.wrong],
    }


def conflicts_line(c) -> str:
    d = c.det
    return f"{d.determination_id} ({d.state}, {d.loss}): rule says {c.rule_says}, documents say {c.documents_say}"


def main() -> None:
    out: list[str] = []
    p = out.append
    ev = CORPUS.eval

    p("1. Elicited rules (r0.9)")
    covered = 0
    for r in R09.rules:
        n = sum(d.topic == r.topic for d in ev)
        covered += n
        p(f"  {r.rule_id:<4} {r.topic:<21} rows {len(r.rows)}  default {r.default:<4}  eval claims {n:>3}")
    p(f"  topics covered: {len(R09.topics())}; eval claims on covered topics: {covered} of {len(ev)}")
    p("")

    p("2. Expert review against the documents (500 reviewed determinations)")
    rows = []
    for book in (R09, R10):
        res = check(book, CORPUS)
        for rc in res:
            rows.append([book.version, rc.rule.rule_id, rc.rule.topic, rc.checked, rc.agree])
        if book is R09:
            for rc in res:
                if rc.conflicts:
                    p(f"  {book.version} {rc.rule.rule_id} {rc.rule.topic}: {rc.agree}/{rc.checked} agree")
                    for c in rc.conflicts:
                        p(f"    {conflicts_line(c)}")
    for book in (R09, R10):
        res = check(book, CORPUS)
        tot, agree = sum(r.checked for r in res), sum(r.agree for r in res)
        p(f"  {book.version}: {agree}/{tot} agree; rules with conflicts: {sum(bool(r.conflicts) for r in res)}")
    with open(HERE / "review.csv", "w", newline="") as f:
        csv.writer(f).writerows([["rulebook", "rule", "topic", "checked", "agree"]] + rows)
    p("")

    p("3. Failure injection: r0.9 ships with R11 as first stated")
    deer = [d for d in CORPUS.determinations if d.topic == "deer_swerve"]
    bull = [d for d in deer if CORPUS.governing(d).doc_id == "WI-OCI-2024-11"]
    p(f"  deer-swerve claims (1-500): {len(deer)}; governed by WI-OCI-2024-11: {len(bull)}")
    p("  model      rulebook   deer (all)  deer (bulletin)  eval 200")
    inj = [["model", "rulebook", "deer_all_wrong_pct", "deer_bulletin_wrong_pct", "eval_wrong_pct"]]
    for m in MODELS:
        for name, book in (("none", None), ("r0.9", R09), ("r1.0", R10)):
            cfg = PipelineConfig(True, m, V22, book)
            dw = run_set(cfg, deer)
            bw = [t for t in dw if t.det in bull]
            e = evaluate(m, book)
            inj.append([m, name, round(pct(sum(t.wrong for t in dw), len(dw)), 1),
                        round(pct(sum(t.wrong for t in bw), len(bw)), 1), round(pct(e['wrong'], e['n']), 1)])
            p(f"  {m:<10} {name:<8}{pct(sum(t.wrong for t in dw), len(dw)):>12.1f}%"
              f"{pct(sum(t.wrong for t in bw), len(bw)):>16.1f}%{pct(e['wrong'], e['n']):>9.1f}%")
    with open(HERE / "injection.csv", "w", newline="") as f:
        csv.writer(f).writerows(inj)
    d30 = CORPUS.by_id["DET-0030"]
    p(f"  example: DET-0030 {d30.name}, {d30.state}, loss {d30.loss}; gold {CORPUS.governing(d30).doc_label}")
    for name, book in (("r0.9", R09), ("r1.0", R10)):
        t = run_set(PipelineConfig(True, "sim-large", V22, book), [d30])[0]
        p(f"    {name} cited {t.answer.citation}")
    p("")

    p("4. Adherence probes added to the eval (claims where a rule selects an exception)")
    pr = probes(R10, CORPUS)
    e0 = evaluate("sim-large", R10)
    p(f"  probes: {len(pr)} (claims 1-500); eval claims covered {e0['covered']}, uncovered {e0['uncovered']}")
    p("  model      rulebook   probe adherence  covered wrong  uncovered wrong  eval wrong  prompt tok")
    prow, adh = [], [["model", "rulebook", "probe_adherence_pct", "covered_wrong_pct", "uncovered_wrong_pct",
                      "eval_wrong_pct", "prompt_tokens"]]
    for m in MODELS:
        for name, book in (("none", None), ("r1.0", R10)):
            e = evaluate(m, book)
            p(f"  {m:<10} {name:<8}{pct(e['probes_ok'], e['probes']):>17.1f}% {pct(e['covered_wrong'], e['covered']):>13.1f}% "
              f"{pct(e['uncovered_wrong'], e['uncovered']):>15.1f}% {pct(e['wrong'], e['n']):>10.1f}%  {e['prompt_tokens']:>10,.0f}")
            adh.append([m, name, round(pct(e['probes_ok'], e['probes']), 1), round(pct(e['covered_wrong'], e['covered']), 1),
                        round(pct(e['uncovered_wrong'], e['uncovered']), 1), round(pct(e['wrong'], e['n']), 1),
                        round(e['prompt_tokens'])])
            prow += [[m, name, t.det.determination_id, t.det.topic, int(not t.wrong)] for t in e["probe_traces"]]
    with open(HERE / "probes.csv", "w", newline="") as f:
        csv.writer(f).writerows([["model", "rulebook", "determination", "topic", "adheres"]] + prow)
    with open(HERE / "adherence.csv", "w", newline="") as f:
        csv.writer(f).writerows(adh)
    p("")

    p("5. Change control")
    for book in (R09, R10):
        p(f"  {book.version} fingerprint {book.fingerprint()}")
    for book in (R09, R10):
        p(f"  changelog {book.version}: {book.changelog}")

    text = "\n".join(out) + "\n"
    (HERE / "output.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
