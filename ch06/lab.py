"""Chapter 6 lab: instruction versions, skills, failure injection and lint, and a release gate.

Writes output.txt (sections 1-4), versions.csv, skills.csv, injection.csv, gate.csv,
instructions_v2.2.md and CHANGELOG-instructions.md. Deterministic; no API key.
Run: python3 lab.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from standin import (CORPUS, PipelineConfig, cost, faithfulness, instruction_tokens, lint,  # noqa: E402
                     mean_tokens, rate, run_set, unsupported)
from contextlab.instructions import V10, V11, V12, V20, V21, V22, V23  # noqa: E402

EVAL = CORPUS.eval
MODELS = ("sim-large", "sim-small")
GATE_EVAL, GATE_HARD = 1.0, 2.0      # block a release if the rate rises by more points than this
out: list[str] = []


def p(line: str = "") -> None:
    out.append(line)


def runs(ins, model, dets=EVAL, **kw):
    return run_set(PipelineConfig(model=model, instructions=ins, **kw), dets)


def gate(base, cand, model) -> dict:
    be, ce = rate(runs(base, model)), rate(runs(cand, model))
    bh, ch = rate(runs(base, model, CORPUS.hard())), rate(runs(cand, model, CORPUS.hard()))
    ok = (ce - be) <= GATE_EVAL and (ch - bh) <= GATE_HARD
    return dict(model=model, eval_base=be, eval_cand=ce, hard_base=bh, hard_cand=ch, gate="pass" if ok else "BLOCK")


def main() -> None:
    # 1. versions
    p("1. Instruction versions (200 determinations)")
    p("version  model       wrong  faithful  unsupported  instr tok  prompt tok  fingerprint")
    rows = []
    for ins in (V10, V11, V12, V20):
        for m in MODELS:
            t = runs(ins, m)
            r = dict(version=ins.version, model=m, wrong=rate(t), faithful=faithfulness(t), unsupported=unsupported(t),
                     instr_tok=instruction_tokens(t), prompt_tok=mean_tokens(t), cost=cost(t),
                     fingerprint=ins.fingerprint())
            rows.append(r)
            p(f"{ins.version:<8} {m:<10} {r['wrong']:5.1f}%  {r['faithful']:8.3f}  {r['unsupported']:11d}  "
              f"{r['instr_tok']:9,.0f}  {r['prompt_tok']:10,.0f}  {r['fingerprint']}")
    p("unsupported = answers that add a sentence the cited provision does not support")
    write_csv("versions.csv", rows)

    # 2. skills
    p("")
    p("2. Skills: always loaded vs progressive disclosure (v2.0)")
    rows = []
    res = {}
    for label, ins in (("all", V20.bump("v2.0-all", disclosure="all")), ("progressive", V20)):
        for m in MODELS:
            t = runs(ins, m)
            sk = sum(len(x.packet.skills()) for x in t) / len(t)
            res[label] = instruction_tokens(t)
            rows.append(dict(disclosure=label, model=m, skills_per_claim=sk, instr_tok=instruction_tokens(t),
                             prompt_tok=mean_tokens(t), wrong=rate(t)))
            p(f"  {label:<12} {m:<10} skills/claim {sk:.2f}  instruction tokens {instruction_tokens(t):5,.0f}  "
              f"prompt {mean_tokens(t):6,.0f}  wrong {rate(t):4.1f}%")
    p(f"  progressive disclosure cuts instruction tokens by {100 * (1 - res['progressive'] / res['all']):.0f}%")
    write_csv("skills.csv", rows)

    # 3. failure injection and lint
    p("")
    p("3. Failure injection: a few-shot example that contradicts a bulletin")
    roof = [d for d in CORPUS.holdout if d.topic == "roof_matching"]
    roof_b = [d for d in roof if CORPUS.governing(d).doc_id.startswith("WI-OCI")]
    p("  version  model       eval 200  roof (all)  roof (WI bulletin)")
    rows = []
    for ins in (V20, V21, V22):
        for m in MODELS:
            r = dict(version=ins.version, model=m, eval=rate(runs(ins, m)), roof=rate(runs(ins, m, roof)),
                     roof_bulletin=rate(runs(ins, m, roof_b)))
            rows.append(r)
            p(f"  {ins.version:<8} {m:<10} {r['eval']:8.1f}%  {r['roof']:9.1f}%  {r['roof_bulletin']:17.1f}%")
    write_csv("injection.csv", rows)
    d = CORPUS.by_id["DET-0074"]
    p(f"  example: {d.determination_id} {d.name}, {d.state}, {d.form}, loss {d.loss}; "
      f"gold {CORPUS.governing(d).doc_label}")
    for ins in (V21, V22):
        t = run_set(PipelineConfig(model="sim-large", instructions=ins), [d])[0]
        p(f"    {ins.version} cited {t.answer.citation}")
    p("")
    for ins in (V21, V22):
        probs = lint(ins.examples)
        if probs:
            p(f"  lint {ins.version}:")
            for x in probs:
                p(f"    {x}")
        else:
            p(f"  lint {ins.version}: no problems")
    p(f"  v2.2 fingerprint {V22.fingerprint()} matches v2.0 ({V20.fingerprint()})"
      if V22.fingerprint() == V20.fingerprint() else "  v2.2 fingerprint differs from v2.0")

    # 4. versioning incident and gate
    p("")
    p("4. Versioning incident: v2.3 tidies the rules")
    p(f"  fingerprints: v2.2 {V22.fingerprint()}  v2.3 {V23.fingerprint()}")
    p("  diff v2.2 -> v2.3:")
    for ln in V22.diff(V23):
        p(f"    {ln}")
    rows = []
    for m in MODELS:
        g = gate(V22, V23, m)
        rows.append(dict(candidate="v2.3", **g))
        p(f"  {m:<10} eval {g['eval_base']:4.1f}% -> {g['eval_cand']:4.1f}%  "
          f"hold-out {g['hard_base']:4.1f}% -> {g['hard_cand']:4.1f}%  gate: {g['gate']}")
    blocked = all(r["gate"] == "BLOCK" for r in rows)
    p("  v2.3 is blocked; production stays on v2.2" if blocked else "  v2.3 passes the gate")
    write_csv("gate.csv", rows)

    (HERE / "instructions_v2.2.md").write_text(V22.render() + "\n")
    (HERE / "output.txt").write_text("\n".join(out) + "\n")
    write_changelog()
    print("\n".join(out))


def write_csv(name, rows):
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def write_changelog():
    """The changelog the release process keeps. Rates are filled from the runs above."""
    def pc(ins, m, dets=EVAL):
        return f"{rate(runs(ins, m, dets)):.1f}%"
    hard = CORPUS.hard()
    roof_b = [d for d in CORPUS.holdout if d.topic == "roof_matching"
              and CORPUS.governing(d).doc_id.startswith("WI-OCI")]
    L, S = "sim-large", "sim-small"
    ul = [unsupported(runs(v, m)) for v in (V11, V12) for m in (L, S)]
    text = f"""v2.3  BLOCKED by gate. Removed rule precedence as redundant.
      eval {pc(V22, L)} -> {pc(V23, L)} (large), {pc(V22, S)} -> {pc(V23, S)} (small); hold-out {pc(V22, L, hard)} -> {pc(V23, L, hard)}, {pc(V22, S, hard)} -> {pc(V23, S, hard)}.
      Reason recorded on rule precedence: only this rule states the order for every topic.
v2.2  Restored Example C (DET-0307, cites WI-OCI-2025-03). Lint added to release.
      Fingerprint {V22.fingerprint()} = v2.0. Known error: DET-0061 (Example A date boundary).
v2.1  Replaced Example C with an archived answer citing HO-3 Conditions D.5.
      Withdrawn: WI roof bulletin claims {pc(V21, L, roof_b)} wrong (large), {pc(V21, S, roof_b)} (small).
v2.0  Added Examples A, B, C and five topic skills, progressive disclosure.
      eval {pc(V12, L)} -> {pc(V20, L)} (large), {pc(V12, S)} -> {pc(V20, S)} (small). Skills: no measured effect; untested.
v1.2  Added rule quote_only. Motivated by DET-0042 (added a 30-day reporting limit).
      Unsupported answers {ul[0]} -> {ul[2]} (large), {ul[1]} -> {ul[3]} (small).
v1.1  Added rules precedence, applicability, cite_exact.
      Motivated by DET-0049, DET-0145, DET-0185: form cited where an endorsement governs.
      eval {pc(V10, L)} -> {pc(V11, L)} (large), {pc(V10, S)} -> {pc(V11, S)} (small).
v1.0  Role and task only. eval {pc(V10, L)} (large), {pc(V10, S)} (small).
"""
    (HERE / "CHANGELOG-instructions.md").write_text(
        "# Changelog: claims assistant instructions (contextlab/instructions.py)\n\n"
        "Newest first. Each entry records what changed, why, and the measured effect.\n\n"
        "```text\n" + text + "```\n")


if __name__ == "__main__":
    main()
