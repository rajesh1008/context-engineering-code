"""Chapter 6 exploration: the listings and figures that go beyond lab.py.

Parts: A, B (v2.2 errors; out_AB.txt), C (ablation; out_CG.txt), D, E, G, F (example
selection, the deer-swerve injection, three models, the second gate; out_DG.txt), H (v1.0 vs
v1.1 and the quote-only rule; out_H.txt), J (skills on sim-small; out_J.txt), K (the
committed instruction file; out_K.txt), L (mismatched examples; out_L.txt), M (additions
that survive; out_M.txt), N (policy filter off; out_N.txt). Run lab.py first.
Run: python3 explore.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from standin import (CORPUS, PipelineConfig, cost, example_from, faithfulness, instruction_tokens,  # noqa: E402
                     lint, lookup, mean_tokens, pool, provision, rate, run_set, select_facts,
                     select_same_topic, unsupported, wrong)
from contextlab.instructions import SKILLS, V10, V11, V12, V20, V22, V22_NO_A, V23B  # noqa: E402

EVAL = CORPUS.eval
L, M, S = "sim-large", "sim-mid", "sim-small"


def runs(ins, model, dets=EVAL, **kw):
    return run_set(PipelineConfig(model=model, instructions=ins, **kw), dets)


def write(name: str, lines: list[str]) -> None:
    (HERE / name).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()


def part_ab() -> list[str]:
    out = ["A. Where the v2.2 errors on sim-large come from (eval 200)"]
    t22 = runs(V22, L)
    errs = [t for t in t22 if t.wrong]
    miss = [t for t in errs if not t.gold_in_packet]
    ex = [t for t in errs if t.gold_in_packet and any(
        e["topic"] == t.det.topic for e in t.packet.examples())]
    other = [t for t in errs if t not in miss and t not in ex]
    out.append(f"  retrieval miss (governing span not in the packet): {len(miss)}")
    out.append(f"  example on the claim's topic cites another document: {len(ex)} "
               f"{[t.det.determination_id for t in ex]}")
    out.append(f"  governing span in the packet, passed over: {len(other)} {[t.det.determination_id for t in other]}")
    out.append("")
    out.append("B. v2.2 sim-large errors on eval 200")
    for t in errs:
        g = t.gold
        out.append(f"  {t.det.determination_id} {t.det.state} {t.det.topic:<16} gold {g.doc_label} ({g.kind}); "
                   f"cited {t.answer.citation}; gold in packet {t.gold_in_packet}")
    w10 = {t.det.determination_id for t in runs(V10, L) if t.wrong}
    w22 = {t.det.determination_id for t in errs}
    out.append(f"  v1.0 wrong {len(w10)}, v2.2 wrong {len(w22)}, fixed {len(w10 - w22)}, "
               f"broken {len(w22 - w10)}, both {len(w10 & w22)}")
    out.append(f"  broken: {sorted(w22 - w10)}")
    return out


V12_EX = V12.bump("v1.2+ex", examples=V22.examples)
V12_SK = V12.bump("v1.2+sk", skills=SKILLS)


def part_c() -> list[str]:
    out = ["C. Ablation on v1.2 (rules), eval 200"]
    for label, ins in (("v1.2", V12), ("v1.2+ex", V12_EX), ("v1.2+sk", V12_SK), ("v2.0", V20),
                       ("v2.2-noA", V22_NO_A)):
        for m in (L, S):
            t = runs(ins, m)
            out.append(f"  {label:<9} {m:<10} wrong {rate(t):4.1f}%  faithful {faithfulness(t):.3f}  "
                       f"instr tok {instruction_tokens(t):4.0f}  fp {ins.fingerprint()}")
    return out


def part_dl() -> tuple[list[str], list[str]]:
    out = ["D. Example selection, v1.2 rules + skills, eval 200"]
    exs = pool()
    sel = {d.determination_id: select_same_topic(d, exs) for d in EVAL}
    fx = {d.determination_id: select_facts(d, exs) for d in EVAL}
    res = {}
    for m in (L, S):
        t = run_set(PipelineConfig(model=m, instructions=V12_SK, examples=sel), EVAL)
        res[m] = t
        out.append(f"  {'selected':<9} {m:<10} wrong {rate(t):4.1f}%  prompt tok {mean_tokens(t):6,.0f}")
    covered = sum(bool(sel[d.determination_id]) for d in EVAL)
    out.append(f"  topics in eval 200 with a pool example: {covered} of {len(EVAL)}")
    out.append(f"  pool examples: {len(exs)}; lint problems {len(lint(exs))}")
    mism = []
    for d in EVAL:
        e = sel[d.determination_id][0]
        eg = CORPUS.governing(CORPUS.by_id[e.source])
        if eg.doc_id != CORPUS.governing(d).doc_id:
            mism.append((d, e, eg))
    tl = res[L]
    wl = [t for t in tl if t.wrong]
    cites_ex = sum(t.answer.provision.doc_id == sel[t.det.determination_id][0].cites for t in wl)
    out.append(f"  sim-large selected: wrong {len(wl)}; example governed by a different document than the claim: "
               f"{len(mism)} of {len(EVAL)}; wrong answers citing the example's document: {cites_ex}")
    for m in (L, S):
        t = run_set(PipelineConfig(model=m, instructions=V12_SK, examples=fx), EVAL)
        n = sum(bool(v) for v in fx.values())
        out.append(f"  facts-matched {m:<10} wrong {rate(t):4.1f}%  claims with an example {n}")
    for m in (L, S):
        out.append(f"  {'none':<9} {m:<10} wrong {rate(runs(V12_SK, m)):4.1f}%")
        out.append(f"  {'fixed 3':<9} {m:<10} wrong {rate(runs(V20, m)):4.1f}%")
    lo = ["L. Claims whose selected example is governed by a different document (same-topic selection)"]
    for d, e, eg in mism:
        lo.append(f"  {d.determination_id} {d.topic:<20} {d.state} gold {CORPUS.governing(d).doc_label:<20} "
                  f"example {e.source} {e.state} cites {eg.doc_label}")
    lo.append(f"  total {len(mism)}")
    return out, lo


def part_e() -> list[str]:
    src = CORPUS.by_id["DET-0252"]
    good = example_from(src, "D")
    bad = example_from(src, "D", provision(src.form_id, src.form_version, src.topic))
    deer = [d for d in CORPUS.holdout if d.topic == "deer_swerve"]
    deer_b = [d for d in deer if CORPUS.governing(d).kind == "bulletin"]
    out = ["E. Second setting: deer swerve",
           f"  source {src.determination_id} {src.name} {src.state} {src.form} loss {src.loss} "
           f"gold {CORPUS.governing(src).doc_label}",
           f"  bad example cites {bad.text.rsplit('Governing provision: ', 1)[1].rstrip('.')}",
           f"  deer hold-out {len(deer)}, bulletin-governed {len(deer_b)}"]
    for label, ins in (("v2.2", V22), ("v2.2+D good", V22.bump("v2.2+D", examples=V22.examples + (good,))),
                       ("v2.2+D bad", V22.bump("v2.2+D", examples=V22.examples + (bad,)))):
        for m in (L, S):
            out.append(f"  {label:<12} {m:<10} deer all {rate(runs(ins, m, deer)):4.1f}%  "
                       f"deer WI bulletin {rate(runs(ins, m, deer_b)):4.1f}%")
    out.append(f"    lint: {lint([bad])}")
    return out


def part_g() -> list[str]:
    out = ["G. Three models"]
    for ins in (V10, V12, V22):
        for m in (L, M, S):
            t = runs(ins, m)
            out.append(f"  {ins.version} {m:<10} wrong {rate(t):4.1f}% faithful {faithfulness(t):.3f} "
                       f"unsupported {unsupported(t):3d} lookup {lookup(t):.1f} cost ${cost(t):.5f}")
    return out


def gate_line(base, cand, m) -> str:
    be, ce = rate(runs(base, m)), rate(runs(cand, m))
    bh, ch = rate(runs(base, m, CORPUS.hard())), rate(runs(cand, m, CORPUS.hard()))
    ok = (ce - be) <= 1.0 and (ch - bh) <= 2.0
    return (f"  {m:<10} eval {be:4.1f}% -> {ce:4.1f}%  hold-out {bh:4.1f}% -> {ch:4.1f}%  "
            f"gate {'pass' if ok else 'BLOCK'}")


def part_f() -> list[str]:
    out = ["F. Gate, second setting: v2.3b drops applicability"]
    out += [f"  {ln}" for ln in V22.diff(V23B)]
    out += [gate_line(V22, V23B, m) for m in (L, S)]
    return out


def part_n() -> list[str]:
    out = ["N. With the policy filter off: does the applicability rule matter? (eval 200)"]
    for m in (L, S):
        for ins in (V22, V23B):
            t = runs(ins, m, policy_filter=False)
            out.append(f"  filter off  {ins.version:<6} {m:<10} wrong {rate(t):4.1f}%")
    return out


def part_h() -> list[str]:
    out = ["H. v1.0 vs v1.1 on sim-large: three determinations motivated by the precedence rule"]
    t10 = {t.det.determination_id: t for t in runs(V10, L)}
    t11 = {t.det.determination_id: t for t in runs(V11, L)}
    fixed = [i for i in t10 if t10[i].wrong and not t11[i].wrong]
    out.append(f"  fixed by v1.1: {len(fixed)} {fixed}")
    for i in ("DET-0049", "DET-0145", "DET-0185"):
        d, g = CORPUS.by_id[i], CORPUS.governing(CORPUS.by_id[i])
        ends = [f"{e} ({v})" for e, v in d.endorsements]
        out.append(f"  {i} {d.topic} {d.state} {d.form} {ends} loss {d.loss} gold {g.doc_id}@{g.version}#{d.topic}")
        out.append(f"    v1.0: {t10[i].answer.citation}")
        out.append(f"    v1.1: {t11[i].answer.citation}")
    out.append("")
    out.append("H2. The quote_only rule on sim-small")
    d = CORPUS.by_id["DET-0042"]
    out.append(f"  {d.determination_id} {d.topic} {d.state}")
    for ins in (V11, V12):
        t = run_set(PipelineConfig(model=S, instructions=ins), [d])[0]
        out.append(f"  {ins.version} answer: {t.answer_text}")
    return out


def part_j() -> list[str]:
    out = ["J. sim-small: all skills loaded vs progressive, which determinations differ"]
    ta = {t.det.determination_id: t for t in runs(V20.bump("v2.0-all", disclosure="all"), S)}
    tp = {t.det.determination_id: t for t in runs(V20, S)}
    for i in ta:
        if ta[i].wrong != tp[i].wrong:
            d = ta[i].det
            out.append(f"  {i} {d.topic} {d.state} gold {ta[i].gold.doc_label}; all: {ta[i].answer.citation}; "
                       f"progressive: {tp[i].answer.citation}")
            out.append(f"    skills loaded progressive: {tp[i].packet.skills()}")
    nos = sum(not t.packet.skills() for t in tp.values())
    out.append(f"  claims with no skill loaded (progressive): {nos} of {len(tp)}")
    return out


def tail(s: str, n: int = 70) -> str:
    return s if len(s) <= n else "..." + s[-n:]


def part_k() -> list[str]:
    committed = (HERE / "instructions_v2.2.md").read_text()
    rendered = V22.render() + "\n"
    out = ["K. Is the committed instruction file the one the code renders?",
           f"  instructions_v2.2.md: matches V22.render(): {committed == rendered}"]
    edited = committed.replace("Ignore underwriting notes.",
                               "Ignore underwriting notes. Mention the deductible in every answer.")
    out.append(f"  hand-edited copy matches V22.render(): {edited == rendered}")
    for a, b in zip(rendered.splitlines(), edited.splitlines()):
        if a != b:
            out.append(f"  - {tail(a, 75)}")
            out.append(f"  + {tail(b, 75)}")
    return out


def part_m() -> list[str]:
    out = ["M. Unsupported additions that survive quote_only, sim-small, v1.2 and v2.2"]
    for ins in (V12, V22):
        t = runs(ins, S)
        ids = [x.det.determination_id for x in t if x.answer.addition]
        out.append(f"  {ins.version}: {len(ids)} answers with an unsupported addition: {ids}")
    t = run_set(PipelineConfig(model=S, instructions=V22), [CORPUS.by_id["DET-0012"]])[0]
    out.append(f"  DET-0012 {t.det.topic} {t.det.state}: {t.answer_text}")
    return out


def main() -> None:
    write("out_AB.txt", part_ab())
    write("out_CG.txt", part_c())
    d, lo = part_dl()
    write("out_DG.txt", d + [""] + part_e() + [""] + part_g() + [""] + part_f())
    write("out_L.txt", lo)
    write("out_H.txt", part_h())
    write("out_J.txt", part_j())
    write("out_K.txt", part_k())
    write("out_M.txt", part_m())
    write("out_N.txt", part_n())


if __name__ == "__main__":
    main()
