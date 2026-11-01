"""Chapter 1 lab: the same question, four assemblies (plus E and the fix).

python 04-code/ch01/lab.py   -> output.txt and assemblies.csv beside this file.
Stand-in model and fixtures: see standin.py (illustrative profile, not measured).
"""
from __future__ import annotations

import csv, sys, time
from pathlib import Path

import standin as S
from standin import Assembler, Kind

HERE = Path(__file__).resolve().parent
RULES = True
BUDGETS = [150, 300, 450, 600, 900, 1200, 2000]
pipe = policy = corpus = W = None


def setup(world: S.World) -> None:
    global pipe, policy, corpus, W
    W, corpus = world, world.corpus
    pipe, policy = S.Pipeline(world), S.Policy()


def packet(view, spans, rules) -> S.Packet:
    return S.Packet(view, list(spans), rules)


def a_prompt_only(det):
    """Instructions, claim header and question; no evidence."""
    return packet(det, [], False), det


def b_top8(det, rules=False):
    """Top eight by keyword ("stuff it all in"): no filter, no rules."""
    hits = pipe.retriever.search(pipe.keyword(det), k=8)
    return packet(det, [s for s, _ in hits], RULES if rules else False), det


def c_engineered(det, budget=900):
    """Filter by policy, add claim history, rules, greedy budget."""
    view = det
    hits = pipe.retriever.search(pipe.query(view), k=40)
    kept = [s for s, _ in hits if policy.applies(s, view).ok] + corpus.history_for(view)
    spans, used = [], 0
    for s in kept:
        if used + s.tokens <= budget:
            spans.append(s)
            used += s.tokens
    return packet(view, spans, RULES), view


def d_no_declarations(det):
    """Failure injection: the declarations call fails silently; endorsements = []."""
    blind = det.model_copy(update={"endorsements": []})
    pk, _ = c_engineered(blind)
    return pk, blind


def missing_endorsements(det, pk) -> list[str]:
    """The fix: every declared endorsement must be in the packet, or refuse it."""
    present = {s.provenance.label for s in pk.spans}
    return sorted(set(det.endorsements) - present)


def d_fixed(det):
    """D with the required-endorsement check: returns (packet or None if refused, packet)."""
    pk, blind = d_no_declarations(det)
    pk.stamp = True
    return (None if missing_endorsements(det, pk) else pk), pk


def e_everything(det):
    """Sidebar: every span in the corpus plus all three rules."""
    return packet(det, corpus.spans, RULES), det


def judge(det, pk):
    """Ask the model, then the citation checker: does the cited span govern this determination?"""
    a = W.model.answer(pk, W.cov[det.determination_id])
    if a.abstained:
        return "none", a
    return ("governing" if a.span.span_id == det.gold_span else "wrong"), a


def category(det, a) -> str:
    g = corpus.by_id[det.gold_span]
    s = a.span
    if s.provenance.doc_id == g.provenance.doc_id and s.provenance.superseded_by and s.topic == g.topic:
        return "superseded edition"
    return "other document"


ASSEMBLIES = [("A prompt only", a_prompt_only), ("B stuff it all in", b_top8),
              ("C engineered", c_engineered), ("D no declarations", d_no_declarations)]


def run_all(world: S.World) -> dict:
    setup(world)
    rows, res = [], {}
    def rec(name, det, pk, verdict=None, refused=False):
        if refused:
            v, cite, concl = "refused", "", "packet refused, adjuster alerted"
        else:
            v, a = verdict if verdict else judge(det, pk)
            cite, concl = (a.span.cite if a.span else ""), a.conclusion
        res.setdefault(name, []).append((det, v, pk.tokens))
        rows.append(dict(assembly=name, determination=det.determination_id, governs=det.governs,
                         result=v, tokens=pk.tokens, spans=len(pk.spans), cited=cite, answer=concl))
    for det in world.dets:
        for name, f in ASSEMBLIES:
            pk, _ = f(det)
            rec(name, det, pk)
        ok, pk = d_fixed(det)
        rec("D + fix", det, pk, refused=ok is None)
        rec("B + rules", det, b_top8(det, rules=True)[0])
        rec("B no header", det, Assembler("markdown").render(b_top8(det)[0], ""))
        pk, _ = c_engineered(det)
        pk.rules = False
        rec("C no rules", det, pk)
        for b in BUDGETS:
            rec(f"C budget {b}", det, c_engineered(det, budget=b)[0])
        rec("E send everything", det, e_everything(det)[0])
    return dict(rows=rows, res=res)


def summary(res, name):
    xs = res[name]
    n = len(xs)
    g = sum(v == "governing" for _, v, _ in xs)
    w = sum(v == "wrong" for _, v, _ in xs)
    na = sum(v == "none" for _, v, _ in xs)
    rf = sum(v == "refused" for _, v, _ in xs)
    tok = sum(t for _, _, t in xs) / n
    return g, w, na, rf, tok, n


def main():
    t0 = time.time()
    world = S.build_world()
    setup(world)
    out = []
    say = out.append
    det = world.dets[0]
    say(f"Question ({det.determination_id}, {det.holder}, {det.state}): {det.question}")
    say(f"Gold: {det.gold}")
    say("")
    for name, f in ASSEMBLIES:
        pk, _ = f(det)
        v, a = judge(det, pk)
        cite = a.span.cite if a.span else "none (abstained)"
        concl = a.conclusion if a.span else "-"
        say(f"{name:18} tokens={pk.tokens:5,} spans={len(pk.spans):2}  -> {concl} | {cite}")
    ok, pk = d_fixed(det)
    miss = missing_endorsements(det, pk)
    say("")
    say(f"Fix on D: missing declared endorsements -> {miss} -> "
        + ("packet refused, adjuster alerted" if ok is None else "packet passed"))
    pk_nh = Assembler("markdown").render(b_top8(det)[0], "")
    say(f"(B with an empty header: {judge(det, pk_nh)[0]}; E: {judge(det, e_everything(det)[0])[0]} at "
        f"{e_everything(det)[0].tokens:,} tokens)")

    R = run_all(world)
    res = R["res"]
    say("")
    say(f"{'assembly':18} {'governing':>9} {'wrong':>6} {'no answer':>9} {'tokens':>7}")
    for name in ["A prompt only", "B stuff it all in", "C engineered", "D no declarations", "D + fix"]:
        g, w, na, rf, tok, n = summary(res, name)
        extra = f"  refused={rf}" if rf else ""
        say(f"{name:18} {g / n:9.1%} {w / n:6.1%} {na / n:9.1%} {tok:7,.0f}{extra}")

    say("")
    say("budget   governing   wrong   no answer   tokens")
    for b in BUDGETS:
        g, w, na, rf, tok, n = summary(res, f"C budget {b}")
        say(f"{b:6,} {g / n:11.1%} {w / n:7.1%} {na / n:11.1%} {tok:8,.0f}")

    say("")
    say("breakdowns (verification: imports the lab without changing it)")
    byname = {k: {d.determination_id: v for d, v, _ in xs} for k, xs in res.items()}
    dets = world.dets
    bw = [d for d in dets if byname["B stuff it all in"][d.determination_id] == "wrong"]
    from collections import Counter
    say(f"B wrong {len(bw)}: by what governs " + ", ".join(f"{k} {v}" for k, v in
        sorted(Counter(d.governs for d in bw).items(), key=lambda t: -t[1])))
    cats = Counter()
    for d in bw:
        pk, _ = b_top8(d)
        cats[category(d, judge(d, pk)[1])] += 1
    say("B wrong by what it cited: " + ", ".join(f"{k} {v}" for k, v in cats.most_common()))
    cw = [d for d in dets if byname["C engineered"][d.determination_id] == "wrong"]
    inpk = sum(any(s.span_id == d.gold_span for s in c_engineered(d)[0].spans) for d in cw)
    say(f"C wrong {len(cw)}: governing span in packet but another chosen {inpk}; governing span not retrieved "
        f"{len(cw) - inpk}")
    dw = [d for d in dets if byname["D no declarations"][d.determination_id] == "wrong"]
    say(f"D wrong {len(dw)}: " + ", ".join(f"{k} {v}" for k, v in Counter(d.governs for d in dw).most_common()))
    for gv in ("form", "endorsement", "bulletin"):
        n = sum(d.governs == gv for d in dets)
        r = sum(d.governs == gv and byname["D no declarations"][d.determination_id] == "governing" for d in dets)
        say(f"  D right on {r} of {n} {gv}-governed")
    ref = [d for d in dets if byname["D + fix"][d.determination_id] == "refused"]
    say(f"D + fix refused {len(ref)}; D had answered {sum(byname['D no declarations'][d.determination_id] == 'governing' for d in ref)} of them correctly")
    say("")
    for name in ["E send everything", "B no header", "B + rules", "C no rules"]:
        g, w, na, rf, tok, n = summary(res, name)
        say(f"{name:18} {g / n:9.1%} {w / n:6.1%} {na / n:9.1%} {tok:7,.0f}")
    cn = sum(byname["C engineered"][d.determination_id] == "governing" and
             byname["C no rules"][d.determination_id] == "wrong" for d in dets)
    say(f"removing the rules from C changes {cn} of 500 determinations from governing to wrong")
    say(f"instruction span with rules: {S.INSTR + S.RULES_TOKENS} tokens; corpus {len(corpus.spans)} spans, "
        f"{sum(s.tokens for s in corpus.spans):,} tokens")
    say(f"lab time {time.time() - t0:.1f} s")
    text = "\n".join(out) + "\n"
    (HERE / "output.txt").write_text(text)
    with open(HERE / "assemblies.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(R["rows"][0]))
        w.writeheader()
        w.writerows(R["rows"])
    print(text, end="")


if __name__ == "__main__":
    main()
