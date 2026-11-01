"""Chapter 5 lab: retrieval against stuffed windows on three simulated models.

Run: python 04-code/ch05/lab.py
Writes output.txt (sections 1-4) and quality.csv, cost_quality.csv, families.csv and
injection.csv next to this file.

STAND-IN: every model is a deterministic SimulatedModel and the retriever is a
stand-in (see standin.py); the corpus and determinations are synthetic; prices,
latency constants and the cache terms are illustrative.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import standin as S                                              # noqa: E402
from contextlab.longctx import Kind, family_of, stuffed_spans, truncate  # noqa: E402
from contextlab.tokens import KNEES, WINDOWS                     # noqa: E402

STRATS = list(S.STRATEGIES)
CUT = 6_000                       # sim-small's rot onset (Chapter 4), the "stay under it" cap
FAMILIES = ["HO-3", "PAP", "BOAT", "BOP"]


def is_form(d) -> bool:
    return d.gold_kind == Kind.FORM.value


def pct(k: int, n: int) -> str:
    return f"{k / n:.1%}"


def _csv(name, header, rows):
    with open(HERE / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def runs(dets):
    """Every strategy on every model: {(model, strategy): (traces, cached tokens)}."""
    out = {}
    for m in S.MODELS:
        for s in STRATS:
            ts = S.run_all(S.pipeline(s, m), dets)
            out[m, s] = ts, S.replay_cache(ts)
    return out


def main(out=print):
    dets = S.CORPUS.eval
    form = [d.determination_id for d in dets if is_form(d)]
    R = runs(dets)

    # 1. quality --------------------------------------------------------------
    out("1. Quality: wrong-clause rate (rejected calls count as wrong)")
    out(f"{'model':<10} {'strategy':<13} {'wrong':>6} {'form clause':>12} {'endorse/bull':>13} "
        f"{'rejected':>9} {'prompt tokens':>14}")
    rows = []
    for m in S.MODELS:
        for s in STRATS:
            ts, _ = R[m, s]
            f = [t for t in ts if t.det.determination_id in form]
            e = [t for t in ts if t.det.determination_id not in form]
            rej = sum(t.answer.rejected for t in ts)
            tok = statistics.fmean(t.answer.tokens_in for t in ts)
            w, wf, we = S.wrong_count(ts), S.wrong_count(f), S.wrong_count(e)
            out(f"{m:<10} {s:<13} {pct(w, len(ts)):>6} {pct(wf, len(f)):>12} {pct(we, len(e)):>13} "
                f"{rej:>9} {tok:>14,.0f}")
            rows.append([m, s, w / len(ts), wf / len(f), we / len(e), rej, round(tok, 1)])
    _csv("quality.csv", ["model", "strategy", "wrong_rate", "form_clause", "endorse_bulletin",
                         "rejected", "mean_prompt_tokens"], rows)
    n_e = len(dets) - len(form)
    out(f"  form clause: {len(form)} determinations governed by a form clause (the needle); "
        f"endorse/bull: {n_e} governed by an endorsement or bulletin")
    ts, _ = R["sim-large", "stuff all"]
    bad = [t for t in ts if not t.right]
    na = sum(not S.Policy().applies(S.CORPUS.span(t.answer.cited_span_id), t.det).ok for t in bad)
    out(f"  sim-large, stuff all: {len(bad)} wrong, {na} of them cite a span that does not apply "
        "to the policy (other edition, state or endorsement)")

    # 2. cost and latency -------------------------------------------------------
    out("")
    out("2. Cost and latency per determination (same runs; simulated prompt cache)")
    out(f"{'model':<10} {'strategy':<13} {'wrong':>6} {'list $':>9} {'cached $':>9} {'cache hit':>9} "
        f"{'latency s':>9} {'cached s':>9}")
    rows, summ = [], {}
    for m in S.MODELS:
        for s in STRATS:
            ts, cached = R[m, s]
            if all(t.answer.rejected for t in ts):
                out(f"{m:<10} {s:<13} {'':>14}all calls rejected: prompt over the window")
                rows.append([m, s, 1.0, "", "", "", "", ""])
                continue
            a = [t.answer for t in ts]
            lst = statistics.fmean(S.trace_cost(t) for t in ts)
            cch = statistics.fmean(S.trace_cost(t, c) for t, c in zip(ts, cached))
            hit = sum(cached) / sum(x.tokens_in for x in a)
            lat = statistics.fmean(S.trace_latency(t) for t in ts)
            lc = statistics.fmean(S.trace_latency(t, c) for t, c in zip(ts, cached))
            w = S.wrong_count(ts) / len(ts)
            summ[m, s] = lst, cch, hit
            out(f"{m:<10} {s:<13} {w:>6.1%} {lst:>9.5f} {cch:>9.5f} {hit:>9.0%} {lat:>9.2f} {lc:>9.2f}")
            rows.append([m, s, w, round(lst, 6), round(cch, 6), round(hit, 4), round(lat, 3), round(lc, 3)])
    _csv("cost_quality.csv", ["model", "strategy", "wrong_rate", "list_usd", "cached_usd", "cache_hit",
                              "latency_s", "cached_latency_s"], rows)
    (la, ca, hit), (lr, cr, _) = summ["sim-large", "stuff all"], summ["sim-large", "retrieve"]
    out(f"  sim-large: stuff all costs {la / lr:.0f}x retrieval at list price and {ca / cr:.1f}x "
        f"with the cache ({hit:.0%} of input tokens cached)")

    # 3. by product family -----------------------------------------------------
    out("")
    out("3. By product family: size, versions, and wrong-clause rate per strategy")
    spans = S.CORPUS.spans
    bull = [s for s in spans if s.kind == Kind.BULLETIN]
    out(f"  bulletins: {len({s.provenance.doc_id for s in bull})} across "
        f"{len({s.provenance.jurisdiction for s in bull})} states, {sum(s.tokens for s in bull):,} tokens, "
        "stuffed with every family")
    out(f"{'family':<6} {'n':>4} {'docs':>5} {'editions':>9} {'text tok':>9} {'model':<10} "
        f"{'retrieve':>9} {'family':>7} {'hybrid':>7} {'stuff ok?':>10}")
    rows, verdict = [], {}
    for fam in FAMILIES:
        own = [s for s in spans if s.kind != Kind.BULLETIN and family_of(s) == fam]
        docs = {(s.provenance.doc_id, s.provenance.version) for s in own}
        eds = {s.provenance.version for s in own if s.kind == Kind.FORM}
        tok = sum(s.tokens for s in own)
        ids = {d.determination_id for d in dets if d.family == fam}
        n = len(ids)
        ok_all = True
        for m in S.MODELS:
            k = {s: sum(not t.right for t in R[m, s][0] if t.det.determination_id in ids)
                 for s in ("retrieve", "stuff family", "hybrid")}
            ok = k["stuff family"] - k["retrieve"] <= max(1, 0.02 * n)
            ok_all &= ok
            out(f"{fam:<6} {n:>4} {len(docs):>5} {len(eds):>9} {tok:>9,} {m:<10} {pct(k['retrieve'], n):>9} "
                f"{pct(k['stuff family'], n):>7} {pct(k['hybrid'], n):>7} {'yes' if ok else 'no':>10}")
            rows.append([fam, n, len(docs), len(eds), tok, m, k["retrieve"], k["stuff family"], k["hybrid"], ok])
        verdict[fam] = ok_all
    _csv("families.csv", ["family", "n", "docs", "editions", "text_tokens", "model", "retrieve_wrong",
                          "family_wrong", "hybrid_wrong", "stuff_ok"], rows)
    out("  stuff ok? = stuffing the family is wrong on at most max(1 claim, 2 points) more claims than retrieval")
    out("  decision (stuff the family only if it holds on all three models):")
    for fam in FAMILIES:
        out(f"    {fam:<5} -> {'whole-family context' if verdict[fam] else 'stay retrieved (or hybrid)'}")

    # 4. failure injection: the family cut at the rot onset ----------------------
    out("")
    out(f"4. Failure injection: HO-3 family stuffed, cut at {CUT:,} tokens (sim-small rot onset {CUT:,})")
    ho = [d for d in dets if d.family == "HO-3"]
    fam_spans = stuffed_spans(S.CORPUS, ho[0], "family")
    kept, dropped = truncate(fam_spans, CUT)
    hb = sum(s.kind == Kind.BULLETIN for s in dropped)
    he = sum(s.kind == Kind.ENDORSEMENT for s in dropped)
    out(f"  HO-3 family: {len(fam_spans)} spans, {sum(s.tokens for s in fam_spans):,} "
        f"tokens of text; the cut keeps {len(kept)} and drops {hb} bulletin spans, {he} endorsement spans")
    keep_ids = {s.span_id for s in kept}
    gone = [d for d in ho if d.gold_span_id not in keep_ids]
    eb = [d for d in ho if not is_form(d)]
    out(f"  gold clause cut away for {len(gone)} of {len(ho)} HO-3 determinations "
        f"({len(eb)} are governed by an endorsement or bulletin)")
    out(f"{'model':<10} {'variant':<26} {'HO-3 wrong':>10} {'endorse/bull':>13} {'prompt tok':>11}")
    eb_ids = {d.determination_id for d in eb}
    rows = []
    variants = [("stuff family (full)", dict(**S.STRATEGIES["stuff family"])),
                (f"stuff family, cut {CUT:,}", dict(**S.STRATEGIES["stuff family"], stuff_cap=CUT)),
                ("hybrid (fix)", dict(**S.STRATEGIES["hybrid"])),
                ("retrieve", dict(**S.STRATEGIES["retrieve"]))]
    traces = {}
    for m in S.MODELS:
        for name, cfg in variants:
            ts = S.run_all(S.Pipeline().with_config(**cfg, model=m), ho)
            traces[m, name] = ts
            w = S.wrong_count(ts)
            we = sum(not t.right for t in ts if t.det.determination_id in eb_ids)
            tok = statistics.fmean(t.answer.tokens_in for t in ts)
            out(f"{m:<10} {name:<26} {pct(w, len(ts)):>10} {pct(we, len(eb)):>13} {tok:>11,.0f}")
            rows.append([m, name, w / len(ts), we / len(eb), round(tok, 1)])
    _csv("injection.csv", ["model", "variant", "ho3_wrong", "endorse_bulletin_wrong", "mean_prompt_tokens"], rows)
    d = S.CORPUS.det("DET-0026")
    g = S.CORPUS.span(d.gold_span_id)
    out("")
    out(f"example {d.determination_id} ({d.holder}, {d.state}, {d.form}, endorsements "
        f"{', '.join(d.endorsements)}, topic {d.topic})")
    out(f"  gold:   {g.provenance.label} ({g.kind.value})")
    tc = traces["sim-large", f"stuff family, cut {CUT:,}"][ho.index(d)]
    th = traces["sim-large", "hybrid (fix)"][ho.index(d)]
    why = "gold cut: past the 6,000-token stuffing cap" if not tc.gold_in_packet else "gold in packet"
    c = S.CORPUS.span(tc.answer.cited_span_id).provenance.citation
    out(f"  cut:    cited {c} ({tc.answer.tokens_in:,} tokens; {why})")
    c = S.CORPUS.span(th.answer.cited_span_id).provenance.citation
    ndoc = len({(s.provenance.doc_id, s.provenance.version) for s in th.packet})
    out(f"  hybrid: cited {c} ({th.answer.tokens_in:,} tokens, {len(th.packet)} spans from {ndoc} documents)")
    _csv("knees.csv", ["model", "knee_prompt_tokens", "window"], [[m, KNEES[m], WINDOWS[m]] for m in S.MODELS])


if __name__ == "__main__":
    lines = []
    main(lines.append)
    (HERE / "output.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
