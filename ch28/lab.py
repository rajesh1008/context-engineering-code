"""Chapter 28 lab: span-level citations and citation precision on 300 determinations.

Run from 04-code:  python3 ch28/lab.py
Stand-in models throughout: the writer is lakeshore28.py, the verifier is
contextlab.citations.verify. Numbers are properties of these stand-ins.
"""
import json
import os
import re
import sys
from collections import Counter

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.dirname(os.path.abspath(__file__))]
from contextlab.citations import (CitationChecker, Manifest, mark, parse, precision_recall,
                                  strip, verify, VERSION)
import lakeshore28 as L

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
DETS = L.build()
TH = 0.6
RESULTS = {}


def toks(s):
    return len(re.findall(r"\w+|[^\w\s]", s))


def gold_labels(det, fmt, order):
    """Per sentence: needs a source, pointers written, pointers that truly support."""
    man, labels = Manifest(order), []
    for s in det.sentences:
        text = L.render(s, det.packet, fmt)
        ptrs = parse(text)
        sup = 0
        for k, ref in ptrs:
            cands = man.resolve(k, ref)
            sup += any(c in s.support for c in cands)
        labels.append(dict(needs=s.kind != "note", total_cites=len(ptrs), supported_cites=sup,
                           text=text, sentence=s, ptrs=ptrs, man=man))
    return labels


def measured(det, fmt, order, checker, in_force=None):
    man = Manifest(order)
    texts = [L.render(s, det.packet, fmt) for s in det.sentences]
    return checker.check(texts, man, in_force), texts


def run_format(fmt, reorder=False, th=TH, reattribute=False):
    chk = CitationChecker(threshold=th, reattribute=reattribute)
    g_all, m_all, amb, cand = [], [], 0, []
    for d in DETS:
        order = L.rerank(d.packet, d.det_id) if reorder else d.packet
        g = gold_labels(d, fmt, order)
        g_all += g
        fs, _ = measured(d, fmt, order, chk)
        for f, lab in zip(fs, g):
            ok = f.status in ("supported", "reattributed")
            m_all.append(dict(needs=lab["needs"], total_cites=lab["total_cites"],
                              supported_cites=lab["total_cites"] if ok else 0))
            for k, ref in lab["ptrs"]:
                n = len(lab["man"].resolve(k, ref))
                cand.append(n)
                amb += n > 1
    gp, gr = precision_recall(g_all)
    mp, mr = precision_recall(m_all)
    return dict(gold_p=gp, gold_r=gr, meas_p=mp, meas_r=mr, ambiguous=amb, pointers=len(cand),
                mean_cands=sum(cand) / max(1, len(cand)))


def section(t):
    print(f"\n== {t}")


section("1. The 300 determinations")
sents = [s for d in DETS for s in d.sentences]
kinds = Counter(s.kind for s in sents)
print(f"determinations {len(DETS)}  sentences {len(sents)}  claim sentences "
      f"{sum(v for k, v in kinds.items() if k != 'note')}  notes {kinds['note']}")
print("by kind:", dict(kinds))
print("by claim type:", dict(Counter(d.ctype for d in DETS)))
print("packet size:", dict(Counter(len(d.packet) for d in DETS)))
print("stand-in mistakes:", dict(Counter(s.mistake for s in sents if s.mistake)))
RESULTS["inventory"] = dict(dets=len(DETS), sentences=len(sents), kinds=dict(kinds),
                            mistakes=dict(Counter(s.mistake for s in sents if s.mistake)))

section("2. Four citation formats, same sentences")
print(f"{'format':8} {'pointers':>8} {'cands/ptr':>9} {'ambiguous':>9} {'tokens/ptr':>10} "
      f"{'gold P':>7} {'checked P':>9}")
RESULTS["formats"] = {}
emily = DETS[187]
for fmt in ("doc", "page", "index", "handle"):
    r = run_format(fmt)
    tk = []
    for d in DETS:
        for s in d.sentences:
            if s.pointers and s.kind != "note":
                tk.append((toks(L.render(s, d.packet, fmt)) - toks(s.text)) / len(set(s.pointers)))
    r["tokens_per_ptr"] = sum(tk) / len(tk)
    RESULTS["formats"][fmt] = r
    print(f"{fmt:8} {r['pointers']:8d} {r['mean_cands']:9.2f} {r['ambiguous']:9d} "
          f"{r['tokens_per_ptr']:10.1f} {r['gold_p']:7.1%} {r['meas_p']:9.1%}")
print("example sentence in each format (DET28-0188, Emily Carter's claim):")
for fmt in ("doc", "page", "index", "handle"):
    print(f"  {fmt:6} {L.render(emily.sentences[3], emily.packet, fmt)}")

section("3. Calibrating the stand-in verifier against gold (handle format)")
rows = []
chk0 = CitationChecker(threshold=0.0, reattribute=False)
for d in DETS:
    man = Manifest(d.packet)
    for s in d.sentences:
        if s.kind == "note" or not s.pointers:
            continue
        f = chk0.check_sentence(L.render(s, d.packet, "handle"), man)
        ok = bool(s.support) and set(s.pointers) <= set(s.support)
        rows.append((f.score, ok, s.mistake))
sup = [x for x, g, _ in rows if g]
uns = [x for x, g, _ in rows if not g]
print(f"cited sentences scored {len(rows)}  gold supported {len(sup)}  gold unsupported {len(uns)}")
print(f"{'thresh':>6} {'measured P':>10} {'flags':>6} {'flag prec':>9} {'flag recall':>11}")
sweep = []
for th in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
    flagged = [(x, g) for x, g, _ in rows if x < th]
    tp = sum(1 for x, g in flagged if not g)
    mp = sum(1 for x, g, _ in rows if x >= th) / len(rows)
    fp_ = tp / len(flagged) if flagged else float("nan")
    fr = tp / len(uns)
    sweep.append(dict(th=th, meas_p=mp, flags=len(flagged), flag_prec=fp_, flag_rec=fr))
    print(f"{th:6.2f} {mp:10.1%} {len(flagged):6d} {fp_:9.1%} {fr:11.1%}")
gold_p = len(sup) / len(rows)
print(f"gold share of cited sentences supported: {gold_p:.1%}")
missed = Counter(m for x, g, m in rows if not g and x >= TH)
false = Counter(m or "none" for x, g, m in rows if g and x < TH)
print(f"at {TH}: unsupported sentences the verifier passes {dict(missed)}; supported it flags {dict(false)}")
RESULTS["calibration"] = dict(sweep=sweep, gold_p=gold_p, sup=sup, uns=uns, missed=dict(missed))

section("4. First measurement: Lakeshore as it ran (page format) and with v6.4 (handles)")
before = run_format("page")
after = run_format("handle", reattribute=True)
chk = CitationChecker(threshold=TH)
st, moved_ok, moved_bad, flags_per_det, shown = Counter(), 0, 0, [], []
for d in DETS:
    fs, _ = measured(d, "handle", d.packet, chk)
    flags_per_det.append(sum(f.status in ("unverified", "missing") for f in fs))
    for f, s in zip(fs, d.sentences):
        st[f.status] += 1
        if s.kind != "note":
            final = [f.moved_to] if f.status == "reattributed" else [e.handle for e in f.resolved]
            good = bool(s.support) and set(final) <= {e.handle for e in s.support}
            shown.append((f.status in ("supported", "reattributed"), good))
        if f.status == "reattributed":
            if any(e.handle == f.moved_to for e in s.support):
                moved_ok += 1
            else:
                moved_bad += 1
gold_after = run_format("handle")
ver = [g for v, g in shown if v]
res_p = sum(ver) / len(ver)
res_r = sum(1 for v, g in shown if v and g) / len(shown)
wrong_unflagged = sum(1 for v, g in shown if v and not g)
print(f"page format, checked at page level   precision {before['meas_p']:.1%}  recall {before['meas_r']:.1%}")
print(f"page format, gold at span level      precision {before['gold_p']:.1%}  recall {before['gold_r']:.1%}")
print(f"handle format, gold                  precision {gold_after['gold_p']:.1%}  recall {gold_after['gold_r']:.1%}")
print(f"handle format, checked (v{VERSION})       precision {after['meas_p']:.1%}  recall {after['meas_r']:.1%}")
print(f"after v{VERSION}, what the adjuster sees as verified: precision {res_p:.1%} "
      f"({wrong_unflagged} wrong of {len(ver)}), recall {res_r:.1%} of {len(shown)} claim sentences")
print("sentence statuses:", dict(st))
print(f"re-attributions to a gold span {moved_ok}, to a wrong span {moved_bad}")
fd = Counter(flags_per_det)
print("flagged sentences per determination:", dict(sorted(fd.items())),
      f"mean {sum(flags_per_det) / len(DETS):.2f}")
RESULTS["first"] = dict(res_p=res_p, res_r=res_r, wrong_unflagged=wrong_unflagged, before=before, after=after, gold_after=gold_after, statuses=dict(st),
                        moved_ok=moved_ok, moved_bad=moved_bad, flags_per_det=dict(fd))

section("5. Failure injection: a reranker reorders spans after indices were assigned")
RESULTS["reorder"] = {}
for fmt in ("index", "handle"):
    a, b = run_format(fmt), run_format(fmt, reorder=True)
    RESULTS["reorder"][fmt] = dict(before=a, after=b)
    print(f"{fmt:6} gold P {a['gold_p']:6.1%} -> {b['gold_p']:6.1%}   checked P "
          f"{a['meas_p']:6.1%} -> {b['meas_p']:6.1%}")
moved = sum(1 for d in DETS if L.rerank(d.packet, d.det_id) != d.packet)
print(f"determinations whose packet order changed: {moved}/{len(DETS)}")

section("6. Right citation, wrong reasoning: which check catches each wrong conclusion")
RULE = {"backup_plain": ("HO-3", 7, 1), "mould_seepage": ("HO-3", 6, 3), "frozen_pipe": ("HO-3", 6, 1),
        "mould_hidden": ("HO-3", 8, 2), "hail_interior": ("HO-3", 5, 2), "backup_endorsed": ("HO 04 95", 1, 2)}
catch, rows6 = {}, []
for d in DETS:
    fs, _ = measured(d, "handle", d.packet, CitationChecker(threshold=TH, reattribute=False))
    for f, s in zip(fs, d.sentences):
        if s.kind != "conclusion" or not s.mistake:
            continue
        gov = L._resolve(RULE[d.ctype], d.ho3, d.e495, None, None)
        hits = {"attribution": f.status != "supported",
                "conclusion check": d.written_decision != d.decision,
                "governing check": gov not in s.pointers}
        caught = [k for k, v in hits.items() if v]
        c = catch.setdefault(s.mistake, Counter())
        c["n"] += 1
        for k in caught:
            c[k] += 1
        c["none"] += not caught
        rows6.append((d.det_id, s.mistake, caught, s.text))
print(f"{'mistake':10} {'n':>3} {'attribution':>11} {'conclusion':>10} {'governing':>9} {'none':>5}")
for m, c in catch.items():
    print(f"{m:10} {c['n']:3d} {c['attribution']:11d} {c['conclusion check']:10d} "
          f"{c['governing check']:9d} {c['none']:5d}")
for r in rows6:
    if not r[2]:
        print("  adjuster only:", r[0], r[3])
        break
RESULTS["reasoning"] = {m: dict(c) for m, c in catch.items()}

section("7. Sidebar: the display that resolves (document, page) to the latest edition")
latest = {"HO-3": "2024-06", "HO 04 95": "2024-06"}
lib = L.library()
shown_wrong, total, dets_hit = 0, 0, set()
for d in DETS:
    for s in d.sentences:
        for e in s.pointers:
            if e.doc not in latest:
                continue
            total += 1
            page = [x.text for x in lib if x.doc == e.doc and x.edition == latest[e.doc] and x.page == e.page]
            if e.text not in page:
                shown_wrong += 1
                dets_hit.add(d.det_id)
older = sum(1 for d in DETS if d.ho3 != "2024-06" or d.e495 == "2019-03")
print(f"form pointers {total}; page shown does not contain the cited span {shown_wrong} "
      f"({shown_wrong / total:.1%}); determinations affected {len(dets_hit)}; "
      f"determinations on an older edition {older}")
chk_ed = CitationChecker(threshold=TH)
ed_flags = 0
for d in DETS:
    fs, _ = measured(d, "handle", d.packet, chk_ed,
                     in_force={"HO-3": d.ho3, **({"HO 04 95": d.e495} if d.e495 else {})})
    ed_flags += sum(f.status == "edition" for f in fs)
print(f"edition check on handles: pointers naming an edition not on the policy {ed_flags} "
      "(stale pointers resolve to nothing and are flagged as unverified first)")
RESULTS["sidebar"] = dict(total=total, wrong=shown_wrong, dets=len(dets_hit), older=older)

section("8. The adjuster's view for one determination")
pick = next(d for d in DETS if d.ctype == "backup_endorsed" and any(s.mistake == "addition" for s in d.sentences))
for d in (emily, pick):
    fs, texts = measured(d, "handle", d.packet, CitationChecker(threshold=TH))
    print(f"{d.det_id} {d.claim_id} {d.ctype} HO-3 ({d.ho3}) HO 04 95 ({d.e495})")
    for f in fs:
        print(f"  {f.status:12} {f.score:4.2f}  {mark(f)[:150]}")
view = []
for d in (emily, pick):
    fs, texts = measured(d, "handle", d.packet, CitationChecker(threshold=TH))
    view.append(dict(det=d.det_id, claim=d.claim_id, sentences=[
        dict(text=strip(t), status=f.status, score=f.score,
             spans=[dict(doc=e.doc, edition=e.edition, page=e.page, n=e.n, text=e.text) for e in f.resolved])
        for f, t in zip(fs, texts)]))
with open(os.path.join(OUT, "view.json"), "w") as fh:
    json.dump(view, fh, indent=1)

section("9. Scoreboard rows added in Chapter 28")
print(f"citation precision, verified pointers (gold)      before {before['gold_p']:.1%} page level, "
      f"{gold_after['gold_p']:.1%} span level  after {res_p:.1%}")
print(f"citation recall (gold, span level)                before {gold_after['gold_r']:.1%}  after {res_r:.1%}")
print(f"flagged sentences per determination (after)       {sum(flags_per_det) / len(DETS):.2f}")
print("wrong-clause rate                                 unchanged (no change to retrieval or rules)")

with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(RESULTS, fh, indent=1, default=str)
