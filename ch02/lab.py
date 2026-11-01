"""Chapter 2 lab: the baseline, the stale form pointer, and the fix (reconcile, then filter).

Writes the chapter's main output to stdout (saved as output.txt). Every model in this lab is a
deterministic stand-in; see standin.py and README.md.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from contextlab.models import Determination                       # noqa: E402
from contextlab.pipeline import Pipeline, claim_facts             # noqa: E402
from contextlab import scoreboard as sb                           # noqa: E402
from standin import load_corpus                                   # noqa: E402

STALE = {"HO-3 (2024-06)": "HO-3 (2022-01)"}     # the pointer that never moved
corpus = load_corpus()
base = Pipeline(corpus=corpus)

def stale_view(det: Determination) -> Determination:
    return det.model_copy(update={"form": STALE.get(det.form, det.form)})

def reconcile(det: Determination) -> tuple[Determination, str | None]:
    """Take the form edition from the policy record; report any mismatch."""
    record = corpus.policies[det.policy_number]["form"]
    if record != det.form:
        return det.model_copy(update={"form": record}), f"{det.form} -> {record}"
    return det, None


fixed = base.with_config(policy_filter=True)
HEADER = "run                                      wrong    rate     95% interval"


def run_all(pipe: Pipeline, dets, view=None):
    return [pipe.run(view(d) if view else d) for d in dets]


def fix_view(d: Determination) -> Determination:
    return reconcile(stale_view(d))[0]


def blank_view(d: Determination) -> Determination:
    return d.model_copy(update={"endorsements": []})


def cited_line(t) -> str:
    s = corpus.span(t.answer.cited_span_id) if t.answer.cited_span_id else None
    verdict = s.meta.get("verdict", "") if s else "no answer"
    return f"{verdict} | {t.answer.citation}"


def main() -> None:
    dets = corpus.eval_set(200)
    by_id = {d.determination_id: d for d in dets}
    d1, d8 = by_id["DET-0001"], by_id["DET-0008"]

    # ---- the packet the baseline sends, and what the filter does to it
    tb, tf = base.run(d1), fixed.run(d1)
    print("== DET-0001, baseline and fixed packets")
    print("baseline", tb.packet.span_ids())
    print("fixed   ", tf.packet.span_ids())
    print("removed", [x for x in tb.packet.span_ids() if x not in tf.packet.span_ids()])
    print("added", [x for x in tf.packet.span_ids() if x not in tb.packet.span_ids()])
    print(f"tokens {tb.rendered.tokens} {tf.rendered.tokens} cited {tb.answer.citation} | {tf.answer.citation}")
    print()
    print(claim_facts(d1))
    print()
    print(tf.answer.text)
    print()

    runs = {
        "baseline": run_all(base, dets),
        "stale pointer": run_all(base, dets, stale_view),
        "stale + filter (no reconcile)": run_all(fixed, dets, stale_view),
        "stale + reconcile (no filter)": run_all(base, dets, lambda d: reconcile(stale_view(d))[0]),
        "stale + reconcile + filter": run_all(fixed, dets, fix_view),
    }

    # ---- packet hygiene
    print("== packet hygiene")
    for name, ts in (("baseline", runs["baseline"]), ("fixed", runs["stale + reconcile + filter"])):
        sup = sum(s.provenance.superseded_by is not None for t in ts for s in t.packet.spans) / len(ts)
        inapp = sum(not base.policy.applies(s, t.det).ok for t in ts for s in t.packet.spans) / len(ts)
        notes = sum(any(s.kind.value == "uw_note" for s in t.packet.spans) for t in ts)
        print(f"{name:<10} superseded/packet {sup:.2f} inapplicable/packet {inapp:.2f} "
              f"packets with uw note {notes}/{len(ts)}")
    print()

    print(HEADER)
    print(sb.run_row("baseline", sb.score(runs["baseline"])))
    print()

    # ---- paired comparison
    pairs = Counter((not sb.correct(a), not sb.correct(b))
                    for a, b in zip(runs["baseline"], runs["stale + reconcile + filter"]))
    print("== paired comparison baseline vs fixed")
    print(pairs)
    print(f"exact sign test two-sided p {sb.sign_test(pairs[(True, False)], pairs[(False, True)]):.4f}")
    for k in (8, 9):
        print(f"if 9 discordant split {k} p= {sb.sign_test(k, 9 - k):.4f}")
    print()

    # ---- the stale pointer
    print(HEADER)
    for name, ts in runs.items():
        print(sb.run_row(name, sb.score(ts)))
    print()

    t8 = fixed.run(stale_view(d8))
    print(f"DET-0008 stale + filter: form {t8.det.form}; cited {t8.answer.citation}")
    for sid, why in list(t8.dropped.items())[:4]:
        print(f"  dropped {sid:<32} {why}")
    print()

    # ---- the scoreboard
    show = {"baseline": runs["baseline"], "stale form pointer": runs["stale pointer"],
            "stale + reconcile + filter": runs["stale + reconcile + filter"]}
    print("Lakeshore scoreboard, 200-determination evaluation set (simulated model sim-large)")
    print()
    print(f"{'run':<30}wrong  lookup  hit@5  faith  tokens    $/det")
    for name, ts in show.items():
        s = sb.score(ts)
        print(f"{name:<30}{100 * s.rate:>4.1f}%{s.lookup:>7.1f}m{s.hit5:>7.2f}{s.faith:>7.3f}"
              f"{s.tokens:>8,.0f}{s.cost:>9.5f}")
    print()
    ho = [i for i, d in enumerate(dets) if corpus.policies[d.policy_number]["form"] == "HO-3 (2024-06)"]
    print(f"HO-3 (2024-06) claims in the set: {len(ho)}")
    for name, ts in show.items():
        wrong = [ts[i] for i in ho if not sb.correct(ts[i])]
        sup = sum(t.answer.cited_span_id is not None
                  and corpus.span(t.answer.cited_span_id).provenance.superseded_by is not None for t in wrong)
        print(f"  {name:<29}wrong {len(wrong):>3}  cites a superseded version {sup:>3}")
    print()
    i8 = dets.index(d8)
    gold = corpus.span(d8.gold_span_id)
    print(f"Example DET-0008 ({d8.claimant}, {d8.state}, {d8.date_of_loss}): {d8.question}")
    print(f"  {'gold':<26}: {gold.provenance.label} -> {gold.meta['verdict']}")
    for name, ts in show.items():
        print(f"  {name:<26}: {cited_line(ts[i8])}")
    print(f"  {'reconcile':<26}: {reconcile(stale_view(d8))[1]}")
    print()
    print(f"Median lookup (proxy) by difficulty, baseline vs. manual {sb.MANUAL_MEDIAN:.1f} min:")
    for diff in ("simple", "complex"):
        ts = [t for t in runs["baseline"] if t.det.difficulty == diff]
        m = median(sb.lookup_minutes(t, sb.exact(t), sb.correct(t)) for t in ts)
        print(f"  {diff:<8} n={len(ts):>3}  {m:.1f} min")
    print()

    # ---- slices of the fixed run
    fx = runs["stale + reconcile + filter"]
    print("== fixed pipeline by slice")
    for name, keep in (("simple", lambda d: d.difficulty == "simple"), ("complex", lambda d: d.difficulty == "complex"),
                       ("no endorsements", lambda d: not d.endorsements), ("endorsements", lambda d: bool(d.endorsements))):
        ts = [t for t in fx if keep(t.det)]
        w = sum(not sb.correct(t) for t in ts)
        lo, hi = sb.wilson(w, len(ts))
        print(f"{name:<16} {len(ts):>3} wrong {w:>2} {100 * w / len(ts):5.1f}% [{100 * lo:.1f}%, {100 * hi:.1f}%]")
    print()

    # ---- second injection: the endorsement list goes missing
    end = [d for d in dets if d.endorsements]
    intact = [fixed.run(fix_view(d)) for d in end]
    blank = [fixed.run(blank_view(fix_view(d))) for d in end]
    blank_b = [base.run(blank_view(d)) for d in end]
    print("== second injection: endorsement list missing from claim facts")
    for name, ts in (("fixed, endorsements intact", intact), ("fixed, endorsements blanked", blank)):
        w = sum(not sb.correct(t) for t in ts)
        cx = [t for t in ts if t.det.difficulty == "complex"]
        cw = sum(not sb.correct(t) for t in cx)
        h5 = sum((sb.gold_rank(t) or 99) <= 5 for t in ts) / len(ts)
        print(f"{name:<33}{w}/{len(ts)} complex {cw}/{len(cx)} hit5 {h5:.2f}")
    print("baseline blanked", sum(not sb.correct(t) for t in blank_b), len(blank_b))
    agree = sum(sorted(corpus.policies[d.policy_number]["endorsements"]) == sorted(d.endorsements) for d in end)
    print(f"policy record endorsements agree with claim facts: {agree} of {len(end)}")
    print()

    # ---- other models (illustrative prices)
    print("== models")
    for m in ("sim-large", "sim-mid", "sim-small"):
        for name, pipe, view in (("baseline", base, None), ("fixed", fixed, fix_view)):
            s = sb.score(run_all(pipe.with_config(model=m), dets, view))
            print(f"{name:<9}{m:<10} wrong {s.wrong:>2}/200 faith {s.faith:.3f} ${s.cost:.5f} "
                  f"per 3,200 a day ${3200 * s.cost:.2f}")
    print()

    # ---- the held-out set, once
    hold = corpus.holdout()
    hb = run_all(base, hold)
    hf = run_all(fixed, hold, fix_view)
    print("== holdout once")
    for name, ts in (("baseline", hb), ("fixed", hf)):
        parts = []
        for lab_, keep in (("all", lambda d: True), ("no-end", lambda d: not d.endorsements),
                           ("with end", lambda d: bool(d.endorsements)), ("simple", lambda d: d.difficulty == "simple"),
                           ("complex", lambda d: d.difficulty == "complex")):
            sel = [t for t in ts if keep(t.det)]
            parts.append(f"{lab_} {sum(not sb.correct(t) for t in sel)} {len(sel)}")
        print(name, " | ".join(parts))
    print("holdout pair", Counter((not sb.correct(a), not sb.correct(b)) for a, b in zip(hb, hf)))


if __name__ == "__main__":
    main()
