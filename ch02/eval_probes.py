"""Chapter 2 probes: proxy sensitivity, underwriting notes, DET-0001 ranks, a miss, the rendered text,
and the frozen evaluation file (eval200.csv)."""
from __future__ import annotations

import argparse
from statistics import mean, median

import lab
from contextlab import scoreboard as sb
from standin import eval_csv, operative

corpus = lab.corpus


def proxy(ts) -> list[float]:
    return [sb.lookup_minutes(t, sb.exact(t), sb.correct(t)) for t in ts]


def sensitivity(dets) -> None:
    fx = lab.run_all(lab.fixed, dets, lab.fix_view)
    bl = lab.run_all(lab.base, dets)
    print("== proxy sensitivity (fixed pipeline median, baseline median)")
    keep = {k: getattr(sb, k) for k in ("VERSION_CHECK", "VERIFY_MEDIAN", "MANUAL_MEDIAN")}
    for name, change in (("as written", {}), ("version check 0", {"VERSION_CHECK": 0.0}),
                         ("version check 1.1", {"VERSION_CHECK": 1.1}), ("version check 3.3", {"VERSION_CHECK": 3.3}),
                         ("verify 2.4", {"VERIFY_MEDIAN": 2.4}), ("manual 8", {"MANUAL_MEDIAN": 8.0}),
                         ("manual 14", {"MANUAL_MEDIAN": 14.0})):
        for k, v in change.items():
            setattr(sb, k, v)
        f, b = proxy(fx), proxy(bl)
        print(f"{name:<20} fixed median {median(f):.2f} mean {mean(f):.2f}  baseline median {median(b):.2f}")
        for k, v in keep.items():
            setattr(sb, k, v)


def notes() -> None:
    for sid in ("UW-NOTE-0001", "UW-NOTE-0002", "UW-NOTE-0004"):
        s = corpus.span(sid)
        print(f"{sid} {s.provenance.effective_from} {s.topic:<13} {s.text[:100]}...")


def ranks(det) -> None:
    t = lab.base.run(det)
    scores = {s.span_id: x for s, x in lab.base.retriever.search(lab.base.query(det))}
    for rank, sid in enumerate(t.retrieved[:10], 1):
        s = corpus.span(sid)
        sup = s.provenance.superseded_by is not None
        print(f"{rank:>2} {sid:<41} {scores[sid]:.2f} sup={sup} {s.kind.value}")


def miss(det) -> None:
    s = corpus.span(det.gold_span_id)
    print(f"{s.text.split('. ')[0]}. {operative(s)}. ...".replace(". ...", " ..."))


def rendered(det) -> None:
    text = lab.base.run(det).rendered.text
    print(text.split("\n\n### [S4]")[0])


def dumps(obj, depth: int = 0) -> str:
    """JSON as the trace store writes it: dicts of several keys open, short values on one line."""
    import json
    if isinstance(obj, dict) and len(obj) > 1 and depth < 2:
        pad = " " * (depth + 1)
        body = ",\n".join(f"{pad}{json.dumps(k)}: {dumps(v, depth + 1)}" for k, v in obj.items())
        return "{\n" + body + "\n" + " " * depth + "}"
    return json.dumps(obj)


def span_json(sid: str) -> None:
    s = corpus.span(sid)
    d = s.model_dump(mode="json", exclude={"tokens"})
    d["provenance"]["entitlement"] = sorted(d["provenance"]["entitlement"], reverse=True)
    print(dumps(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-eval", action="store_true", help="rewrite eval200.csv (freezes the set)")
    args = ap.parse_args()
    dets = corpus.eval_set(200)
    by_id = {d.determination_id: d for d in dets}
    if args.write_eval:
        (lab.HERE / "eval200.csv").write_text(eval_csv(dets))
    sensitivity(dets)
    print()
    span_json("HO 04 95@2024-06#water_backup")
    print()
    print("".join(eval_csv(dets).splitlines(keepends=True)[:5]))
    notes()
    print()
    ranks(by_id["DET-0001"])
    print()
    miss(by_id["DET-0022"])
    print()
    rendered(by_id["DET-0001"])


if __name__ == "__main__":
    main()
