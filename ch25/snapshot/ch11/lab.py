"""Chapter 11 lab: score 300 synthetic documents across twenty layouts,
route them by confidence and measure post-review error.

Run: python 04-code/ch11/lab.py   (writes 04-code/ch11/out/)
Deterministic: every random draw comes from seeded generators. The three
extraction pipelines are stand-ins (sim-ocr-text, sim-vision, sim-vision-ocr),
not real models; their error rates are set so that the shapes match the
research cited in the chapter, and the exact rates belong to the stand-ins.
"""
from __future__ import annotations

import csv, os, random, sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contextlab.docgate import (Document, ExtractedField, Quality, Route, Thresholds,
    auroc, check_digit_ok, field_correct, field_score, normalise_amount,
    normalise_amount_rounding, psi, quality_score, route, route_fail_open, validate)

OUT = os.path.join(os.path.dirname(__file__), "out")
PIPES = ("ocr_text", "vision", "vision_ocr")
FAMILIES = ("born_digital", "scanned_form", "estimate", "medical_bill",
            "police_report", "fax", "handwritten", "phone_photo")

# id, family, description, weight, line items (min, max), pages (min, max)
LAYOUTS = [
    ("L01", "born_digital", "Lakeshore declarations page", 14, (0, 0), (1, 2)),
    ("L02", "born_digital", "Lakeshore online first notice of loss", 12, (0, 0), (1, 1)),
    ("L03", "scanned_form", "agent-scanned property loss notice", 8, (0, 0), (1, 2)),
    ("L04", "scanned_form", "scanned sworn proof of loss", 6, (2, 6), (1, 2)),
    ("L05", "scanned_form", "prior-carrier declarations page", 5, (0, 0), (1, 1)),
    ("L06", "estimate", "estimating-software export", 9, (8, 40), (2, 6)),
    ("L07", "estimate", "roofing contractor estimate", 6, (4, 18), (1, 3)),
    ("L08", "estimate", "plumber invoice", 6, (2, 10), (1, 2)),
    ("L09", "estimate", "water mitigation invoice", 5, (6, 24), (1, 3)),
    ("L10", "medical_bill", "itemised hospital bill", 4, (6, 30), (2, 5)),
    ("L11", "medical_bill", "physician statement", 4, (2, 8), (1, 2)),
    ("L12", "police_report", "Wisconsin crash report", 4, (0, 0), (2, 4)),
    ("L13", "police_report", "Minnesota crash report", 2, (0, 0), (2, 4)),
    ("L14", "police_report", "Illinois crash report", 2, (0, 0), (2, 4)),
    ("L15", "fax", "faxed proof of loss", 3, (2, 6), (1, 2)),
    ("L16", "fax", "faxed repair authorisation", 2, (0, 0), (1, 1)),
    ("L17", "handwritten", "handwritten statement", 2, (0, 0), (1, 3)),
    ("L18", "handwritten", "handwritten contents inventory", 2, (3, 15), (1, 3)),
    ("L19", "phone_photo", "receipt photographed on a phone", 3, (1, 8), (1, 1)),
    ("L20", "phone_photo", "paper estimate photographed on a phone", 1, (3, 12), (1, 2)),
]
FAM_OF = {l[0]: l[1] for l in LAYOUTS}

# page quality profile: (text layer, char conf, dpi choices, skew, garble, layout match)
PROFILE = {
    "born_digital": (1.0, 0.99, (300,), 0.0, 0.002, 0.97),
    "scanned_form": (0.0, 0.93, (200, 300), 1.2, 0.02, 0.90),
    "estimate":     (0.55, 0.94, (200, 300), 0.8, 0.02, 0.84),
    "medical_bill": (0.35, 0.93, (200, 300), 1.0, 0.025, 0.86),
    "police_report": (0.2, 0.90, (200, 300), 1.5, 0.03, 0.88),
    "fax":          (0.0, 0.82, (100, 150, 200), 2.5, 0.06, 0.80),
    "handwritten":  (0.0, 0.70, (150, 200, 300), 2.0, 0.08, 0.62),
    "phone_photo":  (0.0, 0.80, (150, 200), 4.0, 0.06, 0.66),
}
# field error rate at middling quality, per pipeline and family (stand-in values)
BASE = {
    "ocr_text":   dict(born_digital=.002, scanned_form=.030, estimate=.030, medical_bill=.035,
                       police_report=.060, fax=.080, handwritten=.300, phone_photo=.140),
    "vision":     dict(born_digital=.006, scanned_form=.030, estimate=.055, medical_bill=.050,
                       police_report=.030, fax=.050, handwritten=.090, phone_photo=.055),
    "vision_ocr": dict(born_digital=.003, scanned_form=.024, estimate=.018, medical_bill=.020,
                       police_report=.030, fax=.055, handwritten=.090, phone_photo=.060),
}
AMOUNT_MULT = {"ocr_text": 1.0, "vision": 1.7, "vision_ocr": 1.1}   # numbers are fragile
PRIMARY = {f: ("ocr_text" if f == "born_digital" else "vision_ocr") for f in FAMILIES}
CROSS = {f: ("vision_ocr" if f == "born_digital" else "ocr_text") for f in FAMILIES}
COST_PER_PAGE = {"ocr_text": 0.0015, "vision": 0.0060, "vision_ocr": 0.0075}   # US$, illustrative
NAMES = ["Emily Carter", "Daniel Novak", "Grace Lindqvist", "Marcus Bell", "Priya Raman",
         "Owen Schultz", "Hannah Kowalski", "Luis Ortega", "Ruth Anders", "Samuel Okafor"]


def luhn_complete(stem: str) -> str:
    for d in "0123456789":
        if check_digit_ok(stem + d):
            return stem + d
    raise AssertionError


@dataclass
class Truth:
    doc_id: str
    layout: str
    family: str
    latent: float                 # true page condition, 0 (unreadable) to 1
    quality: Quality
    fields: list                  # (name, kind, critical, value)
    start: date
    end: date
    pages: int
    dropped_page: bool = False


def make_truth(i: int, rng: random.Random, layout=None) -> Truth:
    lay = layout or rng.choices(LAYOUTS, weights=[l[3] for l in LAYOUTS])[0]
    lid, fam, _, _, (imin, imax), (pmin, pmax) = lay
    tl, cc, dpis, sk, gb, lm = PROFILE[fam]
    latent = min(1.0, max(0.05, rng.gauss(0.95 if fam == "born_digital" else 0.80, 0.13)))
    text_layer = rng.random() < tl
    q = Quality(text_layer=text_layer,
                char_conf=round(min(0.995, max(0.3, cc - 0.55 * (1 - latent) + rng.gauss(0, .02))), 3),
                dpi=rng.choice(dpis) if not text_layer else 300,
                skew_deg=round(abs(rng.gauss(sk * (1.6 - latent), 0.6)), 2),
                garble=round(max(0.0, gb * (1.8 - latent) + rng.gauss(0, 0.01)), 3),
                layout_match=round(min(1.0, max(0.2, rng.gauss(lm, 0.06))), 3))
    pages = rng.randint(pmin, pmax)
    q.pages_expected = q.pages_found = pages
    start = date(2025, 7, 1) + timedelta(days=rng.randint(0, 420))
    end = start + timedelta(days=365)
    dol = start + timedelta(days=rng.randint(5, 300))
    pol = luhn_complete(f"{rng.randint(10**8, 10**9 - 1)}")
    name = NAMES[i % len(NAMES)] if i % 37 else "Emily Carter"
    fields = [("policy_number", "id", True, pol), ("date_of_loss", "date", True, dol.isoformat()),
              ("claimant_name", "name", True, name),
              ("doc_date", "date", False, (dol + timedelta(days=rng.randint(1, 20))).isoformat()),
              ("issuer", "text", False, f"issuer-{lid}-{rng.randint(1, 40)}")]
    n = rng.randint(imin, imax)
    if n:
        items = [Decimal(rng.randint(40, 120000)) / 100 for _ in range(n)]
        if fam in ("estimate", "medical_bill"):
            items = [v.quantize(Decimal("1")) + Decimal("0.50") if rng.random() < .35 else v for v in items]
        fields += [(f"item_{k:02d}", "amount", False, f"{v:,.2f}") for k, v in enumerate(items)]
        fields.append(("total_amount", "amount", True, f"{sum(items):,.2f}"))
    return Truth(f"DOC-{i:04d}", lid, fam, latent, q, fields, start, end, pages)


def corrupt(value: str, kind: str, rng: random.Random) -> str:
    if kind == "amount":
        digits = [i for i, c in enumerate(value) if c.isdigit()]
        a = rng.choice(digits)
        return value[:a] + str((int(value[a]) + rng.choice((1, 2, 3, 5, 7))) % 10) + value[a + 1:]
    if kind == "date":
        y, m, d = value.split("-")
        return f"{y}-{d}-{m}" if int(d) <= 12 and d != m else f"{y}-{m}-{(int(d) % 28) + 1:02d}"
    if kind == "id":
        k = rng.randrange(len(value))
        return value[:k] + str((int(value[k]) + 1) % 10) + value[k + 1:]
    if kind == "name":
        k = rng.randrange(1, len(value))
        return value[:k] + rng.choice("aeiourn") + value[k + 1:]
    return value + "x"


def extract(t: Truth, pipe: str, rng: random.Random) -> list[ExtractedField]:
    """One stand-in pipeline reading one document."""
    sev = 0.25 + 2.2 * (1 - t.latent)
    if t.quality.layout_match < 0.6:
        sev *= 1.5
    out = []
    for name, kind, crit, value in t.fields:
        fam = "born_digital" if t.quality.text_layer else t.family   # a text layer reads like a born-digital page
        p = BASE[pipe][fam] * sev * (AMOUNT_MULT[pipe] if kind == "amount" else 1)
        wrong = rng.random() < min(0.65, p)
        v = corrupt(value, kind, rng) if wrong else value
        if pipe == "ocr_text":
            conf = rng.gauss(0.70, 0.12) if wrong else rng.gauss(0.86 + 0.12 * t.latent, 0.04)
        elif pipe == "vision":            # verbalised confidence: high whether right or wrong
            conf = rng.gauss(0.92, 0.05) if wrong else rng.gauss(0.95, 0.03)
        else:                             # first-token probability with OCR text in the prompt
            conf = rng.gauss(0.80, 0.10) if wrong else rng.gauss(0.90 + 0.07 * t.latent, 0.05)
        conf = round(min(0.999, max(0.01, conf)), 3)
        missing = (0.08 if wrong else 0.003) * (2 if t.quality.layout_match < 0.6 else 1)
        if pipe != "ocr_text" and rng.random() < missing:
            conf = None                   # the model's JSON left the confidence out
        out.append(ExtractedField(name, kind, v, conf, crit, truth=value))
    return out


def drop_page(t: Truth, fields: list[ExtractedField]) -> list[ExtractedField]:
    """Sidebar case: the continuation page is lost; the page-1 subtotal is read as the total."""
    items = [f for f in fields if f.name.startswith("item_")]
    keep = items[: max(1, len(items) // 2)]
    sub = sum(normalise_amount(f.value) for f in keep)
    total = next(f for f in fields if f.name == "total_amount")
    total.value = f"{sub:,.2f}"
    t.quality.pages_found = max(1, t.pages - 1)
    t.dropped_page = True
    return [f for f in fields if not f.name.startswith("item_") or f in keep]


def build(t: Truth, rng: random.Random, primary=None, cross=True, normalise=normalise_amount,
          reconcile_tol=Decimal("0")):
    pipe = primary or PRIMARY[t.family]
    fields = extract(t, pipe, rng)
    if t.pages > 1 and any(f.name.startswith("item_") for f in fields) and rng.random() < 0.03:
        fields = drop_page(t, fields)
    doc = Document(t.doc_id, t.family, t.layout, t.quality, fields, t.start, t.end)
    validate(doc, normalise=normalise, reconcile_tol=reconcile_tol)
    q = quality_score(t.quality)
    other = {f.name: f.value for f in extract(t, CROSS[t.family], rng)} if cross else {}
    for f in doc.fields:
        agrees = (other.get(f.name) == f.value) if (cross and f.critical) else None
        f.score = field_score(f, q, agrees)
    return doc, q, pipe


def review(doc: Document, rng: random.Random, th: Thresholds) -> int:
    """A reviewer corrects wrong critical fields; flagged ones get more attention."""
    fixed = 0
    for f in doc.fields:
        if f.critical and not field_correct(f):
            flagged = f.score is None or f.score < th.field_auto
            if rng.random() < (0.94 if flagged else 0.75):
                f.value = f.truth
                fixed += 1
    return fixed


# ====================================================================== runs
def run_gate(truths, th=Thresholds(), fail_open=False, tag="main"):
    rows = []
    for t in truths:
        t.quality.pages_found = t.quality.pages_expected = t.pages
        doc, q, pipe = build(t, random.Random(f"{tag}-build-{t.doc_id}"))
        raw_ok = [field_correct(f) for f in doc.fields]
        crit_wrong_raw = sum(1 for f in doc.fields if f.critical and not field_correct(f))
        if fail_open:
            r, why = route_fail_open(doc, th, q), []
        else:
            r, why = route(doc, th, q)
        if r == Route.REVIEW:
            review(doc, random.Random(f"{tag}-rev-{t.doc_id}"), th)
        crit_wrong_after = sum(1 for f in doc.fields if f.critical and not field_correct(f))
        rows.append(dict(doc=t.doc_id, layout=t.layout, family=t.family, q=q, route=r.value,
                         why="; ".join(why), pipe=pipe, n_fields=len(doc.fields),
                         fields_ok=sum(raw_ok), doc_ok=all(raw_ok), crit_wrong_raw=crit_wrong_raw,
                         crit_wrong_after=crit_wrong_after if r != Route.REFUSE else 0,
                         missing_conf=any(f.score is None for f in doc.fields if f.critical),
                         dropped=t.dropped_page, pages=t.pages, doc_obj=doc))
    return rows


def summarise(rows):
    n = len(rows)
    adm = [r for r in rows if r["route"] != "refuse"]
    rev = sum(r["route"] == "review" for r in rows)
    ref = sum(r["route"] == "refuse" for r in rows)
    bad = sum(r["crit_wrong_after"] > 0 for r in adm)
    return dict(n=n, auto=n - rev - ref, review=rev, refuse=ref, post_err=bad, admitted=len(adm),
                raw_bad=sum(r["crit_wrong_raw"] > 0 for r in rows))


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def write_csv(name, rows, keys):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    say = lines.append
    # the lab set: 15 documents from each of the twenty layouts (a stress mix)
    truths = [make_truth(i, random.Random(f"lab-{i}"), layout=LAYOUTS[(i - 1) % 20]) for i in range(1, 301)]
    # the live mix: 4,000 documents drawn at Lakeshore's incoming proportions
    rng = random.Random(12)
    live = [make_truth(i, rng) for i in range(1, 4001)]

    # ---- section 1: the mix
    say("== 1. The 300 documents")
    for fam in FAMILIES:
        ts = [t for t in truths if t.family == fam]
        qs = [quality_score(t.quality) for t in ts]
        say(f"{fam:14s} docs {len(ts):3d}  layouts {len({t.layout for t in ts}):2d}  "
            f"mean quality {sum(qs) / len(qs):.2f}  min {min(qs):.2f}")
    say(f"layouts used: {len({t.layout for t in truths})}")

    # ---- section 2: field accuracy against document accuracy (primary pipeline, raw)
    rows = run_gate(truths)
    nf = sum(r["n_fields"] for r in rows); ok = sum(r["fields_ok"] for r in rows)
    say("\n== 2. Field accuracy and document accuracy, before any review")
    say(f"fields {nf}  field accuracy {pct(ok, nf)}  documents all-correct "
        f"{pct(sum(r['doc_ok'] for r in rows), len(rows))}  mean fields/doc {nf / len(rows):.1f}")
    fa_rows = []
    for fam in FAMILIES:
        rs = [r for r in rows if r["family"] == fam]
        f_n = sum(r["n_fields"] for r in rs); f_ok = sum(r["fields_ok"] for r in rs)
        d_ok = sum(r["doc_ok"] for r in rs)
        fa_rows.append(dict(family=fam, field_acc=f_ok / f_n, doc_acc=d_ok / len(rs),
                            fields_per_doc=f_n / len(rs)))
        say(f"{fam:14s} field {pct(f_ok, f_n):>6s}  document {pct(d_ok, len(rs)):>6s}  "
            f"fields/doc {f_n / len(rs):5.1f}")
    write_csv("field_vs_doc.csv", fa_rows, ["family", "field_acc", "doc_acc", "fields_per_doc"])

    # ---- section 3: heat map, per family x pipeline field error (same truths, no gate)
    say("\n== 3. Field error rate by family and pipeline (all fields, no gate)")
    heat = []
    for fam in FAMILIES:
        ts = [t for t in truths if t.family == fam]
        rec = dict(family=fam)
        for pipe in PIPES:
            errs = tot = 0
            for t in ts:
                fs = extract(t, pipe, random.Random(f"heat-{pipe}-{t.doc_id}"))
                errs += sum(not field_correct(f) for f in fs); tot += len(fs)
            rec[pipe] = errs / tot
        rec["pages"] = sum(t.pages for t in ts) / len(ts)
        heat.append(rec)
        say(f"{fam:14s} " + "  ".join(f"{p} {100 * rec[p]:5.1f}%" for p in PIPES))
    write_csv("heatmap.csv", heat, ["family", *PIPES, "pages"])
    pages = sum(t.pages for t in truths)
    mix_cost = sum(COST_PER_PAGE[PRIMARY[t.family]] * t.pages + COST_PER_PAGE[CROSS[t.family]] * t.pages
                   for t in truths) / len(truths)
    say(f"cost per document, chosen mix with cross-check: US${mix_cost:.4f}; "
        f"vision_ocr everywhere, no cross-check: US${COST_PER_PAGE['vision_ocr'] * pages / len(truths):.4f}")

    # ---- section 4: confidence quality
    say("\n== 4. Confidence: does the score rank wrong fields below right ones? (critical fields)")
    calib = []
    for label, pipe in (("vision verbalised", "vision"), ("vision_ocr first-token", "vision_ocr")):
        sc, pos = [], []
        for t in truths:
            for f in extract(t, pipe, random.Random(f"conf-{pipe}-{t.doc_id}")):
                if f.critical and f.conf is not None:
                    sc.append(f.conf); pos.append(field_correct(f))
        say(f"{label:26s} AUROC {auroc(sc, pos):.3f}  over {len(sc)} fields")
    sc, pos = [], []
    for r in rows:
        for f in r["doc_obj"].fields:
            if f.critical and f.score is not None:
                sc.append(f.score); pos.append(f.truth is not None and (f.value == f.truth or field_correct(f)))
    # rescore against the pre-review values
    sc, pos = [], []
    for t in truths:
        t.quality.pages_found = t.pages
        doc, q, _ = build(t, random.Random(f"main-build-{t.doc_id}"))
        for f in doc.fields:
            if f.critical and f.score is not None:
                sc.append(f.score); pos.append(field_correct(f))
                calib.append(dict(score=f.score, correct=int(field_correct(f)), native=f.conf))
    say(f"{'composite gate score':26s} AUROC {auroc(sc, pos):.3f}  over {len(sc)} fields")
    write_csv("scores.csv", calib, ["score", "correct", "native"])

    # ---- section 5: routing at the chosen thresholds
    th = Thresholds()
    s = summarise(rows)
    say("\n== 5. Routing, thresholds " + str(th))
    say(f"auto {s['auto']}  review {s['review']} ({pct(s['review'], s['n'])})  refuse {s['refuse']} "
        f"({pct(s['refuse'], s['n'])})")
    say(f"documents with a wrong critical field before the gate: {s['raw_bad']} ({pct(s['raw_bad'], s['n'])})")
    say(f"post-review error: {s['post_err']} of {s['admitted']} admitted ({pct(s['post_err'], s['admitted'])})")
    leaks = [r for r in rows if r["route"] == "auto" and r["crit_wrong_after"]]
    for r in leaks:
        bad = [f for f in r["doc_obj"].fields if f.critical and not field_correct(f)]
        say(f"  auto-tier error {r['doc']} {r['layout']} {r['family']} q={r['q']:.2f} "
            + ", ".join(f"{f.name} read {f.value!r} for {f.truth!r} score {f.score}" for f in bad))
    for r in rows:
        if r["route"] == "review" and r["crit_wrong_after"]:
            say(f"  missed in review {r['doc']} {r['family']}")
    reasons = {}
    for r in rows:
        if r["route"] == "review":
            for w in r["why"].split("; "):
                k = w.split(":")[0] if ":" in w else w.split(" ")[0] if w.startswith("quality") else w
                k = "field below threshold" if ":" in w and "no confidence" not in w else (
                    "missing confidence" if "no confidence" in w else k)
                reasons[k] = reasons.get(k, 0) + 1
    say("review reasons (a document can have several): " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
    fam_route = []
    for fam in FAMILIES:
        rs = [r for r in rows if r["family"] == fam]
        fam_route.append(dict(family=fam, n=len(rs), auto=sum(r["route"] == "auto" for r in rs),
                              review=sum(r["route"] == "review" for r in rs),
                              refuse=sum(r["route"] == "refuse" for r in rs)))
        say(f"{fam:14s} auto {fam_route[-1]['auto']:3d} review {fam_route[-1]['review']:3d} "
            f"refuse {fam_route[-1]['refuse']:2d}")
    write_csv("routes_by_family.csv", fam_route, ["family", "n", "auto", "review", "refuse"])
    write_csv("docs.csv", rows, ["doc", "layout", "family", "q", "route", "why", "pipe", "n_fields",
                                 "fields_ok", "crit_wrong_raw", "crit_wrong_after", "dropped", "pages"])
    drops = [r for r in rows if r["dropped"]]
    say(f"dropped continuation pages: {len(drops)}; routed " + ", ".join(f"{r['doc']} {r['route']}" for r in drops))
    lrows = run_gate(live, tag="live")
    sl = summarise(lrows)
    say(f"live mix, 4,000 documents: auto {sl['auto']}  review {sl['review']} ({pct(sl['review'], sl['n'])})  "
        f"refuse {sl['refuse']} ({pct(sl['refuse'], sl['n'])})  wrong critical field before the gate "
        f"{sl['raw_bad']} ({pct(sl['raw_bad'], sl['n'])})  post-review error {sl['post_err']} of {sl['admitted']} "
        f"({pct(sl['post_err'], sl['admitted'])})")
    a_bad = sum(r["route"] == "auto" and r["crit_wrong_after"] > 0 for r in lrows)
    say(f"  of which in the auto tier {a_bad}, missed in review {sl['post_err'] - a_bad}")
    lf = []
    for fam in FAMILIES:
        rs = [r for r in lrows if r["family"] == fam]
        lf.append(dict(family=fam, n=len(rs), review=sum(r["route"] == "review" for r in rs) / len(rs),
                       refuse=sum(r["route"] == "refuse" for r in rs) / len(rs),
                       post_err=sum(r["crit_wrong_after"] > 0 for r in rs if r["route"] != "refuse") / len(rs)))
        say(f"  {fam:14s} share {pct(len(rs), len(lrows)):>6s} review {100 * lf[-1]['review']:5.1f}% "
            f"refuse {100 * lf[-1]['refuse']:4.1f}% post-review error {100 * lf[-1]['post_err']:4.2f}%")
    write_csv("live_by_family.csv", lf, ["family", "n", "review", "refuse", "post_err"])

    # ---- section 6: threshold sweep against review capacity
    say("\n== 6. Threshold sweep on the live mix (field_auto): review share and post-review error")
    sweep = []
    for t_auto in (0.50, 0.55, 0.60, 0.63, 0.65, 0.67, 0.70, 0.75, 0.80, 0.85):
        thx = Thresholds(field_auto=t_auto)
        sx = summarise(run_gate(live, thx, tag="live"))
        sweep.append(dict(t=t_auto, review=sx["review"] / sx["n"], refuse=sx["refuse"] / sx["n"],
                          post_err=sx["post_err"] / sx["admitted"]))
        say(f"t={t_auto:.2f} review {pct(sx['review'], sx['n']):>6s} post-review error "
            f"{pct(sx['post_err'], sx['admitted']):>6s} ({sx['post_err']})")
    write_csv("sweep.csv", sweep, ["t", "review", "refuse", "post_err"])

    # ---- section 7: fail open against fail closed
    say("\n== 7. Missing confidence: fail closed against fail open")
    fo = run_gate(live, fail_open=True, tag="live")
    sfo = summarise(fo)
    miss = [r for r in lrows if r["missing_conf"]]
    say(f"live documents with a critical field missing its confidence: {len(miss)} ({pct(len(miss), len(lrows))}); "
        f"with a wrong critical field among them: {sum(r['crit_wrong_raw'] > 0 for r in miss)}")
    say(f"fail closed: post-review error {sl['post_err']} ({pct(sl['post_err'], sl['admitted'])}), review {pct(sl['review'], sl['n'])}; "
        f"fail open: {sfo['post_err']} ({pct(sfo['post_err'], sfo['admitted'])}), review {pct(sfo['review'], sfo['n'])}")

    # ---- section 8: failure injection, the rounding normaliser
    say("\n== 8. Failure injection: a normaliser that rounds amounts to whole dollars")
    inj = []
    amount_docs = [t for t in truths if any(n.startswith("item_") for n, *_ in t.fields)]
    pass_tol = pass_exact = n_amt = 0
    recon_flag_tol = recon_flag_exact = 0
    for t in amount_docs:
        items = [Decimal(v.replace(",", "")) for n, k, c, v in t.fields if n.startswith("item_")]
        total = Decimal(next(v for n, k, c, v in t.fields if n == "total_amount").replace(",", ""))
        r_items = [normalise_amount_rounding(f"{v}") for v in items]
        r_total = normalise_amount_rounding(f"{total}")
        for a, b in zip(r_items + [r_total], items + [total]):
            n_amt += 1
            pass_tol += abs(a - b) <= Decimal("1.00")
            pass_exact += a == b
        drift = sum(r_items) - total          # what a downstream sum of covered items sees
        recon_flag_tol += abs(sum(r_items) - r_total) > Decimal("0.50") * len(items)
        recon_flag_exact += sum(r_items) != r_total
        inj.append(dict(doc=t.doc_id, family=t.family, items=len(items), total=float(total),
                        drift=float(drift)))
    say(f"amount fields {n_amt}: within the old ±US$1.00 tolerance {pct(pass_tol, n_amt)}; exact to the cent {pct(pass_exact, n_amt)}")
    say(f"documents with line items: {len(amount_docs)}; reconciliation flags with the per-line tolerance "
        f"{recon_flag_tol}; with zero tolerance {recon_flag_exact}")
    drifts = sorted(abs(r["drift"]) for r in inj)
    say(f"sum of rounded items minus printed total: median US${drifts[len(drifts) // 2]:.2f}, "
        f"max US${drifts[-1]:.2f}, documents off by more than US$5: {sum(d > 5 for d in drifts)}")
    worst = max(inj, key=lambda r: abs(r["drift"]))
    say(f"largest drift: {worst['doc']} ({worst['family']}, {worst['items']} line items, printed total "
        f"US${worst['total']:,.2f}, rounded sum off by US${worst['drift']:.2f})")
    write_csv("injection.csv", inj, ["doc", "family", "items", "total", "drift"])
    rt_fail = sum(normalise_amount_rounding(v) != normalise_amount(v)
                  for t in amount_docs for n, k, c, v in t.fields if k == "amount")
    say(f"round-trip test (normalised value must equal the printed value): {rt_fail} failures")

    # ---- section 9: the full mix, document-attributable wrong-clause
    say("\n== 9. Full 1,000-determination mix (Chapter 9): the gate on scanned and carrier documents")
    full_mix(say)

    # ---- section 10: template drift on one layout
    say("\n== 10. Template drift: prior-carrier declarations page (L05), 12 weeks")
    drift_sim(say)

    with open(os.path.join(OUT, "output.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


CH10_FULL_WRONG = 40    # Chapter 10's full-mix close, from 04-code/ch10/out/lab_output.txt (reconciled 1 Nov 2026)
SCAN_LAYOUTS = [l for l in LAYOUTS if l[1] in ("scanned_form", "fax")]


def full_mix(say):
    """Chapter 9's full mix: the gate acts on the 90 scanned and carrier-document determinations."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch09"))
    import fullmix as F
    dets = [d for d in F.build() if d.split == "new" and d.stratum == "scanned"]
    # the gate's catch rate on scanned and faxed documents, measured on the live mix
    live = [make_truth(i, random.Random(12 * 10**6 + i), layout=SCAN_LAYOUTS[i % len(SCAN_LAYOUTS)]) for i in range(1, 1201)]
    rows = run_gate(live, tag="fm")
    bad = [r for r in rows if r["crit_wrong_raw"] > 0]
    caught = sum(r["crit_wrong_after"] == 0 for r in bad) / len(bad)
    before = Counter(F.outcome(d, "ch09") for d in dets)
    after, routed = Counter(), 0
    for d in dets:
        t = make_truth(int(d.det_id[4:]), random.Random(f"fm-{d.det_id}"), layout=SCAN_LAYOUTS[int(d.det_id[4:]) % len(SCAN_LAYOUTS)])
        doc, q, _ = build(t, random.Random(f"fmb-{d.det_id}"))
        r, _ = route(doc, Thresholds(), q)
        u = random.Random(f"fmu-{d.det_id}").random()
        base = F.outcome(d, "ch09")
        if base == "withheld":             # no edition printed: the layout library names the edition, a person confirms
            if r == Route.REFUSE or t.quality.layout_match < 0.80:
                after["withheld"] += 1
                continue
            routed += 1
            after["wrong" if u < 0.05 * 0.06 else "right"] += 1   # wrong template 5%, reviewer misses 6% of those
        elif base == "wrong":              # OCR noise in an admitted scan
            routed += r != Route.AUTO
            after["wrong" if u > caught else "right"] += 1
        else:
            routed += r != Route.AUTO
            after["right"] += 1
    say(f"scanned and carrier-document determinations (Chapter 9 stratum): {len(dets)}; routed to a person {routed}")
    say(f"gate catch rate on scanned and faxed documents with a wrong critical field (live sample of 1,200): {100 * caught:.1f}%")
    say(f"before the gate (Chapter 9 rules): wrong {before['wrong']}, withheld {before['withheld']}; "
        f"after: wrong {after['wrong']}, withheld {after['withheld']}")
    w = CH10_FULL_WRONG - before["wrong"] + after["wrong"]
    say(f"full mix: Chapter 10 close {CH10_FULL_WRONG} of 1000 ({pct(CH10_FULL_WRONG, 1000)}) -> {w} of 1000 ({pct(w, 1000)}); "
        f"withheld {before['withheld'] + 6} -> {after['withheld'] + 6} (6 rejected queries unchanged)")


def drift_sim(say, weeks=12, change=7, per_week=60):
    rng = random.Random(505)
    rows, base_lm, base_ded = [], [], []
    for w in range(1, weeks + 1):
        lm, ded, rev, wrong = [], [], 0, 0
        for _ in range(per_week):
            new = w >= change
            m = min(1.0, max(0.2, rng.gauss(0.72 if new else 0.93, 0.05)))
            true_ded = rng.choice((500, 1000, 1000, 2500))
            # new template: the extractor reads the wind/hail percentage deductible's dollar value
            read = true_ded if not new or rng.random() < 0.55 else rng.choice((3000, 4200, 5600, 7000))
            score = 0.7 * rng.gauss(0.93, 0.05) + 0.3 * (0.55 + 0.45 * m)
            other = rng.random() < 0.05          # reconciliation, missing confidence, quality
            rev += score < Thresholds().field_auto or m < Thresholds().layout_min or other
            wrong += read != true_ded
            lm.append(m); ded.append(min(1.0, read / 8000))
        if w <= 4:
            base_lm += lm; base_ded += ded
        rows.append(dict(week=w, layout_match=sum(lm) / len(lm), review=rev / per_week, wrong=wrong / per_week,
                         psi_layout=psi(base_lm, lm) if w > 4 else 0.0,
                         psi_value=psi(base_ded, ded) if w > 4 else 0.0))
    for r in rows:
        flag = "  ALARM" if r["psi_layout"] > 0.25 or r["psi_value"] > 0.25 else ""
        say(f"week {r['week']:2d} layout match {r['layout_match']:.2f} review {100 * r['review']:4.1f}% "
            f"wrong deductible {100 * r['wrong']:4.1f}% PSI layout {r['psi_layout']:.2f} PSI deductible {r['psi_value']:.2f}{flag}")
    write_csv("drift.csv", rows, ["week", "layout_match", "review", "wrong", "psi_layout", "psi_value"])


if __name__ == "__main__":
    main()
