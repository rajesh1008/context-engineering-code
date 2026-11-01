# Copyright 2026 Lakeshore Mutual. Internal use only.
# SPDX-License-Identifier: LicenseRef-Lakeshore-Internal
# Generated header; do not edit by hand.

"""contextlab.docgate: the document quality gate (Chapter 11, v2.3).

Every incoming document gets a quality score before anything downstream
trusts it, every extracted field gets a confidence and a validation
result, and the gate routes the document to auto, review or refuse.
The gate fails closed: a missing confidence, a failed check or an
unknown layout sends the document to a person, never to the packet.
Self-contained: standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum


class Route(str, Enum):
    AUTO = "auto"
    REVIEW = "review"
    REFUSE = "refuse"


@dataclass
class Quality:
    """Signals measured on the page images and the text layer."""
    text_layer: bool          # born-digital text present and readable
    char_conf: float          # mean OCR character confidence, 0-1
    dpi: int                  # effective resolution of the scan
    skew_deg: float           # absolute skew of the page
    garble: float             # share of tokens that are not words or numbers
    layout_match: float       # similarity to the nearest known template, 0-1
    pages_found: int = 1
    pages_expected: int = 1


@dataclass
class ExtractedField:
    name: str
    kind: str                 # id, date, name, amount, text
    value: str | None
    conf: float | None        # the extractor's own signal; None if absent
    critical: bool = False
    truth: str | None = None  # known only in the lab and in audit
    checks: dict = field(default_factory=dict)
    score: float | None = None   # the gate's composite confidence


@dataclass
class Document:
    doc_id: str
    family: str
    layout: str
    quality: Quality
    fields: list[ExtractedField]
    policy_start: date | None = None
    policy_end: date | None = None


# ---------------------------------------------------------------- quality
def quality_score(q: Quality) -> float:
    """Per-document quality in [0, 1]; computed before any extraction is used."""
    if q.pages_found < q.pages_expected:
        return 0.0                       # a missing page is not a quality issue to average away
    res = min(q.dpi, 300) / 300
    skew = max(0.0, 1 - q.skew_deg / 10)
    clean = max(0.0, 1 - 6 * q.garble)
    read = 1.0 if q.text_layer else q.char_conf
    s = 0.35 * read + 0.15 * res + 0.10 * skew + 0.20 * clean + 0.20 * q.layout_match
    return round(min(1.0, s), 3)


# ------------------------------------------------------------ normalisers
def normalise_amount(text: str | None) -> Decimal | None:
    """Lossless: keeps the cents printed on the page, or returns None."""
    if text is None:
        return None
    t = text.replace("US$", "").replace("$", "").replace(",", "").strip()
    try:
        return Decimal(t).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def normalise_amount_rounding(text: str | None) -> Decimal | None:
    """The Chapter 11 failure injection: rounds to whole dollars."""
    v = normalise_amount(text)
    return None if v is None else v.quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def check_digit_ok(policy_number: str | None) -> bool:
    """Lakeshore policy numbers end in a mod-10 check digit (Luhn)."""
    if not policy_number:
        return False
    digits = [int(c) for c in policy_number if c.isdigit()]
    if len(digits) < 6:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def parse_date(text: str | None) -> date | None:
    try:
        y, m, d = (int(x) for x in (text or "").split("-"))
        return date(y, m, d)
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------ validation
def validate(doc: Document, normalise=normalise_amount,
             reconcile_tol: Decimal = Decimal("0")) -> None:
    """Fill field.checks with True/False for every check that applies."""
    by = {f.name: f for f in doc.fields}
    for f in doc.fields:
        if f.kind == "id":
            f.checks["check_digit"] = check_digit_ok(f.value)
        elif f.kind == "date":
            d = parse_date(f.value)
            ok = d is not None
            if ok and doc.policy_start and doc.policy_end:
                ok = doc.policy_start <= d <= doc.policy_end
            f.checks["in_period"] = ok
        elif f.kind == "amount":
            f.checks["parses"] = normalise(f.value) is not None
    items = [f for f in doc.fields if f.name.startswith("item_")]
    total = by.get("total_amount")
    if items and total is not None:
        vals = [normalise(f.value) for f in items]
        tot = normalise(total.value)
        ok = None not in vals and tot is not None and abs(sum(vals) - tot) <= reconcile_tol
        for f in items + [total]:
            f.checks["reconciles"] = ok      # default: zero tolerance, in cents


# ------------------------------------------------------------ confidence
def field_score(f: ExtractedField, q: float, agrees: bool | None) -> float | None:
    """Composite confidence: perception, layout and validation channels."""
    if f.conf is None:
        return None                       # nothing to fuse; the router fails closed
    s = 0.7 * f.conf + 0.3 * q
    if not all(f.checks.values()):
        s *= 0.25                         # a failed check outranks any perception signal
    if agrees is False:
        s *= 0.8                          # the cross-check extractor read something else
    return round(s, 4)


# --------------------------------------------------------------- routing
@dataclass
class Thresholds:
    refuse_below: float = 0.35            # document quality
    review_below: float = 0.55            # document quality
    field_auto: float = 0.67              # composite confidence, critical fields
    layout_min: float = 0.50              # unknown layout goes to a person


def route(doc: Document, th: Thresholds, q: float) -> tuple[Route, list[str]]:
    """Auto, review or refuse, with the reasons a reviewer will see."""
    if q < th.refuse_below:
        return Route.REFUSE, [f"quality {q:.2f} < {th.refuse_below}"]
    why = []
    if q < th.review_below:
        why.append(f"quality {q:.2f} < {th.review_below}")
    if doc.quality.layout_match < th.layout_min:
        why.append("unknown layout")
    for f in doc.fields:
        if not f.critical:
            continue
        if f.score is None:
            why.append(f"{f.name}: no confidence")      # fail closed
        elif f.score < th.field_auto:
            why.append(f"{f.name}: {f.score:.2f} < {th.field_auto}")
    return (Route.REVIEW, why) if why else (Route.AUTO, [])


def route_fail_open(doc: Document, th: Thresholds, q: float) -> Route:
    """The bug this chapter warns about: a missing score counts as certain."""
    if q < th.refuse_below:
        return Route.REFUSE
    worst = min((f.score or 1.0) for f in doc.fields if f.critical)
    return Route.AUTO if worst >= th.field_auto and q >= th.review_below else Route.REVIEW


# ---------------------------------------------------------------- metrics
def field_correct(f: ExtractedField, amount_tol: Decimal = Decimal("0")) -> bool:
    """Per-field accuracy with a tolerance per kind (amounts: exact cents)."""
    if f.kind == "name":
        return (f.value or "").strip().lower() == (f.truth or "").strip().lower()
    if f.kind == "amount":
        a, b = normalise_amount(f.value), normalise_amount(f.truth)
        return a is not None and b is not None and abs(a - b) <= amount_tol
    return f.value == f.truth


def auroc(scores: list[float], positives: list[bool]) -> float:
    """Probability that a correct field outscores a wrong one."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """Population stability index between two samples of a score in [0, 1]."""
    import math
    def hist(xs):
        h = [0] * bins
        for x in xs:
            h[min(bins - 1, max(0, int(x * bins)))] += 1
        return [(c + 0.5) / (len(xs) + 0.5 * bins) for c in h]
    e, a = hist(expected), hist(actual)
    return sum((ai - ei) * math.log(ai / ei) for ei, ai in zip(e, a))
