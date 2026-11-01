"""Tests for contextlab.docgate and the Chapter 11 lab. Run: python -m pytest 04-code/ch11"""
import os, random, sys
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import lab
from contextlab.docgate import (Document, ExtractedField, Quality, Route, Thresholds, auroc,
    check_digit_ok, field_correct, field_score, normalise_amount, normalise_amount_rounding,
    quality_score, route, route_fail_open, validate)


def doc_with(fields, q=Quality(True, 0.99, 300, 0, 0, 0.95)):
    return Document("D", "estimate", "L06", q, fields)


def test_missing_page_scores_zero():
    q = Quality(True, 0.99, 300, 0, 0, 0.95, pages_found=1, pages_expected=2)
    assert quality_score(q) == 0.0


def test_missing_confidence_fails_closed():
    f = ExtractedField("total_amount", "amount", "10.00", None, True)
    d = doc_with([f]); validate(d); f.score = field_score(f, 0.95, True)
    r, why = route(d, Thresholds(), 0.95)
    assert r == Route.REVIEW and "no confidence" in why[0]
    assert route_fail_open(d, Thresholds(), 0.95) == Route.AUTO      # the bug, kept for the lab


def test_reconciliation_is_exact_to_the_cent():
    items = [ExtractedField("item_00", "amount", "10.50", .99), ExtractedField("item_01", "amount", "4.50", .99)]
    tot = ExtractedField("total_amount", "amount", "15.00", .99, True)
    d = doc_with(items + [tot]); validate(d)
    assert tot.checks["reconciles"]
    tot.value = "15.01"; validate(d)
    assert not tot.checks["reconciles"]


def test_rounding_normaliser_passes_tolerance_and_fails_round_trip():
    v = "1,234.50"
    assert abs(normalise_amount_rounding(v) - normalise_amount(v)) <= Decimal("1.00")
    assert normalise_amount_rounding(v) != normalise_amount(v)


def test_amount_tolerance_is_zero_by_default():
    f = ExtractedField("x", "amount", "100.00", .9, truth="100.40")
    assert not field_correct(f)
    assert field_correct(f, amount_tol=Decimal("1.00"))


def test_check_digit_and_auroc():
    assert check_digit_ok(lab.luhn_complete("123456789"))
    assert auroc([0.9, 0.8, 0.2], [True, True, False]) == 1.0


def test_lab_is_deterministic_and_hits_its_numbers():
    live = [lab.make_truth(i, random.Random(12 * 0 + i)) for i in range(1, 401)]
    a = lab.summarise(lab.run_gate(live, tag="t"))
    b = lab.summarise(lab.run_gate(live, tag="t"))
    assert a == b
    assert 0.04 < a["review"] / a["n"] < 0.25
    assert a["post_err"] / a["admitted"] < a["raw_bad"] / a["n"]


def test_output_matches_chapter():
    lab.main()
    out = open(os.path.join(lab.OUT, "output.txt")).read()
    assert "review 421 (10.5%)" in out
    assert "post-review error 33 of 3963 (0.8%)" in out
    assert "-> 36 of 1000 (3.6%)" in out
    assert "within the old ±US$1.00 tolerance 100.0%" in out


def test_normaliser_round_trips_every_printed_amount():
    for i in range(1, 301):
        t = lab.make_truth(i, random.Random(f"lab-{i}"), layout=lab.LAYOUTS[(i - 1) % 20])
        for name, kind, critical, printed in t.fields:
            if kind == "amount":
                assert normalise_amount(printed) == Decimal(printed.replace(",", ""))
