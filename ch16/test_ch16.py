from decimal import Decimal

import pytest

import lab
import lakeshore16 as L
from contextlab import structured as S


@pytest.fixture(scope="module")
def world():
    db, policies = L.build()
    return db, policies, L.layer(), lab.determinations(policies)


def test_round_trip_check_refuses_thousands_separators():
    rows = [["CL-1", "2025-03-02", "sewer_backup", Decimal("4250.00")]]
    units = ["id", "date", "text", "USD"]
    assert "4250.00" in S.render(["c", "d", "p", "paid"], rows, units)
    with pytest.raises(ValueError):
        S.render(["c", "d", "p", "paid"], rows, units, grouping=True)


def test_verifier_catches_premium_fan_out(world):
    db, _, sl, _ = world
    m = sl.measures["term_premium"]
    ok, why = S.verify(lab.NAIVE["term_premium"], m,
                       L.PUBLISHED | {"v_term_endorsements"}, 2, 1)
    assert not ok and why.startswith("fan-out")
    ok, why = S.verify(lab.NAIVE["term_premium"], m, L.PUBLISHED, 2, 2)
    assert not ok and "not allowed" in why


def test_structured_spans_are_faithful_and_passages_are_not(world):
    db, _, sl, dets = world
    res_s, res_p = [], []
    for d in dets[:60]:
        term, t = lab.truth(db, sl, d)
        res_s.append((lab.structured(db, sl, d, term)[0], t, d["asks"]))
        res_p.append((lab.passages(db, d, term)[0], t, d["asks"]))
    st, rt, _ = lab.score(res_s)
    assert st == rt
    st, rt, _ = lab.score(res_p)
    assert rt < st


def test_unknown_measure_is_withheld_not_improvised(world):
    db, _, sl, dets = world
    d = next(x for x in dets if "paid_all_perils_term" in x["asks"])
    term = lab.term_of(db, d["pol"], d["loss"])
    ans = lab.structured(db, sl, d, term)[0]
    assert ans["paid_all_perils_term"] is None


def test_span_header_carries_provenance(world):
    db, _, sl, dets = world
    d = dets[0]
    term = lab.term_of(db, d["pol"], d["loss"])
    spans = lab.structured(db, sl, d, term)[3]
    h = spans[0].header()
    assert "layer sl-1.2" in h and "query " in h and "as of 2026-10-30" in h
