from datetime import date, datetime, timedelta

import fullmix as F
import inventory_lakeshore as L
import sql_context as Q
from contextlab.knowledge import Record

NOW = datetime(2026, 10, 30, 9, 0)
LOSS = date(2026, 2, 7)


def test_signed_inventory_has_no_gaps_and_draft_has_three():
    assert L.signed().gaps() == []
    assert len(L.draft().gaps()) == 3


def test_inferred_version_fails_closed_and_is_quarantined():
    inv = L.signed()
    rec = Record("LM 112#scan-0457", "scanned-documents", "...",
                 version="2026-03-14", version_origin="file-mtime")
    assert inv.offer(rec, on=LOSS, now=NOW) is None
    assert inv.quarantine[-1][1] == "Claims operations"


def test_missing_version_fails_closed():
    inv = L.signed()
    rec = Record("HO-3#scan-0001", "policy-forms", "...", version=None)
    assert inv.by_name("policy-forms").admit(rec, on=LOSS, now=NOW).reason == "no version"


def test_licence_gate_refuses_accurate_data():
    src = L.signed().by_name("property-data-feed")
    rec = Record("p#roof", src.name, "...", "2026-10-29", version_origin="system",
                 observed_at=NOW - timedelta(days=1))
    assert not src.admit(rec, on=LOSS, now=NOW).ok
    assert src.admit(rec, on=LOSS, now=NOW, role="underwriter",
                     purpose="underwriting").ok


def test_verifier_rejects_query_without_date_predicate():
    db = Q.connect([("P1", "HO-3", "2022-01", "2023-04-01", "2025-06-01", 3),
                    ("P1", "HO-3", "2024-06", "2025-06-01", None, 4)])
    ok, why, _, rows = Q.ask(db, "P1", date(2025, 5, 1))
    assert ok and rows[0][1] == "2022-01"
    ok, why, _, rows = Q.ask(db, "P1", date(2025, 5, 1), fault=True)
    assert not ok and len(rows) == 2


def test_scoreboard_numbers_match_chapter():
    dets = F.build()
    assert len(dets) == 1000
    assert F.score([d for d in dets if d.split == "eval"], "ch08")["wrong"] == 5
    assert F.score([d for d in dets if d.split == "holdout"], "ch08")["wrong"] == 2
    assert F.score(dets, "ch08")["wrong"] == 82
    s = F.score(dets, "ch09")
    assert (s["wrong"], s["withheld"]) == (50, 41)
