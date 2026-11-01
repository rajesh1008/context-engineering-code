"""Tests for contextlab.multimodal and the Chapter 19 lab."""
import lab
import lakeshore19 as L
from contextlab import multimodal as M


def test_image_tokens_match_published_examples():
    assert M.image_tokens(1000, 1000)[0] == 1296
    assert M.image_tokens(1920, 1080)[0] == 1560
    assert M.image_tokens(1920, 1080, "high")[0] == 2691
    assert M.image_tokens(3840, 2160, "high")[0] == 4784


def test_forbidden_kinds_are_refused():
    r = M.Region(0, 0, 10, 10)
    assert M.check(M.Observation("cause", "none", r, 0.99)) is not None
    assert M.check(M.Observation("flood", "floor_drain", r, 0.99)) is not None
    assert M.check(M.Observation("residue_at", "floor_drain", r, 0.5)) is not None
    assert M.check(M.Observation("residue_at", "floor_drain", r, 0.9)) is None


def test_provenance_flags_far_photo():
    s = M.ImageSpan("X-P1", "sha256:0", "X", "policyholder", "2026-06-02T10:00",
                    (44.56, -87.98), "hail damage to window", "m")
    flags = M.provenance_flags(s, (44.5133, -88.0133), "2026-06-01")
    assert flags and flags[0].startswith("captured_") and "km_from_risk" in flags[0]
    s.gps, s.captured_at = None, None
    assert M.provenance_flags(s, (44.5, -88.0), "2026-06-01") == ["no_capture_metadata"]


def test_guard_routes_dropped_negative():
    heard = M.Segment(0.0, 3.6, "Water came in at the window wells.", 0.81, 10.9)
    assert M.guard(heard, L.TERMS) == "listen"
    clean = M.Segment(0.0, 3.6, "No water came in at the window wells.", 0.93, 24.0)
    assert M.guard(clean, L.TERMS) == "use"
    filler = M.Segment(0.0, 3.0, "Homeowner was present.", 0.60, 8.0)
    assert M.guard(filler, L.TERMS) == "use"


def test_wer_hides_a_flip():
    ref = "no water came in at the window wells"
    assert round(M.wer(ref, "water came in at the window wells"), 3) == 0.125


def test_lab_scoreboard_shape():
    cs, rows, *_ , carter = lab.run()
    w = {k[0]: r["wrong"] for k, r in rows.items()}
    assert len(cs) == 240
    assert w["F"] < w["D"] < w["E"] < w["B"] < w["C"] < w["A"]
    assert rows["D + image spans"]["asserts"] == 0
    assert rows["D + image spans"]["flagged_used"] == 0
    assert carter["F + voice spans (guard)"].startswith("HO 04 95")
    assert carter["E + raw transcript"].endswith("(surface water)")


def test_lab_is_deterministic():
    a = [r["wrong"] for r in lab.run()[1].values()]
    b = [r["wrong"] for r in lab.run()[1].values()]
    assert a == b
