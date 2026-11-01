"""Chapter 50 tests. Run per chapter: pytest -q ch50 (from 04-code)."""
from contextlab.casestudies import (ROWS, LayerCard, ablate, compare, fit_band, outcome,
                                    rare_event_exit, route, score, transplant, wilson)
import cases50 as C


def test_card_rows_and_compare():
    c = LayerCard("x", {"unit": "u", "as_of": "d"})
    assert len(ROWS) == 12 and "handback" in c.missing()
    m = compare(C.cards())
    assert m["handback"]["Lakeshore"] is False and m["handback"]["Boston hospital"] is True
    assert m["exit"]["Dutch bank"] is True


def test_outcome_rules():
    assert outcome("A", "A") == "right"
    assert outcome("ASK", "A") == "deferred" and outcome("B", "A") == "wrong"
    assert outcome("HANDBACK", "HANDBACK") == "right"


def test_dutch_releases_and_transplant():
    topics = C.dutch_corpus(); qs = C.dutch_questions(topics)
    run = lambda cfg: score(qs, C.dutch_answer(topics, cfg))  # noqa: E731
    v1, v2, v3, lk = run(C.DUTCH_V1), run(C.DUTCH_V2), run(C.DUTCH_V3), run(C.LAKESHORE)
    assert (v1.wrong, v2.wrong, v3.wrong, lk.wrong) == (205, 113, 1, 168)
    ab = ablate(C.DUTCH_V3, C.LAKESHORE, run)
    assert ab["rank"].wrong == 61 and ab["rank"].wrong_by == {"comply-or-explain": 61}
    assert transplant(C.DUTCH_V3, C.LAKESHORE, ["rank"])["rank"] == "fixed"


def test_boston_dublin_austin():
    gl, qs = C.boston_questions()
    assert score(qs, C.boston_answer(gl, C.BOSTON_V1)).wrong == 142
    assert score(qs, C.boston_answer(gl, C.BOSTON_V2)).wrong == 1
    room = C.dublin_room(); dq = C.dublin_questions(room)
    v2 = score(dq, C.dublin_answer(room, C.DUBLIN_V2))
    assert score(dq, C.dublin_answer(room, C.DUBLIN_V1)).wrong == 39 and (v2.wrong, v2.deferred) == (0, 11)
    custs, aq = C.austin_questions()
    assert score(aq, C.austin_answer(custs, C.AUSTIN_V1)).wrong == 91
    assert score(aq, C.austin_answer(custs, C.AUSTIN_V2)).deferred == 2


def test_gate_and_exit_and_wilson():
    assert route(0.9, 0.8, 0.55) == "auto" and route(0.6, 0.8, 0.55) == "review"
    assert route(0.5, 0.8, 0.55) == "escalate"
    assert fit_band(C.dublin_scans(), 0.55, [0.8, 0.85, 0.9], 200)[0] == 0.8
    assert rare_event_exit([True, False], 0, 6)[0] == "hold"
    assert rare_event_exit([True] * 3, 2, 6)[0] == "human-in-loop"
    assert rare_event_exit([True] * 3, 6, 6)[0] == "exit"
    lo, hi = wilson(1, 600)
    assert lo < 0.002 < hi < 0.01
