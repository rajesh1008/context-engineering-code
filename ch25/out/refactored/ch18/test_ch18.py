import corpus18 as C
import model18 as M
from contextlab.agentic import Budget, route

HARD = C.claims(M.CORPUS, 60, True, 181)


def test_corpus_shape():
    kinds = {s.kind for s in M.CORPUS.values()}
    assert kinds == {"form", "endorsement", "bulletin", "manual"}


def test_loop_beats_fixed_on_hard_claims():
    wrong = {w: sum(sc["wrong"] for _, _, sc in M.evaluate(HARD, w)) for w in ("fixed", "agentic")}
    assert wrong["agentic"] < wrong["fixed"]


def test_always_satisfied_judge_stops_after_one_round():
    for c in HARD[:10]:
        tr = M.agentic(c, judge="always")
        assert tr.rounds == 1 and tr.stop == "sufficient"


def test_budget_stops_a_check_that_never_passes():
    for c in HARD[:10]:
        tr = M.agentic(c, judge="never", budget=Budget(max_rounds=6))
        assert tr.stop in {"round-cap", "token-cap", "call-cap", "sufficient"}
        assert tr.rounds <= 6


def test_isolation_keeps_lead_context_small():
    c = HARD[0]
    assert max(M.agentic(c, isolate=True).lead) < max(M.agentic(c, isolate=False).lead)


def test_slot_check_fills_absences():
    c = HARD[0]
    tr = M.agentic(c)
    assert tr.stop == "sufficient"


def test_router():
    assert route({"perils": 1, "onward_refs": 0}).name == "fixed"
    assert route({"perils": 2, "onward_refs": 0}).name == "agentic"
    assert route({"perils": 1, "onward_refs": 1, "untrusted_edge": True}).name == "agentic"
