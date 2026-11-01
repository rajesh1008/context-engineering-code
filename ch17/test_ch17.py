from datetime import date

import pytest

import lab as L
from contextlab.graph import SPINE, Candidate, Graph, compose, validate
from corpus17 import multi_hop, control

SP, POL, IDX = L.setup()
_, EDGES, G, _ = L.build(POL, SP)
R = L.Runner(SP, POL, IDX, G)
HARD, CTRL = multi_hop(), control()


def test_spine_rejects_a_relation_the_ontology_does_not_allow():
    c = Candidate("CG-HO-022", "MODIFIES", "HO 04 95", "CG-HO-022@v1", 0.9, "", "test")
    e = validate([c], POL)[0]
    assert e.status == "rejected" and "not allowed" in e.reason


def test_low_confidence_is_quarantined_not_published():
    sid = "FVB-HO 01@2023-01#water_backup"
    c = Candidate("FVB-HO 01", "REPLACES", "HO 04 95", sid, 0.5, "", "test")
    assert validate([c], POL)[0].status == "quarantined"


def test_edge_without_a_source_version_is_quarantined():
    c = Candidate("FVB-HO 01", "REPLACES", "HO 04 95", "nowhere@0", 0.95, "", "test")
    assert validate([c], POL)[0].status == "quarantined"


def test_contradictory_limits_in_one_edition_are_quarantined():
    sid = "LM-HO 2311@2025-01#mould"
    a = Candidate("LM-HO 2311", "LIMIT_WITHIN", "HO 04 95", sid, 0.9, "", "t")
    b = Candidate("LM-HO 2311", "LIMIT_IN_ADDITION", "HO 04 95", sid, 0.9, "", "t")
    assert {e.status for e in validate([a, b], POL)} == {"quarantined"}


def test_recursive_walk_reaches_a_bulletin_through_its_clause():
    reached = G.walk({"HO-3"}, date(2025, 9, 1), hops=2, rels=tuple(SPINE))
    assert reached["clause:roof"] == 1 and "WI-OCI-2025-03" in reached


def test_walk_respects_validity_dates():
    assert "WI-OCI-2025-03" not in G.walk({"clause:roof"}, date(2024, 9, 1), hops=1)


def test_unmet_requirement_marks_the_enhancement_inapplicable():
    q = next(q for q in HARD if q.template == "requirement unmet")
    ch = R.hybrid(q)[2]
    assert set(ch.inapplicable) == set(q.inapplicable) and ch.status == "ok"


@pytest.mark.parametrize("t", ["enhancement", "requirement unmet", "shared limit",
                               "additional limit", "endorsement and bulletin", "programme and bulletin"])
def test_hybrid_answers_every_template(t):
    assert all(R.hybrid(q)[0] == "correct" for q in HARD if q.template == t)


def test_hybrid_does_not_break_the_single_span_control():
    assert all(R.hybrid(q)[0] == "correct" for q in CTRL)


def test_lexical_configuration_cannot_answer_multi_hop():
    s, _ = L.score(R, HARD, R.ch16)
    assert s["correct"] == 0


def test_family_level_edges_let_the_superseded_edition_through():
    Ri = L.Runner(SP, POL, IDX, L.build(POL, SP, conflicts=None)[2])
    bad = [q for q in HARD if q.template == "shared limit" and Ri.hybrid(q, scoped=False)[0] == "wrong"]
    assert len(bad) == 8


def test_family_conflict_check_withholds_instead_of_answering_wrong():
    Ri = L.Runner(SP, POL, IDX, L.build(POL, SP, conflicts="family")[2])
    outs = {Ri.hybrid(q, scoped=False)[0] for q in HARD if q.template.endswith("limit")}
    assert outs == {"withheld"}


def test_quarantined_edge_withholds_only_where_it_could_apply():
    R0 = L.Runner(SP, POL, IDX, L.build(POL, SP, reviewed=False)[2])
    out = {(q.template, R0.hybrid(q)[0]) for q in HARD}
    assert ("requirement unmet", "withheld") in out and ("enhancement", "correct") in out


def test_bulletin_cites_the_clause_it_constrains_for_chapter_15s_six():
    from corpus15 import full_mix
    six = [d for d in full_mix() if d.carried and "two governing" in d.carried]
    assert len(six) == 6
    for d in six:
        ch = compose(G, POL, d.req, d.req.cause, cite_constrained=True)
        assert set(ch.evidence) == set(d.gold)


def test_ledger_is_dominated_by_people_not_tokens():
    rows, s = L.ledger(0.02, 0.32, 0.065, 6)
    assert all(r["steward"] + r["gate"] > 20 * r["extract"] for r in rows)
    assert s["per_determination"] < 0.1
