"""Tests for contextlab.assembler (v6.7) and the Chapter 31 stand-in."""
import json
import os

import pytest

import lakeshore31 as W
from contextlab.assembler import (Assembler, CLAIMS_SCHEMA, CollisionError, Span, encode_table,
                                  order_spans, positional_id)
from contextlab.citations import Entry

DETS = W.L.build()
EMILY = DETS[187]
FORMATS = ("plain", "json", "markdown", "xml", "schema")


def asm(det, fmt="schema", order="schema", ids="short", packet=None, **kw):
    return Assembler(fmt, order, ids, **kw).assemble(W.INSTRUCTIONS, W.QUESTIONS[det.ctype],
                                                     W.roles(det, packet or det.packet))


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_span_text_survives_assembly(fmt):
    a = asm(EMILY, fmt, "retrieval", "none" if fmt == "plain" else "handle")
    assert all(e.text in a.text.replace('\\"', '"') for e in EMILY.packet)


def test_short_ids_are_assigned_after_ordering_and_frozen():
    a = asm(EMILY)
    assert a.ids[:3] == ["S1", "S2", "S3"]
    assert a.idmap["S1"] == a.shown[a.order[0].entry]
    h = a.to_handles("The loss is covered. [S5]")
    assert h.endswith(f"[[{a.order[4].entry.handle}]]")
    assert "[[unknown:S99]]" in a.to_handles("x [S99]")


def test_modifier_sits_next_to_the_clause_it_modifies():
    for d in [d for d in DETS if d.ctype == "backup_endorsed"]:
        assert W.relation(asm(d)) == "tagged"
    a = asm(EMILY)
    assert 'modifies="' in a.text and a.text.count("<question>") == 2


def test_positional_ids_collide_after_the_rechunk_and_the_check_refuses():
    packet = W.rechunk(EMILY.packet)
    with pytest.raises(CollisionError):
        asm(EMILY, ids="positional", packet=packet)
    a = asm(EMILY, ids="positional", packet=packet, check_ids=False)
    assert a.collisions and a.idmap[a.collisions[0]].text.startswith("Unless")
    for scheme in ("short", "handle"):
        assert asm(EMILY, ids=scheme, packet=packet).collisions == []


def test_schema_validation_names_what_is_missing():
    spans = W.roles(EMILY)
    assert CLAIMS_SCHEMA.validate(spans) == []
    no_facts = [s for s in spans if s.role != "facts"]
    assert "missing required section claim_facts" in CLAIMS_SCHEMA.validate(no_facts)
    orphan = [s for s in spans if not (s.entry.doc == "HO-3" and s.entry.page == 7 and s.entry.n == 1)]
    assert any("without the clause" in p for p in CLAIMS_SCHEMA.validate(orphan))


def test_elastic_trim_never_touches_governing_spans_by_default():
    a = asm(EMILY, trim_context=5)
    for s in a.order:
        same = a.shown[s.entry].text == s.entry.text
        assert same or s.role == "context"


def test_orders_are_permutations():
    spans = W.roles(EMILY)
    for how in ("retrieval", "document", "edges", "schema"):
        assert sorted(map(id, order_spans(spans, how))) == sorted(map(id, spans))


def test_table_encodings():
    rows = [dict(a=1, b="x, y"), dict(a=2, b="z")]
    toon = encode_table("t", rows, "toon")
    assert toon.splitlines()[0] == "t[2]{a,b}:" and '"x, y"' in toon
    assert json.loads(encode_table("t", rows, "json"))["t"] == rows


def test_reader_is_paired_across_assemblies():
    a, b = asm(EMILY, "markdown", "retrieval", "handle"), asm(EMILY)
    ra, rb = W.read(EMILY, a, 0), W.read(EMILY, b, 0)
    assert len(ra) == len(rb)
    assert W.corrupt("[S3]") == "[S4]" and W.corrupt("[[a#1234abcd]]") != "[[a#1234abcd]]"


def test_printed_results_hold_their_shape():
    p = os.path.join(os.path.dirname(__file__), "out", "results.json")
    if not os.path.exists(p):
        pytest.skip("run lab.py first")
    r = json.load(open(p))
    five = r["five"]
    assert five["schema"]["gold_p"] > five["markdown"]["gold_p"]
    assert abs(five["schema"]["faith"] - five["markdown"]["faith"]) < 0.01
    assert five["markdown"]["meas_p"] == five["xml"]["meas_p"] == five["json"]["meas_p"]
    inj = r["injection"]
    assert inj["positional ids, no check"]["wrong_text"] > 0
    assert inj["positional ids, check"]["refused"] > 0
    assert inj["short ids (v6.7)"]["wrong_text"] == 0
