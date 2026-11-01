from datetime import date
from dataclasses import replace

import corpus15 as C
import lab as L
from contextlab.policy import Policy, Request, lineage

SP = C.governing_spans() + C.context_spans()
POL = Policy(SP)
ID = {s.sid: s for s in SP}


def req(**kw):
    base = dict(role="adjuster", tenant="lakeshore", purpose="coverage-determination", line="HO",
                policy_state="MN", risk_state="MN", garaging_state="MN",
                editions={"HO-3": "2024-06", "PAP": "2018-01"}, endorsements=frozenset(),
                date_of_loss=date(2025, 9, 1), cause="roof")
    base.update(kw)
    return Request(**base)


def test_successor_ends_a_bulletin_that_never_recorded_its_end():
    old = ID["MN-COM-2023-02@2023-02-10#roof"]
    assert old.effective_to is None
    v = POL.in_force(old, date(2025, 9, 1))
    assert not v.ok and "MN-COM-2025-06" in v.reason
    assert POL.in_force(old, date(2024, 9, 1)).ok          # the date of loss decides


def test_bulletin_follows_the_property_and_endorsement_follows_the_contract():
    r = req(policy_state="MN", risk_state="WI", date_of_loss=date(2025, 9, 1))
    assert POL.in_jurisdiction(ID["WI-OCI-2025-03@2025-03-15#roof"], r).ok
    assert not POL.in_jurisdiction(ID["MN-COM-2025-06@2025-06-02#roof"], r).ok
    assert POL.in_jurisdiction(ID["LM-HO 0122 MN@2021-01#notice"], r).ok


def test_unknown_jurisdiction_withholds():
    res = POL.resolve(["HO-3@2024-06#roof"], req(risk_state=None))
    assert res.status == "withhold" and res.reason == "jurisdiction unknown"


def test_precedence_is_scoped_to_the_clause_and_advice_never_governs():
    r = req(cause="mould", endorsements=frozenset({"HO 04 95@2024-06"}))
    res = POL.resolve(["HO 04 95@2024-06#water_backup", "HO-3@2024-06#mould", "NAIC-UCSP@2025"], r)
    assert res.governing == "HO-3@2024-06#mould"
    assert POL.precedence(ID["NAIC-UCSP@2025"], "payment") is None


def test_copied_restricted_text_inherits_its_restriction_and_a_paraphrase_does_not():
    uw = ID["UW-NOTE-0002@2018"]
    copied = replace(uw, sid="N1@1", doc_id="N1", kind="note", roles=C.ADJ,
                     text="Adjuster note, pasted: " + uw.text)
    para = replace(copied, sid="N2@1", doc_id="N2", text="Underwriting says back-up is never covered.")
    p = Policy(SP + [replace(copied, derived_from=lineage(copied, [uw])), para])
    assert not p.entitled(p.spans["N1@1"], req()).ok
    assert p.entitled(p.spans["N2@1"], req()).ok              # the open problem, Chapter 37


def test_full_mix_numbers_and_replay():
    dets = C.full_mix()
    spans, ix, g = L.build(dets)
    pol = Policy(list(spans.values()))
    base = L.run(dets, spans, ix, g, "ch14")
    full = L.run(dets, spans, ix, g, "policy", policy=pol)
    assert L.score(dets, base, spans, "clause")["wrong"] == 34
    assert L.score(dets, base, spans, "audit")["wrong"] == 38
    s = L.score(dets, full, spans, "audit")
    assert s["wrong"] == 16 and s["withheld"] == 15
    d = dets[86]
    a = pol.audit(d.did, d.req, pol.resolve(full[86]["packet"], d.req), full[86]["cited"], "c")
    b = Policy(list(spans.values())).audit(d.did, d.req, pol.resolve(full[86]["packet"], d.req), full[86]["cited"], "c")
    assert a["record"] == b["record"]
