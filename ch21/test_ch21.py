from datetime import date

import lab as L
from contextlab.memory import (Candidate, Memory, MemoryRecord, ReadRequest, WritePolicy,
                               in_scope, read)

D = date(2026, 3, 1)


def cand(ctype, src="claimant", field=None, value=None, when=D, tenant="lakeshore"):
    return Candidate("claimant:C1", ctype, f"Carter: {ctype}", src, "CL-1/E01", when, tenant,
                     field, value)


def test_write_policy_refuses_what_a_system_of_record_owns():
    pol = WritePolicy()
    assert pol.route(cand("policy_fact"))[0] is None
    assert pol.route(cand("status"))[0] is None
    assert pol.route(cand("preference"))[0] == "semantic"
    assert pol.route(cand("event", "system"))[0] == "episodic"


def test_a_model_may_not_certify_its_own_inference():
    kind, reason = WritePolicy().route(cand("inference", "model"))
    assert kind is None and "model" in reason


def test_newer_value_supersedes_and_one_value_stays_current():
    m = Memory()
    a = m.write(cand("preference", field="contact.channel", value="email"))
    b = m.write(cand("preference", field="contact.channel", value="phone", when=date(2026, 5, 1)))
    assert m.current("claimant:C1", "contact.channel").value == "phone"
    assert m.records[a.mid].superseded_by == b.mid and b.supersedes == a.mid


def test_reader_drops_other_tenants_and_superseded_records():
    m = Memory()
    m.write(cand("preference", field="contact.channel", value="email"))
    m.write(cand("preference", field="contact.channel", value="phone", when=date(2026, 5, 1)))
    m.write(cand("household", field="authorised_contact", value="spouse", tenant="fvb"))
    q = ReadRequest({"claimant:C1"}, ("HO", "WI", "water"), "lakeshore", "adjuster", date(2026, 6, 1))
    got = read(m, q, lambda r: 1.0)
    assert [r.value for r in got] == ["phone"]


def test_procedural_scope_matches_wildcards_but_not_other_states():
    r = MemoryRecord("M1", "procedural", "procedure:HO/*/roof", "x", "lakeshore", "consolidation",
                     "c", "job", D, 0.85, scope=("HO", "*", "roof"))
    assert in_scope(r, set(), ("HO", "MN", "roof"), "lakeshore", "adjuster")
    assert not in_scope(r, set(), ("HO", "MN", "water"), "lakeshore", "adjuster")


def test_reader_respects_the_token_budget():
    m = Memory()
    for i in range(30):
        m.write(Candidate("claimant:C1", "event", "Carter: prior claim " + "word " * 20, "system",
                          f"CL-{i}", D))
    q = ReadRequest({"claimant:C1"}, ("HO", "WI", "water"), "lakeshore", "adjuster", D,
                    budget_tokens=80)
    assert sum(len(r.text.split()) for r in read(m, q, lambda r: 1.0)) * 1.33 <= 80


def test_replay_adds_wrong_determinations_only_with_memory_on():
    dets, spans, ix, g, pol, people = L.world()
    eps, cands = L.histories(people)
    mem = L.build_memory(people, cands, WritePolicy())
    L.lessons(mem)
    on = L.L15.score(*L.replay(dets, spans, ix, g, pol, people, mem, True)[:2], spans)
    off = L.L15.score(*L.replay(dets, spans, ix, g, pol, people, mem, False)[:2], spans)
    assert off["wrong"] == 16 and on["wrong"] == 22
