"""Chapter 42 tests. Run per chapter: pytest -q ch42 (from 04-code)."""
import copy

from contextlab.observe import Alarm, Ewma, Ledger, Rule, Store, h, p_limits, reconstruct, reconstructability
from lakeshore42 import BAD, FIXED, PII, Config, Sinks, canary, corpus, make_claim, monitor, render, run
import lab


def test_replay_wrong_clause_and_cause():
    L, S, _ = lab.instrumented(PII)
    wrong = [e for e in L.events if lab.wrong(e)]
    assert len(wrong) == 4
    for e in wrong:
        g = next(o for o in e.offered if o.span_id == e.output["governing"])
        assert g.decision == "dropped" and g.reason == "budget"


def test_reconstruct_exact_and_chain():
    L, S, packets = lab.instrumented(PII, n=200)
    both = lab._both(S)
    assert reconstructability(L.events, both, render)["exact"] == 200
    e = L.events[17]
    assert reconstruct(e, both, render)["packet"] == packets[e.det_id]
    assert L.verify() is None
    T = copy.deepcopy(L); T.events[50].output["status"] = "x"
    assert T.verify() == 50
    T = copy.deepcopy(L); del T.events[80]
    assert T.verify() == 80


def test_bad_redaction_rule_fails_gate_and_scoped_rule_passes():
    L, S, _ = lab.instrumented(PII + [BAD], n=200)
    g = reconstructability(L.events, lab._both(S), render)
    assert g["exact"] < 200
    L2, S2, _ = lab.instrumented(PII + [FIXED], n=200)
    assert reconstructability(L2.events, lab._both(S2), render)["exact"] == 200
    assert lab.leaks(S2.oplog) == 0


def test_no_redaction_leaks_personal_data():
    L, S, _ = lab.instrumented([], n=100)
    assert lab.leaks(S.oplog) > 0


def test_store_dedups_evidence():
    L, S, _ = lab.instrumented(PII, n=300)
    assert len(S.evidence.blobs) < 100 < S.evidence.writes


def test_rule_scope():
    r = Rule("x", r"HO 04 95", ("operational",))
    assert r.apply("HO 04 95", "evidence") == ("HO 04 95", 0)
    assert r.apply("HO 04 95", "operational")[1] == 1


def test_monitor_misses_more_on_long_packets():
    docs = corpus(0)
    short = long = 0
    for i in range(300):
        c = make_claim(5, i, "t")
        ev, _ = run(c, docs, Config(), Sinks(rules=PII), Ledger())
        short += monitor(canary(ev), f"s{i}")
        ev2, _ = run(c, docs, Config(budget=900, k=12), Sinks(rules=PII), Ledger())
        ev2.cache["input"] = 1400
        long += monitor(canary(ev2), f"s{i}")
    assert short > long


def test_alarm_and_ewma():
    lo, hi = p_limits(0.99, 100)
    assert lo < 0.99 < hi
    a = Alarm("x", 0.99, run=2)
    assert not a.observe(0, 99, 100)
    a.observe(1, 80, 100)
    assert a.observe(2, 80, 100) and a.fired == [2]
    e = Ewma(0.99, 0.003)
    assert not e.observe(0, 0.99)
    assert any(e.observe(d, 0.95) for d in range(1, 5))


def test_hash_is_content_addressed():
    s = Store()
    k1 = s.put("same text", "evidence"); k2 = s.put("same text", "evidence")
    assert k1 == k2 == h("same text") and len(s.blobs) == 1
