"""Chapter 46 tests: contract, entitlement and budget behaviour of contextlab.platform."""
import json
import os
from dataclasses import replace

import pytest

from contextlab.platform import (Platform, Principal, issue_token, entitled, render,
                                 verify_packet, sign, SLO, Ledger, Charge, TeamBudget,
                                 simulate, pct, TIERS)
from lakeshore46 import CORPORA, build, people, BUDGETS

SPANS, FORK, AGENCY_OF = build()
CORP = {c.name: c for c in CORPORA}
RES = os.path.join(os.path.dirname(__file__), "out", "results.json")


@pytest.fixture
def plat():
    d = people(AGENCY_OF)
    return Platform(SPANS, CORPORA, d, dict(BUDGETS)), d


def test_identity_comes_from_the_token(plat):
    pf, d = plat
    forged = {"user": "adj-01", "service": "agent-assist", "sig": "0" * 16}
    pk, st = pf.assemble(forged, "P0001", "T01", 2000, 9.0)
    assert pk is None and st.startswith("401")


def test_private_corpus_and_roles(plat):
    pf, d = plat
    pk, _ = pf.assemble(issue_token("agent-01-1", "agent-assist"), "P0041", "T00", 2000, 9.0)
    corpora = {s.corpus for s in pk.spans}
    assert "claim-files" not in corpora and "guidelines" not in corpora
    assert "letter-templates" not in corpora and "producer-guides" in corpora


def test_agent_outside_book_and_other_policyholder(plat):
    pf, d = plat
    pk, _ = pf.assemble(issue_token("agent-01-1", "agent-assist"), "P0002", "T05", 2000, 9.0)
    assert not any(s.policy_id for s in pk.spans)
    pk, _ = pf.assemble(issue_token("ph-0003", "ph-assist"), "P0004", "T05", 1200, 9.0)
    assert not any(s.policy_id for s in pk.spans)
    assert ("dec-P0004", "another policyholder's document") in pk.withheld


def test_superseded_withheld_and_governing_first(plat):
    pf, d = plat
    pk, _ = pf.assemble(issue_token("adj-02", "det-assist"), "P0100", "T03", 3000, 9.0)
    assert ("form-T03@2024", "superseded") in pk.withheld
    tiers = [TIERS.index(s.tier) for s in pk.spans]
    assert tiers == sorted(tiers)


def test_explain_accounts_for_every_candidate(plat):
    pf, d = plat
    pk, _ = pf.assemble(issue_token("agent-07-2", "agent-assist"), "P0407", "T03", 2000, 9.0)
    ex = pf.explain(pk.packet_id)
    n = len(pf.by_topic["T03"]) + len(pf.by_subject["P0407"])
    assert len(ex["included"]) + len(ex["withheld"]) == n
    assert pf.feedback("nope", "missing")["status"] == "404"


def test_verify_packet_rejects_reuse(plat):
    pf, d = plat
    pk, _ = pf.assemble(issue_token("agent-07-2", "agent-assist"), "P0407", "T03", 2000, 9.0)
    me, other = d.users["agent-07-2"], d.users["agent-07-1"]
    assert verify_packet(pk, me, "P0407", 9.1, d.epoch, pf.key) == (True, "ok")
    assert verify_packet(pk, other, "P0407", 9.1, d.epoch, pf.key)[1] == "audience"
    assert verify_packet(pk, me, "P0408", 9.1, d.epoch, pf.key)[1] == "subject"
    assert verify_packet(pk, me, "P0407", 9.5, d.epoch, pf.key)[1] == "expired"
    d.move_policy("P0407", "A08")
    assert verify_packet(pk, me, "P0407", 9.1, d.epoch, pf.key)[1] == "entitlement epoch"
    forged = replace(pk, spans=pk.spans + [SPANS[0]])
    assert verify_packet(forged, me, "P0407", 9.1, 1, pf.key)[1] == "signature"


def test_budget_caps(plat):
    pf, d = plat
    tok = issue_token("agent-07-2", "agent-assist")
    assert pf.assemble(tok, "P0407", "T03", 2001, 9.0)[1].startswith("400")
    pf.budgets["agency"] = TeamBudget("agency", 500, 2000, 3, hard=True)
    assert pf.assemble(tok, "P0407", "T03", 2000, 9.0)[1].startswith("429")
    pf.budgets["agency"] = TeamBudget("agency", 500, 2000, 3, hard=False)
    assert pf.assemble(tok, "P0407", "T03", 2000, 9.0)[1] == "200"


def test_chargeback_allocates_all_shared_cost():
    lg = Ledger()
    lg.add(Charge("a", 1000, 10, 0)); lg.add(Charge("b", 3000, 30, 0))
    prices = {"per_unit": 0.01, "per_token": 0.001}
    for m in ("usage", "even"):
        bill = lg.bill(prices, 100.0, m)
        assert abs(sum(v["shared"] for v in bill.values()) - 100.0) < 1e-9
    assert lg.bill(prices, 100.0, "usage")["b"]["shared"] == pytest.approx(75.0)


def test_contract_v2_breaks_two_consumers():
    r = json.load(open(RES))
    failing = [(c, n) for c, n, ok in r["contract"]["v2"] if not ok]
    assert all(ok for *_, ok in r["contract"]["v1"])
    assert sorted(c for c, _ in failing) == ["agency", "policyholder"]


def test_fair_shares_protect_interactive():
    reqs = [(i * 0.5, "ph", "interactive", 0.2) for i in range(200)]
    reqs += [(0.0 + i * 0.01, "ag", "batch", 0.3) for i in range(1000)]
    fifo = simulate(reqs, 4, "fifo")
    fair = simulate(reqs, 4, "fair", {"ph": 3, "ag": 2})
    p = lambda done: pct([l for t, k, l, _ in done if t == "ph"], 0.95)
    assert p(fair) < 1.0 < p(fifo)


def test_slo_without_error_budget():
    assert SLO("entitlement", 1.0, budgeted=False).evaluate(599, 600)["met"] is False
    assert SLO("availability", 0.999).evaluate(9_995, 10_000)["met"] is True


def test_lab_injection_results():
    r = json.load(open(RES))["injection"]
    assert r["client cache, no check"]["leak_calls"] > 0
    assert r["client cache, gateway verifies"]["leak_calls"] == 0
    assert r["platform cache by entitlement class"]["leak_calls"] == 0
    assert json.load(open(RES))["entitlement"]["platform, identity from token"]["spans"] == 0
