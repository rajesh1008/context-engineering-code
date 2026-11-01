import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import lakeshore47 as L  # noqa: E402
from contextlab.multiagent import (VERSION, Brief, Delegation, delegate, Fact, Handle, HandoffRecord,  # noqa: E402
                                   LoopSpec, PacketSpec, SharedState, brief_gaps, check_spec,
                                   derive, preflight, provenance_coverage, summary_probe,
                                   validate)


def test_version():
    assert VERSION == "10.5"


def test_contract_rejects_missing_and_uncited_and_agent_written():
    rec = HandoffRecord("C1", "intake", "coverage", {
        "peril": Fact("peril", "water", (Handle("s1", "v1"),), by="agent:intake"),
        "endorsements": Fact("endorsements", ["HO 04 95"], (), by="agent:intake")})
    issues = validate(rec, L.CONTRACT)
    assert "missing date_of_loss" in issues
    assert "endorsements has no handle" in issues
    assert any("contract says parser" in i for i in issues)
    assert validate(L.record_for(L.world()[0]), L.CONTRACT) == []


def test_summary_probe_finds_dropped_constraint():
    s = "Water in basement after storm on 14 March; HO 3 policy."
    assert summary_probe(s, ["HO 04 95", "14 March"]) == ["HO 04 95"]


def test_packet_spec():
    spec = PacketSpec("planner", ("record", "status"), 3000, handles_only=("record",),
                      excludes=("spans",))
    bad = {"record": (900, "text"), "spans": (4000, "text")}
    issues = check_spec(spec, bad)
    assert len(issues) == 3
    assert check_spec(spec, {"record": (200, "handle"), "status": (100, "text")}) == []


def test_brief_gaps():
    assert brief_gaps(L.FULL_BRIEF, ("state", "date_of_loss")) == []
    gaps = brief_gaps(L.THIN_BRIEF, ("state", "date_of_loss"))
    assert "fact:date_of_loss" in gaps and "output_format" in gaps


def test_single_writer():
    st = SharedState({"determination": "coverage"})
    assert st.write("coverage", "determination", "d1")
    assert not st.write("review", "determination", "d2")
    assert st.values["determination"] == "d1" and len(st.refused()) == 1


def test_derived_tier_is_minimum():
    d = derive("x", "agent:a", (Handle("s", "v", "governed"), Handle("t", "v", "party")))
    assert d.tier == "party"
    assert derive("y", "agent:a", (Handle("s", "v", "governed"),)).tier == "governed"
    assert derive("z", "agent:a", ()).tier == "public"
    assert provenance_coverage([d, derive("z", "agent:a", ())]) == 0.5


def test_preflight():
    ok = LoopSpec("event:x", "a", "v", "store@current", "done", "s.json", 10, "Tom")
    assert preflight(ok) == []
    assert "verifier shares the assembler's view" in preflight(
        LoopSpec("event:x", "a", "a", "store@current", "done", "s.json", 10, "Tom"))


def test_lab_numbers_reproduce():
    out = subprocess.run([sys.executable, os.path.join(HERE, "lab.py")],
                         capture_output=True, text=True, check=True).stdout
    assert "record" in out
    r = json.load(open(os.path.join(HERE, "out", "results.json")))
    s = r["strategies"]
    assert s["single"]["wrong"] == 3 and s["record"]["after_review"] == 3
    assert s["summary-injected"]["wrong"] > 5 * s["summary"]["wrong"]
    assert s["share"]["coverage_tokens"] > s["record"]["coverage_tokens"]
    assert r["research"]["isolated-full"]["wrong_rate"] < r["research"]["inline"]["wrong_rate"]
    assert r["research"]["isolated-thin"]["wrong_rate"] > r["research"]["inline"]["wrong_rate"]
    assert r["provenance"]["record"]["coverage"] == 1.0
    assert r["loop"]["preflight_good"] == [] and r["loop"]["flagged_shared_view"] == 0


def test_delegation_narrows_never_widens():
    import pytest
    root = Delegation("tom.brennan", "C0001", ("coverage",), frozenset({"forms", "bulletins", "guidelines"}), 30000)
    sub = delegate(root, "research", corpora={"bulletins"}, budget=12000)
    assert sub.user == "tom.brennan" and sub.actors == ("coverage", "research")
    with pytest.raises(ValueError):
        delegate(sub, "deeper", corpora={"bulletins", "guidelines"})
    with pytest.raises(ValueError):
        delegate(sub, "deeper", budget=20000)
