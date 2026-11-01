import copy
from dataclasses import replace

import pytest

import lakeshore43 as L
from contextlab.shadow import (DocStore, EditInPlace, ExitRule, Item, Pack, cell_stats, explain,
                               funnel, merkle, pooled_rule, redact, sha, wilson)

KEY = b"test-key"


@pytest.fixture(scope="module")
def run():
    return L.shadow_run()


def test_wilson_bounds():
    lo, hi = wilson(141, 155)
    assert 0.85 < lo < 0.91 < hi < 0.95
    assert wilson(0, 0) == (0.0, 1.0)


def test_rule_digest_is_stable_and_sensitive():
    r = ExitRule("v1")
    assert r.digest() == ExitRule("v1").digest()
    assert replace(r, max_excess=3).digest() != r.digest()


def test_funnel_adds_up(run):
    arrived, eligible, pairs = run
    f = dict(funnel(arrived, eligible, pairs))
    assert f["agree"] + f["disagree"] == f["final recorded"]
    assert f["candidate right"] + f["production right"] + f["both wrong"] == f["disagree"]


def test_per_cell_rule_holds_the_bad_cell_that_pooling_releases(run):
    _, _, pairs = run
    E = ExitRule("v1").evaluate(pairs)
    assert E["cells"][("HO", "MN")] == "hold: excess wrong"
    assert pooled_rule(pairs)["cells"][("HO", "MN")] == "release"
    assert all(v == "shadow" for c, v in E["cells"].items() if cell_stats(pairs)[c]["n"] < 60)


def test_pin_fixes_the_cell():
    _, _, pairs = L.shadow_run(pin=True)
    assert ExitRule("v1").evaluate(pairs)["cells"][("HO", "MN")] == "release"


def _pack():
    seg = lambda t, cls="", q=False: {"text": t, "cls": cls, "quoted": q}  # noqa: E731
    p = Pack("D-1", [Item("request", "claim", [seg("policyholder "), seg("A. Name", "policyholder")]),
                     Item("span", "note", [seg("Neighbour "), seg("R. O.", "third_party", True)]),
                     Item("tool", "pay", [seg("****"), seg("1234", "payment")])])
    p.seal(KEY)
    return p


def test_commitment_redaction_verifies_and_naive_does_not():
    good, bad = _pack(), _pack()
    r = redact(good, {"payment", "third_party"}, key=KEY)
    assert r["redacted"] == 1 and r["refused"] == [("note", "third_party")]
    assert good.verify(KEY) and "[REDACTED:payment]" in explain(good, KEY)
    redact(bad, {"payment"}, naive=True, key=KEY)
    assert not bad.verify(KEY)


def test_tamper_detected():
    p = _pack()
    p.items[0].segments[1]["text"] = "B. Other"
    assert not p.verify(KEY)


def test_write_once_refuses_edit_in_place():
    ds = DocStore("write_once")
    ds.put("D", "2025-03", "old", "t0")
    with pytest.raises(EditInPlace):
        ds.put("D", "2025-03", "new", "t1")
    ds.put("D", "2025-03r2", "new", "t1")
    assert ds.by_hash(sha("old")) == "old"


def test_mutable_store_recovers_old_bytes_from_snapshot_only():
    ds = DocStore("mutable")
    ds.put("D", "e", "old", "t0")
    ds.snapshot("d1")
    ds.put("D", "e", "new", "t1")
    assert ds.by_hash(sha("old")) == "old"
    assert ds.by_hash(sha("old"), keep_days=["d2"]) is None


def test_explain_is_deterministic():
    p = _pack()
    assert explain(p, KEY) == explain(copy.deepcopy(p), KEY)


def test_merkle_odd_leaves():
    assert merkle(["a", "b", "c"]) == merkle(["a", "b", "c", "c"])
