import json
import pathlib

import agent33 as A
import lab
import lakeshore33 as L
from contextlab import compress as C

HERE = pathlib.Path(__file__).resolve().parent
P = L.build(40)


def test_uncompressed_packets_are_all_right():
    assert all(L.grade(p, lab.texts(p)) == "right" for p in P)


def test_residual_packets_are_always_wrong():
    M = L.build(30, residual=("D0007",))
    p = next(q for q in M if q.pid == "D0007")
    assert L.grade(p, lab.texts(p)) == "wrong"


def test_polarity_diff_sees_an_inversion():
    a = "Cover is void if the sump pump was not maintained."
    b = "Cover applies even if the sump pump was maintained."
    assert C.negation_profile(a) == 2 and C.polarity_diff(a, b) > 0


def test_extract_keeps_order_and_budget():
    s = [C.Sentence(f"x#s{i}", "word " * (4 * (i + 1)), "policy") for i in range(5)]
    out = C.extract(s, lambda x: -x.tokens, 15)
    assert [x.handle for x in out] == ["x#s0", "x#s1"]


def test_clause_groups_keep_every_determination():
    assert all(L.grade(p, lab.extractive(p, 0.33, protect=True)) == "right" for p in P)


def test_force_keep_preserves_negations():
    t = "We do not insure loss caused by water that backs up through a sewer."
    out = C.prune_tokens(t, 0.3, lambda w: 0.0, force=L.FORCE)
    assert "not" in out.split()


def test_probe_reports_changed_answers():
    p = P[0]
    full = lab.texts(p)
    assert C.probe(full, full, lab.probes(p)) == []
    no_loss = [x for x in full if "agreed" not in x]
    assert "what amount was agreed" in C.probe(full, no_loss, lab.probes(p))


def test_shipped_policy_compresses_and_passes_probes():
    for p in P:
        out = lab.shipped(p)
        assert sum(C.tokens(x) for x in out) < 0.5 * p.tokens
        assert C.probe(lab.texts(p), out, lab.probes(p)) == []


def test_trim_results_keeps_handles():
    items = [dict(role="tool", id=f"c{i}", text="long result") for i in range(4)]
    out = C.trim_results(items, 1)
    assert out[-1]["text"] == "long result" and "c0" in out[0]["text"]


def test_pinning_removes_constraint_violations():
    eps = A.episodes(60, attack_share=1.0)
    assert A.summarise(eps, "threshold")["violations"] > 0
    assert A.summarise(eps, "threshold", pin=True)["violations"] == 0


def test_results_file_matches_published_numbers():
    R = json.load(open(HERE / "out" / "results.json"))
    lake = R["lakeshore"]
    first, shipped = list(lake.values())[:2]
    assert first["wrong"] == shipped["wrong"] == 4
    assert R["incident"]["new_wrong"] and len(R["incident"]["new_wrong"]) == 7
