"""Chapter 1 tests: the lab reproduces the chapter's numbers, and the policy fails closed."""
import pytest

import lab as L
import standin as S


@pytest.fixture(scope="module")
def R():
    world = S.build_world()
    return world, L.run_all(world)["res"]


def rate(res, name):
    g, w, na, rf, tok, n = L.summary(res, name)
    return round(100 * g / n, 1), round(tok)


def test_corpus_and_instructions(R):
    world, _ = R
    assert len(world.corpus.spans) == 378
    assert sum(s.tokens for s in world.corpus.spans) == 22_326
    assert S.INSTR + S.RULES_TOKENS == 103
    assert len(world.dets) == 500


@pytest.mark.parametrize("name,gov,tok", [
    ("A prompt only", 0.0, 109), ("B stuff it all in", 87.0, 848), ("C engineered", 98.0, 1579),
    ("D no declarations", 79.8, 1587), ("D + fix", 43.6, 1592), ("E send everything", 84.2, 36074),
    ("B no header", 34.0, 816), ("B + rules", 85.8, 898), ("C no rules", 95.2, 1529)])
def test_table(R, name, gov, tok):
    assert rate(R[1], name) == (gov, tok)


@pytest.mark.parametrize("b,gov,tok", [(150, 90.6, 420), (300, 95.8, 660), (450, 96.2, 896),
                                       (600, 97.8, 1133), (900, 98.0, 1579), (1200, 98.0, 1908),
                                       (2000, 98.0, 2137)])
def test_budget_table(R, b, gov, tok):
    assert rate(R[1], f"C budget {b}") == (gov, tok)


def test_rules_flip_fourteen(R):
    res = R[1]
    c = {d.determination_id: v for d, v, _ in res["C engineered"]}
    nr = {d.determination_id: v for d, v, _ in res["C no rules"]}
    assert sum(c[k] == "governing" and nr[k] == "wrong" for k in c) == 14


def test_carter(R):
    world, _ = R
    det = world.dets[0]
    out = {n: (f(det)[0].tokens, len(f(det)[0].spans), L.judge(det, f(det)[0])[0]) for n, f in L.ASSEMBLIES}
    assert out["B stuff it all in"] == (923, 8, "governing")
    assert out["C engineered"] == (1567, 13, "governing")
    assert out["D no declarations"] == (1616, 14, "wrong")
    assert L.d_fixed(det)[0] is None


def test_policy_fails_closed(R):
    world, _ = R
    det = world.dets[0]
    s = next(x for x in world.corpus.spans if x.kind == S.Kind.FORM)
    import dataclasses
    blank = dataclasses.replace(s, provenance=dataclasses.replace(s.provenance, version=""))
    v = S.Policy().applies(blank, det)
    assert not v.ok and "fail closed" in v.reason


def test_deterministic():
    a = L.run_all(S.build_world())["rows"]
    b = L.run_all(S.build_world())["rows"]
    assert a == b
