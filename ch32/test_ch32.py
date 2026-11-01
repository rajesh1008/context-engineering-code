"""Chapter 32 tests: run with python3 -m pytest -q 04-code/ch32"""
import json, os
from datetime import date
import lakeshore32 as L
from contextlab import budgeter as B

FULL = L.full_mix()
WEEK = L.this_week()


def c(i, tok, rel, tier=2, kind="clause", **kw):
    return B.Candidate(f"s{i}", kw.pop("doc", "FORM"), tok, rel, tier, date(2025, 1, 1), kind, **kw)


def test_knapsack_is_exact_on_a_small_case():
    cs = [c(1, 270, 0.9), c(2, 170, 0.6), c(3, 170, 0.6)]   # 300 / 200 / 200 with headers
    got = {s.span_id for s in B.knapsack(cs, 400, key=lambda s: s.relevance)}
    assert got == {"s2", "s3"}                                 # 1.2 beats 0.9


def test_greedy_by_density_can_starve_a_long_span():
    cs = [c(1, 270, 0.8), c(2, 170, 0.6), c(3, 170, 0.6)]
    got = {s.span_id for s in B.greedy_density(cs, 300, key=lambda s: s.relevance)}
    assert got == {"s2"}


def test_dependency_travels_with_its_dependant():
    d, e = c(1, 100, 0.3, kind="definition"), c(2, 150, 0.9, kind="endorsement", requires=("s1",), doc="END")
    got = {s.span_id for s in B.Budgeter(budget=400, floor=0.1).pack([d, e])[0]}
    assert got == {"s1", "s2"}


def test_one_rendering_per_redundancy_group():
    a, b = c(1, 200, 0.9, group="g"), c(2, 205, 0.89, group="g", doc="FORM-ADMIN")
    got = B.Budgeter(budget=2_000, floor=0.1).pack([a, b])[0]
    assert len(got) == 1


def test_bulletins_are_pinned_even_below_the_floor():
    bl = c(1, 200, 0.05, tier=1, kind="bulletin", doc="BUL")
    assert B.Budgeter(floor=0.99).pack([bl])[0] == [bl]


def test_newest_bulletin_always_packed_this_week():
    bud = B.Budgeter(budget=1_000, floor=0.60)
    for d in WEEK:
        assert any(s.doc_id == "BUL-WI-WB-202610" for s in bud.pack(d.cands)[0]), d.det_id


def test_usage_scorer_drops_the_newest_bulletin():
    bud = B.Budgeter(budget=1_000, floor=0.60, pin_kinds=(), scorer=B.score_usage)
    assert not any(any(s.kind == "bulletin" for s in bud.pack(d.cands)[0]) for d in WEEK)


def test_budgeter_holds_the_wrong_clause_rate_on_the_full_mix():
    bud = B.Budgeter(budget=1_000, floor=0.60)
    wrong = sum(L.read(d, bud.pack(d.cands)[0])["wrong"] for d in FULL)
    assert wrong == 4


def test_cache_needs_an_identical_prefix():
    pc = B.PrefixCache()
    pc.call(0, [("tools", 1_661), ("instr", 640), ("a", 300)], [1, 2])
    hit, _, _ = pc.call(10, [("tools", 1_661), ("instr", 640), ("b", 300)], [1, 2])
    assert hit == 2_301
    hit, _, _ = pc.call(20, [("instr", 640), ("tools", 1_661), ("b", 300)], [1, 2])
    assert hit == 0


def test_cache_entry_expires_after_ttl():
    pc = B.PrefixCache()
    pc.call(0, [("p", 2_000)], [0])
    assert pc.call(301, [("p", 2_000)], [0])[0] == 0


def test_results_file_matches_headline():
    r = json.load(open(os.path.join(os.path.dirname(__file__), "out", "results.json")))
    t = r["tokens_per_det"]
    assert -0.45 < (t["budgeter_and_per_step"] - t["entry"]) / t["entry"] < -0.40
    assert r["matched"]["budgeter"]["faithful"] >= r["entry"]["faithful"]


def test_knapsack_matches_brute_force_on_random_pools():
    import itertools, random
    rng = random.Random(32)
    for _ in range(40):
        cs = [c(i, rng.randint(20, 300), rng.random()) for i in range(rng.randint(3, 12))]
        budget = rng.randint(200, 1_200)
        w = lambda s: -(-(s.tokens + B.FLAT_HEADER) // 10) * 10
        best = max(sum(s.relevance for s in sub) for r in range(len(cs) + 1)
                   for sub in itertools.combinations(cs, r) if sum(w(s) for s in sub) <= budget)
        got = B.knapsack(cs, budget, key=lambda s: s.relevance)
        assert abs(sum(s.relevance for s in got) - best) < 1e-9


def test_a_slot_over_the_cap_is_traced_not_dropped_silently():
    big = [c(i, 900, 0.9, doc=f"END-{i}", kind="endorsement") for i in range(3)]
    spans, trace = B.Budgeter(hard_cap=2_400, floor=0.1).pack(big)
    assert len(spans) == 2 and ("s2", "over cap") in trace
