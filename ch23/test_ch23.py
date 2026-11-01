"""Chapter 23 tests: the numbers the chapter quotes come from this code."""
import json
import pathlib
import sys
from datetime import date

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import session23 as S                                   # noqa: E402
import lifecycle23 as Y                                 # noqa: E402
from contextlab.memguard import MemoryRecord as M, Origin as O   # noqa: E402
from contextlab.statestore import (StateStore, Fact, LifecyclePolicy, RetentionRule, erase,  # noqa: E402
                                   expire, sweep)


def test_full_history_overflows_tom_afternoon():
    r = S.run(S.build_session(23), "full")
    assert r.overflow_minute == 104.9 and r.lost == 7
    assert r.by_hour[1][1] / r.by_hour[1][0] > 3 * r.by_hour[0][1] / r.by_hour[0][0]


def test_structured_keeps_constraints_and_sources():
    tom = S.build_session(23)
    r = S.run(tom, "structured")
    assert r.violations == 0 and r.withheld == 0 and r.peak < S.BUDGET / 2
    assert {c for c, _ in r.store.pinned} == {"K1", "K2", "K3"}


def test_summarise_loses_rules_more_than_facts():
    viol = withheld = 0
    for seed in range(1, 11):
        r = S.run(S.build_session(seed), "summarise")
        viol += r.violations
        withheld += r.withheld
    assert viol > 0 and withheld > 0


def test_checkpoint_round_trip_is_identical():
    st = StateStore("S", "tbrennan")
    st.pin("K1", "never mark ready")
    st.record("C01", Fact("C01.amount", "US$31,200", "P-WI-1/record#amount"))
    a = st.checkpoint(["I3", "I1"], {"HO 04 95": "2024-06"}, 9)
    b = StateStore.restore(st.checkpoints[a], "tbrennan").checkpoint(["I1", "I3"], {"HO 04 95": "2024-06"}, 9)
    assert a == b


def test_lifecycle_takes_mix_from_nine_to_seven():
    out = {}
    for lc, k in ((False, 8), (True, 6)):
        d, s, l, o, _ = Y.replay(lc, k)
        out[lc] = sorted(Y.R.wrong(d, o))
    assert len(out[False]) == 9 and len(out[True]) == 7
    assert set(out[False]) - set(out[True]) == {"D0364", "D0979"}


def test_revalidation_writes_pointers_not_policy_facts():
    rep = {}
    dets = Y.L.make_mix()
    store = Y.lifecycle_store(dets, report=rep)
    new = [m for m in store.values() if m.mid.endswith("-r")]
    assert len(new) == rep["revalidated"]["checked"] == 310
    assert all(Y.is_pointer(m) and m.asserts is None for m in new)
    system = Y.record_truth(dets)
    m = new[0]
    assert Y.resolve_pointer(m, system) == system[m.subject]
    system[m.subject] = not system[m.subject]          # a renewal after the pass
    assert Y.resolve_pointer(m, system) == system[m.subject]   # nothing stale to fix


def test_retention_respects_hold_and_erasure_tombstone():
    pol = LifecyclePolicy({"claimant": RetentionRule("claimant", 30)})
    store = {"a": M("a", "claimant", "P-1", "note", date(2026, 1, 1), (O("determination", "D1"),)),
             "b": M("b", "claimant", "P-2", "note", date(2026, 1, 1), (O("determination", "D2"),)),
             "t": M("t", "team", "WI", "lesson", date(2026, 1, 1),
                    (O("determination", "D2"), O("note", "N1"), O("note", "N2")))}
    assert expire(store, date(2026, 3, 1), pol, holds={"P-1"}) == ["b"]
    rep = erase(store, "P-2", {"D2"}, "ER-1", date(2026, 3, 2), pol)
    assert rep.dropped_derived == ["t"] and "P-2" not in json.dumps(rep.tombstone)
    assert sweep(store, date(2026, 3, 2), pol)["erased"] == 2


def test_lab_output_matches_text():
    res = json.loads((HERE / "out" / "results.json").read_text())
    assert res["mix"]["Chapter 22 close"]["wrong"] == 9
    assert res["mix"]["+ retention 730 days"]["wrong"] == 7
    assert res["sweep"]["corrected"] == 28
    assert res["checkpoint"]["identical"] is True
