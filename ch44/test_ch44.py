"""Chapter 44 tests: the retirement rule, reconciliation, the register and the drills."""
from datetime import date

import lakeshore44 as L
from contextlab.governance import (Register, TermAwarePolicy, reconcile, classify,
                                   fan_out, exit_drill, coverage, calendar, Change)


def pol():
    return TermAwarePolicy(L.SPANS, {L.END: L.RETIRE})


def gov(loss, ts):
    return pol().resolve([s.sid for s in L.SPANS], L.request(loss, ts)).governing


def test_old_term_keeps_endorsement_after_retirement():
    assert gov(date(2026, 10, 2), date(2026, 3, 1)) == L.TRUTH["old"]


def test_new_term_gets_new_edition():
    assert gov(date(2026, 10, 2), date(2026, 9, 5)) == L.TRUTH["new"]


def test_term_aware_never_wrong_over_a_year():
    assert all(gov(l, t) == L.truth(t) for l, t in L.michigan_losses())


def test_clean_catalogue_has_no_findings():
    d, s = L.build_catalogue()
    assert reconcile(d, L.build_copies(d, s), L.LIVE_MODEL) == []


def test_every_injected_defect_is_found_and_nothing_else():
    d, s = L.build_catalogue()
    d2, c2, log = L.inject(d, L.build_copies(d, s))
    found = {f[2] for f in reconcile(d2, c2, L.LIVE_MODEL)}
    assert found == set(log.values())


def test_metadata_change_never_reembeds():
    cls, re, _ = classify(Change("x", "status", "metadata", 46089))
    assert cls == "standard" and not re


def test_purge_is_emergency():
    assert classify(Change("x", "pii", "purge", 1))[0] == "emergency"


def test_register_v10_has_no_gaps_and_v09_does():
    assert Register(L.register("v1.0")).gaps() == []
    assert len(Register(L.register("v0.9")).gaps()) == 9


def test_erasure_has_an_answer_everywhere():
    _, un = fan_out(Register(L.register("v1.0")), L.ERASURE_HOLDINGS)
    assert un == []


def test_exit_drill_selects_by_tenant_everywhere():
    d = exit_drill(Register(L.register("v1.0")), L.EXIT_HOLDINGS)
    assert all(x[0] == "select by tenant" for x in d.values())


def test_coverage_before_is_incomplete():
    assert coverage(L.CONTROLS_BEFORE)[0] == 2


def test_calendar_has_52_reconciliations():
    ev = calendar(2027, {})
    assert sum(1 for e in ev if e[1] == "weekly") == 52
