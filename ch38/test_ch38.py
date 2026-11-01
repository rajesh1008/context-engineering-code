"""Tests for contextlab.gateway v8.4 and the Chapter 38 lab."""
import random

from contextlab.gateway import (KeyIssuer, CredentialProvider, SimGateway, ResidencyPolicy,
                                verify_record, cache_namespace, probe_check, join_traces,
                                UsageRecord, VERSION)
from lakeshore38 import catalogue, EMBED_RETIRE_AT, PROBES, embed, CITE


def gw(**kw):
    iss = KeyIssuer()
    return SimGateway(catalogue(), iss, **kw), iss


def test_version():
    assert VERSION == "8.4"


def test_key_overlap_and_expiry():
    iss = KeyIssuer(168, 24)
    old = iss.current(0)[1]
    assert iss.valid(old, 170) == "key-000"      # inside overlap
    assert iss.valid(old, 193) is None           # after overlap
    assert iss.valid(iss.current(193)[1], 193) == "key-001"


def test_cached_key_fails_and_per_batch_recovers():
    g, iss = gw()
    held = CredentialProvider(iss, "at_start")
    fresh = CredentialProvider(iss, "per_batch")
    assert g.call("job", "open-base@1.3", held.secret(200), 200, "", "x").status == 401
    assert held.refresh(200) is None
    assert g.call("job", "open-base@1.3", fresh.secret(200), 200, "", "x").status == 200


def test_identity_namespace_isolates_tenants():
    for mode, expect in (("pooled", True), ("client_key", True), ("identity", False)):
        g, iss = gw(cache_mode=mode)
        s = iss.current(1)[1]
        p = "z" * 8000
        g.call("tenant-a", "sim-large@2026-04", s, 1, p, "q", client_key="shared")
        r = g.call("tenant-b", "sim-large@2026-04", s, 1, p, "q", client_key="shared")
        assert (r.cached_tokens > 0) is expect
    assert cache_namespace("a") != cache_namespace("b")


def test_fallback_never_crosses_region_unless_any():
    pol = ResidencyPolicy()
    for fb, status in (("in_region", 503), ("any", 200)):
        g, iss = gw(fallback=fb)
        g.down |= {"sim-large@2026-04/us-1", "sim-large@2026-04/us-2"}
        r = g.call("c", "sim-large@2026-04", iss.current(1)[1], 1, "p", "s",
                   allowed_regions={"us"})
        assert r.status == status
        if status == 200:
            assert "region" in verify_record(r, "sim-large@2026-04", "claim_file", pol)


def test_deprecation_modes_are_each_caught():
    t = EMBED_RETIRE_AT + 1
    ref = [embed("open-base@1.2", p.split()) for p in PROBES]
    pol = ResidencyPolicy()
    g, iss = gw(alias_mode="refuse")
    assert g.call("q", "open-base@1.2", iss.current(t)[1], t).status == 404
    g, iss = gw(alias_mode="forward")
    r = g.call("q", "open-base@1.2", iss.current(t)[1], t)
    assert verify_record(r, "open-base@1.2", "claim_file", pol) == ["served_differs"]
    g, iss = gw(alias_mode="forward_masked")
    r = g.call("q", "open-base@1.2", iss.current(t)[1], t)
    assert verify_record(r, "open-base@1.2", "claim_file", pol) == []
    ok, m = probe_check(lambda p: embed("open-base@1.3", p.split()), PROBES, ref)
    assert not ok and m < 0.5
    ok, m = probe_check(lambda p: embed("open-base@1.2", p.split()), PROBES, ref)
    assert ok and m > 0.999


def test_undeclared_transform_flagged():
    rec = UsageRecord("gw-1", "c", "m", "m", "d", "us", "key-000", 200,
                      transforms=("redact", "truncate"))
    assert verify_record(rec, "m", "claim_file", ResidencyPolicy()) == ["undeclared_transform"]


def test_join_counts_unmatched():
    rec = UsageRecord("gw-1", "c", "m", "m", "d", "us", "key-000", 200)
    j = join_traces([{"request_id": "gw-1"}, {"request_id": "gw-9"}], [rec])
    assert j["matched"] == 1 and j["unmatched"] == 1


def test_strict_citation_parser():
    assert CITE.search("x [HO 04 95@2024-06/p1/s3].")
    assert not CITE.search("x [[HO 04 95@2024-06/p1/s3]].")
