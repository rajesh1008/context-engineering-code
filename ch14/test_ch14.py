import numpy as np

import corpus14 as C
from contextlab.embeddings import (EmbeddingRegistry, QueryAdapter, n_tokens,
                                   quantise, search, storage_bytes)

SPANS = C.build_spans()
IDS = [s.sid for s in SPANS]
GOLD = C.gold_set(SPANS)


def hit(model, texts=None, truncate=False):
    D = model.encode(texts or [s.text for s in SPANS], truncate=truncate)
    top = search(model.encode([q for q, _, _ in GOLD]), D, 10)
    return np.array([g in [IDS[j] for j in r] for r, (_, g, _) in zip(top, GOLD)])


def test_encoding_is_deterministic_and_unit_length():
    a, b = C.models()["open-base"], C.models()["open-base"]
    va, vb = a.encode_one("water back-up sump"), b.encode_one("water back-up sump")
    assert np.allclose(va, vb) and abs(np.linalg.norm(va) - 1) < 1e-9


def test_leaderboard_leader_is_not_the_gold_leader():
    M = C.models()
    assert hit(M["open-base"]).mean() > hit(M["gen-large"]).mean()
    docs, lq = C.leaderboard_set()
    def lb(m):
        top = search(m.encode([q for q, _ in lq]), m.encode(docs), 10)
        return np.mean([i in r for r, (_, i) in zip(top, lq)])
    assert lb(M["gen-large"]) > lb(M["open-base"])


def test_silent_provider_update_is_caught_and_kernel_jitter_is_not():
    reg = EmbeddingRegistry(C.PROBES)
    reg.register(C.models()["gen-large"], "api-vendor")
    reg.bind_index("ix", "gen-large@2026-03")
    upd = C.models(provider_update=True)["gen-large"]
    assert upd.model_id == "gen-large@2026-03"          # same name
    assert not reg.check(upd).ok and not reg.may_query("ix", upd)
    jit = C.models()["gen-large"]; jit.jitter = 2e-4
    assert reg.check(jit, "gen-large@2026-03").ok


def test_overlong_spans_are_refused_then_respanned_to_fit():
    reg = EmbeddingRegistry(C.PROBES)
    e = reg.register(C.models()["open-base"], "self-hosted")
    long = [s for s in SPANS if n_tokens(s.text) > 512]
    assert long and all(s.family == "BOAT" for s in long)
    assert not any(reg.admit_span(e.model_id, s.text) for s in long)
    pieces = C.respan(long, 480)
    assert all(n_tokens(p.text) <= 480 and p.text.startswith(p.sid.split("#p")[0])
               for p in pieces)


def test_truncation_hurts_marine_only():
    m = C.models()["open-base"]
    sl = np.array([s for *_, s in GOLD])
    full, cut = hit(m), hit(m, truncate=True)
    assert cut[sl == "marine"].mean() < full[sl == "marine"].mean() - 0.5
    assert (cut[sl != "marine"] == full[sl != "marine"]).all()


def test_adapter_reduces_to_identity_under_heavy_regularisation():
    rng = np.random.default_rng(14)
    Q = rng.standard_normal((50, 16)); D = rng.standard_normal((50, 16))
    A = QueryAdapter(lam=1e9).fit(Q, D)
    assert np.allclose(A.W, np.eye(16), atol=1e-5)


def test_storage_and_quantisation_shapes():
    assert storage_bytes(1, 1024, "binary") == 128
    assert storage_bytes(118_000, 768, "int8") == 90_624_000
    X = np.random.default_rng(1).standard_normal((4, 64))
    assert set(np.unique(quantise(X, 32, "binary"))) <= {-1, 1}
    assert quantise(X, 32, "int8").dtype == np.int8
