"""Chapter 38 fixtures: Lakeshore's catalogue, traffic and stand-in models (seed 38).

Every number here is a stand-in chosen to make the chapter's failure modes
visible; none is a measurement of a real gateway, model or provider.
"""
import random
import re

from contextlab.gateway import Deployment, sim_embed

SEED = 38
DAY = 24.0
DETS_PER_DAY = 3200

# Retirement of the pinned embedding model on day 9 at 17:00 (a Friday).
EMBED_RETIRE_AT = 9 * DAY + 17


def catalogue():
    return [
        Deployment("sim-large@2026-04/us-1", "sim-large@2026-04", "chat", "us"),
        Deployment("sim-large@2026-04/us-2", "sim-large@2026-04", "chat", "us"),
        Deployment("sim-large@2026-04/global", "sim-large@2026-04", "chat", "global"),
        Deployment("sim-large@2026-10/us-1", "sim-large@2026-10", "chat", "us"),
        Deployment("open-base@1.2/us-1", "open-base@1.2", "embed", "us",
                   retire_at=EMBED_RETIRE_AT, successor="open-base@1.3"),
        Deployment("open-base@1.3/us-1", "open-base@1.3", "embed", "us"),
        Deployment("sim-verify@2026-02/global", "sim-verify@2026-02", "verify", "global"),
        Deployment("sim-verify-us@2026-06/us-1", "sim-verify-us@2026-06", "verify", "us"),
        Deployment("sim-letter@2025-11/global", "sim-letter@2025-11", "chat", "global"),
        Deployment("sim-letter@2026-05/us-1", "sim-letter@2026-05", "chat", "us"),
    ]


# What the context layer relied on before the gateway, and how each survives
# the gateway's translated path versus its raw pass-through path.
FEATURES = [
    # feature, chapter, translated path, raw path
    ("Dated model pin (no alias)", 14, "kept", "kept"),
    ("Cache routing key from caller", 34, "lost", "kept"),
    ("Explicit cache breakpoints", 34, "lost", "kept"),
    ("Structured output schema", 28, "degraded", "kept"),
    ("Long-context tier (200K)", 32, "lost", "lost"),
    ("Per-call region choice", 38, "gateway", "gateway"),
    ("Token usage incl. cached", 34, "kept", "kept"),
    ("Tool definitions (MCP)", 26, "kept", "kept"),
    ("Log probabilities", 29, "lost", "lost"),
]

PROBES = ["water backup sublimit endorsement", "named storm deductible percentage",
          "vacancy clause sixty days", "ordinance or law coverage",
          "mould exclusion limited fungi", "sewer drain overflow",
          "matching undamaged siding", "loss of use additional living expense"]

VOCAB = [f"w{i:04d}" for i in range(1500)]


def corpus(rng, n=300, length=28):
    return [rng.sample(VOCAB, length) for _ in range(n)]


def queries(rng, docs, n=200, keep=6, synonym_share=0.75):
    """Queries take words from a target span; three in four of them as synonyms ('~')
    that the dense models understand and BM25 does not."""
    out = []
    for _ in range(n):
        tgt = rng.randrange(len(docs))
        words = rng.sample(docs[tgt], keep) + rng.sample(VOCAB, 2)
        words = [w + "~" if rng.random() < synonym_share else w for w in words]
        out.append((tgt, words))
    return out


def embed(model, words):
    return sim_embed(model, " ".join(w.rstrip("~") for w in words))


def verifier_scores(rng, n=400, unsupported=60):
    """Stand-in scores for 400 labelled cited sentences (Chapter 28's set size
    doubled), from the global verifier and its US-only replacement."""
    gold = [0] * unsupported + [1] * (n - unsupported)
    rng.shuffle(gold)
    old = [min(1, max(0, rng.gauss(0.80, 0.09) if g else rng.gauss(0.40, 0.11))) for g in gold]
    new = [min(1, max(0, rng.gauss(0.70, 0.10) if g else rng.gauss(0.43, 0.11))) for g in gold]
    return gold, old, new


CITE = re.compile(r"(?<!\[)\[(HO[\w ]+@\d{4}-\d{2}/p\d+/s\d+)\](?!\])")


def model_sentences(rng, model, n):
    """Stand-in determination sentences. The successor writes 3% of its
    citations with double brackets, which Chapter 28's parser does not read."""
    out = []
    for i in range(n):
        h = f"HO 04 95@2024-06/p{1 + i % 4}/s{1 + i % 9}"
        if model.endswith("2026-10") and rng.random() < 0.03:
            out.append(f"The limit applies [[{h}]].")
        else:
            out.append(f"The limit applies [{h}].")
    return out
