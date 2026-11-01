"""The full-mix determination ledger that Chapter 20 carries in (synthetic).

The wrong-clause rate on Lakeshore's 1,000-determination full mix is scored by
each chapter's own determination lab. This file carries the chain of closes
and the eight determinations still wrong at Chapter 18's close, each with the
layer its error starts in. The workshop's rebuild is applied to them
mechanically: a determination is fixed only if its cause is in retrieval and
the rebuilt build removes that cause.
"""
from dataclasses import dataclass
from datetime import date

HISTORY = [  # chapter, wrong of 1,000, what moved it, note
    ("Ch 9 close (Part III starts)", 50, "evidence inventory", ""),
    ("Ch 10", 40, "parsing and structural chunking", ""),
    ("Ch 11", 36, "document quality gate", ""),
    ("Ch 12-14", 35, "retrieval ladder, index, embedding", "checked: ch12.md 36, ch14.md 36 -> 35"),
    ("Ch 15", 16, "authority: entitlement, jurisdiction, precedence", ""),
    ("Ch 16", 16, "structured data (amounts, not clauses)", ""),
    ("Ch 17", 10, "policy graph for two-span chains", ""),
    ("Ch 18", 8, "agentic loop for 33 graph-routed claims", ""),
    ("Ch 19", 8, "multimodal evidence (property claims)", "checked: ch19.md 8, unchanged"),
]


@dataclass
class Residual:
    det: str
    stratum: str
    layer: str          # record, document, feed, index
    cause: str
    governing_effective: date | None = None
    next_owner: str = ""


# Composition carried in (the canonical list; Chapters 15, 17 and 18 agree with it):
#   Ch 15 close, 16 = these 8 except R5 (7 record, document and feed errors)
#                   + 6 needing two governing spans + 3 needing a chain of three spans
#                   (ch15/corpus15.py labels them; Chapter 18's lab stands in 65 claims
#                   of the same kind, so R5 is not traced to a Chapter 15 identifier)
#   Ch 17 close, 10 = the 6 two-span determinations compose on the graph;
#                   the 3 chains remain, the graph walk finishing none of them
#   Ch 18 close,  8 = router v2's loop finishes 2 chains; R5 is the one it gives up
RESIDUAL = [
    Residual("R1", "policy-admin", "record", "endorsement schedule misread in the policy summary",
             next_owner="Ch 23 (state)"),
    Residual("R2", "policy-admin", "record", "endorsement schedule misread in the policy summary",
             next_owner="Ch 23 (state)"),
    Residual("R3", "docstore-new", "record", "risk address geocoded across a state line",
             next_owner="Ch 26 (tools)"),
    Residual("R4", "docstore-new", "record", "risk address geocoded across a state line",
             next_owner="Ch 26 (tools)"),
    Residual("R5", "multi-state", "index", "Iowa bulletin effective after the graph's last build",
             governing_effective=date(2026, 8, 1), next_owner="fixed by the rebuild"),
    Residual("R6", "policy-admin", "record", "endorsement missing from the policy record",
             next_owner="Ch 23 (state)"),
    Residual("R7", "scanned", "document", "OCR misread the endorsement on the declarations",
             next_owner="Ch 11 gate audit, Ch 28 (exact page)"),
    Residual("R8", "third-party", "feed", "licensed feed misdated the policy change",
             next_owner="Ch 24 (freshness)"),
]

GRAPH_BUILD_BEFORE = date(2026, 6, 30)   # last full build at Chapter 18's close
GRAPH_BUILD_REBUILT = date(2026, 10, 26)  # the workshop's clean rebuild


def fixed_by_rebuild(r: Residual, build: date) -> bool:
    """Only an index cause can be fixed by rebuilding the index."""
    return (r.layer == "index" and r.governing_effective is not None
            and r.governing_effective <= build)
