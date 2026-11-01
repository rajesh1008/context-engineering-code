"""Lakeshore knowledge inventory, draft v0.9 and signed v1.0 (Chapter 9)."""
from dataclasses import replace
from datetime import timedelta as td
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from contextlab.knowledge import (Source, Inventory, EvidenceClass as E,
                                  Access as A, Volatility as V, VersionRule as R)

ADJ = frozenset({"adjuster", "supervisor"})
UW = frozenset({"underwriter"})
CLAIMS = frozenset({"claims_determination"})

SOURCES = [
    Source("policy-forms", "document", "Product forms", "document store",
           E.GOVERNING, A.RETRIEVE, V.LOW, R.EDITION, roles=ADJ, precedence=10),
    Source("endorsements", "document", "Product forms", "document store",
           E.GOVERNING, A.RETRIEVE, V.LOW, R.EDITION, roles=ADJ, precedence=20),
    Source("state-bulletins", "document", "Compliance", "regulatory feed",
           E.GOVERNING, A.RETRIEVE, V.MODERATE, R.EFFECTIVE,
           max_age=td(days=2), roles=ADJ, precedence=30),
    Source("naic-guidance", "document", "Compliance", "regulatory feed",
           E.ADVISORY, A.RETRIEVE, V.LOW, R.EDITION, roles=ADJ),
    Source("scanned-documents", "document", "Claims operations", "imaging archive",
           E.GOVERNING, A.RETRIEVE, V.LOW, R.EDITION, roles=ADJ, precedence=20),
    Source("policy-in-force", "structured", "Policy administration",
           "policy admin (Oracle)", E.FACTUAL, A.QUERY, V.MODERATE, R.ROW,
           version_origin="system", max_age=td(minutes=15), roles=ADJ),
    Source("claim-record", "api", "Claims platform", "claims system API",
           E.FACTUAL, A.TOOL, V.HIGH, R.AS_OF, version_origin="system",
           max_age=td(minutes=5), roles=ADJ),
    Source("adjuster-notes", "structured", "Claims operations", "claims system API",
           E.ADVISORY, A.TOOL, V.HIGH, R.AS_OF, version_origin="system",
           max_age=td(minutes=5), roles=ADJ,
           note="statements about the loss; never governing"),
    Source("weather-reports", "api", "Claims platform", "public weather API",
           E.FACTUAL, A.TOOL, V.HIGH, R.AS_OF, version_origin="system",
           max_age=td(days=1), roles=ADJ),
    Source("claims-rulebook", "people", "Claims standards", "rulebook repo",
           E.PROCEDURAL, A.RULEBOOK, V.LOW, R.EDITION, roles=ADJ),
    Source("underwriting-notes", "document", "Underwriting", "SharePoint",
           E.ADVISORY, A.RETRIEVE, V.LOW, R.NONE, roles=UW),
    Source("property-data-feed", "api", "Underwriting", "licensed data feed",
           E.FACTUAL, A.TOOL, V.MODERATE, R.AS_OF, version_origin="system",
           max_age=td(days=30), roles=ADJ | UW,
           purposes=frozenset({"underwriting"}),
           note="licence covers underwriting only"),
    Source("crm-correspondence", "structured", "Customer service", "CRM",
           E.ADVISORY, A.EXCLUDE, V.HIGH, R.AS_OF, version_origin="system",
           roles=ADJ, note="policyholder statements; turn-time only"),
    Source("generated-summaries", "generated", "Claims platform", "context store",
           E.ADVISORY, A.EXCLUDE, V.MODERATE, R.AS_OF, version_origin="system",
           roles=ADJ, note="model output never becomes evidence"),
    Source("current-form-pointer", "structured", None, "claims system table",
           E.FACTUAL, A.RETRIEVE, V.MODERATE, R.NONE, roles=ADJ,
           note="nightly copy of the latest form; Section 2.6"),
]


def draft() -> Inventory:
    """v0.9: what the walk-through found, before the owner interviews."""
    rows = [replace(s, access=A.RETRIEVE, note="nightly export, indexed")
            if s.name == "claim-record" else s for s in SOURCES]
    return Inventory("v0.9", rows)


def signed() -> Inventory:
    """v1.0: signed by Laura Jensen's team; every row has an owner."""
    rows = []
    for s in SOURCES:
        if s.name == "current-form-pointer":
            s = replace(s, owner="Claims platform", access=A.EXCLUDE,
                        note="registered so that it can be refused")
        rows.append(s)
    return Inventory("v1.0", rows)
