"""Deterministic stand-ins for Chapter 44: Lakeshore's Michigan forms, its catalogue
and derived copies (with eight injected defects), its asset register, and the
holdings an erasure request and a partner exit have to find. Seeded; no network."""
import random
from datetime import date, timedelta

from contextlab.policy import Span
from contextlab.governance import Asset, Doc, Copy, Change, TermRequest, h

RETIRE = date(2026, 9, 1)          # first Michigan term on the new edition
LIVE_MODEL = "open-base@1.3"

# ---------------------------------------------------------------- the Michigan forms
OLD, NEW, END = "05 11", "MI 09 26", "LM-HO 0122 MI"
SPANS = [
    Span(f"HO-3@{OLD}#loss-payment", "HO-3", OLD, "form", "loss-payment",
         "We will pay within 60 days after we receive your proof of loss.",
         edition_family="HO-3", effective_from=date(2011, 5, 1)),
    Span(f"HO-3@{NEW}#loss-payment", "HO-3 MI", NEW, "form", "loss-payment",
         "We will pay within 30 days after we receive your proof of loss. If we need "
         "more information we will tell you in writing within 15 days.",
         edition_family="HO-3", effective_from=RETIRE),
    Span(f"{END}@04 23#loss-payment", END, "04 23", "endorsement", "loss-payment",
         "Loss Payment is replaced: we will pay within 30 days after we receive your "
         "proof of loss.", jurisdiction="MI", scope="loss-payment",
         effective_from=date(2023, 4, 1)),
]
TRUTH = {"old": f"{END}@04 23#loss-payment", "new": f"HO-3@{NEW}#loss-payment"}


def michigan_losses(start=RETIRE, end=date(2027, 8, 31), per_week=9, seed=44):
    """Michigan loss-payment determinations: date of loss and the term it falls in."""
    rng = random.Random(seed)
    out, d = [], start
    while d <= end:
        for _ in range(per_week):
            loss = d + timedelta(days=rng.randrange(7))
            if loss > end:
                continue
            term_start = loss - timedelta(days=rng.randrange(365))
            out.append((loss, term_start))
        d += timedelta(days=7)
    return sorted(out)


def request(loss, term_start):
    edition = NEW if term_start >= RETIRE else OLD
    return TermRequest(role="adjuster", tenant="lakeshore", purpose="coverage-determination",
                       line="HO", policy_state="MI", risk_state="MI", garaging_state=None,
                       editions={"HO-3": edition}, endorsements=frozenset(),
                       date_of_loss=loss, cause="loss-payment", term_start=term_start)


def truth(term_start):
    return TRUTH["new"] if term_start >= RETIRE else TRUTH["old"]


# ---------------------------------------------------------------- catalogue and copies
KINDS = (("form", 38, "HO"), ("endorsement", 146, "END"), ("bulletin", 61, "BUL"),
         ("guideline", 115, "GL"))


def build_catalogue(seed=44):
    rng = random.Random(seed)
    docs, spans_of = [], {}
    for kind, n, pfx in KINDS:
        for i in range(n):
            did = f"{pfx}-{i:03d}"
            v = f"v{rng.randrange(1, 4)}"
            docs.append(Doc(did, v, kind, text_hash=h(did + v),
                            effective_from=date(2020 + rng.randrange(6), 1, 1)))
            spans_of[did] = rng.randrange(3, 13)
            if kind in ("bulletin", "guideline") and rng.random() < 0.15:
                old = f"v0{rng.randrange(1, 9)}"
                docs.append(Doc(did, old, kind, state="superseded", text_hash=h(did + old),
                                effective_to=docs[-1].effective_from))
    docs.append(Doc(END, "04 23", "endorsement", text_hash=h(END),
                    effective_from=date(2023, 4, 1)))
    spans_of[END] = 6
    return docs, spans_of


def build_copies(docs, spans_of, seed=44):
    rng = random.Random(seed + 1)
    copies = []
    for d in docs:
        for j in range(spans_of[d.doc_id]):
            key = f"{d.doc_id}@{d.version}#{j}"
            copies.append(Copy("passages", key, d.doc_id, d.version, d.text_hash,
                               state=d.state, effective_to=d.effective_to, tenant=d.tenant))
            copies.append(Copy("vectors", key, d.doc_id, d.version, d.text_hash,
                               model=LIVE_MODEL, state=d.state, tenant=d.tenant))
    live = [d for d in docs if d.state == "published"]
    for i in range(600):
        d = rng.choice(live)
        copies.append(Copy("cache", f"qa-{i:04d}", d.doc_id, d.version, tenant=d.tenant))
    for i in range(1144):
        d = rng.choice([x for x in live if x.doc_id != END])
        copies.append(Copy("gold", f"g-{i:04d}", d.doc_id, d.version, tenant=d.tenant))
    for i in range(2900):
        d = rng.choice(live)
        copies.append(Copy("memory", f"m-{i:04d}", d.doc_id, d.version, tenant=d.tenant))
    return copies


def inject(docs, copies):
    """The eight defects, each a thing that happened somewhere in the last year."""
    from dataclasses import replace
    docs = list(docs)
    cs = list(copies)
    log = {}
    # 1 retired in the catalogue, nothing told the copies (the incident)
    docs = [replace(d, state="retired", retired_for_terms_from=RETIRE) if d.doc_id == END else d
            for d in docs]
    extra = [Copy("cache", f"qa-r{i}", END, "04 23") for i in range(3)] + \
            [Copy("memory", f"m-r{i}", END, "04 23") for i in range(2)]
    cs += extra
    log["retired endorsement still current"] = "retired-open"
    # 2 orphan vectors left after a deletion batch half-failed
    cs += [Copy("vectors", f"GL-900@v1#{j}", "GL-900", "v1", "x", model=LIVE_MODEL)
           for j in range(4)]
    log["vectors outlive deleted guideline"] = "orphan"
    # 3 a new Iowa bulletin published, ingest failed silently
    docs.append(Doc("BUL-IA-2610", "v1", "bulletin", text_hash=h("ia")))
    log["new bulletin never ingested"] = "missing"
    # 4 a guideline edited in place: passages re-parsed, vectors not
    g = next(d for d in docs if d.doc_id == "GL-017")
    docs = [replace(d, text_hash=h("edited")) if d is g else d for d in docs]
    cs = [replace(c, text_hash=h("edited")) if c.doc_id == "GL-017" and c.store == "passages"
          else c for c in cs]
    log["guideline edited in place"] = "stale-hash"
    # 5 rows the model migration skipped
    n = 0
    for i, c in enumerate(cs):
        if c.store == "vectors" and c.doc_id.startswith("END-04") and n < 6:
            cs[i] = replace(c, model="open-base@1.2"); n += 1
    log["vectors left on the retired model"] = "wrong-model"
    # 6 a superseded bulletin whose end date was typed into the passage store
    b = next(d for d in docs if d.kind == "bulletin" and d.state == "superseded")
    cs = [replace(c, effective_to=b.effective_to + timedelta(days=31))
          if c.store == "passages" and (c.doc_id, c.version) == (b.doc_id, b.version) else c
          for c in cs]
    log["end date mistyped in the passage store"] = "end-mismatch"
    # 7 a partner-only guideline indexed as Lakeshore's
    docs = [replace(d, tenant="partner-b") if d.doc_id == "GL-088" else d for d in docs]
    log["partner guideline indexed for everyone"] = "tenant-mismatch"
    # 8 an endorsement published in error and voided
    docs = [replace(d, state="void") if d.doc_id == "END-131" else d for d in docs]
    log["voided endorsement still indexed"] = "void-present"
    return docs, cs, log


# Which defects did the event-driven controls of Chapters 13, 14 and 37 catch?
EVENT_CAUGHT = {"orphan": "deletion check after the batch",
                "stale-hash": "monthly content-hash audit"}


# ---------------------------------------------------------------- the register
def register(version="v1.0"):
    CT, PT, CL, PO, FL, CO = ("context team", "platform team", "claims operations",
                              "privacy office", "forms library", "Compliance")
    a = [
        Asset("source inventory", "source", CT, CT, False, True, None, "none-held", "annual"),
        Asset("forms catalogue", "source", FL, FL, False, True, None, "none-held", "monthly"),
        Asset("passage store", "derived", CT, CT, False, True, None, "none-held", "weekly",
              ("forms catalogue",)),
        Asset("vector index", "derived", CT, PT, False, True, None, "none-held", "weekly",
              ("passage store",)),
        Asset("shadow index", "derived", CT, PT, False, True, 30, "age-out", "quarterly",
              ("passage store",)),
        Asset("knowledge graph", "derived", CT, CT, False, True, None, "none-held", "weekly",
              ("passage store",)),
        Asset("determinations index", "derived", CL, CT, True, True, 730, "delete", "monthly",
              ("evidence packs",)),
        Asset("result cache", "derived", CT, PT, True, True, 7, "age-out", "weekly",
              ("vector index", "determinations index")),
        Asset("claim memory", "derived", CL, CT, True, True, 730, "delete", "monthly",
              ("determinations index",)),
        Asset("team memory and held queue", "derived", CL, CT, True, True, 730, "delete",
              "weekly"),
        Asset("checkpoints", "log", CT, CT, True, True, 365, "redact", "monthly",
              ("claim memory",)),
        Asset("traces", "log", PT, PT, True, True, 90, "age-out", "monthly"),
        Asset("gateway logs", "log", PT, PT, True, True, 30, "age-out", "monthly"),
        Asset("gold set", "evaluation", CT, CL, True, True, None, "delete", "quarterly",
              ("evidence packs",)),
        Asset("evidence packs", "evidence", CO, CT, True, True, None, "exempt", "annual",
              exemption="claim-file retention set by Compliance"),
        Asset("backups", "backup", PT, PT, True, True, 35, "age-out", "quarterly",
              ("passage store", "claim memory", "checkpoints")),
    ]
    if version == "v1.0":
        return a
    from dataclasses import replace
    broken = {
        "shadow index": dict(owner=None, retention_days=None, erasure=None, review=None),
        "determinations index": dict(erasure=None),
        "result cache": dict(tenant_field=False, erasure=None),
        "team memory and held queue": dict(owner=None),
        "gateway logs": dict(retention_days=None, erasure="age-out"),
        "gold set": dict(erasure=None),
        "evidence packs": dict(exemption=None),
    }
    return [replace(x, **broken.get(x.name, {})) for x in a]


ERASURE_HOLDINGS = {            # ER-2026-0131, a Wisconsin claimant
    "determinations index": 4, "result cache": 2, "claim memory": 14, "checkpoints": 3,
    "traces": 22, "gateway logs": 9, "evidence packs": 2, "backups": 5, "shadow index": 4,
}
EXIT_HOLDINGS = {               # Northwoods Savings, partner-b
    "passage store": 412, "vector index": 412, "knowledge graph": 57,
    "determinations index": 96, "result cache": 31, "claim memory": 258,
    "team memory and held queue": 11, "checkpoints": 140, "traces": 2210,
    "gateway logs": 5340, "gold set": 18, "evidence packs": 96, "backups": 35,
}

# ---------------------------------------------------------------- changes in the quarter
CHANGES = [
    Change("CH-0901", "nightly re-embed of edited spans", "content", 212),
    Change("CH-0902", "forms-library status changes (retire, supersede)", "metadata", 31),
    Change("CH-0907", "lifecycle field added to every vector record", "metadata", 46089),
    Change("CH-0915", "Iowa bulletin batch after a filing", "content", 640),
    Change("CH-0922", "open-base@1.2 to open-base@1.3", "model", 46089, "retire-by 2026-12-31"),
    Change("CH-1006", "endorsements re-spanned at clause level", "spanning", 9310),
    Change("CH-1013", "title prefix added to the embedded text", "template", 46089),
    Change("CH-1020", "partner-b corpus moved to its own partition", "tenant", 412),
    Change("CH-1024", "claim note embedded by mistake, purged", "purge", 4),
]

CONTROLS_BEFORE = {
    "retired forms never govern new terms": {"rule": "forms calendar", "enforcement point": None,
        "telemetry": None, "evidence": None, "regression test": None},
    "erasure reaches every copy in 30 days": {"rule": "privacy policy", "enforcement point": "erase()",
        "telemetry": "erasure log", "evidence": "tombstones", "regression test": None},
    "no partner data outside its tenant": {"rule": "partner contracts", "enforcement point": "Policy",
        "telemetry": "denials", "evidence": None, "regression test": "ch21 tests"},
    "no vector on a retired model": {"rule": "registry retire_by", "enforcement point": "gateway",
        "telemetry": "notices", "evidence": "registry", "regression test": "ch38 tests"},
    "every asset has an owner": {"rule": None, "enforcement point": None, "telemetry": None,
        "evidence": None, "regression test": None},
    "index changes are recorded and gated": {"rule": "change policy", "enforcement point": "shadow gate",
        "telemetry": "gate report", "evidence": "change records", "regression test": "ch14 tests"},
}
