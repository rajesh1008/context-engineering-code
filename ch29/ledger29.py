"""Chapter 29 ledger: the 82 wrong determinations under Chapter 8's configuration,
the incidents to date, the ten class probes and the two-class injection.

The 82 come from Chapter 9's full mix (ch09/fullmix.py, unchanged). Each carries
the open code an analyst wrote from its trace and a stand-in packet built to match
the cause the later chapters found. Note: deterministic stand-in, no model runs.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ch09"))
import fullmix                                          # noqa: E402
from standin29 import Case, Span                        # noqa: E402

G = Span("gold", "gold", 320, 0.85)


def pads(n, sim=0.70, tokens=300):
    return [Span(f"p{i}", "pad", tokens, sim) for i in range(n)]


def look(sim=0.90, sid="look"):
    return Span(sid, "lookalike", 320, sim, applies=False)


# open code -> (axial class, stand-in packet builder, fixing chapter)
CODES = {
    "edition cited is not the edition in force on the date of loss":
        ("confusion", lambda: [look(), G] + pads(3), "Ch 9 verified query"),
    "base form cited; endorsement is a scan never indexed":
        ("missing_evidence", lambda: [Span("base", "pad", 320, 0.80)] + pads(3), "Ch 9 admission"),
    "another state's bulletin cited":
        ("confusion", lambda: [look(), G] + pads(3), "Ch 15 jurisdiction"),
    "another state's bulletin cited; rule needs two spans":
        ("confusion", lambda: [look(), Span("gold", "gold", 320, 0.85, complete=False)] + pads(3),
         "Ch 15 + Ch 17 graph"),
    "state's bulletin absent from the packet":
        ("missing_evidence", lambda: pads(4), "Ch 14 embedding model"),
    "governing sentence cut at a span boundary":
        ("fragmentation", lambda: [Span("gold", "gold", 320, 0.85, complete=False)] + pads(4),
         "Ch 10 structural spans"),
    "table broken across a column or page":
        ("fragmentation", lambda: [Span("gold", "gold", 320, 0.85, complete=False)] + pads(4),
         "Ch 10 structural spans"),
    "underwriting guidance cited as governing":
        ("confusion", lambda: [look(0.88, "guidance"), G] + pads(3), "Ch 15 authority"),
    "roof age in the claim header wrong (licensed feed)":
        ("poisoning", lambda: [Span("roof", "false", 120, 0.90, true=False, origin="feed"), G]
         + pads(3), "Ch 9 licence gate"),
    "change date in the partner feed wrong":
        ("drift", lambda: [Span("gold", "gold", 320, 0.85, true=False, lag=True)] + pads(4),
         "Ch 24 change log"),
    "endorsement schedule misread in the policy summary":
        ("poisoning", lambda: [Span("summ", "false", 200, 0.90, true=False, origin="summary"), G]
         + pads(3), "open: policy record"),
    "endorsement missing from the policy record":
        ("missing_evidence", lambda: pads(4), "open: policy record"),
    "no defect visible in the trace":
        ("not_context", lambda: [G] + pads(4), "not recorded"),
}


# what a symptom-first triage wrote before the flowchart (the answer's visible defect)
SYMPTOM = {
    "edition cited is not the edition in force on the date of loss": "confusion",
    "base form cited; endorsement is a scan never indexed": "confusion",
    "another state's bulletin cited": "confusion",
    "another state's bulletin cited; rule needs two spans": "confusion",
    "state's bulletin absent from the packet": "missing_evidence",
    "governing sentence cut at a span boundary": "not_context",
    "table broken across a column or page": "not_context",
    "underwriting guidance cited as governing": "clash",
    "roof age in the claim header wrong (licensed feed)": "not_context",
    "change date in the partner feed wrong": "poisoning",
    "endorsement schedule misread in the policy summary": "not_context",
    "endorsement missing from the policy record": "not_context",
    "no defect visible in the trace": "not_context",
}


def ledger():
    """[(det_id, stratum, open_code, hand_class, Case, fixed_by)] for the 82."""
    dets = fullmix.build()
    wrong = [d for d in dets if fullmix.outcome(d, "ch08") == "wrong"]
    by = {}
    for d in wrong:
        by.setdefault(d.stratum, []).append(d)
    plan = []
    pa = by["policy-admin"]
    plan += [(d, "edition cited is not the edition in force on the date of loss")
             for d in pa if d.pointer_wrong]
    rest = [d for d in pa if not d.pointer_wrong]
    plan += list(zip(rest, ["endorsement schedule misread in the policy summary"] * 2
                     + ["endorsement missing from the policy record",
                        "no defect visible in the trace"]))
    plan += [(d, "base form cited; endorsement is a scan never indexed") for d in by["scanned"]]
    ms = by["multi-state"]
    plan += [(d, "another state's bulletin cited") for d in ms[:16]]
    plan += [(d, "another state's bulletin cited; rule needs two spans") for d in ms[16:22]]
    plan += [(d, "state's bulletin absent from the packet") for d in ms[22:]]
    for d in by["docstore"]:
        code = ("table broken across a column or page" if d.split == "holdout"
                else "governing sentence cut at a span boundary")
        plan.append((d, code))
    plan += [(d, "underwriting guidance cited as governing") for d in by["guidance"]]
    tp = by["third-party"]
    plan += [(d, "roof age in the claim header wrong (licensed feed)") for d in tp[:3]]
    plan += [(tp[3], "change date in the partner feed wrong")]
    out = []
    for d, code in plan:
        cls, build, fix = CODES[code]
        case = Case(d.det_id, tuple(build()), slip=(cls == "not_context"))
        out.append((d.det_id, d.stratum, code, cls, case, fix))
    return out, dets


# ---------------------------------------------------------------- incidents to date
def incidents():
    """Lakeshore incidents, each as a stand-in packet. (id, chapter, story, root, Case)."""
    L = lambda *s: tuple(s)
    near = Span("near", "near", 320, 0.80)
    return [
        ("I01", "Ch 1", "2025 chatbot cites HO-3 (2019-03) exclusion", "retrieval, no versions",
         Case("I01", L(look(0.90, "ho3-2019"), Span("excl", "rival", 300, 0.84), *pads(6)))),
        ("I02", "Ch 2", "current-form pointer never moved", "unowned copy",
         Case("I02", L(Span("gold", "gold", 320, 0.85, true=False, lag=True), *pads(3)))),
        ("I03", "Ch 6", "v2.3 tidy-up deletes the precedence sentence", "instruction change",
         Case("I03", L(Span("rival", "rival", 320, 0.88), G, *pads(3)))),
        ("I04", "Ch 7", "roof skill: bulletins are 'guidance for complaint handling'",
         "instruction against evidence",
         Case("I04", L(Span("skill", "rival", 120, 0.92), G, *pads(3)))),
        ("I05", "Ch 10", "Iowa deductible: 2% read for 1% across a page break", "parser",
         Case("I05", L(Span("gold", "gold", 320, 0.85, complete=False), *pads(4)))),
        ("I06", "Ch 13", "post-filter: small tenant gets fewer than k rows", "vector store filter",
         Case("I06", L(*pads(3)))),
        ("I07", "Ch 14", "marine spans truncated at 512 tokens; marine retrieval 0.000",
         "overflow at the embedder",
         Case("I07", L(*pads(4)))),
        ("I08", "Ch 16", "remaining back-up limit US$6,150 for US$10,000", "summary drops the term",
         Case("I08", L(Span("gold", "gold", 200, 0.85, complete=False), *pads(3)))),
        ("I09", "Ch 18", "Iowa EB bulletin missing from the graph", "graph behind the store",
         Case("I09", L(Span("amend", "pad", 320, 0.80), *pads(4)))),
        ("I10", "Ch 19", "voice note loses 'No' (conf 0.81, SNR 10.9 dB)", "transcription",
         Case("I10", L(Span("gold", "gold", 200, 0.85, true=False, origin="extraction"),
                       *pads(3)))),
        ("I11", "Ch 22", "team memory read as a rule (the slow slide)", "memory write",
         Case("I11", L(Span("mem", "false", 150, 0.92, true=False, origin="generated"), G,
                       *pads(3)))),
        ("I12", "Ch 24", "partner feed misdates an endorsement change", "partner copy",
         Case("I12", L(Span("gold", "gold", 320, 0.85, true=False, lag=True), *pads(3)))),
    ]


# ---------------------------------------------------------------- probes and injection
def probes(n=40):
    """Ten families of n cases, one defect each, varied deterministically."""
    out = {}
    for k in range(n):
        f = k / n
        near = Span("near", "near", 320, 0.80)
        mid = 2 + (k * 7) % 21
        out.setdefault("overflow", []).append(Case(f"OV{k}", tuple(
            [G] + pads(26)), window=8_000 + 200 * (k % 5)))
        out.setdefault("missing_evidence", []).append(Case(f"ME{k}", tuple(
            [Span("base", "pad", 320, 0.78 + 0.04 * f)] + pads(3))))
        out.setdefault("drift", []).append(Case(f"DR{k}", tuple(
            [Span("gold", "gold", 320, 0.85, true=False, lag=True)] + pads(3 + k % 3))))
        origin = ["generated", "adversarial", "extraction", "feed", "summary"][k % 5]
        poison = ([Span("gold", "gold", 320, 0.85, true=False, origin="extraction")]
                  if origin == "extraction" else
                  [Span("bad", "false", 150, 0.82 + 0.12 * f, true=False, origin=origin), G])
        out.setdefault("poisoning", []).append(Case(f"PO{k}", tuple(poison + pads(3))))
        out.setdefault("confusion", []).append(Case(f"CO{k}", tuple(
            [look(0.80 + 0.12 * f), G] + pads(3))))
        out.setdefault("clash", []).append(Case(f"CL{k}", tuple(
            [G, Span("rival", "rival", 300, 0.82 + 0.10 * f)] + pads(3))))
        out.setdefault("fragmentation", []).append(Case(f"FR{k}", tuple(
            [Span("gold", "gold", 320, 0.85, complete=False)] + pads(4))))
        out.setdefault("distraction", []).append(Case(f"DI{k}", tuple(
            [Span("hist", "offq", 400, 0.82 + 0.10 * f), G] + pads(3))))
        p = pads(24)
        out.setdefault("lost_in_middle", []).append(Case(f"LM{k}", tuple(
            p[:mid] + [G] + p[mid:] + [near])))
        out.setdefault("rot", []).append(Case(f"RO{k}", tuple(
            [near] + pads(34 + 4 * (k % 10)) + [G])))
    return out


def injection(n=40):
    """Gold span mid-packet AND an inapplicable look-alike: two classes at once.
    The first 25 have the gold span in the middle; the last 15 at the edge."""
    out = []
    for k in range(n):
        s = 0.78 + 0.14 * (k % 8) / 7          # look-alike similarity 0.78 to 0.92
        p = pads(24)
        if k < 25:
            spans = p[:12] + [G] + p[12:] + [look(s)]
        else:
            spans = [look(s)] + p + [G]
        out.append(Case(f"INJ{k:02d}", tuple(spans)))
    return out


def mixed(n=100):
    """Realistic packets: one causal defect plus harmless ones that still fire a signal
    (a stale pad copy, a low-ranked look-alike, a weak rival). (injected class, Case)."""
    fam = probes(40)
    harmless = [Span("stale", "pad", 300, 0.60, lag=True, true=False),
                Span("oldlook", "lookalike", 300, 0.60, applies=False),
                Span("weak", "rival", 300, 0.60)]
    out, k = [], 0
    classes = list(fam)
    while len(out) < n:
        cls = classes[k % len(classes)]
        c = fam[cls][(k * 7) % 40]
        k += 1
        from standin29 import run
        if run(c):
            continue
        extra = [harmless[k % 3]] + ([harmless[(k + 1) % 3]] if k % 2 else [])
        spans = list(c.spans)
        out.append((cls, Case(c.cid + "x", tuple(spans[:1] + extra + spans[1:]),
                              window=c.window + 900)))
    return out
