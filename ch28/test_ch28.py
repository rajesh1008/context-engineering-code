"""Chapter 28 tests: the Citation Checker and the lab's claims."""
from contextlab.citations import (CitationChecker, Entry, Manifest, mark, parse,
                                  precision_recall, verify)
import lakeshore28 as L

E1 = Entry("HO 04 95", "2024-06", 1, 3, "A $1,000 deductible applies.")
E2 = Entry("HO 04 95", "2024-06", 1, 4, "We do not pay for loss caused by your negligent failure to maintain the sump pump.")


def test_handle_is_content_addressed():
    assert E1.handle.startswith("HO 04 95@2024-06/p1/s3#")
    changed = Entry("HO 04 95", "2024-06", 1, 3, "A $500 deductible applies.")
    assert changed.handle != E1.handle


def test_parse_formats():
    s = f"A $1,000 deductible applies. [[{E1.handle}]] [2] [HO 04 95 (2024-06), page 1] [HO-3 (2024-06)]"
    kinds = [k for k, _ in parse(s)]
    assert kinds == ["handle", "index", "page", "doc"]


def test_verifier_numbers_and_polarity():
    assert verify("A $1,000 deductible applies.", E1.text) == 1.0
    assert verify("A $500 deductible applies.", E1.text) < 0.6
    assert verify("We pay for loss caused by failure to maintain the sump pump.", E2.text) < 0.6


def test_checker_flags_not_deletes():
    man = Manifest([E1, E2])
    f = CitationChecker().check_sentence(f"A $500 deductible applies. [[{E1.handle}]]", man)
    assert f.status == "unverified"
    assert mark(f).startswith("A $500 deductible applies.") and "UNVERIFIED" in mark(f)


def test_reattribution_moves_pointer():
    man = Manifest([E1, E2])
    f = CitationChecker().check_sentence(f"A $1,000 deductible applies. [[{E2.handle}]]", man)
    assert f.status == "reattributed" and f.moved_to == E1.handle


def test_missing_and_exempt():
    man = Manifest([E1])
    c = CitationChecker()
    assert c.check_sentence("A deductible applies.", man).status == "missing"
    assert c.check_sentence("Please send photographs. [no-source]", man).status == "exempt"


def test_precision_recall():
    labels = [dict(needs=True, total_cites=2, supported_cites=1),
              dict(needs=True, total_cites=0, supported_cites=0),
              dict(needs=False, total_cites=0, supported_cites=0)]
    assert precision_recall(labels) == (0.5, 0.5)


def test_fixture_is_seeded_and_has_emily():
    a, b = L.build(), L.build()
    assert [s.text for d in a for s in d.sentences] == [s.text for d in b for s in d.sentences]
    assert len(a) == 300 and a[187].claim_id == "CLM-2026-01187"


def test_reorder_breaks_indices_not_handles():
    d = L.build()[0]
    order = L.rerank(d.packet, d.det_id)
    s = next(x for x in d.sentences if x.kind == "policy" and x.pointers)
    idx = Manifest(order).resolve(*parse(L.render(s, d.packet, "index"))[0])
    hdl = Manifest(order).resolve(*parse(L.render(s, d.packet, "handle"))[0])
    assert hdl == [s.pointers[0]]
    assert idx != hdl or order.index(s.pointers[0]) == d.packet.index(s.pointers[0])


def test_older_edition_pages_shift():
    assert L.ho3_page("2022-01", 7) == 8 and L.ho3_page("2024-06", 7) == 7
