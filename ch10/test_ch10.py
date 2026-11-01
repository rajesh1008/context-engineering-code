"""Chapter 10 checks: run after make_corpus.py (OCR not needed)."""
import json, os, re
import pytest
from contextlab.parsing import parse_layout, parse_text_layer, ParsedDoc, Block
from contextlab.chunker import Chunker

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
TRUTH = json.load(open(os.path.join(CORPUS, "truth.json")))
DOCS = {d["doc_id"]: d for d in TRUTH["docs"]}
import hashlib
for _d in TRUTH["docs"]:
    _d["sha256"] = hashlib.sha256(open(os.path.join(CORPUS, _d["doc_id"] + ".pdf"), "rb").read()).hexdigest()
CITE = re.compile(r"^.+ \(\d{4}-\d{2}\), .+, page \d+$")


def pdf(i):
    return os.path.join(CORPUS, i + ".pdf")


@pytest.fixture(scope="module")
def form22():
    return parse_layout(pdf("HO-3_2022-01"), "HO-3_2022-01")


def test_text_layer_interleaves_the_2022_columns():
    tl = " ".join(b.text for b in parse_text_layer(pdf("HO-3_2022-01"), "HO-3_2022-01").blocks if b.page == 2)
    assert "freezing of a plumbing, heating" not in re.sub(r"\s+", " ", tl)


def test_structural_chunk_keeps_perils_12_whole(form22):
    chunks = Chunker("structural", 512, prefix=True).chunk(form22, DOCS["HO-3_2022-01"])
    c = next(c for c in chunks if c.path and c.path[-1] == "12. Freezing"
             and c.path[0] == "Section I Perils Insured Against")
    assert "maintain heat in the building" in c.text and "frozen pipe, fitting or appliance" in c.text
    assert c.pages == [2] and c.prefix.endswith("> 12. Freezing")


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "structural", "semantic"])
def test_every_chunk_is_citable(form22, strategy):
    for c in Chunker(strategy, 256).chunk(form22, DOCS["HO-3_2022-01"]):
        assert c.doc_id and c.label and c.edition and c.pages and c.parser == "layout"
        assert len(c.source_sha) == 12
        assert CITE.match(c.citation()), c.citation()


def test_table_rows_carry_header_and_own_page():
    d = "LM-DED_2024-06"
    chunks = Chunker("structural", 512, prefix=True).chunk(parse_layout(pdf(d), d), DOCS[d])
    iowa = [c for c in chunks if c.text.startswith("State |") and "Iowa" in c.text]
    assert len(iowa) == 1
    assert "State" in iowa[0].text and re.search(r"Iowa\W+.*1% of Coverage A", iowa[0].text)
    assert iowa[0].pages == [3]


def test_footer_is_stripped_and_kept_only_on_request():
    d = "HO0454_2022-01"
    phrase = "Coverage under this endorsement applies only when"
    on = Chunker("structural", 512).chunk(parse_layout(pdf(d), d), DOCS[d])
    off = Chunker("structural", 512).chunk(parse_layout(pdf(d), d, strip_furniture=False), DOCS[d])
    assert not any(phrase in c.text for c in on)
    assert any(phrase in c.text for c in off)


def test_same_input_same_chunks(form22):
    a = [c.digest() for c in Chunker("structural", 512, prefix=True).chunk(form22, DOCS["HO-3_2022-01"])]
    b = [c.digest() for c in Chunker("structural", 512, prefix=True).chunk(
        parse_layout(pdf("HO-3_2022-01"), "HO-3_2022-01"), DOCS["HO-3_2022-01"])]
    assert a == b


def test_one_edited_sentence_touches_one_structural_chunk():
    d = "HO-3_2024-06"
    base = parse_layout(pdf(d), d)
    rev = ParsedDoc(d, "layout", [Block(b.kind, b.text, b.page, b.level, b.rows, b.continued) for b in base.blocks], 0, base.pages)
    k = next(i for i, b in enumerate(rev.blocks) if b.text.startswith("We insure for loss caused by windstorm or hail."))
    rev.blocks[k].text = rev.blocks[k].text.replace("windstorm or hail.", "windstorm or hail, including a derecho.")
    ch = Chunker("structural", 512, prefix=True)
    before = {c.digest() for c in ch.chunk(base, DOCS[d])}
    assert sum(c.digest() not in before for c in ch.chunk(rev, DOCS[d])) == 1
