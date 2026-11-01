import hashlib, json, sys
sys.path.insert(0, "..")
from contextlab.parsing import parse_layout
from contextlab.chunker import Chunker

meta = next(d for d in json.load(open("corpus/truth.json"))["docs"] if d["doc_id"] == "HO-3_2022-01")
meta["sha256"] = hashlib.sha256(open("corpus/HO-3_2022-01.pdf", "rb").read()).hexdigest()
doc = parse_layout("corpus/HO-3_2022-01.pdf", "HO-3_2022-01")
chunks = Chunker("structural", size=512, prefix=True).chunk(doc, meta)
c = next(c for c in chunks if c.path[-1] == "12. Freezing")
print(len(chunks), "chunks;", c.chunk_id, c.pages, c.parser, c.source_sha, c.tokens, "tokens")
print(c.citation())
print(c.prefix)
