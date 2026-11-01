"""OCR every page once and cache the lines (slow: a few minutes on a laptop CPU)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from contextlab.parsing import ocr_page_lines

HERE = os.path.dirname(os.path.abspath(__file__))
truth = json.load(open(os.path.join(HERE, "corpus", "truth.json")))
cache, timing = {}, {}
for d in truth["docs"]:
    path = os.path.join(HERE, "corpus", d["doc_id"] + ".pdf")
    t0 = time.perf_counter()
    cache[d["doc_id"]] = [ocr_page_lines(path, i) for i in range(d["pages"])]
    timing[d["doc_id"]] = time.perf_counter() - t0
    print(d["doc_id"], d["pages"], f"{timing[d['doc_id']]:.1f}s", flush=True)
json.dump({"lines": cache, "seconds": timing}, open(os.path.join(HERE, "corpus", "ocr_cache.json"), "w"))
print("total seconds", sum(timing.values()))
