"""Chapter 13: repeated latency runs on the tables lab.py built.

The box running the lab is shared, so single-run tail latencies are noisy.
This script runs the 1,000 determinations three times per engine at k = 20
and reports the median of the three runs' p50 and p99.
Run after lab.py:  .venv-ch13/bin/python ch13/latency.py
"""
import numpy as np

import lab
from lab import C, DB, FlatStore, LanceStore, Row, SimEncoder, V7, lat, run, v7

spans, dets = C.build()
enc = SimEncoder()
qv = {d.qid: enc.encode(d.text).astype(np.float32) for d in dets}
flat = FlatStore()
flat.upsert([Row(s.span_id, s.tenant, s.doc_type, s.text, enc.encode(s.text).astype(np.float32))
             for s in spans.values()])
engines = [("flat (from scratch)", flat), ("ivf_pq, defaults", LanceStore(DB, V7)),
           ("ivf_hnsw_sq, ef=80", LanceStore(lab.HN, nprobes=16, ef=80)),
           ("ivf_pq, nprobes=100, refine=20", v7()), ("spans-v7 + exact < 5,000", v7(exact_below=5000)),
           ("post-filter, no routing", v7(prefilter=False))]
rows, lines = [], []
for name, st in engines:
    run(st, dets[:50], qv, k=20)                       # warm caches
    reps = [lat(run(st, dets, qv, k=20)) for _ in range(3)]
    p50, p99 = float(np.median([r[0] for r in reps])), float(np.median([r[1] for r in reps]))
    rows.append([name, p50, p99])
    lines.append(f"{name:32} p50 {p50:5.1f} ms  p99 {p99:5.1f} ms  (runs: "
                 + ", ".join(f"{a:.1f}/{b:.1f}" for a, b in reps) + ")")
    print(lines[-1], flush=True)
lab.write_csv("latency.csv", ["engine", "p50", "p99"], rows)
with open(lab.os.path.join(lab.OUT, "latency.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
