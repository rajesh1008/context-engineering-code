# Chapter 13 lab: a vector store that keeps the delta at zero

    python3 -m venv .venv-ch13 && .venv-ch13/bin/pip install lancedb==0.41.* pyarrow numpy pydantic pytest
    cd ch13 && PYTHONPATH=.. ../.venv-ch13/bin/python lab.py > output.txt     # about five minutes
    PYTHONPATH=.. ../.venv-ch13/bin/python latency.py                          # Table 13.1 latency, three runs
    PYTHONPATH=.. ../.venv-ch13/bin/python -m pytest -q test_ch13.py           # nine contract tests

## Why the numbers repeat
LanceDB exposes no seed for index training (k-means sampling for IVF, level assignment for HNSW),
so two builds of the same index on the same data differ. The first run trains both indexes and
freezes the build in `out/frozen/`; every later run restores a fresh copy of it, so every quality
number in the chapter (recall, evidence hit, lost/gained counts, scoreboard) repeats exactly.
`python lab.py --rebuild` (or deleting `out/frozen/`) trains new indexes: the two untuned engines'
lost/gained counts then move from build to build, while the tuned IVF-PQ and `spans-v7` lose none.
The frozen build used for the book is distributed separately as `ch13-frozen-pq.zip` and `ch13-frozen-hnsw.zip`
(unzip both into `ch13/out/`); without it your first run freezes a build of its own.

Latency is wall-clock and depends on the machine. The chapter's Table 13.1 latency columns are the
median of three `latency.py` runs on the book's machine; figure 13.2 plots those values of record
(`03-figures/src/ch13/latency_of_record.csv`).
