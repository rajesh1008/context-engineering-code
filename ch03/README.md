# Chapter 3 lab: tokens, attention, the KV cache and the system turn

    python3 tokens.py         # Listings 3.1-3.3 (tiktoken columns need tiktoken installed)
    python3 embeddings.py     # Listing 3.4
    python3 lab.py            # Listing 3.6, the failure injection and the fix (about a minute)
    python3 fix_ablation.py   # Listing 3.7
    python3 variance.py       # Listing 3.8
    python3 kv_memory.py      # Listing 3.9
    python3 system_turn.py    # Listing 3.11
    python3 cache_layout.py   # Listings 3.10 and 3.12
    python3 cag_snapshot.py   # Listing 3.13
    python3 order.py          # Listing 3.14
    python3 flip_probe.py     # Listing 3.15
    python3 run_all.py        # all of the above; writes out/*.txt and output.txt
    python3 -m pytest -q      # the chapter's numbers
    python3 calibrate.py      # optional, several minutes: rebuilds calibration.json

Needs Python 3.11+, pydantic 2 and numpy; tiktoken (optional) for the cl100k_base and
o200k_base columns of Listings 3.1 and 3.2.

- `standin.py`: Lakeshore's synthetic corpus (378 spans, 22,326 tokens, 99 of them effective
  after 2024-05-31), the 200 reviewed determinations, the Chapter 2 pipeline (retrieve,
  applicability filter, assemble in four formats and five orders) and `SimulatedModel`, a
  deterministic stand-in reader.
- `experiments.py`: the padded packets of the position sweep and the retrieved packets of the
  order experiment, shared by the scripts and by `calibrate.py`.
- `../contextlab/llm.py`: the position weight of Listing 3.5; `../contextlab/hashembed.py`: the
  offline embedder of Listing 3.4; `../contextlab/cache.py`: the prompt cache.
- `calibrate.py`: builds the determinations, span sizes, retrieval candidates and misread
  fixtures, and fits them so that the lab reproduces the chapter's profile.

The stand-in's mechanism is real and inspectable: every answer is computed from the packet the
pipeline built, and the governing provision, the filter, padding, ordering, token counts, the
prompt cache, prices and latency are all computed. Its behaviour profile (which packets a model
misreads, and the rival it then cites) is a calibrated fixture, keyed by the exact sequence of
spans the model was shown, illustrative rather than measured, and says nothing about how any
hosted model would score.
