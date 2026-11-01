# Chapter 2 lab: the baseline, the stale pointer and the fix

    ./run_all.sh              # a few seconds; writes output.txt and out/*.txt
    python3 -m pytest -q      # the chapter's numbers, the smoke checks and the fail-closed routing rule
    python3 calibrate.py POOLS.pkl   # optional and slow (MILP, 10-30 min): rebuilds data/calibration.json

Needs Python 3.11+ and pydantic 2 (numpy and scipy only for calibrate.py). The scripts put
`04-code` on the path themselves, so `contextlab` is imported from `../contextlab`.

- `world.py`: Lakeshore's synthetic world (seed 2026): forms, endorsements, bulletins, NAIC
  guidance and underwriting notes (378 spans, 22,326 tokens), 500 policies, 157 claim-history
  records and the 500 determinations.
- `standin.py`: builds the corpus from the world and the fixture, the applicability policy, the
  BM25-style retriever and `SimulatedModel`, a deterministic stand-in reader.
- `../contextlab/pipeline.py`, `../contextlab/scoreboard.py`: `PipelineConfig`, the assembler and
  `Pipeline.run` (Listings 2.6 and 2.7), and the lookup proxy and scoreboard (Listing 2.1).
- `lab.py`: the baseline, the stale form pointer, `reconcile` (Listing 2.10), the fixed pipeline,
  the scoreboard, slices, the second injection, the model comparison and the held-out check
  (`output.txt`).
- `stage_variants.py`: the baseline with `query_facts=False` (`out/stage_variants.txt`).
- `ablation.py`: the k sweep (Listing 2.8), format and order (`out/ablation.txt`).
- `retrieval_modes.py`: the retrieval-modes table (BM25, TF-IDF, dense with the hashing stand-in
  embedder, hybrid) through the baseline pipeline (`out/retrieval_modes.txt`).
- `eval_probes.py`: the lookup-proxy sensitivity, the DET-0001 ranked list, the gold-rank
  distribution, the DET-0022 miss and the first rows of `eval200.csv` (`out/eval_probes.txt`);
  `--write-eval` refreezes `eval200.csv`.
- `smoke.py`: the source registry (Listing 2.4) and the ten smoke checks (Listing 2.9);
  `--stale` runs them on the stale view (`out/smoke.txt`, `out/smoke_stale.txt`).
- `trace_det8.py`: one fixed-pipeline trace (Listing 2.11). `flag.py`: the pilot's routing rule
  (Listing 2.12).
- `calibrate.py`: fits the fixture in `data/calibration.json` (per-question ranking strengths,
  the reader's misreadings per model, the embellished answers and the span lengths) so that the
  lab reproduces the chapter's profile.

The stand-in's mechanism is real and inspectable: retrieval, the applicability filter, packet
assembly, rendering and scoring all run on the packet the pipeline built, so changing the packet
(a stale form edition, no filter, a smaller k, another format) changes the result. What the
reader gets wrong, and how strongly each span matches its question, is a calibrated fixture,
illustrative rather than measured, and says nothing about how any hosted model would score.
The prices are illustrative lab constants, not any provider's price list.
