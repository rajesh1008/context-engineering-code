# Chapter 5 code: long context versus retrieval

> **Stand-in note.** Every model here is a deterministic `SimulatedModel` and the retriever is
> a stand-in (standin.py); the corpus (351 binding spans, 21,354 tokens, in 42 documents) and
> the 500 determinations are synthetic. Every prompt is rendered and counted for real, the
> stuffing, applicability filter, cut, cache and cost arithmetic run in code, and a model that
> gets a prompt over its window rejects it. Only the simulated models' errors (which span
> each one cites when it misreads a given packet) and the retriever's seeds come from
> `calibrate.py`, which embeds the fixtures that reproduce the chapter's illustrative profile
> and writes `calibration.json`. Prices (US$ per million tokens), latency constants, cache
> terms and lookup minutes are illustrative lab constants, not any provider's figures.

    python3 lab.py               # writes output.txt and the CSVs behind the chapter's figures
    python3 explore.py           # and the other scripts below; each writes out/<name>.txt
    python3 -m pytest -q         # the chapter's numbers, Listing 5.1, the cache replay, the scorer
    python3 calibrate.py         # optional: rewrites calibration.json from the embedded fixtures

| File | What it does |
| --- | --- |
| `standin.py` | corpus, determinations, stand-in retriever, the four assemblies, `SimulatedModel`, `Pipeline`, cache replay |
| `spec_corpus.py` | which documents, sections and topics exist |
| `calibrate.py` | document lengths and the fitted fixtures; writes `calibration.json` |
| `lab.py` | Sections 5.2-5.4: quality, cost and latency, families, failure injection (`output.txt`, CSVs) |
| `why_wrong.py` | Listing 5.4: why each strategy's wrong citations were wrong (`out/why_wrong.txt`) |
| `explore.py` | slices, paired comparison, holdout, stale pointer, evidence budget, Carter (`out/explore.txt`) |
| `change_rate.py` | Listing 5.12 change rate and Listing 5.11 corpus size (`out/change_rate.txt`) |
| `families.py` | Listing 5.14 per-family runs with the cache (`out/families.txt`) |
| `scoreboard_ch05.py` | Listing 5.15 and the scoreboard rows (`out/scoreboard.txt`) |
| `benchmark.py` | Listing 5.16: exports the benchmark JSONL to `ch05-benchmark/` (regenerated on every run, git-ignored; about 48 MB) and scores replies; `--send MODEL` posts to an OpenAI-compatible endpoint |
| `../contextlab/longctx.py` | `stuffed_spans` (Listing 5.1), the cache and corpus orders |
| `test_ch05.py` | `python -m pytest 04-code/ch05 -q` |

Needs Python 3.11+ and nothing outside the standard library and `contextlab`.
