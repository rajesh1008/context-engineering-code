# The Context Engineering Handbook for LLMs and AI Agents — companion code

Code for the book **The Context Engineering Handbook for LLMs and AI Agents** (AI at Work series) by **Rajesh Ranjan Mahapatra**.

Repository: <https://github.com/rajesh1008/context-engineering-code>

## What's here

- `contextlab/` — the small library the book grows chapter by chapter: retrieval, chunking, vector stores, embeddings registry, memory, freshness, citations, assembly, budgeting, compression, caching, defences, evaluation, observability, governance and more. Its own tests live in `contextlab/tests/`.
- `chNN/` — one folder per chapter lab. A typical folder holds `lab.py` (the experiment the chapter reports), helper modules (the Lakeshore Mutual stand-in corpus, models and scenarios), `test_chNN.py` (the checks behind the chapter's claims), `conftest.py`, and `output.txt` or `out/` (the reference output printed in the book).

All examples follow **Lakeshore Mutual**, a fictional regional property insurer. Every company, policy, claim and person in the data is invented.

### Deterministic stand-in models

The labs do not call hosted models. Wherever the book needs a model (a reader, planner, embedder, reranker, judge or attacker), the lab uses a **deterministic stand-in**: a small, seeded, rule-based model whose behaviour is calibrated to the effects the chapter describes. That keeps every number in the book reproducible on a laptop, free and offline. The chapters say plainly which parts are real (parsers, indexes, token counts, databases, timings) and which are stand-ins. A few optional side scripts run real open models (for example `ch12/real_models.py`) and need extra packages and downloads.

## Requirements

- Python 3.11 or newer (tested on Python 3.13).
- Linux or macOS; Windows should work under WSL.

## Setup

```bash
git clone https://github.com/rajesh1008/context-engineering-code.git
cd context-engineering-code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers every lab and test. The optional packages listed at its end are only for the real-model side scripts.

## Running

Run every test from the repository root:

```bash
pytest -q
```

Run one chapter's tests, or its lab:

```bash
pytest -q ch17
cd ch17 && PYTHONPATH=..:. python lab.py
```

Running labs from inside the chapter folder with `PYTHONPATH=..:.` works for every chapter (it puts `contextlab/` and the chapter's own modules on the path). A root `conftest.py` keeps each chapter's modules separate, so chapters that reuse module names such as `lab` or `standin` can all be tested in one `pytest` run.

### Chapter notes

- **ch10** — the 34 fixture PDFs (300 pages) and `corpus/truth.json` are not stored in the repository. They are rebuilt deterministically by `python ch10/make_corpus.py`, and the chapter's `conftest.py` does this automatically the first time you run its tests. The OCR results are shipped in `ch10/corpus/ocr_cache.json`; re-running OCR (`ch10/ocr_all.py`) needs `rapidocr-onnxruntime`.
- **ch13** — needs `lancedb` and `pyarrow` (in `requirements.txt`). The trained LanceDB indexes (`out/frozen/`, about 130 MB) are not stored here. The first `python lab.py` (about five minutes) trains and freezes a build of its own; LanceDB index training is not seeded, so the untuned engines' lost/gained counts may differ slightly from the book, while the tuned index results repeat. See `ch13/README.md`.
- **ch25** — `snapshot/` is a pinned copy of the book's code that the code-map lab reads; `gold_jedi.py` needs `jedi`.
- **ch31** — real token counts use `tiktoken` (`o200k_base`), which downloads the encoding on first use. Without `tiktoken` the lab falls back to an approximate count and its output will differ from the book.
- **ch34, ch38** — run the lab from the chapter folder with `PYTHONPATH=..:.` as above.
- **ch36** — the optional DSPy test is skipped unless `dspy==3.4.1` is installed.

## Chapter index

| Ch | Title | Code | Tests |
|---:|---|---|---|
| 1 | From Prompts to Context | [`ch01/`](ch01/) | yes |
| 2 | An End-to-End Context Project | [`ch02/`](ch02/) | yes |
| 3 | How the Model Reads the Window | [`ch03/`](ch03/) | yes |
| 4 | Context Rot, Measured | [`ch04/`](ch04/) | yes |
| 5 | Long Context or Retrieval: When the Window Is Enough | [`ch05/`](ch05/) | yes |
| 6 | Instructions | [`ch06/`](ch06/) | yes |
| 7 | How Instructions Break | [`ch07/`](ch07/) | yes |
| 8 | Writing Instructions with the Experts: The Rulebook as Context | [`ch08/`](ch08/) | yes |
| 9 | External Knowledge: What Counts as Evidence | [`ch09/`](ch09/) | yes |
| 10 | Parsing and Chunking: The Unglamorous Half | [`ch10/`](ch10/) | yes |
| 11 | Documents at Scale: Fifty Formats, Quality Scores and Confidence | [`ch11/`](ch11/) | yes |
| 12 | Retrieval from First Principles | [`ch12/`](ch12/) | no |
| 13 | Vector Databases and Indexes in Production | [`ch13/`](ch13/) | yes |
| 14 | Embedding Models: Choosing, Adapting and Migrating | [`ch14/`](ch14/) | yes |
| 15 | Similarity Is Not Authority | [`ch15/`](ch15/) | yes |
| 16 | Structured Data as Context: Text-to-SQL, Semantic Layers and Tables | [`ch16/`](ch16/) | yes |
| 17 | Knowledge Graphs and GraphRAG: Relationships When Passages Are Not Enough | [`ch17/`](ch17/) | yes |
| 18 | Agentic Retrieval | [`ch18/`](ch18/) | yes |
| 19 | Multimodal Context: Photos, Diagrams and Voice in the Packet | [`ch19/`](ch19/) | yes |
| 20 | Workshop: Rebuilding Lakeshore Retrieval | [`ch20/`](ch20/) | yes |
| 21 | Memory: Giving the System a Past | [`ch21/`](ch21/) | yes |
| 22 | How Memory Breaks | [`ch22/`](ch22/) | yes |
| 23 | Memory Lifecycle and State | [`ch23/`](ch23/) | yes |
| 24 | Freshness: Live Data, Streams and Change in the Corpus | [`ch24/`](ch24/) | yes |
| 25 | Code and Workspaces as Context | [`ch25/`](ch25/) | yes |
| 26 | Tools as Context | [`ch26/`](ch26/) | yes |
| 27 | The User's Turn | [`ch27/`](ch27/) | yes |
| 28 | Citations and Attribution: Every Sentence Points to Its Page | [`ch28/`](ch28/) | yes |
| 29 | A Taxonomy of Context Failures | [`ch29/`](ch29/) | yes |
| 30 | Reproducing the Failures | [`ch30/`](ch30/) | yes |
| 31 | Assembling the Packet | [`ch31/`](ch31/) | yes |
| 32 | Budgeting and Selection | [`ch32/`](ch32/) | yes |
| 33 | Compression | [`ch33/`](ch33/) | yes |
| 34 | Prompt Caching in Depth: Layout, Invalidation and Cost | [`ch34/`](ch34/) | yes |
| 35 | The Cost–Quality Frontier | [`ch35/`](ch35/) | yes |
| 36 | Optimising Context Programmatically | [`ch36/`](ch36/) | yes |
| 37 | Defensive Context | [`ch37/`](ch37/) | yes |
| 38 | Context Behind a Model Gateway: Approved Models, Residency and Deprecations | [`ch38/`](ch38/) | yes |
| 39 | Evaluating Context | [`ch39/`](ch39/) | yes |
| 40 | The Test Pyramid for the Context Layer | [`ch40/`](ch40/) | yes |
| 41 | Error Analysis as the Engine | [`ch41/`](ch41/) | yes |
| 42 | Context Observability | [`ch42/`](ch42/) | yes |
| 43 | Shadow Mode and Evidence for the Examiner | [`ch43/`](ch43/) | yes |
| 44 | Governing Knowledge, Embeddings and Memory | [`ch44/`](ch44/) | yes |
| 45 | Operating the Context Layer for Twelve Months | [`ch45/`](ch45/) | yes |
| 46 | The Context Layer as a Platform: Serving Many Teams | [`ch46/`](ch46/) | yes |
| 47 | Context in Multi-Step and Multi-Agent Systems | [`ch47/`](ch47/) | yes |
| 48 | Patterns and Anti-Patterns | [`ch48/`](ch48/) | yes |
| 49 | The Context Engineer's Interview, from Both Sides | [`ch49/`](ch49/) | yes |
| 50 | End-to-End Case Studies | [`ch50/`](ch50/) | yes |
| 51 | The Last Twenty Percent, and Where This Is Going | [`ch51/`](ch51/) | yes |

## Errata and corrections

Found a bug, a number that doesn't reproduce, or an error in the book? Please open an issue on this repository or email **mahapatra.rajeshranjan@gmail.com**.

## Licence

The code is released under the MIT licence (see `LICENSE`), copyright 2026 Rajesh Ranjan Mahapatra. The book's text and figures are not part of this repository and are not covered by that licence.
