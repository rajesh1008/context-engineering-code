"""Listing 5.16: export the window-or-retrieve benchmark and check its scorer.

Run: python 04-code/ch05/benchmark.py [--send MODEL] [--out DIR]
Writes one JSON Lines file per strategy (default: ch05/ch05-benchmark/, regenerated
deterministically on every run and git-ignored, since stuff_all.jsonl is about 31 MB),
with each prompt, the map from each [S#] label to its span ID and
the gold span. Without --send it scores the simulated reader from its reply text,
which must reproduce the scoreboard counts (12, 55, 33, 3). With --send MODEL it
posts every prompt to an OpenAI-compatible endpoint (OPENAI_BASE_URL,
OPENAI_API_KEY) and scores the replies the same way. Not run for the book.

STAND-IN: the prompts come from the synthetic corpus and the simulated reader (see standin.py).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import standin as S                     # noqa: E402
from contextlab.instructions import count_tokens  # noqa: E402

FILES = {"retrieve": "retrieve", "stuff_all": "stuff all", "stuff_family": "stuff family", "hybrid": "hybrid"}


def score_reply(row: dict, reply: str) -> bool:
    """Right only if the cited label maps to the governing span (the scoreboard's test).

    No label, or a label that is not in the packet, counts as wrong, the same
    as an abstention: the adjuster falls back to the manual lookup.
    """
    m = re.search(r"\[(S\d+)\]", reply)
    return bool(m) and row["labels"].get(m.group(1)) == row["gold"]


def export(out_dir: Path) -> dict:
    """Write the JSONL files; return {name: (rows, simulated replies)}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, strat in FILES.items():
        rows, replies = [], []
        for t in S.run_all(S.pipeline(strat), S.CORPUS.eval):
            rows.append({"id": t.det.determination_id, "prompt": t.rendered.text,
                         "labels": t.rendered.labels,
                         "gold": t.det.gold_span_id})
            replies.append(t.answer.text)
        path = out_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
        result[name] = rows, replies, path.stat().st_size
    return result


def send(model: str, prompt: str) -> str:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", body, {
        "Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main(out=print, out_dir: Path | None = None, model: str | None = None):
    res = export(out_dir or HERE / "ch05-benchmark")
    for name, (rows, _, size) in res.items():
        mean = statistics.fmean(count_tokens(r["prompt"]) for r in rows)
        out(f"{name:<13} {len(rows)} prompts  mean {mean:>8,.0f} tokens {size / 1e6:>6.1f} MB")
    for name, (rows, replies, _) in res.items():
        if model:
            replies = [send(model, r["prompt"]) for r in rows]
        wrong = sum(not score_reply(r, x) for r, x in zip(rows, replies))
        who = model or "simulated reader"
        out(f"{name:<13} {who}, scored from its reply text: wrong {wrong}/{len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", metavar="MODEL")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    lines = []
    main(lines.append, a.out, a.send)
    if not a.send:
        (HERE / "out").mkdir(exist_ok=True)
        (HERE / "out" / "benchmark.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
