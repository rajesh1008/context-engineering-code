"""Shared helpers for lab.py and explore.py (Chapter 7)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from standin import CORPUS, Pipeline, PipelineConfig  # noqa: E402

corpus = CORPUS
MODELS = ("sim-large", "sim-mid", "sim-small")
SEEDS = tuple(range(2026, 2036))
ROOF = ("roof_matching", "roof_settlement", "hail_cosmetic", "wind_hail", "ice_dam")
WATER = ("water_backup", "frozen_pipes", "flood", "mold")


def pipe(ins, model: str, seed: int = 2026) -> Pipeline:
    return Pipeline(PipelineConfig(model=model, instructions=ins, seed=seed))


def traces(ins, model: str, dets, seed: int = 2026) -> list:
    p = pipe(ins, model, seed)
    return [p.run(d) for d in dets]


def adh(pipeline: Pipeline, dets) -> int:
    """Probes answered correctly: the cited provision is the governing one."""
    return sum(not pipeline.run(d).wrong for d in dets)
