"""contextlab.llm (Chapter 3): the stand-in reader's model profiles and position weight.

SimulatedModel in the chapter labs does not run attention. It scores the spans in a
packet and multiplies each score by a position weight. The weight below is built to
reproduce the published U shape (high at both edges, a dip in the middle that deepens
with packet length); it is an assumption of the lab, not a measurement of any model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    middle_dip: float
    window: int


PROFILES = {
    "sim-large": ModelProfile("sim-large", 0.30, 200_000),
    "sim-mid": ModelProfile("sim-mid", 0.35, 128_000),
    "sim-small": ModelProfile("sim-small", 0.45, 32_000),
}


def position_weight(mid: float, total: int, profile: ModelProfile) -> float:
    """1 at the edges, lowest in the middle; the dip deepens with length."""
    depth = profile.middle_dip * (1 - math.exp(-total / 15_000))
    return 1 - depth * math.sin(math.pi * mid) ** 2
