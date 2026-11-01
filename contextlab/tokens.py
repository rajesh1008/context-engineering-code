"""contextlab.tokens (v0.8, Chapters 3-5): illustrative prices, windows and latency.

Prices are illustrative list prices in US dollars per million tokens, set inside the
range of the real prices in Chapter 5's Table 5.1. The latency model charges a fixed
overhead, a prefill rate for input tokens (cached tokens prefill ten times faster) and
a decode rate for output. Its constants are placeholders, not measurements: replace
them with your provider's measured numbers before any latency claim leaves the lab.
"""
from __future__ import annotations

from .cache import DISCOUNT

PRICES = {  # (input, output) US$ per million tokens
    "sim-large": (3.00, 15.00),
    "sim-mid": (1.00, 5.00),
    "sim-small": (0.25, 1.25),
}
WINDOWS = {"sim-large": 200_000, "sim-mid": 128_000, "sim-small": 32_000}
KNEES = {"sim-large": 36_336, "sim-mid": 14_447, "sim-small": 14_447}   # Chapter 4, prompt tokens
LATENCY = {  # overhead s, prefill tokens/s, decode tokens/s
    "sim-large": (0.28, 5_000.0, 50.0),
    "sim-mid": (0.17, 10_000.0, 90.0),
    "sim-small": (0.10, 20_000.0, 150.0),
}


def cost_usd(model: str, tokens_in: int, tokens_out: int, cached: int = 0) -> float:
    p_in, p_out = PRICES[model]
    return ((tokens_in - cached) * p_in + cached * p_in * DISCOUNT + tokens_out * p_out) / 1e6


def latency_s(model: str, tokens_in: int, tokens_out: int, cached: int = 0) -> float:
    over, prefill, decode = LATENCY[model]
    return over + (tokens_in - cached + cached / 10) / prefill + tokens_out / decode
