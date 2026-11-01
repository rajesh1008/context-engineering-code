"""contextlab.cache (v0.8, Chapters 3 and 5): a simulated provider prompt cache.

Works the way hosted caches bill, with illustrative terms: the prompt is hashed in
prefix blocks of 1,024 tokens, a block seen within the last 50 calls is a hit, and
cached input tokens are billed at 10% of the list input price (contextlab.tokens).
A block is identified by every segment that starts before its end, so any change at
or before a block invalidates it and every block after it. Chapter 34 replaces this
with a fuller model (contextlab.caching).
"""
from __future__ import annotations

import hashlib

BLOCK = 1_024
LIFETIME = 50          # calls
DISCOUNT = 0.10        # cached input billed at this share of list


class PromptCache:
    def __init__(self, block: int = BLOCK, lifetime: int = LIFETIME):
        self.block, self.lifetime = block, lifetime
        self.seen: dict[str, int] = {}
        self.calls = 0

    def lookup(self, pieces: list[tuple[str, int]]) -> int:
        """Cached tokens for one prompt given as (text, tokens) segments; records the call."""
        self.calls += 1
        total = sum(n for _, n in pieces)
        h, pos, i, keys = hashlib.sha256(), 0, 0, []
        for b in range(self.block, total + 1, self.block):
            while i < len(pieces) and pos < b:          # every segment that starts before b
                h.update(pieces[i][0].encode())
                h.update(b"\x00")
                pos += pieces[i][1]
                i += 1
            k = h.copy()
            k.update(str(b).encode())
            keys.append(k.hexdigest())
        cached = 0
        for k in keys:
            last = self.seen.get(k)
            if last is None or self.calls - last > self.lifetime:
                break
            cached += self.block
        for k in keys:
            self.seen[k] = self.calls
        return cached
