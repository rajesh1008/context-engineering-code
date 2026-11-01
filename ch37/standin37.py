"""Chapter 37 deterministic stand-in for the model reading a packet.

It is not a model. It answers from the governing span unless an injected
payload is present in what it was given, and then it complies with a
probability set by how the payload is framed and where it sits. The draw is
a hash of the attack id, so every run gives the same result. The numbers are
assumptions chosen to be plausible, not measurements of any model:

  framing        flat prompt   tier-labelled data block
  role forgery       0.85          0.10
  addressed          0.60          0.15
  reworded           0.50          0.20
  plausible (party)  0.55          0.30
  plausible (gov.)   0.90          0.90   # a governed span is meant to be believed
  tool description   0.60          0.60   # descriptions are not data blocks
"""
import hashlib

P = {"flat": {"role": .85, "addressed": .60, "reworded": .50, "plausible": .55,
              "plausible_gov": .90, "tool_desc": .60},
     "blocked": {"role": .10, "addressed": .15, "reworded": .20, "plausible": .30,
                 "plausible_gov": .90, "tool_desc": .60}}


def draw(aid: str) -> float:
    return int(hashlib.sha256(aid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def complies(aid: str, framing: str, technique: str, force: bool = False) -> bool:
    """force=True is the worst case: a reader that always obeys."""
    return force or draw(aid) < P[framing][technique]
