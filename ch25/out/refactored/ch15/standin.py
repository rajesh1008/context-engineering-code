"""Deterministic stand-in models for the Chapter 15 lab.

No model is called. A stand-in reads a packet the way the chapter's real
runs did: it leans on similarity, honours the authority cues in its
instructions most of the time, and follows a GOVERNS label when one is
given, at a rate that falls with model size. Every draw is a hash of the
model name and the determination, so every run is identical.
"""
import hashlib

CUE = {"bulletin": 1.0, "endorsement": 0.7, "form": 0.0,
       "guideline": -0.6, "advisory": -0.6, "note": -0.6, "summary": -0.6}
PROFILES = {  # chance of ignoring the "context only" instruction; chance of following a label
    "sim-large": {"disobey": 0.003, "follow": 1.00},
    "sim-mid":   {"disobey": 0.012, "follow": 0.98},
    "sim-small": {"disobey": 0.035, "follow": 0.93},
}


def draw(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 2**32


def cite(model, did, packet, spans, sims, governs=None, topic=None, state=None):
    """packet: sids in rank order; sims: bm25 scores; governs: the labelled sid;
    state: the instruction's 'apply the bulletin for the policy's state' (Chapter 14)."""
    p = PROFILES[model]
    if governs is not None and draw(model, did, "follow") < p["follow"]:
        return governs
    top = max(sims.values()) or 1.0
    on_topic = [x for x in packet if spans[x].topic == topic] if topic else packet
    packet = on_topic or packet             # a reader can tell what a span is about
    lead = max(packet, key=lambda x: sims[x]) if packet else None
    pd = p["disobey"] * (3 if lead and CUE[spans[lead].kind] < 0 else 1)
    def score(sid):
        k = spans[sid].kind
        cue = CUE[k]
        if cue < 0 and draw(model, did, "disobey") < pd:
            cue = 1.2                       # treats context as if it governed
        if state and k == "bulletin" and spans[sid].jurisdiction != state:
            cue = -0.5                      # 'the policy state's bulletin applies'
        return sims[sid] / top + cue + 0.05 * draw(model, did, sid)
    return max(packet, key=score) if packet else None
