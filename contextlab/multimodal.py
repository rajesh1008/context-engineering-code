"""contextlab.multimodal (v4.3): photos, diagrams and voice notes as spans.

An image span carries observations, never conclusions. Each observation has
a kind from a closed vocabulary, a region in the original image's pixels and
a confidence. Provenance (hash, uploader, capture time and place) travels in
the header because the model never sees an image's metadata. An audio span
keeps per-segment confidence so that a decisive sentence heard badly can be
sent to a person to listen to, rather than trusted or dropped.
Chapter 19 of The Context Engineering Handbook for LLMs and AI Agents.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

# What a vision model may say about a claim photo: things a person standing
# in the room could point at. Cause, coverage, age and identity are absent.
OBSERVE = {"water_at", "residue_at", "debris_at", "tide_line", "pipe_split",
           "spray_pattern", "mould", "impact_marks", "curled_shingles",
           "granule_loss", "glass_cracked", "vehicle_interior", "no_water"}
FORBIDDEN = {"cause", "coverage", "age", "subject", "flood", "storm_damage"}
FEATURES = {"floor_drain", "window_well", "door", "supply_line", "wall",
            "ceiling", "roof_slope", "gutter", "window", "none"}


@dataclass(frozen=True)
class Region:
    """A box in the ORIGINAL image's pixel coordinates (x0, y0, x1, y1)."""
    x0: int
    y0: int
    x1: int
    y1: int

    def scaled(self, s: float) -> tuple[int, int, int, int]:
        return tuple(round(v * s) for v in (self.x0, self.y0, self.x1, self.y1))

    def size(self) -> tuple[int, int]:
        return self.x1 - self.x0, self.y1 - self.y0


@dataclass(frozen=True)
class Observation:
    kind: str
    target: str
    region: Region
    conf: float
    value: str = ""          # e.g. a tide-line height, "30cm"


def check(obs: Observation) -> str | None:
    """Return a reason to refuse the observation, or None if it may enter."""
    if obs.kind in FORBIDDEN or obs.kind not in OBSERVE:
        return f"kind '{obs.kind}' is not an observation"
    if obs.target not in FEATURES:
        return f"target '{obs.target}' is not in the feature list"
    if obs.conf < 0.6:
        return "below confidence floor 0.6"
    return None


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()[:16]


@dataclass
class ImageSpan:
    media_id: str
    digest: str
    claim_id: str
    uploader: str
    captured_at: str | None      # from capture metadata, not from the model
    gps: tuple[float, float] | None
    stated_subject: str          # what the uploader said the photo shows
    model: str                   # observation model and prompt version
    observations: list[Observation] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def header(self) -> str:
        where = "none" if self.gps is None else f"{self.gps[0]:.4f},{self.gps[1]:.4f}"
        return (f"[IMAGE {self.media_id} {self.digest} claim={self.claim_id} "
                f"by={self.uploader} captured={self.captured_at or 'none'} "
                f"gps={where} subject='{self.stated_subject}' via={self.model}"
                + (f" FLAGS={','.join(self.flags)}" if self.flags else "") + "]")

    def text(self) -> str:
        rows = [f"- {o.kind}({o.target}{', ' + o.value if o.value else ''}) "
                f"region={o.region.x0},{o.region.y0},{o.region.x1},{o.region.y1} "
                f"conf={o.conf:.2f}" for o in self.observations]
        return self.header() + "\n" + "\n".join(rows or ["- (no observations)"])


def admit(span: ImageSpan, raw: list[Observation]) -> list[str]:
    """Keep allowed observations; return the refusal reasons for the rest."""
    refused = []
    for o in raw:
        why = check(o)
        (refused.append(why) if why else span.observations.append(o))
    return refused


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres (haversine)."""
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def provenance_flags(span: ImageSpan, insured: tuple[float, float],
                     loss_date: str, max_km: float = 1.0) -> list[str]:
    flags = []
    if span.gps is None or span.captured_at is None:
        flags.append("no_capture_metadata")
    elif km(span.gps, insured) > max_km:
        flags.append(f"captured_{km(span.gps, insured):.1f}km_from_risk")
    if span.captured_at and span.captured_at[:10] < loss_date:
        flags.append("captured_before_loss")
    if any(o.kind == "vehicle_interior" for o in span.observations):
        flags.append("vehicle_interior")
    span.flags = flags
    return flags


# Image token budgets, after one provider's published rule (read 1 Nov 2026):
# 28x28 patches, ceil(w/28)*ceil(h/28) tokens, long-edge and token caps by tier.
TIERS = {"standard": (1568, 1568), "high": (2576, 4784)}


def image_tokens(w: int, h: int, tier: str = "standard") -> tuple[int, int, int]:
    """Return (tokens, sent_w, sent_h) after the provider-side downscale."""
    edge, cap = TIERS[tier]
    s = min(1.0, edge / max(w, h))
    while True:
        sw, sh = max(1, math.floor(w * s)), max(1, math.floor(h * s))
        t = math.ceil(sw / 28) * math.ceil(sh / 28)
        if t <= cap:
            return t, sw, sh
        s *= 0.99


def crop_tokens(region: Region, pad: float = 0.15, tier: str = "standard") -> int:
    w, h = region.size()
    return image_tokens(round(w * (1 + 2 * pad)), round(h * (1 + 2 * pad)), tier)[0]


# Voice notes: segments keep the recogniser's confidence and audio offsets.
NEGATION = re.compile(r"\b(no|not|never|none|nothing|isn't|wasn't|aren't|"
                      r"weren't|didn't|doesn't|without)\b", re.I)


@dataclass(frozen=True)
class Segment:
    t0: float
    t1: float
    text: str
    conf: float              # mean word confidence for the segment
    snr_db: float            # signal-to-noise ratio estimate


def decisive(seg: Segment, terms: set[str]) -> bool:
    """True when the sentence names a feature the determination turns on."""
    flat = seg.text.lower()
    return any(t.replace("_", " ") in flat for t in terms)


def guard(seg: Segment, terms: set[str], min_conf: float = 0.85,
          min_snr: float = 15.0) -> str:
    """'use', or 'listen' when a decisive sentence was heard badly."""
    if not decisive(seg, terms):
        return "use"
    if seg.conf < min_conf or seg.snr_db < min_snr:
        return "listen"
    return "use"


@dataclass
class AudioSpan:
    media_id: str
    digest: str
    claim_id: str
    speaker: str
    recorded_at: str
    asr: str                  # recogniser and version
    segments: list[Segment]

    def text(self, terms: set[str]) -> str:
        head = (f"[AUDIO {self.media_id} {self.digest} claim={self.claim_id} "
                f"by={self.speaker} recorded={self.recorded_at} via={self.asr}]")
        rows = []
        for s in self.segments:
            tag = guard(s, terms)
            body = s.text if tag == "use" else "(withheld: listen required)"
            rows.append(f"- {s.t0:05.1f}-{s.t1:05.1f}s conf={s.conf:.2f} {body}")
        return head + "\n" + "\n".join(rows)


def wer(ref: str, hyp: str) -> float:
    """Word error rate by edit distance."""
    r, h = ref.lower().split(), hyp.lower().split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))
