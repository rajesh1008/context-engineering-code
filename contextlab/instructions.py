"""Versioned runtime instructions for the claims assistant (Chapters 6-8).

An Instructions object is rendered from a single source: the committed
instructions_v*.md files are produced by render(), never edited by hand, and the
fingerprint (sha256 of the rendered body, version line excluded) is what every
trace records. Chapter 6 builds the version history; Chapters 7 and 8 use V22.
"""
from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field, replace

BASE = ("You are the claims assistant for Lakeshore Mutual Insurance. Answer the adjuster's "
        "coverage question from the documents provided and name the governing provision.")

RULE_TEXT = {
    "precedence": ("When documents conflict, a state bulletin in force on the date of loss overrides "
                   "an endorsement, and an endorsement overrides the policy form."),
    "applicability": ("Use only documents that apply to this policy: its form edition, its "
                      "endorsements, and bulletins for its state. Ignore underwriting notes."),
    "cite_exact": "Cite the exact document version and page, for example HO-3 (2024-06), page 7.",
    "quote_only": ("State only what the cited provision says. Do not add deductibles, limits, or "
                   "conditions that the provision does not state."),
}

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """The labs' tokenizer: words and punctuation marks, one token each."""
    return len(_TOKEN.findall(text))


@dataclass(frozen=True)
class Span:
    span_id: str
    text: str
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


@dataclass(frozen=True)
class Skill:
    name: str
    topics: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Example:
    example_id: str            # "A", "B", "C"
    text: str
    source: str                # the reviewed determination it was taken from
    state: str
    topic: str
    loss: str                  # ISO date of loss in the example's facts
    cites: str                 # doc_id the example's answer cites


@dataclass(frozen=True)
class Instructions:
    version: str
    rules: tuple[str, ...] = ()
    examples: tuple[Example, ...] = ()
    skills: tuple[Skill, ...] = ()
    disclosure: str = "progressive"
    base: str = BASE
    rule_text: dict = field(default_factory=lambda: dict(RULE_TEXT), compare=False, hash=False)

    def loaded_skills(self, det=None) -> list[Skill]:
        if self.disclosure == "all" or det is None:
            return list(self.skills)
        return [s for s in self.skills if det.topic in s.topics]

    def system_text(self) -> str:
        return "\n".join([self.base] + [f"rule {r}: {self.rule_text[r]}" for r in self.rules])

    def _span(self, sid: str, text: str, meta: dict) -> Span:
        return Span(f"INSTR-{self.version}-{sid}", text, {"kind": "instruction", **meta})

    def spans(self, det=None) -> list[Span]:
        out = [self._span("rules", self.system_text(), {"rules": list(self.rules), "tier": 0})]
        out += [self._span(f"skill-{s.name}", s.text, {"skill": s.name, "tier": 1, "topics": s.topics})
                for s in self.loaded_skills(det)]
        out += [self._span(f"ex-{e.example_id}", e.text,
                           {"tier": 1, "example": {"id": e.example_id, "source": e.source,
                                                   "state": e.state, "topic": e.topic,
                                                   "loss": e.loss, "cites": e.cites}})
                for e in self.examples]
        return out

    def render(self) -> str:
        parts = [f"# instructions {self.version}", self.base]
        parts += [f"rule {r}: {self.rule_text[r]}" for r in self.rules]
        parts += [f"## skill {s.name} ({', '.join(s.topics)})\n{s.text}" for s in self.skills]
        parts += [f"## example\n{e.text}" for e in self.examples]
        return "\n".join(parts)

    def fingerprint(self) -> str:
        body = self.render().split("\n", 1)[1]   # the version line is not content
        return hashlib.sha256(body.encode()).hexdigest()[:12]

    def diff(self, other: "Instructions") -> list[str]:
        a = self.render().splitlines()[1:]
        b = other.render().splitlines()[1:]
        return [ln for ln in difflib.unified_diff(a, b, lineterm="", n=0)
                if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]

    def bump(self, version: str, **changes) -> "Instructions":
        return replace(self, version=version, **changes)


SKILLS = (
    Skill("roof", ("roof_matching", "roof_settlement", "hail_cosmetic", "wind_hail", "ice_dam"),
          "Roof claims: check whether the policy settles roofs at actual cash value by an endorsement, "
          "and whether a state bulletin requires matching of undamaged roofing or limits cosmetic hail "
          "exclusions. Name the bulletin when one is in force."),
    Skill("water", ("water_backup", "frozen_pipes", "flood", "mold"),
          "Water claims: separate backup of sewers and drains, which needs the water backup endorsement, "
          "from flood, which the forms exclude. For frozen pipes, check heat maintenance and any bulletin "
          "on utility interruptions. Mold has sublimits."),
    Skill("auto", ("deer_swerve", "deer_collision", "rental_reimbursement", "rideshare"),
          "Auto claims: contact with an animal is other than collision; leaving the road to avoid one is "
          "collision unless a state bulletin says otherwise. Rental and rideshare questions turn on the "
          "endorsements the policy carries."),
    Skill("marine", ("navigation_limits", "hull_damage"),
          "Boat claims: check the navigation warranty in the declarations and any endorsement that "
          "extends the navigation period or waters."),
    Skill("commercial", ("employee_dishonesty", "spoilage", "equipment_breakdown"),
          "Businessowners claims: employee dishonesty and spoilage are excluded by the form unless the "
          "policy carries the endorsement that adds them back. Check the declarations for it."),
)

EX_A = Example(
    "A",
    "Example A. Question: Pipes froze and burst in the upstairs bathroom while the owners were away for "
    "a week. Is the freeze damage covered? Facts: form HO-3 (2024-06); endorsements HO 04 90 (2022-01); "
    "state IL; date of loss 2024-11-25. Answer: Governing provision: HO-3 (2024-06), Section I Perils "
    "Insured Against 12, page 2.",
    source="DET-0236", state="IL", topic="frozen_pipes", loss="2024-11-25", cites="HO-3")
EX_B = Example(
    "B",
    "Example B. Question: Water came up through the basement floor drain during a storm and soaked the "
    "finished basement. Is the backup covered? Facts: form HO-3 (2024-06); endorsements HO 04 95 "
    "(2024-06); state MN; date of loss 2025-06-19. Answer: Governing provision: HO 04 95 (2024-06), "
    "Coverage, page 1.",
    source="DET-0288", state="MN", topic="water_backup", loss="2025-06-19", cites="HO 04 95")
EX_C = Example(
    "C",
    "Example C. Question: Insured wants all siding replaced because the new panels will not match. "
    "What do we owe for matching? Facts: form HO-3 (2024-06); endorsements HO 04 98 (2024-06); state WI; "
    "date of loss 2025-07-11. Answer: Governing provision: WI-OCI-2025-03 (2025-03), Requirements, page 1.",
    source="DET-0307", state="WI", topic="roof_matching", loss="2025-07-11", cites="WI-OCI-2025-03")

V22 = Instructions("v2.2", rules=("precedence", "applicability", "cite_exact", "quote_only"),
                   examples=(EX_A, EX_B, EX_C), skills=SKILLS)
V22_NO_A = V22.bump("v2.2-noA", examples=(EX_B, EX_C))
