"""Chapter 41: open-code notes as an adjuster writes them, and two codebooks.

The note templates are stand-ins written for the lab; each states the first thing
that went wrong in a determination, in an adjuster's words.
"""
import random
from contextlab.erroranalysis import Category, Codebook

TEMPLATES = {
    "second_peril": [
        "Wind took the shingles and rain came in. Determination covers the roof, says nothing about the ceiling and contents.",
        "Two causes on this loss, hail then interior water. Answered the hail part only.",
        "No clause for the water damage in the packet, just the windstorm one. Water half of the claim not addressed.",
        "Tree came down on the garage and the freezer spoiled in the outage. Answer only deals with the garage.",
    ],
    "deductible_misapplied": [
        "Used the flat deductible; {state} wind/hail on this {form} is the 2% one.",
        "Deductible applied twice across the house and the detached garage.",
        "Percentage deductible taken off the loss instead of off the dwelling limit.",
    ],
    "wrong_limit_line": [
        "Limit read from the other scheduled item on the declarations.",
        "Quoted the Coverage C limit; the question was the special limit for jewellery.",
    ],
    "condition_missed": [
        "Did not check the notice condition; reported 14 months after the loss.",
        "Mitigation condition ignored, pipe left leaking for weeks.",
    ],
    "extraction": [
        "Read the date of loss wrong from the scanned first notice.",
        "Policy number OCR'd wrong, pulled the neighbour's declarations.",
    ],
    "should_withhold": [
        "Answered without the endorsement schedule in the file; should have sent it back to me.",
    ],
    "superseded_edition": ["Cited the 2019 edition of the {form}; this policy is on the 2024 edition."],
    "precedence_missed": ["{state} bulletin on matching overrides the form here and the answer went with the form."],
    "not_a_failure": [
        "Determination was right. I overrode because the insured sent new photos after it ran.",
        "Fine as written. Overrode to keep the agent happy, my call.",
    ],
}


def note_for(cat, d):
    t = TEMPLATES[cat][int(d["v"] * 1000) % len(TEMPLATES[cat])]
    return t.format(state=d.get("state", "WI"), form=d.get("form", "HO-3"))


_V1 = [  # drafted from the first 40 notes, with Chapter 29's classes as lenses
    Category("missing_evidence", "governing text absent from the packet",
             ("no clause", "not in the packet", "without the"), 3, "retrieval"),
    Category("deductible_misapplied", "deductible type or base wrong", ("deductible",), 3, "determination"),
    Category("wrong_limit_line", "limit taken from the wrong line", ("limit",), 3, "extraction"),
    Category("condition_missed", "a policy condition not checked", ("condition",), 2, "determination"),
    Category("extraction", "a fact misread from a document", ("scanned", "ocr"), 1, "parsing"),
    Category("superseded_edition", "an old edition cited", ("edition",), 3, "retrieval"),
    Category("precedence_missed", "binding sources, wrong one followed", ("bulletin",), 2, "assembly"),
]
_V2_NEW = [
    Category("second_peril", "the claim raises two coverage questions and the answer settles one; "
             "the clause for the other is in the corpus and the index",
             ("only", "says nothing about", "not addressed", "two causes"), 3, "question formation"),
    Category("not_a_failure", "the determination was right; the override had another reason",
             ("was right", "fine as written"), 0, "none"),
    Category("should_withhold", "answered when a required document was missing",
             ("should have sent it back",), 2, "assembly"),
]
CODEBOOK_V1 = Codebook("v1", list(_V1))
CODEBOOK_V2 = Codebook("v2", _V2_NEW + [c for c in _V1 if c.name != "missing_evidence"] + [_V1[0]])

# How the second coder (an evaluation engineer) mislabels, before and after the
# alignment session that wrote the v2 definitions: truth -> (confused with, rate)
CONFUSE = {
    "before": {"second_peril": ("missing_evidence", .50), "not_a_failure": ("condition_missed", .40),
               "deductible_misapplied": ("wrong_limit_line", .25), "should_withhold": ("missing_evidence", .50)},
    "after": {"second_peril": ("missing_evidence", .05), "not_a_failure": ("condition_missed", .10),
              "deductible_misapplied": ("wrong_limit_line", .05), "should_withhold": ("missing_evidence", .10)},
}


def coder_label(truth, who, phase, rng):
    if who == "adjuster":
        return truth if rng.random() > .03 else "condition_missed"
    other, p = CONFUSE[phase].get(truth, (None, 0))
    if other and rng.random() < p:
        return other
    return truth if rng.random() > .04 else "extraction"


def code_pair(truths, phase, seed=7):
    rng = random.Random(seed)
    return ([coder_label(t, "adjuster", phase, rng) for t in truths],
            [coder_label(t, "engineer", phase, rng) for t in truths])
