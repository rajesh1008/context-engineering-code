"""Chapter 41's fix for the second-peril class: one coverage question per peril.

The released pipeline built one retrieval question from the loss description, and
the description leads with the first peril named. The fix forms a question for
every peril code on the first notice of loss, gives each its own share of the
packet budget, and refuses to finish until every peril has an answer.
"""
PERIL_TEXT = {"WIND": "windstorm damage to the roof", "HAIL": "hail damage",
              "WATER": "water entering through storm-created openings",
              "TREE": "a falling tree", "POWER": "spoilage after a power outage"}


def questions_for(claim, budget=1600):
    perils = claim["perils"] or ["UNSPECIFIED"]
    share = budget // len(perils)
    return [dict(claim=claim["id"], peril=p, budget=share,
                 text=f"Is {PERIL_TEXT.get(p, p.lower())} covered under {claim['form']} in {claim['state']}?")
            for p in perils]


def complete(claim, answers):
    """Every peril on the claim has a determination; otherwise hand off."""
    missing = [p for p in claim["perils"] if p not in {a["peril"] for a in answers}]
    return (not missing), missing
