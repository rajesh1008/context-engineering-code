"""Listing 36.1 and 36.2 in runnable form (needs dspy; see .venv-ch36)."""
import dspy


class Determine(dspy.Signature):
    """Decide coverage for one claim from the policy packet."""
    claim: str = dspy.InputField(desc="facts of loss, date of loss, state")
    packet: str = dspy.InputField(desc="numbered spans with form and edition")
    outcome: str = dspy.OutputField(desc="covered, excluded or refer")
    cited_span: str = dspy.OutputField(desc="id of the span that governs")
    rationale: str = dspy.OutputField(desc="sentences, each supported by that span")


def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    if pred.cited_span != example.gold_span:
        return dspy.Prediction(score=0.0, feedback=f"cited {pred.cited_span}, "
                               f"governing span is {example.gold_span}")
    terms = example.support[example.gold_span]
    unsupported = [s for s in pred.rationale.split(". ")
                   if not any(t in s.lower() for t in terms)]
    score = 1.0 if not unsupported else 0.0
    return dspy.Prediction(score=score, feedback=f"span {pred.cited_span}; "
                           f"unsupported: {unsupported or 'none'}")
