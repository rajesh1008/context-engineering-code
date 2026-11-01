"""Chapter 25 task set: change tasks drawn from the snapshot, locate questions by hand."""

# Locate questions: plain words, no identifiers, each with the symbol that answers it.
LOCATE = [
    ("Where do we combine two ranked lists using reciprocal ranks?", "contextlab.retrieval.fusion.rrf"),
    ("Which function decides whether an image observation may enter the packet?", "contextlab.multimodal.check"),
    ("Where is the delay between a commit and the assistant seeing it added up?", "contextlab.freshness.time_to_visible"),
    ("Where do we find the largest padded length before accuracy falls off?", "contextlab.diagnostics.knee"),
    ("How is the memory store restored to how it stood before a date?", "contextlab.memguard.rollback"),
    ("Where is the policy number's check digit verified?", "contextlab.docgate.check_digit_ok"),
    ("Where are printed money amounts turned into numbers without losing the cents?", "contextlab.docgate.normalise_amount"),
    ("Where is the population stability index between two score samples computed?", "contextlab.docgate.psi"),
    ("Which function removes one data subject from memory and checkpoints?", "contextlab.statestore.erase"),
    ("Where are page headers and footers that recur on most pages detected?", "contextlab.parsing.repeated_lines"),
    ("Where is word error rate of a transcript measured by edit distance?", "contextlab.multimodal.wer"),
    ("Where does a question get sent to the loop only if it needs more than one hop?", "contextlab.agentic.route"),
    ("How are vectors truncated and stored as int8 or binary to save space?", "contextlab.embeddings.quantise"),
    ("Where do we check that no hit came back from another tenant?", "contextlab.vectorstore.check"),
    ("Which function walks the graph to gather the governing spans for one determination?", "contextlab.graph.compose"),
    ("Where are rows rendered as comma-separated text and refused if they do not round-trip?", "contextlab.structured.render"),
]


def change_text(name: str) -> str:
    return f"Change the signature of {name} and update every place that uses it."
