"""Tokenising and analysing text for the inverted index."""
import re

TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
# A short stop list. It deliberately keeps "not", "no" and "without":
# many stock English lists drop them, and Section 12.8 shows the cost.
STOP = set("a an and are as at be by for from in into is it its of on or "
           "such that the their then there these this to was will with "
           "we you your our us".split())


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; hyphenated identifiers are also split."""
    out = []
    for tok in TOKEN.findall(text.lower()):
        out.append(tok)
        if "-" in tok:
            out.extend(p for p in tok.split("-") if p)
    return out


def stem(tok: str) -> str:
    """A crude suffix stripper: enough to join 'pipes' and 'pipe'."""
    if tok.isdigit() or len(tok) <= 4:
        return tok
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def analyse(text: str, stop: set[str] = STOP) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in stop]


# Adjuster shorthand -> document vocabulary (Chapter 12's glossary; Chapter 18 reuses it).
GLOSSARY = {"bsmt": "basement", "dol": "date of loss", "wks": "weeks", "mos": "months",
            "rc": "replacement cost", "pp": "personal property", "pol": "proof of loss",
            "wbu": "water back-up sump discharge endorsement",
            "o&l": "ordinance or law endorsement"}


def rewrite(query: str, glossary: dict = GLOSSARY) -> str:
    """Replace shorthand tokens with the words the documents use."""
    return " ".join(glossary.get(w.lower().strip(",.?;:"), w) for w in query.split())
