"""Chapter 31 stand-in: Chapter 28's 300 determinations, given roles, read by a reader.

The world is Chapter 28's (same seed, same packets, same retrieval order). What is
new is the reader: a deterministic stand-in for the drafting model whose mistakes
depend on how the packet was assembled. Every multiplier below is a parameter of
the stand-in, chosen to point the way published studies point, not a measurement.
The three structured syntaxes (JSON, Markdown, XML) are treated identically.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "ch28")]
import lakeshore28 as L                                    # noqa: E402
from contextlab.assembler import Span, count_tokens                      # noqa: E402
from contextlab.citations import Entry, NOSOURCE, verify, strip   # noqa: E402

INSTRUCTIONS = ("You are the claims assistant for Lakeshore Mutual Insurance. Answer the adjuster's "
                "coverage question from the documents provided. After every sentence that states a "
                "fact or a policy term, cite the id of the span that supports it.")
QUESTIONS = {
    "backup_endorsed": "Is the basement water damage from the sump pump failure covered, and what limit and deductible apply?",
    "backup_plain": "Is the basement water damage from the floor drain back-up covered?",
    "frozen_pipe": "Is the burst pipe, and the repair to the kitchen wall, covered?",
    "mould_hidden": "Is the mould inside the bathroom wall covered?",
    "mould_seepage": "Is the mould behind the shower tiles covered?",
    "hail_interior": "Is the rain damage in the kitchen after the hail covered?",
}
# Reader parameters (stand-in, stated in the chapter's Note box).
BASE = {"neighbour": 0.05, "missing": 0.03, "addition": 0.03, "number": 0.015}
POSITION = (1.0, 1.6, 1.2)          # missing-pointer multiplier: first, middle, last third
RELATION = {"tagged": 0.3, "adjacent": 0.6, "near": 1.0, "far": 1.5}   # polarity, back-up claims
COPY = 0.0015                       # chance per token of an identifier that the copy is corrupted
SEEDS = 5                           # reader runs per determination
SPLIT = ", unless "                 # where the re-chunker cuts HO-3 page 7 span 1


def ho3_governing_pages(ctype):
    return {s[1] for _, _, sup in L.TYPES[ctype]["sentences"] for s in sup if s[0] == "HO-3"}


def roles(det, packet=None):
    """Roles from the claim type's governing clauses (the applicability map)."""
    packet = packet or det.packet
    pages = {L.ho3_page(det.ho3, p) for p in ho3_governing_pages(det.ctype)}
    excl = next((e for e in packet if e.doc == "HO-3" and e.page == L.ho3_page(det.ho3, 7)
                 and e.n == 1), None)
    out = []
    for e in packet:
        if e.edition in ("record", "2025-06"):
            out.append(Span(e, "facts", topic="0"))
        elif e.doc == "HO 04 95":
            out.append(Span(e, "modifier", modifies=excl, topic="7"))
        elif e.doc == "HO-3" and e.page in pages:
            out.append(Span(e, "governing", topic=str(e.page)))
        else:
            out.append(Span(e, "context", topic=str(e.page)))
    return out


def gold(det):
    """Chapter 28's sentences before any writer mistake: (kind, text, support)."""
    T = L.TYPES[det.ctype]
    lim, ded = (10000, 1000) if det.e495 == "2024-06" else (5000, 500)
    fmt = dict(lim=lim, ded=ded, breakdown=", even if caused by mechanical breakdown of the sump pump"
               if det.e495 == "2024-06" else "")
    claim = [e for e in det.packet if e.edition == "record"]
    decs = next(e for e in det.packet if e.edition == "2025-06")
    return [(k, t.format(**fmt), [L._resolve(s, det.ho3, det.e495, claim, decs) for s in sup])
            for k, t, sup in T["sentences"]], fmt


def rechunk(packet):
    """The re-chunk: HO-3 page 7 span 1 becomes two spans; both keep number 1."""
    out = []
    for e in packet:
        if e.doc == "HO-3" and SPLIT in e.text:
            a, b = e.text.split(SPLIT)
            out += [Entry(e.doc, e.edition, e.page, e.n, a + "."),
                    Entry(e.doc, e.edition, e.page, e.n, "Unless " + b)]
        else:
            out.append(e)
    return out


def split_support(sup, packet):
    """After a re-chunk, a sentence that rested on the whole clause rests on both halves."""
    out = []
    for e in sup:
        halves = [x for x in packet if x.doc == e.doc and x.edition == e.edition and x.page == e.page
                  and x.n == e.n and x.text != e.text and x.text.rstrip(".") in e.text + "Unless"]
        out += halves if (e not in packet and halves) else [e]
    return out


def relation(asm):
    """How the back-up endorsement sits relative to the exclusion it modifies."""
    ev = [s for s in asm.order if s.role != "facts"]
    ex = [i for i, s in enumerate(ev) if s.entry.doc == "HO-3" and s.entry.page in (7, 8) and s.entry.n == 1
          and "backs up" in s.entry.text]
    en = [i for i, s in enumerate(ev) if s.entry.doc == "HO 04 95" and s.entry.n == 2 and s.entry.page == 1]
    if not ex or not en:
        return "far"
    d = min(abs(a - b) for a in ex for b in en)
    if 'modifies="' in asm.text and d <= 2:
        return "tagged"
    return "adjacent" if d <= 2 else "near" if d <= 4 else "far"


def pointer(e, asm):
    """How the reader writes a pointer to entry e in this assembly."""
    ids = [i for i, s in zip(asm.ids, asm.order) if s.entry == e]
    if not ids or ids[0] is None:
        return f"[{e.doc} ({e.edition}), page {e.page}]"
    i = ids[0]
    return f"[{i}]" if asm.ids[0] == "S1" else f"[[{i}]]"


def corrupt(tag):
    """A copying slip: one character of the identifier changes."""
    if tag.startswith("[S"):
        return f"[S{int(tag[2:-1]) + 1}]"
    return tag[:-3] + ("0" if tag[-3] != "0" else "1") + tag[-2:]


def read(det, asm, seed, packet=None):
    """One reading: sentences with pointers, gold support and the mistake made."""
    packet = packet or det.packet
    sents, fmt = gold(det)
    rel = relation(asm)
    out = []
    for k, (kind, text, sup) in enumerate(sents):
        r = L.rng(f"ch31/{det.det_id}/{seed}/{k}")
        u = {m: r.random() for m in ("content", "missing", "neighbour", "pick", "c0", "c1", "c2")}
        uc = {m: r.random() for m in L.CONCLUSION_RATES}
        sup = split_support(sup, packet)
        if kind == "note":
            out.append(dict(kind=kind, text=f"{text} {NOSOURCE}", claim=text, support=[], mistake=None,
                            intended=[]))
            continue
        ptrs, mistake = list(sup), None
        if kind == "conclusion":
            for m, p in L.CONCLUSION_RATES.items():
                if m == "polarity" and det.ctype == "backup_endorsed":
                    p *= RELATION[rel]
                if det.ctype in L.WRONG[m] and uc[m] < p:
                    t, wp, ok = L.WRONG[m][det.ctype]
                    text, mistake = t.format(**fmt), m
                    if wp:
                        ptrs = [L._resolve(x, det.ho3, None, None, None) for x in wp]
                        ptrs = split_support(ptrs, packet)
                    sup = list(ptrs) if ok else []
                    break
        elif u["content"] < BASE["addition"]:
            text, sup, mistake = text.rstrip(".") + L.ADDITION + ".", [], "addition"
        elif u["content"] < BASE["addition"] + BASE["number"]:
            for a, b in {"$10,000": "$5,000", "$5,000": "$10,000", "$1,000": "$500", "$500": "$1,000",
                         "60 days": "30 days", "14 or more": "30 or more", "5 days": "15 days"}.items():
                if a in text:
                    text, sup, mistake = text.replace(a, b, 1), [], "number"
                    break
        if mistake is None and ptrs:
            third = min(2, int(3 * asm.position(ptrs[0]) - 1e-9)) if len(asm.order) > 1 else 0
            if u["missing"] < BASE["missing"] * POSITION[max(0, third)]:
                ptrs, mistake = [], "missing"
            elif u["neighbour"] < BASE["neighbour"]:
                t = ptrs[0]
                cands = [e for e in packet if e.doc == t.doc and e.page == t.page and e not in sup] or \
                        [e for e in packet if e not in sup and e.edition != "record"]
                ptrs, mistake = [cands[int(u["pick"] * len(cands))]], "neighbour"
        tags = list(dict.fromkeys(pointer(e, asm) for e in ptrs))
        for j, t in enumerate(tags[:3]):
            if "page" not in t and u[f"c{j}"] < COPY * count_tokens(t):
                tags[j], mistake = corrupt(t), mistake or "copy"
        tags = " ".join(tags)
        out.append(dict(kind=kind, text=f"{text} {tags}".strip(), claim=text, support=sup,
                        mistake=mistake, intended=ptrs))
    return out


def faithful(sent, asm):
    """Supported by the packet as the reader saw it (shown text, possibly trimmed)."""
    if not sent["support"]:
        return False
    shown = " ".join(asm.shown[e].text for e in sent["support"] if e in asm.shown)
    return verify(sent["claim"], shown) >= 0.6
