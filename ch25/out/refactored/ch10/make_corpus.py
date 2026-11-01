"""Build the Chapter 10 fixture corpus: 300 PDF pages with a ground truth.

Run: python 04-code/ch10/make_corpus.py   (writes 04-code/ch10/corpus/)
Deterministic. The PDFs are drawn with reportlab so that each one has the
layout feature a real corpus throws at a parser: two columns written row by
row, running headers and footers, ruled tables written column by column, a
table that breaks across a page, and image-only scans with no text layer.
"""
from __future__ import annotations
import io, json, os, random, re
from dataclasses import dataclass, field
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import clauses as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus")
W, H = 612, 792
TARGET_PAGES = 300
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


@dataclass
class Doc:
    doc_id: str          # file stem
    label: str           # form or bulletin number as cited
    edition: str
    kind: str            # form, endorsement, bulletin, schedule, manual, guideline
    layout: str          # single, two_col, two_col_rows
    header: str | None
    footer: str          # template with {page} and {pages}
    scanned: bool = False
    state: str | None = None
    blocks: list = field(default_factory=list)


def para(doc: Doc, path: list[str], text: str, cid: str | None = None, section: str | None = None):
    cid = cid or f"{doc.label}@{doc.edition}#{slug(path[-1])}"
    doc.blocks.append(("p", cid, text, list(path), section or " ".join(path)))


def head(doc: Doc, text: str, level: int = 1):
    doc.blocks.append(("h", level, text))


# ------------------------------------------------------------------ layout
def wrap(text, width, font="Helvetica", size=9.5):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if stringWidth(t, font, size) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def layout(doc: Doc):
    """Place every block. Returns pages: list of list of placed items."""
    two = doc.layout != "single"
    ncol, gap, left, right = (2 if two else 1), 18, 54, W - 54
    colw = (right - left - gap * (ncol - 1)) / ncol
    top, bottom, lead = H - 72, 66, 12.0
    pages, page, col, y = [], [], 0, top

    def newcol():
        nonlocal col, y, page
        col += 1; y = top
        if col >= ncol:
            pages.append(page); page = []; col = 0

    def place(text, font, size, ref, kind, x_off=0):
        nonlocal y
        if y - lead < bottom:
            newcol()
        x = left + col * (colw + gap) + x_off
        page.append(dict(x=x, y=y, text=text, font=font, size=size, ref=ref, kind=kind, col=col))
        y -= lead

    for b in doc.blocks:
        if b[0] == "h":
            if y - 3 * lead < bottom:
                newcol()
            y -= 4
            for ln in wrap(b[2], colw, "Helvetica-Bold", 10.5):
                place(ln, "Helvetica-Bold", 10.5, None, f"h{b[1]}")
        elif b[0] == "p":
            for ln in wrap(b[2], colw):
                place(ln, "Helvetica", 9.5, b[1], "p")
            y -= 5
        elif b[0] == "colbreak":
            newcol()
        elif b[0] == "space":
            y -= b[1]
            if y < bottom:
                newcol()
        elif b[0] == "pagebreak":
            if page or col:
                pages.append(page); page, col = [], 0
            y = top
        elif b[0] == "table":
            _, tid, hdr, rows, order, split, path, section = b
            cw = (right - left) / len(hdr)
            def put_rows(rr, with_head):
                nonlocal y
                if y - lead * (len(rr) + 2) < bottom:
                    if page:
                        pages.append(page)
                    reset_page()
                tab = dict(kind="table", tid=tid, x=left, y=y, cw=cw, order=order,
                           cells=[], ref=tid, path=path, section=section)
                rws = ([hdr] if with_head else []) + list(rr)
                for i, r in enumerate(rws):
                    for j, c in enumerate(r):
                        tab["cells"].append(dict(i=i, j=j, text=c, head=(with_head and i == 0)))
                tab["nrows"], tab["ncols"] = len(rws), len(hdr)
                page.append(tab)
                y -= lead * 1.6 * len(rws) + 8
            def reset_page():
                nonlocal page, y, col
                page, col, y = [], 0, top
            if split:
                put_rows(rows[:split], True)
                pages.append(page); reset_page()
                put_rows(rows[split:], False)
            else:
                put_rows(rows, True)
    if page:
        pages.append(page)
    return pages


# ------------------------------------------------------------------ drawing
def draw_pdf(doc: Doc, pages, path):
    c = canvas.Canvas(path, pagesize=(W, H), invariant=1)
    c.setTitle(f"{doc.label} ({doc.edition})")
    n = len(pages)
    for i, items in enumerate(pages, 1):
        foot = doc.footer.format(page=i, pages=n)
        if doc.scanned:
            img = render_scan(doc, items, foot, i)
            c.drawImage(ImageReader(img), 0, 0, W, H)
            c.showPage(); continue
        c.setFont("Helvetica", 7.5); c.drawString(54, 36, foot)          # footer first
        texts = [t for t in items if t["kind"] != "table"]
        if doc.layout == "two_col_rows":                                   # row by row
            rows = {}
            for t in texts:
                rows.setdefault(round(t["y"]), []).append(t)
            texts = [t for yk in sorted(rows, reverse=True) for t in sorted(rows[yk], key=lambda t: t["x"])]
        for t in texts:
            c.setFont(t["font"], t["size"]); c.drawString(t["x"], t["y"], t["text"])
        for tab in [t for t in items if t["kind"] == "table"]:
            draw_table(c, tab)
        if doc.header:
            c.setFont("Helvetica-Bold", 8); c.drawString(54, H - 44, doc.header)
        c.showPage()
    c.save()


def table_geometry(tab):
    rh = 12.0 * 1.6
    return rh, [(tab["x"] + j * tab["cw"]) for j in range(tab["ncols"] + 1)]


def draw_table(c, tab):
    rh, xs = table_geometry(tab)
    y0 = tab["y"] + 10
    c.setLineWidth(0.6)
    for i in range(tab["nrows"] + 1):
        c.line(xs[0], y0 - i * rh, xs[-1], y0 - i * rh)
    for x in xs:
        c.line(x, y0, x, y0 - tab["nrows"] * rh)
    cells = tab["cells"]
    if tab["order"] == "col":                       # column-major content stream
        cells = sorted(cells, key=lambda d: (d["j"], d["i"]))
    for d in cells:
        c.setFont("Helvetica-Bold" if d["head"] else "Helvetica", 8.5)
        c.drawString(xs[d["j"]] + 4, y0 - d["i"] * rh - 13, d["text"])


def render_scan(doc, items, foot, pageno):
    s = 150 / 72
    im = Image.new("L", (int(W * s), int(H * s)), 250)
    d = ImageDraw.Draw(im)
    def f(sz, bold=False):
        return ImageFont.truetype(FONT_B if bold else FONT, int(sz * s))
    for t in items:
        if t["kind"] == "table":
            rh, xs = table_geometry(t); y0 = t["y"] + 10
            for i in range(t["nrows"] + 1):
                d.line([(xs[0] * s, (H - y0 + i * rh) * s), (xs[-1] * s, (H - y0 + i * rh) * s)], fill=40)
            for x in xs:
                d.line([(x * s, (H - y0) * s), (x * s, (H - y0 + t["nrows"] * rh) * s)], fill=40)
            for cl in t["cells"]:
                d.text(((xs[cl["j"]] + 4) * s, (H - y0 + cl["i"] * rh + 5) * s), cl["text"], fill=20, font=f(8.5, cl["head"]))
        else:
            d.text((t["x"] * s, (H - t["y"] - t["size"]) * s), t["text"], fill=20, font=f(t["size"], "Bold" in t["font"]))
    d.text((54 * s, (H - 36 - 7.5) * s), foot, fill=20, font=f(7.5))
    if doc.header:
        d.text((54 * s, (H - 44 - 8) * s), doc.header, fill=20, font=f(8, True))
    rng = random.Random(pageno * 7919 + len(doc.doc_id))
    im = im.rotate(rng.uniform(-0.6, 0.6), fillcolor=250, resample=Image.BICUBIC)
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    px = im.load()
    for _ in range(int(im.size[0] * im.size[1] * 0.002)):
        px[rng.randrange(im.size[0]), rng.randrange(im.size[1])] = rng.choice((30, 120))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=60); buf.seek(0)
    return Image.open(buf)


# ------------------------------------------------------------------ documents
def ho3(edition: str, n_excl: int, n_cond: int, n_sec2: int, spacer: float = 0, dpos=(0, 9, 25)) -> Doc:
    rows = edition == "2022-01"
    d = Doc(f"HO-3_{edition}", "HO-3", edition, "form", "two_col_rows" if rows else "two_col",
            "HOMEOWNERS 3 SPECIAL FORM",
            f"HO-3 ({edition})          Lakeshore Mutual Insurance Company          Page {{page}} of {{pages}}")
    rng = random.Random(3 if rows else 4)
    head(d, "Homeowners 3 Special Form")
    para(d, ["Agreement"], "We will provide the insurance described in this policy in return for the premium and compliance with all applicable provisions of this policy.")
    head(d, "Definitions")
    for term, txt in [("Insured", "Insured means you and residents of your household who are your relatives or other persons under the age of 21 in your care."),
                      ("Residence Premises", "Residence premises means the one family dwelling where you reside, shown as the residence premises in the Declarations."),
                      ("Occurrence", "Occurrence means an accident, including continuous or repeated exposure to substantially the same general harmful conditions."),
                      ("Actual Cash Value", "Actual cash value means the cost to replace damaged property with new property of like kind and quality, less depreciation for age, condition and obsolescence.")]:
        para(d, ["Definitions", term], txt)
    d.blocks.append(("colbreak",))
    d.blocks.append(("space", spacer))
    head(d, "Section I Perils Insured Against")
    para(d, ["Section I Perils Insured Against", "Coverage A And B"], "We insure against direct physical loss to property described in Coverages A and B unless the loss is excluded in Section I Exclusions. For Coverage C we insure for direct physical loss caused by a peril listed below.",
         section="Section I Perils Insured Against")
    for n, h, body in C.PERILS:
        path = ["Section I Perils Insured Against", f"{n}. {h}"]
        cid, sec = f"HO-3@{edition}#perils-{n}", f"Section I Perils Insured Against {n}"
        head(d, f"{n}. {h}", 2)
        if n == 12 and rows:
            para(d, path, C.FREEZING_2022_A, cid, sec)
            d.blocks.append(("colbreak",))
            para(d, path, C.FREEZING_2022_B, cid, sec)
        elif n == 12:
            para(d, path, C.FREEZING_2024, cid, sec)
        else:
            para(d, path, body, cid, sec)
    head(d, "Section I Exclusions")
    for k in range(n_excl):
        h, body, _ = C.gen_clause(rng, "form")
        lab = f"{chr(65 + k // 20)}.{k % 20 + 1}"
        head(d, f"{lab} {h}", 2)
        para(d, ["Section I Exclusions", f"{lab} {h}"], body, f"HO-3@{edition}#excl-{k+1}", f"Section I Exclusions {lab}")
    head(d, "Section I Conditions")
    dist = list(C.CONDITIONS_DISTRACTORS.items())
    for k in range(n_cond):
        if rows and k in dpos and dist:          # distractors spread across pages
            h, body = dist.pop(0)
        elif k == 29:
            h, body = "Roof Surfacing", (C.ROOF_2022 if rows else C.ROOF_2024)
        else:
            h, body, _ = C.gen_clause(rng, "form")
        lab = f"{chr(65 + k // 9)}.{k % 9 + 1}"
        head(d, f"{lab} {h}", 2)
        para(d, ["Section I Conditions", f"{lab} {h}"], body, f"HO-3@{edition}#cond-{k+1}", f"Section I Conditions {lab}")
    head(d, "Section II Liability Coverages")
    for k in range(n_sec2):
        h, body, _ = C.gen_clause(rng, "form")
        head(d, f"{k+1}. {h}", 2)
        para(d, ["Section II", f"{k+1}. {h}"], body.replace("loss to", "liability for damage to"), f"HO-3@{edition}#sec2-{k+1}", f"Section II {k+1}")
    return d


def endorsement(form, ed, scanned=False) -> Doc:
    d = Doc(f"{form.replace(' ', '')}_{ed}", form, ed, "endorsement", "single",
            "THIS ENDORSEMENT CHANGES THE POLICY. PLEASE READ IT CAREFULLY.",
            f"{form} ({ed})     Coverage under this endorsement applies only when the endorsement is shown in the Declarations.     Page {{page}} of {{pages}}",
            scanned=scanned)
    head(d, f"{form} ({ed})")
    for sec, txt in C.ENDORSEMENTS[form][ed]:
        if txt is None:   # roof schedule
            d.blocks.append(("table", f"{form}@{ed}#schedule", ("Age of roof surfacing", "Percentage of replacement cost paid"),
                             C.ROOF_SCHEDULE, "row", 0, [sec], "Schedule"))
            continue
        head(d, sec, 2)
        para(d, [sec], txt, f"{form}@{ed}#{slug(sec)}", sec)
    d.blocks.append(("pagebreak",))
    head(d, "All Other Provisions")
    para(d, ["All Other Provisions"], "All other provisions of the policy apply. This endorsement is part of the policy only when its form number and edition appear in the Declarations for the policy period.", f"{form}@{ed}#other", "All Other Provisions")
    return d


def state_amendatory(st, ed, rng) -> Doc:
    name = C.STATES[st]
    d = Doc(f"SP-{st}_{ed}", f"HO 01 {40 + list(C.STATES).index(st)} {st}", ed, "endorsement", "single",
            f"SPECIAL PROVISIONS {name.upper()}", f"Special Provisions {name} ({ed})     Page {{page}} of {{pages}}")
    head(d, f"Special Provisions {name}")
    for k in range(18):
        h, body, _ = C.gen_clause(rng, "form")
        head(d, f"{k+1}. {h}", 2)
        para(d, [f"Special Provisions {name}", f"{k+1}. {h}"], body, None, f"Special Provisions {name} {k+1}")
    return d


def bulletin(b, scanned=False) -> Doc:
    bid, st, eff, end, topic, text = b
    d = Doc(bid, bid, eff, "bulletin", "single", None,
            f"{bid}     {C.STATES[st]} insurance regulator bulletin (fixture)     Page {{page}} of {{pages}}",
            scanned=scanned, state=st)
    head(d, f"Bulletin {bid}")
    para(d, ["Applicability"], f"This bulletin applies to homeowners policies issued or renewed in {C.STATES[st]} for losses on or after {eff}" + (f" and before {end}." if end else "."), f"{bid}#applicability", "Applicability")
    head(d, "Guidance", 2)
    para(d, ["Guidance"], text, f"{bid}#guidance", "Guidance")
    d.blocks.append(("pagebreak",))
    head(d, "Background", 2)
    rng = random.Random(len(bid) * 13 + int(eff[:4]))
    for k in range(4):
        h, body, _ = C.gen_clause(rng, "manual")
        para(d, ["Background", h], body.replace("must", "should"), f"{bid}#bg-{k+1}", "Background")
    return d


def schedule_doc() -> Doc:
    d = Doc("LM-DED_2024-06", "LM DED", "2024-06", "schedule", "single", "LAKESHORE MUTUAL DEDUCTIBLE SCHEDULE",
            "LM DED (2024-06)     Page {page} of {pages}")
    head(d, "Homeowners Deductible Schedule (2024-06)")
    para(d, ["Purpose"], "This schedule states the minimum deductible that applies by state for homeowners policies on HO-3 (2024-06). Where the Declarations show a higher deductible, the Declarations apply.", "LM DED@2024-06#purpose", "Purpose")
    rng = random.Random(77)
    for k in range(30):     # fill so the table starts near the foot of page 2
        h, body, _ = C.gen_clause(rng, "underwriting")
        para(d, ["Notes", h], body, f"LM DED@2024-06#note-{k+1}", f"Notes {k+1}")
    d.blocks.append(("table", "LM DED@2024-06#table-1", C.DEDUCTIBLE_HEAD, C.DEDUCTIBLE_ROWS, "col", 3,
                     ["Table 1 Minimum deductibles by state"], "Table 1"))
    for k in range(6):
        h, body, _ = C.gen_clause(rng, "underwriting")
        para(d, ["Notes", h], body, f"LM DED@2024-06#post-{k+1}", f"Notes {31+k}")
    return d


def long_doc(doc_id, label, ed, kind, style, n_sections, seed, header) -> Doc:
    d = Doc(doc_id, label, ed, kind, "single", header, f"{label} ({ed})     Internal use only     Page {{page}} of {{pages}}")
    rng = random.Random(seed)
    for s in range(n_sections):
        chap = f"Part {s // 12 + 1}"
        if s % 12 == 0:
            head(d, f"{chap}")
        h, body, _ = C.gen_clause(rng, style)
        head(d, f"{s+1}. {h}", 2)
        para(d, [chap, f"{s+1}. {h}"], body, f"{label}@{ed}#s{s+1}", f"{chap} {s+1}")
    return d


FORM_ARGS = {}


def tune_form(edition, n_excl=None):
    """Choose the spacer so Perils 12 sits at the foot of page 2, left column,
    and place the 2022 distractor Conditions on pages 12, 13 and 15."""
    if n_excl is None:
        for n in range(150, 40, -5):
            try:
                tune_form(edition, n)
                return
            except (ValueError, SystemExit):
                continue
        raise SystemExit("no exclusion count places the distractors")
    n_cond, n_sec2 = 90, 60
    for sp in range(0, 700, 6):
        d = ho3(edition, n_excl, n_cond, n_sec2, sp)
        pages = layout(d)
        locs = [(pi + 1, t["col"], t["y"]) for pi, pg in enumerate(pages) for t in pg
                if t.get("ref") == f"HO-3@{edition}#perils-12"]
        if edition == "2022-01":
            ok = {(p, c) for p, c, _ in locs} == {(2, 0), (2, 1)} and min(y for p, c, y in locs if c == 0) < 66 + 36
        else:
            ok = {p for p, _, _ in locs} == {2} and len({c for _, c, _ in locs}) == 1
        if ok:
            break
    else:
        raise SystemExit(f"could not place perils 12 for {edition}")
    dpos = (0, 9, 25)
    if edition == "2022-01":
        where = {}
        for pi, pg in enumerate(pages):
            for t in pg:
                m = re.match(r"HO-3@2022-01#cond-(\d+)$", t.get("ref") or "")
                if m:
                    where.setdefault(int(m.group(1)) - 1, pi + 1)
        dpos = tuple(min(k for k, p in where.items() if p == target and k != 29) for target in (12, 13, 15))
    FORM_ARGS[edition] = (n_excl, n_cond, n_sec2, sp, dpos)


def build_docs(manual_sections: int):
    if not FORM_ARGS:
        tune_form("2022-01"); tune_form("2024-06")
    docs = [ho3("2022-01", *FORM_ARGS["2022-01"]), ho3("2024-06", *FORM_ARGS["2024-06"])]
    for form, eds in C.ENDORSEMENTS.items():
        for ed in eds:
            docs.append(endorsement(form, ed, scanned=(form, ed) in {("HO 04 95", "2019-03"), ("HO 04 61", "2022-01")}))
    rng = random.Random(11)
    for st in C.STATES:
        docs.append(state_amendatory(st, "2024-06", rng))
    for b in C.BULLETINS:
        docs.append(bulletin(b, scanned=b[0] in {"WI-OCI-2022-05", "MI-DIFS-2023-09", "IL-DOI-2023-10", "IN-DOI-2024-03"}))
    docs.append(schedule_doc())
    docs.append(long_doc("UW-GUIDE_2024-01", "LM UW GUIDE", "2024-01", "guideline", "underwriting", 150, 21, "UNDERWRITING GUIDELINES"))
    docs.append(long_doc("CLAIMS-MANUAL_2025-09", "LM CLAIMS MANUAL", "2025-09", "manual", "manual", manual_sections, 31, "CLAIMS HANDLING MANUAL"))
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    lo, hi = 50, 4000
    while lo < hi:                                    # size the manual to hit 300 pages
        mid = (lo + hi + 1) // 2
        total = sum(len(layout(d)) for d in build_docs(mid))
        lo, hi = (mid, hi) if total <= TARGET_PAGES else (lo, mid - 1)
    docs = build_docs(lo)
    total, truth, gpage = 0, {"docs": [], "clauses": {}, "pages": []}, 0
    for d in docs:
        pages = layout(d)
        draw_pdf(d, pages, os.path.join(OUT, d.doc_id + ".pdf"))
        truth["docs"].append(dict(doc_id=d.doc_id, label=d.label, edition=d.edition, kind=d.kind,
                                  layout=d.layout, scanned=d.scanned, state=d.state, pages=len(pages),
                                  header=d.header, footer=d.footer))
        for b in d.blocks:
            if b[0] == "p":
                cl = truth["clauses"].setdefault(b[1], dict(doc_id=d.doc_id, label=d.label, edition=d.edition,
                        kind=d.kind, path=b[3], section=b[4], text="", pages=[]))
                cl["text"] = (cl["text"] + " " + b[2]).strip()
            if b[0] == "table":
                truth["clauses"][b[1]] = dict(doc_id=d.doc_id, label=d.label, edition=d.edition, kind=d.kind,
                        path=b[6], section=b[7], text=" | ".join(b[2]) + " || " + " || ".join(" | ".join(r) for r in b[3]),
                        pages=[], table=dict(head=list(b[2]), rows=[list(r) for r in b[3]]))
        for pno, items in enumerate(pages, 1):
            gpage += 1
            lines = []
            order = sorted([t for t in items if t["kind"] != "table"], key=lambda t: (t["col"], -t["y"]))
            for t in order:
                lines.append(dict(kind=t["kind"], ref=t["ref"], text=t["text"]))
            for tab in [t for t in items if t["kind"] == "table"]:
                rows = {}
                for cl in tab["cells"]:
                    rows.setdefault(cl["i"], {})[cl["j"]] = cl["text"]
                lines.append(dict(kind="table", ref=tab["tid"], rows=[[rows[i][j] for j in sorted(rows[i])] for i in sorted(rows)]))
            for ln in lines:
                if ln["ref"] and pno not in truth["clauses"][ln["ref"]]["pages"]:
                    truth["clauses"][ln["ref"]]["pages"].append(pno)
            truth["pages"].append(dict(gid=gpage, doc_id=d.doc_id, page=pno, lines=lines,
                                       header=d.header, footer=d.footer.format(page=pno, pages=len(pages))))
        total += len(pages)
    with open(os.path.join(OUT, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print(f"{len(docs)} documents, {total} pages, manual sections {lo}")
    fz = truth["clauses"]["HO-3@2022-01#perils-12"]
    print("HO-3 (2022-01) perils 12 pages:", fz["pages"])
    for cid, cl in truth["clauses"].items():
        if cid.startswith("HO-3@2022-01#cond-") and ("pipe" in cl["text"]):
            print(cid, cl["path"][-1], cl["section"], cl["pages"])
    print("pages per doc:", {d["doc_id"]: d["pages"] for d in truth["docs"]})
    print("form args:", FORM_ARGS)
    print("deductible table pages:", truth["clauses"]["LM DED@2024-06#table-1"]["pages"])


if __name__ == "__main__":
    main()
