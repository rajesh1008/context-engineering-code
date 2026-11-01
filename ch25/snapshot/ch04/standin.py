"""Chapter 4 stand-in lab: synthetic Lakeshore corpus, determinations, simulated
models and a minimal pipeline (retrieve -> pad -> Budgeter -> render -> model).

STAND-IN NOTICE. Nothing here is a language model. SimulatedModel reduces each
determination to a margin: how far the governing span leads its nearest rival.
Three things eat that margin:
  * length noise: an amplitude that grows linearly once the prompt passes the
    model's onset (slope * (prompt_tokens - onset) / 1000);
  * position: a weight for every place the governing span sits below first;
  * look-alike pull: grows with the inapplicable look-alike tokens in the packet.
If look-alike pull plus noise plus position overturns the look-alike margin, the
model cites the most similar look-alike; else if noise plus position overturns
the rival margin, it cites the rival (an applicable span, wrong clause).
The per-determination margins are calibrated (calibrate.py -> calibration.json)
so the stand-ins reproduce the chapter's illustrative profile exactly. The
mechanism is real and inspectable; the profile is illustrative, not measured.
"""
from __future__ import annotations

import hashlib, json, math, random
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- tokens
INSTR_TOKENS = 260          # system instructions and output contract
SPAN_HDR = 18               # provenance header per evidence span
PAGE_HDR = 40               # header per claim-file page
FEED_LINE = 30              # provenance line the related-documents feed adds per document
FEED_BANNER = 0             # feed manifest; set at the end of the module (exact-tuning knob)
OUT_TOKENS = 57             # answer with citation
OUT_TOKENS_LA = 66          # answers that cite an endorsement not on the policy add a caveat

# ---------------------------------------------------------------- corpus
@dataclass(frozen=True)
class Span:
    span_id: str
    family: str
    form: str
    edition: str
    kind: str              # form | endorsement | bulletin | page
    section: str
    page: int
    topic: str
    tokens: int
    valid_from: date | None = None
    valid_to: date | None = None
    state: str | None = None
    effect: str = ""       # grant | exclusion | general
    binding: bool = True

    @property
    def label(self) -> str:
        return f"{self.form} ({self.edition}), {self.section}, page {self.page}"


FAMILIES = {
    "HO-3": dict(
        base=[("HO-3", "2022-01", date(2022, 1, 1), date(2024, 5, 31)),
              ("HO-3", "2024-06", date(2024, 6, 1), None)],
        topics=["water_damage", "sewer_backup", "wind_hail", "roof", "theft", "fire", "mold", "liability"],
        endorsements=[("HO 04 95", "2022-01", "sewer_backup", date(2022, 1, 1), date(2024, 5, 31)),
                      ("HO 04 95", "2024-06", "sewer_backup", date(2024, 6, 1), None),
                      ("HO 04 90", "2024-06", "fire", date(2024, 6, 1), None),
                      ("HO 04 16", "2024-06", "theft", date(2024, 6, 1), None),
                      ("HO 04 36", "2024-06", "roof", date(2024, 6, 1), None),
                      ("HO 04 32", "2024-06", "mold", date(2024, 6, 1), None),
                      ("HO 04 11", "2024-06", "water_damage", date(2024, 6, 1), None)],
        bulletin_topics=["roof", "wind_hail", "water_damage"]),
    "PAP": dict(
        base=[("PP 00 01", "2018-03", date(2018, 3, 1), date(2023, 12, 31)),
              ("PP 00 01", "2024-01", date(2024, 1, 1), None)],
        topics=["collision", "comprehensive", "uninsured_motorist", "rental", "towing", "liability"],
        endorsements=[("PP 04 03", "2024-01", "uninsured_motorist", date(2024, 1, 1), None),
                      ("PP 03 06", "2024-01", "rental", date(2024, 1, 1), None),
                      ("PP 03 13", "2024-01", "towing", date(2024, 1, 1), None)],
        bulletin_topics=["uninsured_motorist"]),
    "BOAT": dict(
        base=[("BT 00 01", "2019-01", date(2019, 1, 1), date(2022, 12, 31)),
              ("BT 00 01", "2023-01", date(2023, 1, 1), None)],
        topics=["hull_damage", "theft", "liability", "navigation"],
        endorsements=[("BT 04 10", "2023-01", "navigation", date(2023, 1, 1), None)],
        bulletin_topics=[]),
    "BOP": dict(
        base=[("BOP", "2019-07", date(2019, 7, 1), date(2024, 12, 31)),
              ("BOP", "2025-01", date(2025, 1, 1), None)],
        topics=["employee_dishonesty", "equipment_breakdown", "spoilage", "fire", "theft", "liability"],
        endorsements=[("BP 04 50", "2025-01", "employee_dishonesty", date(2025, 1, 1), None),
                      ("BP 04 30", "2025-01", "equipment_breakdown", date(2025, 1, 1), None),
                      ("BP 04 17", "2025-01", "spoilage", date(2025, 1, 1), None)],
        bulletin_topics=["fire"]),
}
STATES = ["IL", "IA", "WI", "MN", "OH", "IN", "MI"]
N_SPANS, CORPUS_TOKENS = 378, 22_326


def _rng(*key) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:12], 16))


def build_corpus() -> list[Span]:
    r = _rng("corpus")
    spans: list[Span] = []
    for fam, f in FAMILIES.items():
        for form, ed, vf, vt in f["base"]:
            page = 1
            for t in f["topics"]:
                spans.append(Span(f"{form} ({ed}) {t} grant", fam, form, ed, "form",
                                  "Section I Perils Insured Against" if t != "liability" else "Section II Coverage E",
                                  page, t, 0, vf, vt, effect="grant"))
                clause = "B.2.h" if (fam, t) == ("BOP", "employee_dishonesty") else f"B.{1 + len(spans) % 5}.{'abcdefgh'[len(spans) % 8]}"
                spans.append(Span(f"{form} ({ed}) {t} exclusion", fam, form, ed, "form",
                                  f"Section I Exclusions {clause}", page + 2, t, 0, vf, vt, effect="exclusion"))
                page += 1
            for g in ["Definitions", "Conditions", "Declarations", "Loss Settlement", "Duties After Loss",
                      "Other Insurance", "Policy Period", "Concealment or Fraud", "Appraisal", "Mortgage Clause"]:
                spans.append(Span(f"{form} ({ed}) {g}", fam, form, ed, "form", g, page, "general", 0, vf, vt,
                                  effect="general"))
                page += 1
        for form, ed, t, vf, vt in f["endorsements"]:
            for i, sec in enumerate(["Coverage", "Limit of Liability", "Exclusions", "Conditions", "Schedule", "Definitions"]):
                spans.append(Span(f"{form} ({ed}) {sec}", fam, form, ed, "endorsement", sec, 1 + i // 2, t, 0, vf, vt,
                                  effect="grant" if sec == "Coverage" else "general"))
        for t in f["bulletin_topics"]:
            for st in STATES:
                for yr, (vf, vt) in (("2024", (date(2024, 1, 1), date(2025, 2, 28))), ("2025", (date(2025, 3, 1), None))):
                    spans.append(Span(f"LB-{st}-{yr} {fam} {t}", fam, f"Bulletin {st}-{yr}-{t}", yr, "bulletin",
                                      "Guidance", 1, t, 0, vf, vt, state=st, effect="grant"))
    # pad / trim to exactly N_SPANS with extra general form spans
    extra = 0
    while len(spans) < N_SPANS:
        fam = list(FAMILIES)[extra % 4]
        form, ed, vf, vt = FAMILIES[fam]["base"][1]
        spans.append(Span(f"{form} ({ed}) Schedule {extra}", fam, form, ed, "form", f"Schedule {extra}", 40 + extra,
                          "general", 0, vf, vt, effect="general"))
        extra += 1
    spans = spans[:N_SPANS]
    sizes = [max(20, int(r.lognormvariate(math.log(52), 0.45))) for _ in spans]
    scale = CORPUS_TOKENS / sum(sizes)
    sizes = [max(20, round(s * scale)) for s in sizes]
    sizes[-1] += CORPUS_TOKENS - sum(sizes)
    return [replace(s, tokens=n) for s, n in zip(spans, sizes)]


CORPUS = build_corpus()
BY_ID = {s.span_id: s for s in CORPUS}

# ---------------------------------------------------------------- determinations
FIRST = ["Grace", "Tom", "Ana", "Wei", "Priya", "Samuel", "Ines", "Kofi", "Marta", "Omar", "Lena", "Diego",
         "Hannah", "Ravi", "Chloe", "Yusuf", "Elena", "Jonah", "Mei", "Felix"]
LAST = ["Murphy", "Novak", "Okafor", "Lindqvist", "Reyes", "Chen", "Patel", "Kowalski", "Haddad", "Brennan",
        "Silva", "Andersen", "Moreau", "Ibrahim", "Kaur", "Fischer", "Rossi", "Tanaka", "Walsh", "Duarte"]


@dataclass
class Determination:
    determination_id: str
    holder: str
    state: str
    family: str
    form: str
    edition: str
    topic: str
    task_type: str               # "form clause" | "endorsement or bulletin"
    date_of_loss: date
    endorsements: tuple[tuple[str, str], ...]
    gold_span_id: str
    rival_span_id: str
    evidence: list[str] = field(default_factory=list)
    retrieval: dict[str, float] = field(default_factory=dict)
    header_tokens: int = 60

    @property
    def gold_label(self) -> str:
        return f"{self.form} ({self.edition})" if self.task_type == "form clause" else BY_ID[self.gold_span_id].form


def applies(s: Span, d: Determination) -> bool:
    if s.kind == "page":
        return True
    if s.family != d.family:
        return False
    in_force = (s.valid_from is None or s.valid_from <= d.date_of_loss) and (s.valid_to is None or d.date_of_loss <= s.valid_to)
    if s.kind == "form":
        return (s.form, s.edition) == (d.form, d.edition)
    if s.kind == "endorsement":
        return (s.form, s.edition) in d.endorsements and in_force
    return s.state == d.state and in_force


FAMILY_QUOTA = ["HO-3"] * 113 + ["PAP"] * 48 + ["BOAT"] * 22 + ["BOP"] * 17
ENDORSE_QUOTA = {"HO-3": 30, "PAP": 9, "BOAT": 3, "BOP": 5}   # 47 endorsement-or-bulletin determinations


def build_dets() -> list[Determination]:
    r = _rng("dets")
    fams = FAMILY_QUOTA[:]
    r.shuffle(fams)
    # DET-0004 is Grace Murphy's businessowners claim, the chapter's worked example
    i_bop = fams.index("BOP")
    fams[3], fams[i_bop] = fams[i_bop], fams[3]
    left = dict(ENDORSE_QUOTA)
    remaining = {f: fams.count(f) for f in FAMILY_QUOTA}
    dets = []
    for i, fam in enumerate(fams):
        f = FAMILIES[fam]
        did = f"DET-{i + 1:04d}"
        # endorsement-type determinations: spread evenly through the set
        p_end = left[fam] / remaining[fam]
        remaining[fam] -= 1
        etype = i != 3 and r.random() < p_end and left[fam] > 0
        if etype:
            left[fam] -= 1
        holder = "Grace Murphy" if i == 3 else f"{r.choice(FIRST)} {r.choice(LAST)}"
        state = "IA" if i == 3 else r.choice(STATES)
        dol = date(2025, 1, 1) + (date(2026, 9, 30) - date(2025, 1, 1)) * r.random()
        dol = date.fromordinal(int(dol.toordinal())) if isinstance(dol, date) else dol
        form, ed = [(b[0], b[1]) for b in f["base"] if b[2] <= dol and (b[3] is None or dol <= b[3])][0]
        live_end = [(e[0], e[1], e[2]) for e in f["endorsements"]
                    if e[3] <= dol and (e[4] is None or dol <= e[4])]
        if etype:
            opts = live_end + ([("BULLETIN", None, t) for t in f["bulletin_topics"]] if f["bulletin_topics"] else [])
            choice = r.choice(opts)
            topic = choice[2]
        else:
            topic = "employee_dishonesty" if i == 3 else r.choice(f["topics"])
        on_policy = tuple((e[0], e[1]) for e in live_end if e[2] != topic and r.random() < 0.35)
        if etype and choice[0] != "BULLETIN":
            on_policy = on_policy + ((choice[0], choice[1]),)
        d = Determination(did, holder, state, fam, form, ed, topic,
                          "endorsement or bulletin" if etype else "form clause", dol, on_policy, "", "")
        base_grant = f"{form} ({ed}) {topic} grant"
        base_excl = f"{form} ({ed}) {topic} exclusion"
        if etype and choice[0] == "BULLETIN":
            yr = "2025" if dol >= date(2025, 3, 1) else "2024"
            gold, rival = f"LB-{state}-{yr} {fam} {topic}", base_excl
        elif etype:
            gold, rival = f"{choice[0]} ({choice[1]}) Coverage", base_excl
        else:
            gold, rival = (base_excl, base_grant) if (i == 3 or r.random() < 0.5) else (base_grant, base_excl)
        d.gold_span_id, d.rival_span_id = gold, rival
        others = [s.span_id for s in CORPUS if applies(s, d) and s.span_id not in (gold, rival)
                  and (s.topic in (topic, "general") or (s.kind == "endorsement" and s.section == "Coverage"))]
        r.shuffle(others)
        d.evidence = [gold] + others[:1] + [rival] + others[1:1 + r.randint(3, 6)]
        d.header_tokens = r.randint(40, 80)
        sc, d.retrieval = 0.97 - 0.12 * r.random(), {}
        for sid in d.evidence:                      # retrieval order, descending scores
            d.retrieval[sid] = round(sc, 3)
            sc -= 0.01 + 0.09 * r.random()
        dets.append(d)
    # exact clean-packet means: 1,046 prompt tokens on the first 100 and on all 200
    for lo, hi, fix in ((0, 100, 0), (100, 200, 100)):
        gap = 1046 * (hi - lo) - sum(clean_tokens(d) for d in dets[lo:hi])
        for k, d in enumerate(dets[lo:hi]):           # spread the correction over the claim headers
            d.header_tokens += gap // (hi - lo) + (1 if k < gap % (hi - lo) else 0)
    return dets


def clean_tokens(d: Determination) -> int:
    return INSTR_TOKENS + d.header_tokens + sum(BY_ID[s].tokens + SPAN_HDR for s in d.evidence)


# ---------------------------------------------------------------- padding
NEUTRAL_TARGETS = [2_000, 4_000, 8_000, 12_000, 16_000, 24_000, 32_000, 48_000, 64_000, 96_000, 128_000, 160_000]
NEUTRAL_RENDERED = [2_346, 4_675, 9_039, 13_401, 17_756, 26_486, 35_290, 52_914, 70_451, 105_618, 140_766, 175_903]


def _page_lengths() -> list[int]:
    """One sequence of claim-file page lengths, shared by every claim (the pages
    differ in content, not length), so the neutral curve's x-axis is the same
    for every determination. Pages are appended until the target is reached."""
    r = _rng("pages")
    pages: list[int] = []
    body = 0
    for t, rendered in zip(NEUTRAL_TARGETS, NEUTRAL_RENDERED):
        k = (rendered - t) // PAGE_HDR
        while rendered - PAGE_HDR * k - t < 1:
            k -= 1
        total_body = rendered - PAGE_HDR * k
        n_new, need = k - len(pages), total_body - body
        sizes = [r.randint(180, 620) for _ in range(n_new)]
        scale = need / sum(sizes)
        sizes = [max(30, round(s * scale)) for s in sizes]
        sizes[-1] += need - sum(sizes)
        over = total_body - t
        if sizes[-1] <= over:                       # the last page must be the one that crosses the target
            shift = over + 1 - sizes[-1]
            sizes[-1] += shift
            sizes[0] -= shift
        pages += sizes
        body = total_body
    while body < 400_000:
        s = r.randint(250, 560)
        pages.append(s)
        body += s
    return pages


PAGES = _page_lengths()


def pages_to(d: Determination, target: int, start_body: int = 0) -> list[Span]:
    out, body, i = [], start_body, 0
    while body < target:
        n = PAGES[i]
        out.append(Span(f"{d.determination_id} claim file p{i + 1}", d.family, "claim file", "", "page",
                        "Claim file", i + 1, "claim_file", n, binding=False, effect="general"))
        body += n
        i += 1
    return out


def neutral(d: Determination, target: int) -> list[Span]:
    """Claim-file pages for the claim being decided, appended until `target` padded tokens."""
    return pages_to(d, target)


def similarity(s: Span, d: Determination) -> float:
    return (1.0 if s.topic == d.topic else 0.35 if s.topic == "general" else 0.5) + \
           (0.12 if s.kind == "endorsement" else 0.06 if s.kind == "bulletin" else 0.0) + \
           (0.05 if s.section == "Coverage" else 0.0)


def pool(d: Determination) -> list[Span]:
    """Inapplicable binding spans of the claim's line of business, most similar first."""
    cands = [s for s in CORPUS if s.family == d.family and s.binding and not applies(s, d)]
    return sorted(cands, key=lambda s: (-similarity(s, d), s.span_id))


def lookalike(d: Determination, target: int) -> list[Span]:
    """Look-alikes most similar to the question first, topped up with claim-file pages."""
    out, body = [], 0
    for s in pool(d):
        if body + s.tokens > target:
            break
        out.append(s)
        body += s.tokens
    return out + pages_to(d, target, body)


# ---------------------------------------------------------------- packet, model, pipeline
@dataclass
class Rendered:
    tokens: int
    spans: list[Span]


@dataclass
class Answer:
    cited_span_id: str | None
    kind: str          # right | rival | lookalike
    output_tokens: int


@dataclass
class Trace:
    det: Determination
    model: "SimulatedModel"
    rendered: Rendered
    answer: Answer

    @property
    def billed_tokens(self) -> int:
        return self.rendered.tokens + self.answer.output_tokens


def retrieval_score(s: Span, d: Determination) -> float:
    if s.kind == "page":
        return 0.0
    if s.span_id in d.retrieval:
        return d.retrieval[s.span_id]
    return 0.5 * similarity(s, d)


AUTHORITY = {"endorsement": 1.25, "bulletin": 1.15, "form": 1.0, "page": 0.0}


def budget_score(s: Span, d: Determination) -> float:
    return retrieval_score(s, d) * AUTHORITY[s.kind]


def pack(spans: list[Span], d: Determination, budget: int) -> list[Span]:
    """The Budgeter: keep scored spans by score until the evidence budget is full;
    unscored padding (claim-file pages) only if room is left, so it goes first."""
    scored = sorted([s for s in spans if s.kind != "page"], key=lambda s: -budget_score(s, d))
    kept, used = [], 0
    for s in scored:
        if used + s.tokens <= budget:
            kept.append(s)
            used += s.tokens
    for s in spans:
        if s.kind == "page" and used + s.tokens <= budget:
            kept.append(s)
            used += s.tokens
    return kept


def render(d: Determination, spans: list[Span]) -> Rendered:
    toks = INSTR_TOKENS + d.header_tokens
    toks += sum(s.tokens + (PAGE_HDR if s.kind == "page" else SPAN_HDR) for s in spans)
    fed = sum(1 for s in spans if s.kind != "page" and not applies(s, d))
    if fed:   # the feed attaches a provenance line to each document and a short manifest
        toks += FEED_LINE * fed + FEED_BANNER
    return Rendered(toks, spans)


PROFILES = {   # window, onset, noise slope per 1k tokens, look-alike pull per 1k tokens, position weight
    "sim-large": dict(window=200_000, onset=24_000, slope=0.010, pull=0.020, position=0.0, price_in=3.00, price_out=15.00),
    "sim-mid": dict(window=128_000, onset=12_000, slope=0.012, pull=0.030, position=0.100, price_in=1.00, price_out=5.00),
    "sim-small": dict(window=32_000, onset=6_000, slope=0.030, pull=0.080, position=0.0, price_in=0.25, price_out=1.25),
}
CALIB = json.loads((HERE / "calibration.json").read_text()) if (HERE / "calibration.json").exists() else {}


class SimulatedModel:
    def __init__(self, name: str):
        self.name = name
        self.__dict__.update(PROFILES[name])

    def __repr__(self):
        return self.name

    def __format__(self, spec):
        return format(self.name, spec)

    def scores(self, d: Determination, r: Rendered) -> tuple[float, float, float, bool]:
        """(noise + position, look-alike pressure, look-alike tokens, any look-alike present)."""
        noise = self.slope * max(0, r.tokens - self.onset) / 1000
        ids = [s.span_id for s in r.spans]
        rank = ids.index(d.gold_span_id) if d.gold_span_id in ids else 0
        u = noise + self.position * rank
        la = [s for s in r.spans if s.kind != "page" and not applies(s, d)]
        e = sum(s.tokens for s in la)
        return u, u + self.pull * e / 1000, e, bool(la)

    def answer(self, d: Determination, r: Rendered) -> Answer:
        m = CALIB.get(self.name, {}).get(d.determination_id, {"rot": math.inf, "la": math.inf})
        u, v, _, has_la = self.scores(d, r)
        ids = {s.span_id for s in r.spans}
        if has_la and v >= m["la"]:
            top = max((s for s in r.spans if s.kind != "page" and not applies(s, d)), key=lambda s: similarity(s, d))
            return Answer(top.span_id, "lookalike", OUT_TOKENS_LA)
        if u >= m["rot"] or d.gold_span_id not in ids:
            return Answer(d.rival_span_id, "rival", OUT_TOKENS)
        return Answer(d.gold_span_id, "right", OUT_TOKENS)


class Pipeline:
    """Retrieve (frozen Chapter 2 fixed-pipeline evidence) -> pad -> Budgeter -> render -> model."""

    def __init__(self, corpus=CORPUS, model: str | SimulatedModel = "sim-large", pad_fn=None, budget=None):
        self.corpus = corpus
        self.model = model if isinstance(model, SimulatedModel) else SimulatedModel(model)
        self.pad_fn, self.budget = pad_fn, budget

    def with_config(self, **kw) -> "Pipeline":
        cfg = dict(corpus=self.corpus, model=self.model, pad_fn=self.pad_fn, budget=self.budget)
        cfg.update(kw)
        return Pipeline(**cfg)

    def packet(self, d: Determination) -> Rendered:
        spans = [BY_ID[s] for s in d.evidence] + (self.pad_fn(d) if self.pad_fn else [])
        if self.budget is not None:
            spans = pack(spans, d, self.budget)
        return render(d, spans)

    def run(self, d: Determination) -> Trace:
        r = self.packet(d)
        if r.tokens > self.model.window:
            raise ValueError(f"prompt {r.tokens:,} tokens exceeds {self.model.name} window {self.model.window:,}")
        return Trace(d, self.model, r, self.model.answer(d, r))


DETS_ALL = build_dets()

# exact-tuning knob: the related-documents banner (rendered once when the feed has
# added spans) is sized so the injected packet's mean is 22,205 prompt tokens.
FEED_BANNER = 0
FEED_BANNER = round((22_205 * 100 - sum(render(d, [BY_ID[s] for s in d.evidence] + lookalike(d, 16_000)).tokens
                                        for d in DETS_ALL[:100])) / 100)


# ---------------------------------------------------------------- scoring helpers
VERIFY_MEDIAN, VERSION_CHECK, MANUAL_MEDIAN = 1.2, 2.2, 11.0   # Chapter 2 proxy constants, minutes


def lookup_minutes(t: Trace | None, d: Determination) -> float:
    rng = random.Random(int(hashlib.sha256(d.determination_id.encode()).hexdigest()[:12], 16))
    jitter = rng.lognormvariate(0, 0.35)
    if t is None or t.answer.kind != "right":
        return MANUAL_MEDIAN * jitter + 0.5
    return VERIFY_MEDIAN * jitter + VERSION_CHECK


def cost(t: Trace) -> float:
    m = t.model
    return (t.rendered.tokens * m.price_in + t.answer.output_tokens * m.price_out) / 1e6
