"""Chapter 19 lab: photos, sketches and voice notes as spans with provenance.

Note: deterministic stand-in. No vision, speech or language model runs. The
'vision model' reports what each synthetic photo shows with fixed detection
and error rates, and in free-caption mode it adds conclusions (flood, storm
damage) at a fixed rate. The 'recogniser' substitutes words and, in noisy
recordings, sometimes drops a leading negation. The 'determination model'
scores causes from the facts in its packet by fixed weights. Token counts for
images follow one provider's published patch rule; price is a stated constant.
Run: python3 lab.py > output.txt
"""
import pathlib
import random
import re
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import lakeshore19 as L
from contextlab import multimodal as M

PRICE_PER_MTOK = 3.00                       # US$ per million input tokens (stand-in)
CONFIGS = ["A text only", "B + raw photos", "C + free captions", "D + image spans",
           "E + raw transcript", "F + voice spans (guard)"]
W_OBS, W_SKETCH, W_VOICE = 1, 2, 4
FOREIGN = {"sewer_backup": "surface_water", "surface_water": "sewer_backup",
           "burst_pipe": "sewer_backup", "seepage": "burst_pipe",
           "hail": "wear_tear", "wear_tear": "hail"}


def section(t):
    print(f"\n== {t}")


def fnol_prior(text, cause):
    t = text.lower()
    for cue, c in [("sewer", "sewer_backup"), ("drain", "sewer_backup"),
                   ("rain water", "surface_water"), ("flooded", "surface_water"),
                   ("pipe", "burst_pipe"), ("staining", "seepage"), ("hail", "hail"),
                   ("storm", "hail")]:
        if cue in t:
            return c
    return "sewer_backup" if cause in L.WATER else "hail"


def signature(kind, target, value):
    """Which cause one observation speaks for (None if it says nothing)."""
    if (kind, target) in {("tide_line", "wall"), ("water_at", "wall")} and value != "rings":
        return None                             # seen in several causes
    for cause, seen in L.SEEN.items():
        if (kind, target, value) in seen:
            return cause
    return {"glass_cracked": "hail"}.get(kind)


def decide(c, facts, asserts):
    """facts: list of (cause, weight). asserts: free-text conclusions."""
    for a in asserts:                           # a stated conclusion wins
        return {"flood": "surface_water", "storm_damage": "hail",
                "hail_on_window": "hail"}[a]
    score = {}
    for cause, w in facts:
        if cause:
            score[cause] = score.get(cause, 0) + w
    best = max(score.values(), default=0)
    if best <= 0:
        return fnol_prior(c["fnol"], c["cause"])
    top = sorted(k for k, v in score.items() if v == best)
    prior = fnol_prior(c["fnol"], c["cause"])
    return prior if prior in top else top[0]


def free_vision(rng, c, mention, overreach):
    facts, asserts, n_assert = set(), [], 0
    for p in c["photos"]:
        if p["car"]:
            facts.add(("hail", "glass_cracked"))
            asserts.append("hail_on_window") if c["cause"] == "wear_tear" else None
            n_assert += 1
            continue
        for o in p["seen"]:
            if rng.random() < mention:
                facts.add((signature(*o), o[0] + ":" + o[1]))
        if rng.random() < overreach:
            asserts.append("flood" if c["cause"] in L.WATER else "storm_damage")
            n_assert += 1
    return [(f[0], W_OBS) for f in facts], asserts, n_assert


EMILY_P1 = {"water_at": M.Region(1180, 1640, 2860, 2620),
            "residue_at": M.Region(1720, 1900, 2240, 2290),
            "tide_line": M.Region(0, 820, 4032, 1010)}


def image_spans(rng, c):
    spans, refused = [], []
    for p in c["photos"]:
        s = M.ImageSpan(p["media_id"], M.sha(p["media_id"].encode()), c["id"], p["uploader"],
                        p["captured_at"], p["gps"], p["subject"], "obs-model v2 / prompt 19.3")
        raw = []
        for kind, target, value in p["seen"]:
            conf = rng.uniform(0.62, 0.97) if rng.random() < 0.78 else rng.uniform(0.35, 0.59)
            x, y = rng.randrange(200, 2800), rng.randrange(200, 2000)
            box = EMILY_P1.get(kind) if p["media_id"] == L.EMILY + "-P1" else None
            raw.append(M.Observation(kind, target, box or M.Region(x, y, x + 600, y + 450),
                                     round(conf, 2), value))
        if rng.random() < 0.06:                 # a reflection read as water, say
            k, t, v = L.SEEN[FOREIGN[c["cause"]]][0]
            raw.append(M.Observation(k, t, M.Region(100, 100, 700, 550),
                                     round(rng.uniform(0.6, 0.72), 2), v))
        if rng.random() < 0.10:                 # the model tries to conclude
            raw.append(M.Observation("cause", "none", M.Region(0, 0, 10, 10), 0.9))
        refused += M.admit(s, raw)
        M.provenance_flags(s, c["home"], c["loss"])
        spans.append(s)
    return spans, refused


def span_facts(spans):
    seen = set()
    for s in spans:
        if s.flags:
            continue
        for o in s.observations:
            seen.add((o.kind, o.target, o.value))
    return [(signature(*o), W_OBS) for o in seen]


NEG = re.compile(r"^(no|not)\s+", re.I)
CONFUSE = {"the": "a", "was": "is", "of": "off", "for": "four", "and": "an",
           "is": "", "with": "", "on": "in", "each": "eats", "sit": "set",
           "by": "buy", "are": "our", "whole": "hole", "in": "and"}


def record(rng, c, force_drop=False):
    """Return (segments_truth, segments_heard) for a voice note."""
    lines = rng.sample(L.FILLER, 3)
    lines.insert(rng.randrange(0, 4), L.SAID[c["cause"]])
    t, truth, heard = 0.0, [], []
    for line in lines:
        dur = round(0.35 * len(line.split()) + rng.uniform(0.2, 0.8), 1)
        snr = rng.uniform(6, 14) if c["noisy"] else rng.uniform(16, 30)
        conf = rng.uniform(0.70, 0.88) if c["noisy"] else rng.uniform(0.86, 0.97)
        words = line.split()
        sub = 0.30 if c["noisy"] else 0.08
        hw = [CONFUSE.get(w, w) if rng.random() < sub else w for w in words]
        text = " ".join(x for x in hw if x)
        p_drop = 0.35 if c["noisy"] else 0.03
        if NEG.match(line) and (force_drop or rng.random() < p_drop):
            text = NEG.sub("", text, count=1).capitalize()
            conf -= 0.06
        truth.append(M.Segment(t, t + dur, line, 1.0, snr))
        heard.append(M.Segment(t, t + dur, text, round(conf, 2), round(snr, 1)))
        t += dur
    return truth, heard


def voice_facts(segments):
    out = []
    for s in segments:
        low = s.text.lower()
        for term, cause in L.TERM_CAUSE.items():
            if term in low:
                out.append((cause, -W_VOICE if NEG.match(low) else W_VOICE))
    return out


def run():
    rng = random.Random(1919)
    cs = L.claims()
    rows = {k: {"wrong": 0, "asserts": 0, "flagged_used": 0, "tokens": 0,
                "routed": 0, "listen_s": 0.0, "by_cause": {}, "ids": set()} for k in CONFIGS}
    refusals, flips, wer_n, wer_c = [], 0, [], []
    carter = {}
    global caught, sweep, ablate
    caught, sweep, ablate = {}, [], {}
    for c in cs:
        truth = L.clause(c["cause"], c["endorsed"])
        r_free = random.Random(zlib.crc32(c["id"].encode()))
        em = 1.0 if c["id"] == L.EMILY else None   # her captions are fixed text
        f_raw, a_raw, n_raw = free_vision(random.Random(r_free.random()), c, 0.80, em or 0.08)
        f_cap, a_cap, n_cap = free_vision(random.Random(r_free.random()), c, 0.55, em or 0.15)
        spans, refused = image_spans(rng, c)
        refusals += refused
        sf = span_facts(spans)
        sk = [(L.SEEN_CAUSE[c["sketch"]], W_SKETCH)] if c["sketch"] else []
        if c["voice"]:
            tr, heard = record(rng, c, force_drop=(c["id"] == L.EMILY))
            (wer_n if c["noisy"] else wer_c).append(
                M.wer(" ".join(s.text for s in tr), " ".join(s.text for s in heard)))
            flips += sum(1 for a, b in zip(tr, heard)
                         if bool(NEG.match(a.text)) != bool(NEG.match(b.text)))
            guarded, routed, secs = [], 0, 0.0
            for s_t, s_h in zip(tr, heard):
                flip = bool(NEG.match(s_t.text)) != bool(NEG.match(s_h.text))
                tag = M.guard(s_h, L.TERMS)
                caught[(flip, tag)] = caught.get((flip, tag), 0) + 1
                if M.decisive(s_h, L.TERMS):
                    sweep.append((flip, s_h.conf, s_h.snr_db))
                if tag == "listen":
                    routed, secs = routed + 1, secs + (s_h.t1 - s_h.t0) + 10
                    guarded.append(s_t)          # the adjuster listened: truth
                else:
                    guarded.append(s_h)
            vf_raw, vf_g = voice_facts(heard), voice_facts(guarded)
            v_tokens = sum(len(s.text) for s in heard) // 4 + 30
        else:
            vf_raw = vf_g = []
            routed, secs, v_tokens = 0, 0.0, 0
        img_tok = sum(M.image_tokens(p["w"], p["h"])[0] for p in c["photos"])
        cap_tok = 45 * len(c["photos"])
        span_tok = sum(len(s.text()) for s in spans) // 4
        plan = {"A text only": ([], [], 0, 0, 0),
                "B + raw photos": (f_raw, a_raw, n_raw, img_tok, sum(p["car"] for p in c["photos"])),
                "C + free captions": (f_cap, a_cap, n_cap, cap_tok, sum(p["car"] for p in c["photos"])),
                "D + image spans": (sf + sk, [], 0, span_tok, 0),
                "E + raw transcript": (sf + sk + vf_raw, [], 0, span_tok + v_tokens, 0),
                "F + voice spans (guard)": (sf + sk + vf_g, [], 0, span_tok + v_tokens, 0)}
        no_sk = L.clause(decide(c, sf, []), c["endorsed"]) != truth
        ablate["D without sketches"] = ablate.get("D without sketches", 0) + no_sk
        for k, (facts, asserts, n_a, tok, car_used) in plan.items():
            got = L.clause(decide(c, facts, asserts), c["endorsed"])
            r = rows[k]
            bad = got != truth
            r["wrong"] += bad
            if bad:
                r["ids"].add(c["id"])
            r["by_cause"][c["cause"]] = r["by_cause"].get(c["cause"], 0) + bad
            r["asserts"] += n_a
            r["flagged_used"] += car_used
            r["tokens"] += tok
            if k.startswith("F"):
                r["routed"] += routed
                r["listen_s"] += secs
            if c["id"] == L.EMILY:
                carter[k] = got
        if c["id"] == L.EMILY:
            carter["spans"], carter["heard"] = spans, heard if c["voice"] else []
    return cs, rows, refusals, flips, wer_n, wer_c, carter


def main():
    cs, rows, refusals, flips, wer_n, wer_c, carter = run()
    n = len(cs)
    section("Media slice")
    print(f"determinations {n}; photos {sum(len(c['photos']) for c in cs)}; "
          f"voice notes {sum(c['voice'] for c in cs)} "
          f"(noisy {sum(c['voice'] and c['noisy'] for c in cs)}); "
          f"sketches {sum(bool(c['sketch']) for c in cs)}; "
          f"windscreen photos {sum(p['car'] for c in cs for p in c['photos'])}")
    section("Scoreboard (media slice)")
    print(f"{'configuration':26s} {'wrong':>5s} {'rate':>6s} {'asserts':>7s} "
          f"{'flagged':>7s} {'tok/det':>7s} {'US$/1k':>7s} {'routed':>6s}")
    for k, r in rows.items():
        tok = r["tokens"] / n
        print(f"{k:26s} {r['wrong']:5d} {100 * r['wrong'] / n:5.1f}% {r['asserts']:7d} "
              f"{r['flagged_used']:7d} {tok:7.0f} {tok * PRICE_PER_MTOK / 1000:7.2f} "
              f"{r['routed']:6d}")
    print(f"ablation: D without sketches {ablate['D without sketches']} wrong "
          f"({100 * ablate['D without sketches'] / n:.1f}%)")
    section("Wrong clauses by cause")
    causes = list(L.CAUSES)
    print(f"{'configuration':26s} " + " ".join(f"{c[:10]:>10s}" for c in causes))
    for k, r in rows.items():
        print(f"{k:26s} " + " ".join(f"{r['by_cause'].get(c, 0):10d}" for c in causes))
    section("Observation gate")
    kinds = {}
    for why in refusals:
        key = ("not an observation kind" if why.startswith("kind") else
               "target not in feature list" if why.startswith("target") else why)
        kinds[key] = kinds.get(key, 0) + 1
    for k, v in sorted(kinds.items()):
        print(f"refused: {k:40s} {v}")
    section("Voice notes")
    avg = lambda xs: 100 * sum(xs) / len(xs)
    print(f"WER clean {avg(wer_c):.1f}% (n={len(wer_c)}); noisy {avg(wer_n):.1f}% (n={len(wer_n)})")
    print(f"negations dropped in decisive sentences: {flips}")
    f = rows["F + voice spans (guard)"]
    print(f"guard: clips routed {f['routed']}, adjuster listening {f['listen_s'] / 60:.0f} min "
          f"({f['listen_s'] / max(1, f['routed']):.0f} s per clip incl. 10 s overhead)")
    print(f"guard on decisive flips: caught {caught.get((True, 'listen'), 0)}, "
          f"missed {caught.get((True, 'use'), 0)}; clean clips routed (false alarms) "
          f"{caught.get((False, 'listen'), 0)}")
    print("threshold sweep (decisive sentences, snr floor 15 dB):")
    for mc in (0.75, 0.80, 0.85, 0.90, 0.95):
        hit = [f for f, cf, sn in sweep if cf < mc or sn < 15]
        print(f"  min_conf {mc:.2f}: routed {len(hit):3d}, flips caught {sum(hit)} of "
              f"{sum(f for f, _, _ in sweep)}")
    hit = [f for f, cf, sn in sweep if cf < 0.85]
    print(f"  confidence only (no snr floor) at 0.85: routed {len(hit)}, caught {sum(hit)}")
    d, e, g = (rows[k]["ids"] for k in CONFIGS[3:])
    print(f"E vs D: fixed {len(d - e)}, introduced {len(e - d)}; "
          f"F vs D: fixed {len(d - g)}, introduced {len(g - d)}")
    section("Image token budget (one 4032x3024 phone photo)")
    for tier in ("standard", "high"):
        t, w, h = M.image_tokens(4032, 3024, tier)
        print(f"{tier:8s} full photo -> sent {w}x{h}, {t} tokens")
    crop = M.Region(1400, 1900, 2000, 2350)
    print(f"crop 600x450 (+15% pad) -> {M.crop_tokens(crop)} tokens; "
          f"observation line ~{len('- residue_at(floor_drain) region=1400,1900,2000,2350 conf=0.91') // 4} tokens")
    section("CLM-2026-01187 (Emily Carter)")
    for k in CONFIGS:
        print(f"{k:26s} -> {carter[k]}")
    print(carter["spans"][0].text())
    for s in carter["heard"]:
        print(f"heard {s.t0:5.1f}-{s.t1:5.1f}s conf={s.conf:.2f} snr={s.snr_db:4.1f} "
              f"{M.guard(s, L.TERMS):6s} {s.text}")


if __name__ == "__main__":
    main()
