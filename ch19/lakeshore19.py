"""Synthetic media slice for Chapter 19: 240 property determinations drafted
before an adjuster has typed a cause of loss, with photos, sketches and voice
notes. Seeded and deterministic. Every name and number here is invented."""
import random
from datetime import date, timedelta

CAUSES = {"sewer_backup": 60, "surface_water": 40, "burst_pipe": 45,
          "seepage": 25, "hail": 40, "wear_tear": 30}
WATER = {"sewer_backup", "surface_water", "burst_pipe", "seepage"}


def clause(cause, endorsed):
    if cause == "sewer_backup":
        return "HO 04 95 (2024-06) water back-up" if endorsed else "HO-3 water exclusion (back-up)"
    return {"surface_water": "HO-3 water exclusion (surface water)",
            "burst_pipe": "HO-3 accidental discharge (Peril 12)",
            "seepage": "HO-3 seepage and leakage exclusion",
            "hail": "HO-3 windstorm or hail (Peril 2)",
            "wear_tear": "HO-3 wear and tear exclusion"}[cause]


# What a person in the room could see, by cause: (kind, target, value)
SEEN = {
    "sewer_backup": [("water_at", "floor_drain", ""), ("residue_at", "floor_drain", ""),
                     ("tide_line", "wall", "")],
    "surface_water": [("water_at", "window_well", ""), ("debris_at", "window_well", ""),
                      ("tide_line", "wall", "")],
    "burst_pipe": [("pipe_split", "supply_line", ""), ("spray_pattern", "ceiling", ""),
                   ("water_at", "wall", "")],
    "seepage": [("tide_line", "wall", "rings"), ("mould", "wall", "")],
    "hail": [("impact_marks", "roof_slope", ""), ("impact_marks", "gutter", "")],
    "wear_tear": [("curled_shingles", "roof_slope", ""), ("granule_loss", "roof_slope", "")],
}
# The sentence in the adjuster's voice note that decides the cause.
SAID = {
    "sewer_backup": "No water came in at the window wells.",
    "surface_water": "No residue at the floor drain.",
    "burst_pipe": "The supply line is split above the laundry.",
    "seepage": "Tide lines on the wall in several rings.",
    "hail": "Impact marks on the north roof slope.",
    "wear_tear": "No hail marks on the roof slope.",
}
# Which cause a feature named in a voice sentence speaks for.
TERM_CAUSE = {"window well": "surface_water", "floor drain": "sewer_backup",
              "supply line": "burst_pipe", "tide line": "seepage",
              "impact marks": "hail", "hail marks": "hail"}
FILLER = ["Basement is finished with carpet and drywall.",
          "Homeowner was present for the whole visit.",
          "Took photos of each wall and the mechanical room.",
          "Furnace and water heater sit on a raised pad.",
          "Contents are stacked by the stairs for the contractor."]
TERMS = set(TERM_CAUSE)
# A sketch's water entry point speaks for one cause.
SEEN_CAUSE = {"floor_drain": "sewer_backup", "window_well": "surface_water",
              "supply_line": "burst_pipe", "wall": "seepage"}
# What policyholders say at first notice of loss, true and loose versions.
SAY_TRUE = {"sewer_backup": "the sewer backed up through the drain",
            "surface_water": "rain water came in at the window",
            "burst_pipe": "a pipe burst", "seepage": "we found staining on the wall",
            "hail": "hail hit the roof", "wear_tear": "the roof is leaking"}
SAY_LOOSE = {"sewer_backup": "the basement flooded", "surface_water": "water in the basement",
             "burst_pipe": "water everywhere in the basement",
             "seepage": "the basement flooded", "hail": "storm damage to the roof",
             "wear_tear": "storm damage to the roof"}
GREEN_BAY = (44.5133, -88.0133)
EMILY = "CLM-2026-01187"


def _photo(rng, cid, k, cause, home, loss):
    seen = [o for o in SEEN[cause] if rng.random() < 0.35]
    shot = loss + timedelta(days=rng.choice([0, 0, 1, 2]))
    return {"media_id": f"{cid}-P{k}", "w": 4032, "h": 3024, "seen": seen,
            "gps": (home[0] + rng.uniform(-2e-4, 2e-4), home[1] + rng.uniform(-2e-4, 2e-4)),
            "captured_at": f"{shot.isoformat()}T{rng.randrange(8, 19):02d}:{rng.randrange(60):02d}",
            "subject": "damage at the insured home", "car": False,
            "uploader": rng.choice(["policyholder", "tbrennan", "field_adjuster"])}


def claims(seed=19):
    rng = random.Random(seed)
    out, n = [], 0
    for cause, count in CAUSES.items():
        for _ in range(count):
            n += 1
            cid = EMILY if (cause == "sewer_backup" and n == 1) else f"CLM-2026-{3000 + n:05d}"
            loss = date(2026, 3, 1) + timedelta(days=rng.randrange(0, 200))
            home = (GREEN_BAY[0] + rng.uniform(-0.15, 0.15), GREEN_BAY[1] + rng.uniform(-0.2, 0.2))
            c = {"id": cid, "cause": cause, "endorsed": rng.random() < 0.7,
                 "loss": loss.isoformat(), "home": home,
                 "fnol": SAY_TRUE[cause] if rng.random() < 0.55 else SAY_LOOSE[cause]}
            c["photos"] = [_photo(rng, cid, k, cause, home, loss)
                           for k in range(1, rng.randint(2, 5) + 1)]
            c["sketch"] = (SEEN[cause][0][1] if cause in WATER and rng.random() < 0.35 else None)
            c["voice"] = rng.random() < 0.7
            c["noisy"] = rng.random() < 0.3
            out.append(c)
    em = out[0]
    em.update(endorsed=True, loss="2026-03-14", fnol="the city sewer backed up during heavy rain",
              home=(44.5192, -87.9760), voice=True, noisy=True, sketch=None)
    em["photos"] = [_photo(random.Random(1187), EMILY, k, "sewer_backup", em["home"],
                           date(2026, 3, 14)) for k in range(1, 5)]
    for p in em["photos"]:
        p["seen"] = list(SEEN["sewer_backup"])
    # The photo taken in a car: eight roof claims carry a windscreen photo.
    roof = [c for c in out if c["cause"] in ("wear_tear", "hail")]
    for c in rng.sample(roof, 8):
        p = _photo(rng, c["id"], len(c["photos"]) + 1, c["cause"], c["home"],
                   date.fromisoformat(c["loss"]))
        p.update(seen=[("glass_cracked", "window", ""), ("vehicle_interior", "none", "")],
                 gps=(c["home"][0] + rng.uniform(0.02, 0.06), c["home"][1] + rng.uniform(-0.05, 0.05)), car=True,
                 subject="hail damage to window", uploader="policyholder")
        c["photos"].append(p)
    return out
