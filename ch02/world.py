"""Chapter 2 world: the catalogue of Lakeshore's synthetic documents and the 500 determinations.

STAND-IN. Every document, policy and claim here is synthetic. This module fixes the facts
(which forms, editions, endorsements, bulletins, notes and topics exist, and what each clause
says); calibrate.py sizes the texts and writes the fixtures; standin.py loads the result.
"""
from __future__ import annotations

import hashlib
import random
from datetime import date

SEED = 2026


def rng(*key) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, (SEED,) + key)).encode()).hexdigest()[:12], 16))


STATES = ["WI", "IL", "MI", "MN", "IA", "IN"]

# ------------------------------------------------------------------ base forms
FORMS = {  # doc_id: (line, [(version, effective_from)], spans per edition)
    "HO-3": ("homeowners", [("2019-03", date(2019, 3, 1)), ("2022-01", date(2022, 1, 1)),
                            ("2024-06", date(2024, 6, 1))], 31),
    "PAP": ("auto", [("2018-09", date(2018, 9, 1)), ("2023-01", date(2023, 1, 1))], 22),
    "BOAT": ("marine", [("2021-05", date(2021, 5, 1)), ("2025-03", date(2025, 3, 1))], 18),
    "BOP": ("commercial", [("2020-04", date(2020, 4, 1)), ("2025-01", date(2025, 1, 1))], 15),
}
LINE_FORM = {v[0]: k for k, v in FORMS.items()}

# topic: (title, section, page, holding by edition group {"old": ..., "new": ...}, verdict old, verdict new)
# "new" is the latest edition; "old" every earlier one (their texts are identical).
T = dict
HO_TOPICS = {
    "water_backup": T(title="Water Damage", section="Section I Exclusions A.3", page=9,
                      old="We do not insure for loss caused by water which backs up through sewers or drains",
                      vo="excluded", vn="excluded"),
    "flood": T(title="Surface Water", section="Section I Exclusions A.3.a", page=9,
               old="We do not insure for loss caused by flood, surface water or waves",
               vo="excluded", vn="excluded"),
    "freezing": T(title="Freezing", section="Section I Perils Insured Against A.2.c", page=6,
                  old="We insure for loss caused by freezing of a plumbing, heating, air conditioning or automatic fire protective sprinkler system or of a household appliance only if, while the dwelling is vacant, unoccupied or being constructed, you used reasonable care to maintain heat in the building",
                  new="We insure for loss caused by freezing of a plumbing, heating, air conditioning or automatic fire protective sprinkler system or of a household appliance, and this includes frozen pipes that burst, only if you used reasonable care to maintain heat in the building",
                  vo="covered if heat maintained", vn="covered if heat maintained"),
    "hail_cosmetic": T(title="Cosmetic Hail Damage", section="Section I Perils Insured Against A.2.b", page=6,
                       old="We insure for direct physical loss by hail to roof surfacing, including dents and marks that do not let water in",
                       new="We do not insure for cosmetic loss by hail to metal roof surfacing, meaning dents or marks that do not allow water to enter the building",
                       vo="covered", vn="excluded for metal roofs"),
    "ice_dam": T(title="Ice Dams", section="Section I Perils Insured Against A.2.d", page=7,
                 old="We insure for interior water damage caused by an ice dam, but not for removing the ice",
                 vo="interior water damage covered", vn="interior water damage covered"),
    "wind_hail": T(title="Windstorm or Hail", section="Section I Perils Insured Against A.2.a", page=6,
                   old="We insure for direct physical loss caused by windstorm or hail",
                   vo="covered", vn="covered"),
    "tree_debris": T(title="Trees and Debris", section="Section I Additional Coverages E.1", page=5,
                     old="We will pay up to $1,000 for the removal of a tree felled by windstorm or hail that damages a covered structure or blocks a driveway",
                     new="We will pay up to $1,500 for the removal of a tree felled by windstorm, hail or the weight of ice that damages a covered structure or blocks a driveway",
                     vo="covered up to $1,000", vn="covered up to $1,500"),
    "mold": T(title="Fungi and Mold", section="Section I Exclusions A.9", page=10,
              old="We do not insure for loss caused by mold unless it results from a covered water loss",
              vo="excluded unless from a covered water loss", vn="excluded unless from a covered water loss"),
    "roof_settlement": T(title="Roof Surfacing Loss Settlement", section="Section I Conditions D.3", page=4,
                         old="Loss to roof surfacing is settled at replacement cost without deduction for depreciation",
                         new="Loss to roof surfacing is settled at actual cash value when the roof surfacing is more than 15 years old at the time of loss",
                         vo="replacement cost", vn="actual cash value if roof over 15 years"),
    "roof_matching": T(title="Matching of Undamaged Material", section="Section I Conditions D.4", page=4,
                       old="We will pay to replace undamaged roof surfacing when replacement material cannot reasonably match the repaired area",
                       new="We do not pay to replace undamaged roof surfacing on any slope or elevation that was not damaged, even if the material does not match",
                       vo="matching required", vn="matching limited to damaged slopes"),
    "earth_movement": T(title="Earth Movement", section="Section I Exclusions A.2", page=9,
                        old="We do not insure for loss caused by earth movement, including earthquake and landslide",
                        vo="excluded", vn="excluded"),
    "food_spoilage": T(title="Refrigerated Property", section="Section I Exclusions A.4", page=9,
                       old="We do not insure for loss to food caused by an off-premises power failure",
                       vo="excluded", vn="excluded"),
    "theft": T(title="Theft", section="Section I Perils Insured Against B.9", page=7,
               old="We insure for loss of personal property by theft, including attempted theft",
               vo="covered", vn="covered"),
    "jewelry_limit": T(title="Special Limits of Liability", section="Section I Coverage C.3.h", page=3,
                       old="We will pay no more than $1,500 in total for theft of jewellery and watches",
                       vo="limited to $1,500", vn="limited to $1,500"),
    "collapse": T(title="Collapse", section="Section I Additional Coverages E.8", page=5,
                  old="We insure for direct physical loss involving abrupt collapse of a building caused by the weight of ice, snow or sleet",
                  vo="covered", vn="covered"),
    "dog_bite": T(title="Personal Liability", section="Section II Coverage E", page=11,
                  old="We will pay damages for bodily injury caused by an animal owned by an insured",
                  vo="covered up to the liability limit", vn="covered up to the liability limit"),
    "ale": T(title="Loss of Use", section="Section I Coverage D", page=3,
             old="We cover additional living expense if a covered loss makes the residence unfit to live in",
             vo="covered for the shortest repair time", vn="covered for the shortest repair time"),
    "vandalism_vacancy": T(title="Vandalism and Malicious Mischief", section="Section I Perils Insured Against B.8", page=7,
                           old="We do not insure for vandalism and malicious mischief if the dwelling has been vacant for more than 60 consecutive days immediately before the loss",
                           vo="excluded after 60 days vacant", vn="excluded after 60 days vacant"),
    "fire": T(title="Fire or Lightning", section="Section I Perils Insured Against B.1", page=6,
              old="We insure for direct physical loss caused by fire or lightning", vo="covered", vn="covered"),
    "smoke": T(title="Smoke", section="Section I Perils Insured Against B.7", page=7,
               old="We insure for sudden and accidental damage from smoke, but not smoke from agricultural smudging or industrial operations",
               vo="covered", vn="covered"),
    "debris_removal": T(title="Debris Removal", section="Section I Additional Coverages E.2", page=5,
                        old="We will pay your reasonable expense for the removal of debris of covered property after a covered loss",
                        vo="covered", vn="covered"),
    "trees_shrubs": T(title="Trees, Shrubs and Other Plants", section="Section I Additional Coverages E.3", page=5,
                      old="We cover trees, shrubs, plants or lawns on the residence premises for loss by fire, lightning, explosion, theft or vandalism, up to $500 for any one tree",
                      vo="covered up to $500 a tree", vn="covered up to $500 a tree"),
}
PAP_TOPICS = {
    "collision": T(title="Collision", section="Part D Coverage for Damage to Your Auto", page=8,
                   old="We will pay for loss to your covered auto caused by collision", vo="covered less deductible", vn="covered less deductible"),
    "comprehensive": T(title="Other Than Collision", section="Part D Coverage for Damage to Your Auto", page=8,
                       old="We will pay for loss to your covered auto caused by other than collision, including theft", vo="covered", vn="covered"),
    "deer_strike": T(title="Contact with an Animal", section="Part D Coverage for Damage to Your Auto", page=9,
                     old="Loss caused by contact with a bird or animal is treated as loss caused by other than collision", new="Contact with a deer or other animal is treated as other than collision", vo="other than collision", vn="other than collision, no collision deductible"),
    "glass": T(title="Glass Breakage", section="Part D Coverage for Damage to Your Auto", page=9,
               old="Breakage of glass may be paid under collision or other than collision", vo="your choice of coverage", vn="your choice of coverage"),
    "rental_reimbursement": T(title="Transportation Expenses", section="Part D Transportation Expenses", page=10,
                              old="We will pay up to $20 a day, to a maximum of $600, for temporary transportation expenses after a covered theft of your auto",
                              new="We will pay up to $30 a day, to a maximum of $900, for transportation expenses", vo="up to $20 a day after theft", vn="up to $30 a day"),
    "rideshare": T(title="Public or Livery Conveyance", section="Part A Exclusions A.5", page=4,
                   old="We do not provide liability coverage while the auto is used as a public or livery conveyance", vo="excluded", vn="excluded"),
    "towing": T(title="Towing and Labour", section="Part D Towing and Labour Costs", page=10,
                old="We will pay up to $75 for towing and labour costs each time your covered auto is disabled", vo="up to $75 a disablement", vn="up to $75 a disablement"),
    "uninsured_motorist": T(title="Uninsured Motorists", section="Part C", page=7, old="We will pay damages which an insured is legally entitled to recover from the owner or operator of an uninsured motor vehicle", vo="covered", vn="covered"),
    "medical_payments": T(title="Medical Payments", section="Part B", page=6, old="We will pay reasonable expenses for necessary medical services because of bodily injury caused by accident and sustained by an insured", vo="covered", vn="covered"),
    "liability_limit": T(title="Limit of Liability", section="Part A Limit of Liability", page=5, old="The limit of liability shown in the declarations for each person is our maximum for all damages from bodily injury to any one person", vo="per-person limit applies", vn="per-person limit applies"),
    "custom_equipment": T(title="Custom Equipment", section="Part D Limit of Liability", page=11, old="We will pay no more than $1,000 for custom furnishings or equipment in or upon a pickup or van", vo="limited to $1,000", vn="limited to $1,000"),
    "racing": T(title="Racing", section="Part D Exclusions 13", page=12, old="We will not pay for loss to your covered auto while it is located inside a facility designed for racing", vo="excluded", vn="excluded"),
    "business_use": T(title="Business Use", section="Part A Exclusions B.2", page=4, old="We do not provide liability coverage for the ownership or use of a vehicle while it is maintained or used in any business other than farming", vo="excluded", vn="excluded"),
    "territory": T(title="Policy Territory", section="Part F General Provisions", page=14, old="This policy applies only to accidents and losses which occur within the United States of America, its territories or possessions, or Canada", vo="US and Canada only", vn="US and Canada only"),
}
BOAT_TOPICS = {
    "navigation_limits": T(title="Navigation Limits", section="Section A Navigation", page=3,
                           old="Coverage applies only while the boat is afloat on the Great Lakes and their connecting waters", new="Coverage applies only while the boat is afloat on the Great Lakes, their connecting waters and inland lakes and rivers of the United States", vo="Great Lakes only", vn="Great Lakes and inland waters"),
    "ice_damage": T(title="Ice and Freezing", section="Section A Exclusions 4", page=5, old="We do not cover loss caused by ice or freezing unless the boat was laid up as the lay-up warranty requires", vo="excluded unless laid up", vn="excluded unless laid up"),
    "towing_assistance": T(title="Towing and Assistance", section="Section C Towing", page=7, old="We will pay up to $500 for towing when the boat is disabled on the water", vo="up to $500", vn="up to $500"),
    "theft_boat": T(title="Theft of the Boat", section="Section A Perils 2", page=4, old="We cover theft of the boat when there is visible evidence of forced entry", vo="covered with evidence of forced entry", vn="covered with evidence of forced entry"),
    "fuel_spill": T(title="Fuel Spill Liability", section="Section B Liability 3", page=6, old="We will pay clean-up costs required by law for a discharge of fuel from the boat", vo="covered", vn="covered"),
    "trailer": T(title="Boat Trailer", section="Section A Property Covered 3", page=3, old="We cover the boat trailer shown in the declarations while it is attached to or detached from a vehicle", vo="covered", vn="covered"),
    "lay_up": T(title="Lay-up Period", section="Section D Warranties 1", page=8, old="The boat must be laid up and out of commission from 1 November to 1 April", vo="warranty applies", vn="warranty applies"),
    "racing_boat": T(title="Racing", section="Section A Exclusions 7", page=5, old="We do not cover loss while the boat is used in any race or speed test", vo="excluded", vn="excluded"),
    "wear_tear": T(title="Wear and Tear", section="Section A Exclusions 1", page=5, old="We do not cover wear and tear, gradual deterioration, osmosis, blistering or marine life", vo="excluded", vn="excluded"),
    "mooring": T(title="Mooring", section="Section A Perils 5", page=4, old="We cover sinking while moored at a dock or mooring when the sinking is caused by a covered peril", vo="covered", vn="covered"),
}
BOP_TOPICS = {
    "employee_dishonesty": T(title="Dishonesty", section="Section I Exclusions B.2.c", page=12, old="We will not pay for loss resulting from dishonest acts by you or your employees", vo="excluded", vn="excluded"),
    "business_income": T(title="Business Income", section="Section I Additional Coverages 5.f", page=6, old="We will pay for the actual loss of business income due to a suspension of operations", vo="up to 12 months", vn="up to 12 months"),
    "equipment_breakdown": T(title="Mechanical Breakdown", section="Section I Exclusions B.2.d", page=12, old="We will not pay for loss caused by mechanical breakdown, including rupture or bursting caused by centrifugal force", vo="excluded", vn="excluded"),
    "spoilage_bop": T(title="Spoilage", section="Section I Optional Coverages G.4", page=9, old="We will pay for loss of perishable stock caused by breakdown of refrigeration equipment only if Spoilage Coverage is shown in the declarations", vo="optional", vn="optional"),
    "money_securities": T(title="Money and Securities", section="Section I Optional Coverages G.2", page=8, old="We will pay for loss of money and securities used in your business while at a bank or on your premises, up to the limit in the declarations", vo="optional", vn="optional"),
    "signs": T(title="Outdoor Signs", section="Section I Optional Coverages G.1", page=8, old="We will pay for direct physical loss of outdoor signs at the described premises, up to $2,500 for any one sign", vo="up to $2,500 a sign", vn="up to $2,500 a sign"),
    "utility_interruption": T(title="Utility Services", section="Section I Additional Coverages 5.k", page=7, old="We will pay for loss caused by the interruption of utility service to the described premises if the interruption results from direct physical loss away from the premises", vo="covered", vn="covered"),
    "ordinance": T(title="Ordinance or Law", section="Section I Exclusions B.1.a", page=11, old="We will not pay for loss caused by the enforcement of any ordinance or law regulating the construction, use or repair of any property", vo="excluded", vn="excluded"),
}
FORM_TOPICS = {"HO-3": HO_TOPICS, "PAP": PAP_TOPICS, "BOAT": BOAT_TOPICS, "BOP": BOP_TOPICS}
COND = {"HO-3": range(3, 12), "PAP": range(1, 9), "BOAT": range(1, 9), "BOP": range(1, 8)}
COND_TITLES = ["Duties After Loss", "Loss Payment", "Appraisal", "Other Insurance", "Suit Against Us",
               "Our Option", "Abandonment of Property", "Mortgage Clause", "No Benefit to Bailee",
               "Concealment or Fraud", "Policy Period"]

EVAL_TOPICS = {
    "homeowners": ["water_backup", "flood", "freezing", "hail_cosmetic", "ice_dam", "wind_hail", "tree_debris",
                   "mold", "roof_settlement", "roof_matching", "earth_movement", "food_spoilage", "theft",
                   "jewelry_limit", "collapse", "dog_bite", "ale", "vandalism_vacancy"],
    "auto": ["collision", "comprehensive", "deer_strike", "glass", "rental_reimbursement", "rideshare", "towing"],
    "marine": ["navigation_limits", "ice_damage", "towing_assistance", "theft_boat"],
    "commercial": ["employee_dishonesty", "business_income", "equipment_breakdown"],
}

# ------------------------------------------------------------------ endorsements
# doc_id: (line, [versions], topic it amends or None, holding, verdict, extra section names)
E = dict
ENDORSEMENTS = {
    "HO 04 95": E(line="homeowners", versions=["2019-03", "2024-06"], topic="water_backup", extras=[]),
    "HO 04 54": E(line="homeowners", versions=["2022-01"], topic="earth_movement", extras=["deductible", "definitions", "exclusions"],
                  title="Earthquake", holding="We insure for direct physical loss to covered property caused by earthquake, including land shock waves or tremors before, during or after a volcanic eruption", verdict="covered (earthquake)"),
    "HO 04 98": E(line="homeowners", versions=["2022-01"], topic="food_spoilage", extras=["limit", "deductible", "conditions"],
                  title="Refrigerated Property Coverage", holding="We insure up to $500 for loss to food in a freezer or refrigerator caused by interruption of electrical service, including a power failure off the residence premises", verdict="covered up to $500"),
    "LM HO 17": E(line="homeowners", versions=["2024-06"], topic="roof_settlement", extras=["definitions", "roof_matching", "conditions", "exclusions"],
                  title="Roof Replacement Cost Restoration", holding="Loss to roof surfacing is settled at replacement cost regardless of the age of the roof surfacing", verdict="replacement cost regardless of age"),
    "HO 04 61": E(line="homeowners", versions=["2019-03"], topic="jewelry_limit", extras=["limit", "conditions", "definitions"],
                  title="Scheduled Personal Property", holding="We cover the jewellery and watches described in the schedule for their scheduled amounts, and the special limit for theft of jewellery does not apply to them", verdict="scheduled amount"),
    "HO 04 90": E(line="homeowners", versions=["2019-03"], topic="theft", extras=["conditions", "definitions", "limit"],
                  title="Personal Property Replacement Cost", holding="Covered losses to personal property, including loss by theft, are settled at replacement cost without deduction for depreciation", verdict="replacement cost"),
    "LM HO 12": E(line="homeowners", versions=["2022-01"], topic="mold", extras=["limit", "conditions", "definitions"],
                  title="Limited Fungi Coverage", holding="We will pay up to $10,000 for loss caused by mold, fungus or wet rot that results from a covered water loss, whenever it is reported", verdict="covered up to $10,000"),
    "HO 04 81": E(line="homeowners", versions=["2022-01"], topic="ale", extras=["limit", "conditions"],
                  title="Additional Living Expense Extension", holding="We cover additional living expense for up to 24 months after a covered loss, even if repairs could be completed sooner", verdict="up to 24 months"),
    "HO 04 50": E(line="homeowners", versions=["2019-03"], topic="vandalism_vacancy", extras=["conditions", "schedule2"],
                  title="Vacancy Permission", holding="Vandalism and malicious mischief remain covered while the dwelling is vacant during the permission period shown in the schedule", verdict="covered during the permission period"),
    "LM HO 30": E(line="homeowners", versions=["2024-06"], topic="dog_bite", extras=["definitions", "conditions", "limit", "exclusions"],
                  title="Animal Liability Limitation", holding="Personal liability for bodily injury caused by any dog owned by an insured is limited to $25,000 for each occurrence", verdict="limited to $25,000"),
    "LM HO 21": E(line="homeowners", versions=["2024-06"], topic="ice_dam", extras=["conditions", "limit"],
                  title="Ice Dam Removal", holding="We will pay up to $2,000 for the reasonable cost of removing an ice dam that is causing water to enter the dwelling", verdict="removal covered up to $2,000"),
    "LM HO 33": E(line="homeowners", versions=["2024-06"], topic="hail_cosmetic", extras=["conditions", "definitions", "deductible"],
                  title="Cosmetic Hail Buy-Back", holding="The cosmetic hail exclusion does not apply to metal roof surfacing", verdict="covered subject to a 2% deductible"),
    "PP 03 02": E(line="auto", versions=["2023-01"], topic="rental_reimbursement", extras=["limit", "conditions", "definitions"],
                  title="Extended Transportation Expenses", holding="We will pay up to $50 a day, to a maximum of $1,500, for a rental vehicle while your covered auto is out of use after any covered loss", verdict="up to $50 a day"),
    "PP 03 13": E(line="auto", versions=["2018-09"], topic="glass", extras=["deductible", "conditions"],
                  title="Full Glass Coverage", holding="No deductible applies to the repair or replacement of glass in your covered auto", verdict="no deductible"),
    "LM PP 40": E(line="auto", versions=["2023-01"], topic="rideshare", extras=["definitions", "limit", "conditions", "exclusions"],
                  title="Transportation Network Driver Coverage", holding="The public or livery conveyance exclusion does not apply while you are logged in to a transportation network company platform and not yet matched with a passenger", verdict="covered while logged in and unmatched"),
    "PP 03 03": E(line="auto", versions=["2018-09"], topic="towing", extras=["limit", "conditions"],
                  title="Roadside Assistance", holding="We will pay up to $150 for towing and labour, including lock-out service", verdict="up to $150 a disablement"),
    "BT 05": E(line="marine", versions=["2021-05"], topic="navigation_limits", extras=["definitions", "conditions", "schedule2", "exclusions"],
               title="Extended Navigation", holding="The navigation limits are extended to the coastal waters of the United States and Canada within 50 miles of shore", verdict="coastal waters within 50 miles"),
    "BT 12": E(line="marine", versions=["2021-05"], topic="ice_damage", extras=["conditions", "definitions"],
               title="In-Water Winter Storage", holding="Loss caused by ice or freezing is covered while the boat is stored in the water with a working de-icing system", verdict="covered with a de-icer"),
    "BT 20": E(line="marine", versions=["2025-03"], topic="towing_assistance", extras=["limit", "conditions", "definitions"],
               title="Towing and Salvage Extension", holding="We will pay up to $2,500 for towing, salvage and emergency fuel delivery when the boat is disabled on the water", verdict="up to $2,500"),
    "BP 04 50": E(line="commercial", versions=["2025-01"], topic="employee_dishonesty", extras=["limit", "definitions", "conditions", "exclusions"],
                  title="Employee Dishonesty", holding="We will pay for loss of money and business personal property from employee dishonesty", verdict="covered up to $25,000"),
    "BP 04 30": E(line="commercial", versions=["2020-04"], topic="equipment_breakdown", extras=["limit", "definitions", "conditions", "deductible"],
                  title="Equipment Breakdown Protection", holding="We will pay for direct physical damage to covered property caused by a breakdown of covered equipment, including mechanical breakdown", verdict="covered"),
    "BP 05 15": E(line="commercial", versions=["2025-01"], topic="business_income", extras=["limit", "conditions"],
                  title="Extended Business Income", holding="The period of business income coverage is extended to 18 months, and to 60 days after operations resume", verdict="up to 18 months"),
}
HO0495 = {
    "2019-03": "We will pay up to $5,000 for direct physical loss caused by water or waterborne material which backs up through sewers or drains or overflows or is discharged from a sump, sump pump or related equipment",
    "2024-06": "We will pay up to $10,000 for direct physical loss caused by water or waterborne material which backs up through sewers or drains or overflows or is discharged from a sump, sump pump or related equipment, even if caused by mechanical breakdown of the sump pump",
}
HO0495_TAIL = {"2019-03": "A $500 deductible applies.",
               "2024-06": "A $1,000 deductible applies. We do not pay for loss caused by your negligent failure to maintain the sump pump."}
HO0495_VERDICT = {"2019-03": "covered up to $5,000", "2024-06": "covered up to $10,000"}

# ------------------------------------------------------------------ bulletins (3 spans each)
# doc_id, state, effective_from, effective_to (exclusive) or None, topic, holding, verdict, superseded_by
BULLETINS = [
    ("WI Bulletin 25-03", "WI", date(2025, 3, 1), None, "roof_matching",
     "Insurers must replace undamaged roof surfacing on adjoining slopes when the replacement material does not reasonably match in colour or texture", "matching required on adjoining slopes", None),
    ("WI Bulletin 22-04", "WI", date(2022, 4, 1), date(2025, 3, 1), "roof_matching",
     "Insurers should consider matching of adjoining roof surfacing when settling a partial roof loss", "matching to be considered", "WI Bulletin 25-03"),
    ("MN Bulletin 24-09", "MN", date(2024, 9, 1), None, "hail_cosmetic",
     "An exclusion for cosmetic hail damage to roof surfacing may not be applied to a homeowners policy in force in Minnesota", "covered (exclusion unenforceable)", None),
    ("MN Bulletin 21-02", "MN", date(2021, 2, 1), date(2024, 9, 1), "hail_cosmetic",
     "Insurers must disclose any cosmetic hail limitation on the declarations page at renewal", "disclosure required", "MN Bulletin 24-09"),
    ("IA Bulletin 23-07", "IA", date(2023, 7, 1), None, "deer_strike",
     "A loss caused by a collision with a deer must be adjusted under other than collision coverage without a collision deductible", "other than collision, no collision deductible", None),
    ("IA Bulletin 20-11", "IA", date(2020, 11, 1), date(2023, 7, 1), "deer_strike",
     "Insurers are reminded that animal collisions are ordinarily covered under other than collision coverage", "other than collision", "IA Bulletin 23-07"),
    ("IN Bulletin 24-03", "IN", date(2024, 3, 1), None, "rideshare",
     "A personal auto policy may not exclude coverage while a driver is logged in to a transportation network platform without a matched passenger", "covered while logged in and unmatched", None),
    ("MI Bulletin 23-10", "MI", date(2023, 10, 1), None, "navigation_limits",
     "Coverage for a boat on Michigan inland lakes may not be excluded solely by a Great Lakes navigation limit", "inland lakes covered", None),
    ("MI Bulletin 25-01", "MI", date(2025, 1, 1), None, "ice_dam",
     "Insurers must pay the reasonable cost of removing an ice dam that is causing interior water damage", "removal covered", None),
    ("IL Bulletin 25-05", "IL", date(2025, 5, 1), None, "freezing",
     "A frozen pipe loss may not be denied for failure to maintain heat when the heating system failed without the insured's knowledge", "covered if heating failed unknowingly", None),
]
BULLETIN_LINE = {"roof_matching": "homeowners", "hail_cosmetic": "homeowners", "deer_strike": "auto",
                 "rideshare": "auto", "navigation_limits": "marine", "ice_dam": "homeowners", "freezing": "homeowners"}

# ------------------------------------------------------------------ guidance and notes
GUIDANCE = [
    ("NAIC MDL-902", "2021-04", ["roof_matching", "roof_settlement", "hail_cosmetic", "wind_hail", "ice_dam"]),
    ("NAIC MDL-900", "2019-08", ["freezing", "water_backup", "mold", "flood"]),
    ("NAIC Bulletin 23-1", "2023-01", ["rideshare", "collision", "deer_strike", "towing"]),
]
NOTES = [
    ("2020-08-31", "water_backup", "Reminder from the team: sewer backup is never covered on HO-3, decline add-on requests unless the dwelling has a working backflow valve."),
    ("2022-10-26", "water_backup", "Water back-up endorsement pays $5,000 with a $500 deductible; push it at renewal for finished basements."),
    ("2019-11-14", "roof_settlement", "Older roofs still get full replacement cost on HO-3, so inspect before binding anything over 20 years."),
    ("2021-04-02", "hail_cosmetic", "Cosmetic hail on metal roofs is excluded on the new HO-3 everywhere, including Minnesota; no premium credit."),
    ("2023-02-17", "earth_movement", "Earthquake is excluded on every homeowners form; we do not write the endorsement in Wisconsin."),
    ("2022-06-08", "freezing", "Frozen pipes are always paid if the furnace failed; do not ask about heat."),
    ("2020-01-21", "food_spoilage", "Freezer food is covered up to $500 on all homeowners policies after any outage."),
    ("2021-09-30", "roof_matching", "We always match the whole roof in Wisconsin; tell agents to expect it."),
    ("2019-05-12", "navigation_limits", "Boat policies cover inland lakes too; the Great Lakes wording is old."),
    ("2024-01-09", "rideshare", "Rideshare drivers are excluded, full stop, in every state."),
    ("2022-03-03", "employee_dishonesty", "Employee theft is covered on the BOP; sell the endorsement only for large payrolls."),
    ("2020-10-19", "mold", "Mold is excluded on HO-3 with no exceptions."),
    ("2023-08-25", "deer_strike", "Deer hits go under collision; the deductible applies."),
    ("2021-12-01", "ice_dam", "Ice dam removal is paid on all homeowners policies."),
]

FIRST = ["Emily", "Eric", "Sarah", "James", "Linda", "Michael", "Karen", "David", "Susan", "Robert", "Laura",
         "Daniel", "Nancy", "Paul", "Megan", "Kevin", "Rachel", "Brian", "Julie", "Mark", "Anna", "Scott",
         "Heather", "Greg", "Amy", "Tom", "Diane", "Ryan", "Carol", "Peter"]
LAST = ["Carter", "Reid", "Olson", "Larson", "Nielsen", "Schmidt", "Kowalski", "Brooks", "Meyer", "Hansen",
        "Novak", "Jensen", "Fischer", "Peterson", "Walsh", "Dahl", "Becker", "Lund", "Murphy", "Krueger",
        "Hoffman", "Sorensen", "Weber", "Nash", "Quinn", "Ryan", "Klein", "Vogel", "Berg", "Moore"]
ADJUSTERS = ["tbrennan", "mhale", "jkowal", "rlund", "sdahl", "pmeyer", "kwalsh", "anovak"]

QUESTIONS = {
    "water_backup": ["Water came up through the basement floor drain during the storm last night. Is the sewer back-up covered?",
                     "The sump pump failed and the finished basement flooded. Do we pay for the water damage?",
                     "Sewage backed up into the lower level after heavy rain. What does the policy pay?"],
    "flood": ["The river overflowed and water came in under the doors. Is the flood damage covered?",
              "Surface water from the street ran into the garage. Do we pay?"],
    "freezing": ["Frozen pipe burst during the January cold snap after the furnace went out. Do we pay?",
                 "A pipe froze and split in the crawl space while the owners were away. Is it covered?"],
    "hail_cosmetic": ["Hail dented the metal roof but nothing leaks. Is the cosmetic damage covered?",
                      "The adjuster found hail marks on the steel roof panels with no water entry. Do we pay?"],
    "ice_dam": ["An ice dam pushed water through the ceiling. What do we pay, and is removal covered?",
                "Water from an ice dam stained the bedroom wall. Is the ice removal covered too?"],
    "wind_hail": ["Wind tore shingles off and rain came into the attic. Is the interior damage covered?",
                  "A hailstorm broke the skylight and soaked the carpet. Do we pay?"],
    "tree_debris": ["A tree fell on the garage in the windstorm. How much do we pay to remove it?",
                    "Ice brought down an oak across the driveway. Is the removal covered?"],
    "mold": ["Mold showed up behind the dishwasher months after a leak. Is it covered?",
             "The basement has mold after last spring's water loss. Do we pay for remediation?"],
    "roof_settlement": ["How do we settle the roof surfacing loss on an older roof after the windstorm?",
                        "The roof is 18 years old and the hail damaged it. Replacement cost or actual cash value?"],
    "roof_matching": ["Only the north slope was damaged but the shingles are discontinued. Do we replace the whole roof?",
                      "The new shingles will not match the undamaged slopes. Must we pay to match?"],
    "earth_movement": ["The foundation cracked after the ground shifted. Is the earth movement covered?",
                       "A small earthquake cracked the basement walls. Do we pay?"],
    "food_spoilage": ["The power was out for three days after the storm and the freezer food spoiled. Is it covered?",
                      "A utility outage ruined everything in the chest freezer. Do we pay for the food?"],
    "theft": ["A bicycle and tools were stolen from the shed. How is the theft settled?",
              "Electronics were stolen in a break-in. Do we pay replacement cost?"],
    "jewelry_limit": ["A diamond ring was stolen from the bedroom. How much can we pay?",
                      "Watches worth $9,000 were taken in a burglary. What is the limit?"],
    "collapse": ["Snow load collapsed part of the garage roof. Is the collapse covered?"],
    "dog_bite": ["The insured's dog bit a neighbour's child. Is the liability claim covered, and up to what?",
                 "A delivery driver was bitten by the insured's dog. What does liability pay?"],
    "ale": ["The family has to live in a hotel during repairs. How long do we pay living expenses?",
            "Repairs will take five months. Is the rental house covered for the whole time?"],
    "vandalism_vacancy": ["The house was vacant for three months and was vandalised. Is it covered?",
                          "Vandals broke in while the home sat empty after the owner moved. Do we pay?"],
    "collision": ["The insured backed into a pole. What does collision pay?",
                  "The car slid on ice into a guardrail. Is it covered, and what deductible applies?"],
    "comprehensive": ["A falling branch smashed the hood in the driveway. Which coverage pays?",
                      "The car was stolen from the mall lot and found stripped. Is it covered?"],
    "deer_strike": ["The insured hit a deer on the county road. Collision or comprehensive, and which deductible?",
                    "A deer ran into the side of the truck at dusk. What coverage applies?"],
    "glass": ["A stone cracked the windshield on the interstate. What do we pay?",
              "The rear window shattered in the cold. Does a deductible apply?"],
    "rental_reimbursement": ["The car is in the body shop for two weeks. Do we pay for a rental, and how much?",
                             "The insured needs a rental after the collision. What is the daily limit?"],
    "rideshare": ["The insured was driving for a rideshare app when the crash happened. Is liability covered?",
                  "The driver was logged in to the app waiting for a ride request. Are we on the hook?"],
    "towing": ["The car broke down on the highway and was towed. How much do we reimburse?",
               "The insured was locked out and needed a tow. What does the policy pay?"],
    "navigation_limits": ["The boat was damaged on an inland lake in Wisconsin. Is it within the navigation limits?",
                          "The insured took the boat down the Mississippi and hit a log. Is it covered?"],
    "ice_damage": ["Ice crushed the hull while the boat was left in the water in December. Is it covered?",
                   "The outdrive froze and cracked over the winter. Do we pay?"],
    "towing_assistance": ["The boat lost power on the lake and had to be towed in. How much do we pay?",
                          "The insured ran out of fuel and needed a tow and salvage. What is covered?"],
    "theft_boat": ["The boat was stolen from the marina. What do we need to pay the theft?",
                   "The outboard motor was taken off the boat overnight. Is it covered?"],
    "employee_dishonesty": ["An employee stole cash from the register over several months. Is the theft covered?",
                            "The bookkeeper embezzled from the business account. Do we pay?"],
    "business_income": ["The bakery was closed for 14 months after the fire. How long do we pay lost income?",
                        "The shop lost a year of income after the storm. What is the limit on time?"],
    "equipment_breakdown": ["The walk-in cooler compressor broke down. Is the breakdown covered?",
                            "A motor in the production line failed mechanically. Do we pay?"],
}


# ================================================================== builders
def form_spans() -> list[dict]:
    """One dict per form span: id, kind, doc, version, topic, section, page, title, holding, verdict."""
    out = []
    for doc, (line, eds, n) in FORMS.items():
        topics = FORM_TOPICS[doc]
        conds = list(COND[doc])
        assert len(topics) + len(conds) == n, (doc, len(topics), len(conds))
        for i, (ver, eff) in enumerate(eds):
            latest = i == len(eds) - 1
            sup = None if latest else eds[i + 1][0]
            for t, d in topics.items():
                hold = d.get("new", d["old"]) if latest else d["old"]
                verdict = d["vn"] if latest else d["vo"]
                out.append(dict(span_id=f"{doc}@{ver}#{t}", kind="form", doc_id=doc, version=ver, line=line,
                                effective_from=eff, superseded_by=f"{doc} ({sup})" if sup else None,
                                topic=t, section=d["section"], page=d["page"], title=d["title"],
                                holding=hold, verdict=verdict, precedence=10))
            for j, c in enumerate(conds):
                title = COND_TITLES[(c - 1) % len(COND_TITLES)]
                sec = {"HO-3": "Section I Conditions", "PAP": "Part E Duties", "BOAT": "Section E Conditions",
                       "BOP": "Section III Common Policy Conditions"}[doc] + f" {chr(64 + c)}"
                out.append(dict(span_id=f"{doc}@{ver}#cond-{c}", kind="form", doc_id=doc, version=ver, line=line,
                                effective_from=eff, superseded_by=f"{doc} ({sup})" if sup else None,
                                topic=None, section=sec, page=2 + (c % 5), title=title,
                                holding=f"The {title.lower()} condition applies to every claim under this policy",
                                verdict="", precedence=10))
    return out


ENDO_EXTRA_TITLES = {"deductible": "Deductible", "definitions": "Definitions", "exclusions": "Exclusions",
                     "limit": "Limit of Insurance", "conditions": "Conditions", "schedule2": "Schedule of Locations",
                     "roof_matching": "Matching", "conditions2": "Additional Conditions", "definitions2": "Additional Definitions"}
N_ENDO_SPANS = 118


def endorsement_spans() -> list[dict]:
    out = []
    extras_total = sum(2 + len(e["extras"]) for d, e in ENDORSEMENTS.items() if d != "HO 04 95") + 4
    need = N_ENDO_SPANS - extras_total
    padded = [d for d in ENDORSEMENTS if d != "HO 04 95"][:need]
    for doc, e in ENDORSEMENTS.items():
        for i, ver in enumerate(e["versions"]):
            yr, mo = map(int, ver.split("-"))
            eff = date(yr, mo, 1)
            sup = f"{doc} ({e['versions'][i + 1]})" if i + 1 < len(e["versions"]) else None
            base = dict(kind="endorsement", doc_id=doc, version=ver, line=e["line"], effective_from=eff,
                        superseded_by=sup, page=1, precedence=30)
            if doc == "HO 04 95":
                title = "Water Back-Up and Sump Discharge or Overflow"
                out.append(dict(base, span_id=f"{doc}@{ver}#schedule", topic=None, section="Schedule", title=title,
                                holding="This endorsement changes the policy", verdict="", fixed_text=(
                                    f"{title}. This endorsement changes the policy. It modifies insurance provided "
                                    f"under the Homeowners 3 Special Form (HO-3). All other provisions of the policy apply. "
                                    f"[{doc} ({ver}), page 1]")))
                out.append(dict(base, span_id=f"{doc}@{ver}#water_backup", topic="water_backup", section="Coverage",
                                title=title, holding=HO0495[ver], verdict=HO0495_VERDICT[ver], fixed_text=(
                                    f"{title}. {HO0495[ver]}. {HO0495_TAIL[ver]} [{doc} ({ver}), page 1]")))
                continue
            title = e["title"]
            out.append(dict(base, span_id=f"{doc}@{ver}#schedule", topic=None, section="Schedule", title=title,
                            holding="This endorsement changes the policy", verdict=""))
            out.append(dict(base, span_id=f"{doc}@{ver}#{e['topic']}", topic=e["topic"], section="Coverage",
                            title=title, holding=e["holding"], verdict=e["verdict"], page=1))
            extras = list(e["extras"]) + (["conditions2" if "conditions2" not in e["extras"] else "definitions2"]
                                          if doc in padded else [])
            for j, x in enumerate(extras):
                t = ENDO_EXTRA_TITLES[x]
                out.append(dict(base, span_id=f"{doc}@{ver}#{x}", topic=None, section=t, title=title,
                                holding=f"The {t.lower()} in this endorsement apply in addition to those of the policy",
                                verdict="", page=1 + (j + 1) // 2))
    assert len(out) == N_ENDO_SPANS, len(out)
    return out


def bulletin_spans() -> list[dict]:
    out = []
    for doc, st, eff, end, topic, hold, verdict, sup in BULLETINS:
        ver = f"{eff.year}-{eff.month:02d}"
        base = dict(kind="bulletin", doc_id=doc, version=ver, line=BULLETIN_LINE[topic], effective_from=eff,
                    effective_to=end, superseded_by=sup, jurisdiction=st, precedence=50,
                    source_system="regulatory-feed")
        out.append(dict(base, span_id=f"{doc}@{ver}#{topic}", topic=topic, section="Directive", page=1,
                        title=f"{st} Bulletin", holding=hold, verdict=verdict))
        out.append(dict(base, span_id=f"{doc}@{ver}#scope", topic=None, section="Scope", page=1,
                        title=f"{st} Bulletin", holding="This bulletin applies to every insurer admitted in the state", verdict=""))
        out.append(dict(base, span_id=f"{doc}@{ver}#effective", topic=None, section="Effective Date", page=2,
                        title=f"{st} Bulletin", holding=f"This bulletin applies to losses on or after {eff.isoformat()}", verdict=""))
    return out


def guidance_spans() -> list[dict]:
    out = []
    for doc, ver, topics in GUIDANCE:
        yr, mo = map(int, ver.split("-"))
        for i, t in enumerate(topics):
            out.append(dict(span_id=f"{doc}@{ver}#{t}", kind="guidance", doc_id=doc, version=ver, line=None,
                            effective_from=date(yr, mo, 1), topic=t, section=f"Section {i + 1}", page=i + 1,
                            title="Model guidance", holding=f"Insurers should explain the settlement of {t.replace('_', ' ')} claims in writing",
                            verdict="", precedence=5, source_system="naic"))
    return out


def note_spans() -> list[dict]:
    out = []
    for i, (d, topic, text) in enumerate(NOTES, 1):
        yr, mo, dy = map(int, d.split("-"))
        out.append(dict(span_id=f"UW-NOTE-{i:04d}", kind="uw_note", doc_id=f"UW-NOTE-{i:04d}", version=d, line=None,
                        effective_from=date(yr, mo, dy), topic=topic, section=None, page=None, title="",
                        holding=text.rstrip("."), verdict="", precedence=0, entitlement=["underwriting"],
                        source_system="sharepoint", trust=0.6, fixed_text=text))
    return out


def all_span_specs() -> list[dict]:
    return form_spans() + endorsement_spans() + bulletin_spans() + guidance_spans() + note_spans()


# ================================================================== determinations
EDITIONS = {"HO-3 (2024-06)": 62, "HO-3 (2022-01)": 51, "PAP (2023-01)": 38, "BOAT (2021-05)": 12,
            "BOAT (2025-03)": 10, "PAP (2018-09)": 10, "BOP (2025-01)": 9, "BOP (2020-04)": 8}
STATE_N = {"WI": 85, "IL": 29, "MI": 26, "MN": 26, "IA": 18, "IN": 16}
TOPIC_N = {"water_backup": 13, "freezing": 13, "roof_settlement": 13, "collapse": 1, "hail_cosmetic": 7, "ice_dam": 5,
           "wind_hail": 5, "tree_debris": 4, "mold": 5, "roof_matching": 7, "earth_movement": 8, "food_spoilage": 6,
           "theft": 4, "jewelry_limit": 4, "dog_bite": 4, "ale": 4, "vandalism_vacancy": 4, "flood": 6,
           "collision": 8, "comprehensive": 7, "deer_strike": 8, "glass": 6, "rental_reimbursement": 7,
           "rideshare": 6, "towing": 6, "navigation_limits": 8, "ice_damage": 5, "towing_assistance": 5,
           "theft_boat": 4, "employee_dishonesty": 7, "business_income": 5, "equipment_breakdown": 5}
LINE_OF_TOPIC = {t: ln for ln, ts in EVAL_TOPICS.items() for t in ts}
ENDO_FOR_TOPIC = {e["topic"]: d for d, e in ENDORSEMENTS.items()}

# pinned working-set rows: id -> (state, form, topic, gold kind, endorsements, extra)
PINNED = {
    1: ("WI", "HO-3 (2024-06)", "water_backup", "endorsement", ["HO 04 95 (2024-06)"],
        dict(claimant="Emily Carter", claim_id="CLM-2026-01187", policy_number="LMH-WI-0048213",
             date_of_loss=date(2026, 3, 14), adjuster="tbrennan",
             question="Water came up through the basement floor drain during the storm last night. Is the sewer back-up covered?")),
    2: ("MI", "BOAT (2021-05)", "navigation_limits", "form", [], dict(date_of_loss=date(2023, 6, 17))),
    3: ("IL", "BOP (2020-04)", "employee_dishonesty", "form", [], {}),
    4: ("IA", "BOP (2025-01)", "employee_dishonesty", "form", [], {}),
    8: ("MN", "HO-3 (2024-06)", "roof_settlement", "form", ["HO 04 54 (2022-01)"],
        dict(claimant="Eric Reid", date_of_loss=date(2025, 8, 14),
             question="How do we settle the roof surfacing loss on an older roof after the windstorm?")),
    22: ("WI", "HO-3 (2022-01)", "freezing", "form", ["HO 04 54 (2022-01)"],
         dict(question="Frozen pipe burst during the January cold snap after the furnace went out. Do we pay?")),
    63: ("WI", "HO-3 (2022-01)", "freezing", "form", ["HO 04 95 (2019-03)"], {}),
    163: ("WI", "HO-3 (2022-01)", "freezing", "form", ["LM HO 12 (2022-01)"], {}),
    178: ("WI", "HO-3 (2022-01)", "freezing", "form", ["HO 04 81 (2022-01)"], {}),
    181: ("IL", "HO-3 (2022-01)", "freezing", "form", ["HO 04 54 (2022-01)"], dict(date_of_loss=date(2025, 1, 21))),
    53: ("IN", "HO-3 (2022-01)", "freezing", "form", ["HO 04 98 (2022-01)"], {}),
    83: ("IN", "HO-3 (2022-01)", "freezing", "form", ["HO 04 98 (2022-01)"], {}),
    134: ("WI", "BOP (2020-04)", "employee_dishonesty", "form", [], {}),
    75: ("IN", "HO-3 (2024-06)", "roof_matching", "form", [], {}),
    49: ("WI", "BOAT (2025-03)", "navigation_limits", "endorsement", ["BT 05 (2021-05)"], {}),
    69: ("MN", "HO-3 (2022-01)", "water_backup", "endorsement", ["HO 04 95 (2019-03)"], {}),
    71: ("IL", "HO-3 (2024-06)", "earth_movement", "endorsement", ["HO 04 54 (2022-01)"], {}),
    113: ("IL", "HO-3 (2022-01)", "earth_movement", "endorsement", ["HO 04 54 (2022-01)"], {}),
    131: ("WI", "HO-3 (2024-06)", "earth_movement", "endorsement", ["HO 04 54 (2022-01)"], {}),
    145: ("WI", "HO-3 (2024-06)", "food_spoilage", "endorsement", ["HO 04 98 (2022-01)"], {}),
    185: ("MI", "HO-3 (2024-06)", "food_spoilage", "endorsement", ["HO 04 98 (2022-01)"], {}),
    188: ("WI", "HO-3 (2024-06)", "roof_settlement", "endorsement", ["LM HO 17 (2024-06)"], {}),
    # the six bulletin-governed determinations
    12: ("MN", "HO-3 (2024-06)", "hail_cosmetic", "bulletin", ["HO 04 95 (2024-06)"], dict(date_of_loss=date(2025, 6, 2))),
    37: ("WI", "HO-3 (2024-06)", "roof_matching", "bulletin", [], dict(date_of_loss=date(2025, 7, 9))),
    91: ("IL", "HO-3 (2024-06)", "freezing", "bulletin", ["HO 04 95 (2024-06)"], dict(date_of_loss=date(2026, 1, 12))),
    127: ("MI", "HO-3 (2024-06)", "ice_dam", "bulletin", [], dict(date_of_loss=date(2026, 2, 3))),
    150: ("IA", "PAP (2023-01)", "deer_strike", "bulletin", ["PP 03 02 (2023-01)"], dict(date_of_loss=date(2025, 10, 30))),
    172: ("MI", "BOAT (2025-03)", "navigation_limits", "bulletin", [], dict(date_of_loss=date(2025, 7, 19))),
}
# (state, topic) pairs a bulletin governs over some window: other determinations must avoid them
BULLETIN_BLOCK = {("WI", "roof_matching"), ("MN", "hail_cosmetic"), ("IA", "deer_strike"), ("IN", "rideshare"),
                  ("IL", "freezing"), ("MI", "ice_dam"), ("MI", "navigation_limits")}

DOL_WINDOW = {"HO-3 (2024-06)": (date(2024, 7, 1), date(2026, 6, 30)), "HO-3 (2022-01)": (date(2022, 3, 1), date(2025, 6, 30)),
              "PAP (2023-01)": (date(2023, 2, 1), date(2026, 6, 30)), "PAP (2018-09)": (date(2018, 10, 1), date(2023, 6, 30)),
              "BOAT (2021-05)": (date(2021, 6, 1), date(2025, 6, 30)), "BOAT (2025-03)": (date(2025, 4, 1), date(2026, 6, 30)),
              "BOP (2020-04)": (date(2020, 5, 1), date(2025, 3, 31)), "BOP (2025-01)": (date(2025, 2, 1), date(2026, 6, 30))}
FORM_LINE = {"HO-3": "homeowners", "PAP": "auto", "BOAT": "marine", "BOP": "commercial"}
LINE_EDITIONS = {"homeowners": ["HO-3 (2024-06)", "HO-3 (2022-01)"], "auto": ["PAP (2023-01)", "PAP (2018-09)"],
                 "marine": ["BOAT (2021-05)", "BOAT (2025-03)"], "commercial": ["BOP (2025-01)", "BOP (2020-04)"]}
LINE_CODE = {"homeowners": "H", "auto": "A", "marine": "M", "commercial": "C"}


def endo_label(doc: str, form: str) -> str:
    vers = ENDORSEMENTS[doc]["versions"]
    if doc == "HO 04 95":
        return f"{doc} ({'2024-06' if form == 'HO-3 (2024-06)' else '2019-03'})"
    return f"{doc} ({vers[-1]})"


def _rand_date(r: random.Random, lo: date, hi: date) -> date:
    return date.fromordinal(r.randint(lo.toordinal(), hi.toordinal()))


def _simple_endorsements(r: random.Random, line: str, form: str, topic: str) -> list[str]:
    docs = [d for d, e in ENDORSEMENTS.items() if e["line"] == line and e["topic"] != topic]
    n = 1 if r.random() < 0.75 else 2
    return sorted(endo_label(d, form) for d in r.sample(docs, n))


def build_working() -> list[dict]:
    r = rng("working")
    topic_left = dict(TOPIC_N)
    ed_left = dict(EDITIONS)
    st_left = dict(STATE_N)
    for st, form, topic, kind, ends, extra in PINNED.values():
        topic_left[topic] -= 1
        ed_left[form] -= 1
        st_left[st] -= 1
    assert min(topic_left.values()) >= 0 and min(ed_left.values()) >= 0 and min(st_left.values()) >= 0
    # slots: (edition, topic) for the unpinned
    slots = []
    for line, eds in LINE_EDITIONS.items():
        tops = [t for t in EVAL_TOPICS[line] for _ in range(topic_left[t])]
        r.shuffle(tops)
        ed_list = [e for e in eds for _ in range(ed_left[e])]
        assert len(tops) == len(ed_list), line
        slots += list(zip(ed_list, tops))
    r.shuffle(slots)
    endo_target = {"HO-3 (2024-06)": 8, "HO-3 (2022-01)": 12, "auto": 6, "marine": 3, "commercial": 3}
    kinds = []
    for ed, t in slots:
        line = LINE_OF_TOPIC[t]
        key = ed if line == "homeowners" else line
        if t in ENDO_FOR_TOPIC and ENDORSEMENTS[ENDO_FOR_TOPIC[t]]["line"] == line and endo_target.get(key, 0) > 0:
            endo_target[key] -= 1
            kinds.append("endorsement")
        else:
            kinds.append("form")
    assert not any(endo_target.values()), endo_target
    simple_idx = [i for i, k in enumerate(kinds) if k == "form"]
    with_end = set(r.sample(simple_idx, 56))
    # states, avoiding bulletin-governed (state, topic) pairs
    states = [s for s, n in st_left.items() for _ in range(n)]
    for _ in range(1000):
        r.shuffle(states)
        if not any((states[i], slots[i][1]) in BULLETIN_BLOCK for i in range(len(slots))):
            break
    else:
        raise RuntimeError("state assignment")
    free_ids = [i for i in range(1, 201) if i not in PINNED]
    rows = {}
    for n, i in enumerate(free_ids):
        ed, t = slots[n]
        line = LINE_OF_TOPIC[t]
        if kinds[n] == "endorsement":
            ends = sorted({endo_label(ENDO_FOR_TOPIC[t], ed)} | (set(_simple_endorsements(r, line, ed, t)[:1]) if r.random() < 0.3 else set()))
        elif n in with_end:
            ends = _simple_endorsements(r, line, ed, t)
        else:
            ends = []
        rows[i] = (states[n], ed, t, kinds[n], ends, {})
    rows.update(PINNED)
    return [_det_row(i, *rows[i], r=r) for i in range(1, 201)]


def build_holdout() -> list[dict]:
    """The 300 held out: 232 simple, 68 complex (62 endorsement, 6 bulletin); 174 with endorsements."""
    r = rng("holdout")
    lines = ["homeowners"] * 170 + ["auto"] * 72 + ["marine"] * 33 + ["commercial"] * 25
    r.shuffle(lines)
    bull = [("MN", "HO-3 (2024-06)", "hail_cosmetic", date(2025, 7, 1)), ("WI", "HO-3 (2024-06)", "roof_matching", date(2025, 9, 3)),
            ("IL", "HO-3 (2024-06)", "freezing", date(2026, 1, 30)), ("IA", "PAP (2023-01)", "deer_strike", date(2024, 9, 9)),
            ("IN", "PAP (2023-01)", "rideshare", date(2025, 3, 3)), ("MI", "HO-3 (2024-06)", "ice_dam", date(2025, 12, 20))]
    states = [s for s, n in STATE_N.items() for _ in range(round(n * 1.5))][:300]
    while len(states) < 300:
        states.append("WI")
    r.shuffle(states)
    rows = []
    n_endo, n_with_simple = 62, 174 - 62 - 3
    for i in range(300):
        did = 201 + i
        if i < 6:
            st, ed, t, dol = bull[i]
            ends = _simple_endorsements(r, FORM_LINE[ed.split(" (")[0]], ed, t) if i < 3 else []
            rows.append(_det_row(did, st, ed, t, "bulletin", ends, dict(date_of_loss=dol), r=r))
            continue
        line = lines[i]
        ed = r.choice(LINE_EDITIONS[line])
        topics = EVAL_TOPICS[line]
        st = states[i]
        if n_endo > 0 and (300 - i) * r.random() < n_endo * 2.2:
            cand = [t for t in topics if t in ENDO_FOR_TOPIC and ENDORSEMENTS[ENDO_FOR_TOPIC[t]]["line"] == line
                    and (st, t) not in BULLETIN_BLOCK]
            if cand:
                t = r.choice(cand)
                n_endo -= 1
                rows.append(_det_row(did, st, ed, t, "endorsement", [endo_label(ENDO_FOR_TOPIC[t], ed)], {}, r=r))
                continue
        cand = [t for t in topics if (st, t) not in BULLETIN_BLOCK]
        t = r.choice(cand)
        ends = _simple_endorsements(r, line, ed, t) if n_with_simple > 0 and r.random() < 0.5 else []
        if ends:
            n_with_simple -= 1
        rows.append(_det_row(did, st, ed, t, "form", ends, {}, r=r))
    return rows


def _det_row(i: int, state: str, form: str, topic: str, kind: str, ends: list[str], extra: dict, r: random.Random) -> dict:
    rr = rng("det", i)
    line = LINE_OF_TOPIC[topic]
    lo, hi = DOL_WINDOW[form]
    dol = extra.get("date_of_loss") or _rand_date(rr, lo, hi)
    if (state, topic) in BULLETIN_BLOCK and kind != "bulletin" and "date_of_loss" not in extra:
        raise ValueError((i, state, topic))
    doc = form.split(" (")[0]
    q = extra.get("question") or rr.choice(QUESTIONS[topic])
    return dict(
        determination_id=f"DET-{i:04d}",
        claim_id=extra.get("claim_id") or f"CLM-{dol.year}-{rr.randint(1000, 39999):05d}",
        claimant=extra.get("claimant") or f"{rr.choice(FIRST)} {rr.choice(LAST)}",
        policy_number=extra.get("policy_number") or f"LM{LINE_CODE[line]}-{state}-{rr.randint(10000, 9999999):07d}",
        state=state, line=line, form=form, endorsements=list(ends), date_of_loss=dol,
        adjuster=extra.get("adjuster") or rr.choice(ADJUSTERS), question=q, topic=topic, kind=kind)
