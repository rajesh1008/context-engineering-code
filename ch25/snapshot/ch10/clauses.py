"""Clause text for the Chapter 10 fixture corpus.

Every document here is a synthetic fixture written for Lakeshore Mutual's
lab corpus. None of it is a real insurer's form, a real regulator's bulletin
or a licensed industry form. Hand-written clauses carry the determinations
the chapter discusses; the rest is generated from seeded templates so that
the 300 pages have realistic volume and realistic distractors.
"""
from __future__ import annotations
import random

# ---------------------------------------------------------------- hand-written
FREEZING_2022_A = ("Freezing. We insure for loss caused by freezing of a plumbing, heating, "
    "air conditioning or automatic fire protective sprinkler system or of a household "
    "appliance only if, while the dwelling is vacant, unoccupied or being constructed, "
    "you used reasonable care to maintain heat in the building or to shut off the water "
    "supply and drain all systems and appliances of water.")
FREEZING_2022_B = ("This peril includes the water damage that follows when a frozen pipe, "
    "fitting or appliance cracks or bursts, and the cost of tearing out and replacing any "
    "part of the building needed to repair the system, but not the system itself.")
FREEZING_2024 = ("Freezing. We insure for loss caused by freezing of a plumbing, heating, "
    "air conditioning or automatic fire protective sprinkler system or of a household "
    "appliance. This includes frozen pipes that burst. If the dwelling is vacant, "
    "unoccupied or being constructed, this peril applies only if you used reasonable care "
    "to maintain heat in the building or to shut off the water supply and drain all "
    "systems and appliances of water.")

PERILS = [  # (number, heading, body) for Section I Perils Insured Against, Coverage C
    (1, "Fire Or Lightning", "We insure for direct physical loss caused by fire or lightning."),
    (2, "Windstorm Or Hail", "We insure for loss caused by windstorm or hail. This peril does not include loss to the inside of a building or the property contained in a building caused by rain, snow, sleet, sand or dust unless the direct force of wind or hail makes an opening in a roof or wall and the rain, snow, sleet, sand or dust enters through this opening."),
    (3, "Explosion", "We insure for loss caused by explosion, other than the rupture of a steam boiler or water heater, which is covered under peril 13."),
    (4, "Riot Or Civil Commotion", "We insure for loss caused by riot or civil commotion, including direct loss from pillage and looting during and at the site of the riot."),
    (5, "Aircraft", "We insure for loss caused by aircraft, including self-propelled missiles and spacecraft."),
    (6, "Vehicles", "We insure for loss caused by vehicles. This peril does not include loss to a fence, driveway or walk caused by a vehicle owned or operated by a resident of the residence premises."),
    (7, "Smoke", "We insure for sudden and accidental damage from smoke, including the emission or puffback of smoke, soot or fumes from a boiler, furnace or related equipment. This peril does not include loss caused by smoke from agricultural smudging or industrial operations."),
    (8, "Vandalism Or Malicious Mischief", "We insure for loss caused by vandalism or malicious mischief. This peril does not apply if the dwelling has been vacant for more than 60 consecutive days immediately before the loss."),
    (9, "Theft", "We insure for loss caused by theft, including attempted theft and loss of property from a known place when it is likely that the property has been stolen. This peril does not include loss caused by theft committed by an insured or from that part of a residence premises rented to someone other than an insured."),
    (10, "Falling Objects", "We insure for loss caused by falling objects. This peril does not include loss to property contained in a building unless the roof or an outside wall of the building is first damaged by a falling object. Damage to the falling object itself is not included."),
    (11, "Weight Of Ice, Snow Or Sleet", "We insure for loss caused by the weight of ice, snow or sleet which causes damage to property contained in a building. Ice damming along the eaves is covered only when it causes such weight damage."),
    (12, "Freezing", None),  # filled per edition
    (13, "Sudden And Accidental Tearing Apart", "We insure for the sudden and accidental tearing apart, cracking, burning or bulging of a steam or hot water heating system, an air conditioning or automatic fire protective sprinkler system, or an appliance for heating water. We do not cover loss caused by freezing under this peril."),
    (14, "Accidental Discharge Or Overflow Of Water Or Steam", "We insure for accidental discharge or overflow of water or steam from within a plumbing, heating, air conditioning or automatic fire protective sprinkler system or from within a household appliance. We do not cover loss to the system or appliance from which the water or steam escaped, and we do not cover loss on the residence premises caused by accidental discharge or overflow which occurs off the residence premises."),
    (15, "Sudden And Accidental Damage From Artificial Electrical Current", "We insure for sudden and accidental damage from artificially generated electrical current. This peril does not include loss to tubes, transistors, electronic components or circuitry that is a part of appliances, fixtures, computers, home entertainment units or other types of electronic apparatus."),
    (16, "Volcanic Eruption", "We insure for loss caused by volcanic eruption other than loss caused by earthquake, land shock waves or tremors."),
]

CONDITIONS_DISTRACTORS = {  # 2022 Section I Conditions that share the claim's words
    "Duties After Loss": ("In case of a loss to covered property, you must give prompt notice to us or our agent, protect the property from further damage, and make reasonable and necessary repairs to protect the property, for example by shutting off the water to a burst pipe. You must keep an accurate record of repair expenses and show the damaged property as often as we reasonably require."),
    "Vacancy And Unoccupancy": ("If the dwelling will be vacant or unoccupied in winter, you should arrange for heat or shut off and drain the water supply. A pipe that has frozen and burst in a vacant dwelling is a common source of loss, and we may inspect the dwelling to confirm that heat was maintained before we pay a claim for burst pipe damage."),
    "Our Option": ("If we give you written notice within 30 days after we receive your signed, sworn proof of loss, we may repair or replace any part of the damaged property with material or property of like kind and quality, including a burst pipe or fitting, rather than pay for the loss in cash."),
}

ROOF_2022 = "Roof Surfacing. Loss to roof surfacing caused by windstorm or hail is settled at replacement cost, subject to the deductible shown in the Declarations."
ROOF_2024 = "Roof Surfacing. Loss to roof surfacing caused by windstorm or hail is settled at actual cash value when the roof surfacing is more than 15 years old at the time of loss. Otherwise it is settled at replacement cost."

ENDORSEMENTS = {  # form -> edition -> list of (section, text)
    "HO 04 95": {
        "2019-03": [("Coverage", "Water Back-Up And Sump Discharge Or Overflow. We will pay up to $5,000 for direct physical loss caused by water or waterborne material which backs up through sewers or drains or overflows or is discharged from a sump, sump pump or related equipment."),
                    ("Deductible", "The deductible shown in the Declarations applies to each loss under this endorsement."),
                    ("Exclusions", "We do not pay for loss caused by your failure to maintain a sump pump in working order, or for flood, surface water or waves.")],
        "2024-06": [("Coverage", "Water Back-Up And Sump Discharge Or Overflow. We will pay up to $10,000 for direct physical loss caused by water or waterborne material which backs up through sewers or drains or overflows or is discharged from a sump, sump pump or related equipment, even if caused by mechanical breakdown of the sump pump."),
                    ("Deductible", "A deductible of $1,000 applies to each loss under this endorsement unless a different amount is shown in the Schedule."),
                    ("Exclusions", "We do not pay for loss caused by flood, surface water, waves, tidal water or overflow of a body of water, whether or not driven by wind.")],
    },
    "HO 04 54": {"2022-01": [("Coverage", "Earthquake. We insure for direct physical loss to covered property caused by earthquake, including land shock waves or tremors before, during or after a volcanic eruption. One or more earthquake shocks that occur within a 72-hour period are a single earthquake."),
                             ("Deductible", "The earthquake deductible is 10% of the Coverage A limit of liability and applies separately to each earthquake."),
                             ("Exclusions", "We do not cover loss resulting directly or indirectly from flood or tidal wave, even if caused by earthquake.")]},
    "HO 04 98": {"2024-06": [("Coverage", "Refrigerated Property. We insure covered property stored in freezers or refrigerators on the residence premises for direct loss caused by interruption of electrical service or mechanical failure, up to $500."),
                             ("Conditions", "You must maintain the refrigeration unit in proper working condition, and the deductible does not apply to this coverage.")]},
    "LM HO 17": {"2024-06": [("Coverage", "Roof Surface Payment Schedule. Loss to roof surfacing caused by windstorm or hail is paid as a percentage of replacement cost shown in the Schedule according to the age of the roof surfacing at the time of loss."),
                             ("Schedule", None)]},
    "HO 04 90": {"2022-01": [("Coverage", "Personal Property Replacement Cost. Covered losses to personal property are settled at replacement cost without deduction for depreciation, subject to the limits of liability."),
                             ("Conditions", "We will pay no more than the actual cash value until actual repair or replacement is complete, and you may make a claim within 180 days for the additional amount.")]},
    "HO 04 61": {"2022-01": [("Coverage", "Scheduled Personal Property. We cover jewellery, furs, cameras, musical instruments, silverware, golfer's equipment, fine arts and stamp collections listed in the Schedule against risks of direct physical loss."),
                             ("Conditions", "Newly acquired property of a class already scheduled is covered for 30 days for up to 25% of the amount of insurance for that class.")]},
    "HO 04 16": {"2024-06": [("Coverage", "Premises Alarm Or Fire Protection System. We acknowledge the installation of an alarm or automatic sprinkler system on the residence premises, and you agree to maintain it in working order."),
                             ("Conditions", "You must tell us promptly of any change to the system or if it is removed, because the premium credit depends on it.")]},
    "HO 04 35": {"2024-06": [("Coverage", "Loss Assessment. We will pay up to $1,000 for your share of a loss assessment charged during the policy period against you by a corporation or association of property owners."),
                             ("Conditions", "The assessment must result from direct loss to property owned by all members collectively, of the type that would be covered by this policy.")]},
}

STATES = {"WI": "Wisconsin", "IL": "Illinois", "MI": "Michigan", "MN": "Minnesota",
          "IA": "Iowa", "IN": "Indiana"}

BULLETINS = [  # id, state, effective, ends, topic, operative paragraph
    ("WI-OCI-2023-07", "WI", "2023-07-01", "2025-03-15", "matching",
     "When a covered loss damages siding, insurers should replace undamaged siding on the same elevation where the replacement material cannot reasonably match the remaining siding in colour and texture."),
    ("WI-OCI-2024-11", "WI", "2024-11-01", None, "water",
     "Insurers should apply water back-up endorsements to losses where municipal sewer surcharge and sump discharge occur together, and should not deny such losses on the ground that surface water was present outside the dwelling."),
    ("WI-OCI-2025-03", "WI", "2025-03-15", None, "matching",
     "When a covered loss damages siding or roofing, the insurer shall replace undamaged material on all elevations visible from the street where a reasonable match in colour, texture and quality cannot be obtained."),
    ("IL-DOI-2025-05", "IL", "2025-05-01", None, "freezing",
     "Where a utility interruption of more than 24 hours leaves a dwelling without heat, insurers shall not treat the dwelling as one in which the insured failed to maintain heat when pipes freeze during the interruption."),
    ("IA-ID-2024-08", "IA", "2024-08-01", None, "hail",
     "Insurers shall disclose any cosmetic damage exclusion for metal roofing on the declarations page, and a cosmetic damage exclusion not so disclosed shall not be applied to hail losses."),
    ("MN-COMM-2026-01", "MN", "2026-01-01", None, "roof",
     "An insurer that settles roof surfacing losses at actual cash value shall state the depreciation schedule used and shall provide it to the insured with the first payment."),
    ("MI-DIFS-2023-09", "MI", "2023-09-01", "2025-08-01", "ice",
     "Insurers should treat interior water damage from ice dams as a weight of ice, snow or sleet loss where the ice dam caused the roof to deflect or the gutters to tear away."),
    ("MI-DIFS-2025-08", "MI", "2025-08-01", None, "ice",
     "Interior water damage caused by an ice dam is a covered loss under homeowners forms that insure against weight of ice, snow or sleet, whether or not the roof deflected, unless the policy contains an express ice dam exclusion."),
    ("IN-DOI-2025-02", "IN", "2025-02-01", None, "deductible",
     "Insurers shall state any percentage wind or hail deductible as a dollar amount on the declarations page in addition to the percentage."),
    ("MN-COMM-2024-04", "MN", "2024-04-01", None, "vacancy",
     "For vacancy clauses, a dwelling under active renovation with workers present at least weekly shall not be treated as vacant."),
    ("IL-DOI-2023-10", "IL", "2023-10-01", None, "mold",
     "Insurers shall pay for mold remediation that results from a covered water loss up to the limit stated for mold, fungi or wet rot, and shall not apply the mold exclusion to that remediation."),
    ("IA-ID-2025-06", "IA", "2025-06-01", None, "derecho",
     "Losses from a derecho are windstorm losses, and insurers shall apply a single windstorm deductible to all damage from one derecho event."),
    ("WI-OCI-2022-05", "WI", "2022-05-01", "2024-11-01", "water",
     "Insurers should consider whether a sump discharge loss is caused by mechanical failure before applying the surface water exclusion."),
    ("IN-DOI-2024-03", "IN", "2024-03-01", None, "theft",
     "Catalytic converter theft from a vehicle parked at the residence is an auto loss and is not covered under the homeowners theft peril."),
]

# deductible schedule: state -> (all other perils, wind/hail, named-storm n/a in midwest -> "hail only")
DEDUCTIBLE_ROWS = [
    ("Wisconsin", "$1,000", "1% of Coverage A", "$2,500"),
    ("Illinois", "$1,000", "2% of Coverage A", "$2,500"),
    ("Michigan", "$1,000", "1% of Coverage A", "$2,000"),
    ("Minnesota", "$1,500", "2% of Coverage A", "$3,000"),
    ("Iowa", "$1,000", "1% of Coverage A", "$2,500"),
    ("Indiana", "$1,000", "2% of Coverage A", "$2,000"),
]
DEDUCTIBLE_HEAD = ("State", "All other perils", "Wind or hail", "Minimum wind or hail")

ROOF_SCHEDULE = [("0 to 5 years", "100%"), ("6 to 10 years", "90%"), ("11 to 15 years", "75%"),
                 ("16 to 20 years", "55%"), ("21 years or more", "40%")]

# ---------------------------------------------------------------- generated text
THINGS = ["detached garage", "fence", "swimming pool", "outdoor antenna", "satellite dish",
    "lawn and shrubs", "trees", "awnings", "patio furniture", "boat and trailer",
    "golf cart", "watercraft", "business property", "money and bank notes", "securities",
    "jewellery", "firearms", "silverware", "electronic data", "credit cards",
    "grave markers", "foundation", "retaining wall", "driveway", "septic system",
    "solar panels", "heat pump", "well pump", "chimney", "gutters", "skylight",
    "basement finish", "carpeting", "wallpaper", "kitchen cabinets", "water heater",
    "furnace", "air conditioning unit", "sprinkler system", "storage shed",
    "dock", "greenhouse", "hot tub", "playground set", "wood stove", "propane tank",
    "rental unit", "home office equipment", "student dormitory property", "bicycles"]
CAUSES = ["wear and tear", "mechanical breakdown", "settling or cracking", "birds or rodents",
    "insects", "mould or wet rot", "smog or rust", "pollution", "earth movement", "flood",
    "surface water", "power failure", "neglect", "intentional loss", "governmental action",
    "nuclear hazard", "war", "faulty workmanship", "collapse", "ordinance or law",
    "continuous seepage", "freezing of a fence", "domestic animals", "theft by a tenant",
    "vandalism of a vacant building", "lightning surge", "ice damming", "hail", "windstorm"]
CONDS = ["the loss is reported within 60 days", "you have taken reasonable steps to protect it",
    "it is located on the residence premises", "it is shown in the Declarations",
    "the dwelling was occupied at the time of loss", "a licensed contractor did the work",
    "you kept receipts or photographs", "the damage was sudden and accidental",
    "we have given written consent", "the item is not used for business",
    "the limit in the Schedule has not been exhausted", "a police report was filed within 72 hours"]
ACTIONS = ["inspect the damaged property", "obtain two written estimates", "photograph the loss",
    "confirm the policy edition on the Declarations", "record the date of loss",
    "check for an applicable state bulletin", "verify the deductible in the Schedule",
    "request the proof of loss form", "refer the file to the special investigations unit",
    "confirm occupancy at the time of loss", "notify the mortgagee", "issue a reservation of rights letter",
    "calculate depreciation", "contact the policyholder", "escalate to a senior adjuster",
    "request plumber or contractor invoices", "check prior claims history", "obtain weather data for the date of loss"]
AMOUNTS = ["$200", "$250", "$500", "$1,000", "$1,500", "$2,500", "$5,000", "10% of the Coverage A limit"]


def third(cond: str, who: str) -> str:
    return cond.replace("you have", f"{who} has").replace("you ", f"{who} ")


def gen_clause(rng: random.Random, style: str) -> tuple[str, str, dict]:
    """Return (heading, body, facets) for a generated clause."""
    thing, cause, cond = rng.choice(THINGS), rng.choice(CAUSES), rng.choice(CONDS)
    action, amt = rng.choice(ACTIONS), rng.choice(AMOUNTS)
    if style == "form":
        heading = f"{thing.title()} And {cause.split()[0].title()}"
        s = [f"We do not insure loss to {thing} caused by {cause} unless {cond}.",
             f"The most we will pay for loss to {thing} under this provision is {amt} for any one loss.",
             f"If {cause} and a covered peril both contribute to the loss, we pay only for the part caused by the covered peril.",
             f"This provision does not reduce the limit of liability shown for Coverage A."]
        body = " ".join(s[: rng.randint(2, 4)])
    elif style == "manual":
        heading = f"Handling {thing} claims involving {cause}"
        s = [f"When a claim involves {thing} and {cause}, the adjuster must {action} before making a determination.",
             f"Adjusters should {rng.choice(ACTIONS)} and note the result in the claim file.",
             f"Payments above {amt} require the approval of a senior adjuster or the claims manager.",
             f"Where the policy is silent, apply the governing form edition and any state bulletin in force on the date of loss.",
             f"Document whether {third(cond, 'the insured')} before closing the file."]
        body = " ".join(rng.sample(s, rng.randint(3, 5)))
    else:  # underwriting
        heading = f"Eligibility for {thing}"
        s = [f"Risks with {thing} are eligible only when {third(cond, 'the applicant')}.",
             f"Decline applications where the {thing} shows evidence of {cause}.",
             f"Underwriters must {action.replace('damaged property', 'property')} for any risk with prior losses from {cause}.",
             f"A surcharge applies when the minimum deductible is below {amt}."]
        body = " ".join(s[: rng.randint(2, 4)])
    return heading, body, {"thing": thing, "cause": cause}
