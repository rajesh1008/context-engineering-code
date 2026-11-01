"""Chapter 26 fixtures: Lakeshore's tools, the platform catalogue and the steps.

Synthetic and seeded. Descriptions come in two versions: as shipped (terse,
written by each owning team) and rewritten to the chapter's five-part form.
"""
import random

from contextlab.tools import Encoder, ToolSpec

# Synonyms the stand-in treats as one concept (a crude model of what any
# embedding or language model would also treat as close).
SYN = {
    "policy": "cover", "coverage": "cover", "covered": "cover", "cover": "cover",
    "coverages": "cover", "insured": "cover", "declarations": "decs", "decs": "decs",
    "dec": "decs", "page": "decs", "limit": "limit", "limits": "limit",
    "deductible": "limit", "deductibles": "limit", "endorsement": "endorse",
    "endorsements": "endorse", "rider": "endorse", "riders": "endorse",
    "add": "endorse", "ons": "endorse",
    "edition": "edition", "editions": "edition", "version": "edition",
    "versions": "edition", "superseded": "edition", "supersedes": "edition",
    "date": "when", "dated": "when", "happen": "when", "force": "when", "when": "when", "happened": "when",
    "day": "when", "past": "when", "historical": "when", "time": "when",
    "loss": "loss", "damage": "loss", "flooded": "loss", "incident": "loss",
    "claim": "claim", "claims": "claim", "file": "claim",
    "prior": "prior", "previous": "prior", "before": "prior", "earlier": "prior",
    "history": "prior", "industry": "industry", "carriers": "industry",
    "insurers": "industry", "other": "industry", "database": "industry",
    "flood": "floodzone", "zone": "floodzone", "fema": "floodzone",
    "floodplain": "floodzone", "sfha": "floodzone",
    "storm": "weather", "hail": "weather", "wind": "weather", "weather": "weather",
    "noaa": "weather", "gust": "weather", "hailstone": "weather",
    "estimate": "estimate", "contractor": "estimate", "quote": "estimate",
    "invoice": "bill", "line": "lineitem", "items": "lineitem",
    "item": "lineitem", "itemised": "lineitem", "repair": "repair", "repairs": "repair",
    "premium": "premium", "billing": "bill", "bill": "bill", "paid": "pay",
    "payment": "pay", "payments": "pay", "reserve": "reserve", "reserves": "reserve",
    "rule": "rule", "rulebook": "rule", "guidance": "rule",
    "text": "wording", "wording": "wording", "clause": "wording",
    "exclusion": "wording", "exclude": "wording", "excluded": "wording",
    "span": "span", "id": "span", "identifier": "span",
    "say": "wording", "much": "money", "money": "money", "amount": "money",
    "amounts": "money", "reserved": "reserve", "left": "remaining",
    "remaining": "remaining", "handle": "handling", "handling": "handling",
    "handled": "handling", "another": "industry", "elsewhere": "industry",
    "came": "industry", "report": "reported", "reported": "reported",
    "changed": "edition", "still": "edition",
    "current": "now", "today": "now", "renewal": "renew", "renews": "renew",
    "customer": "person", "policyholder": "person", "contact": "person",
    "phone": "person", "email": "person", "address": "place", "location": "place",
    "property": "place", "house": "place", "home": "place",
}
ENC = Encoder(SYN)

# name, as-shipped description, rewritten description, params, server, trust, examples
CORE = [
 ("search", "Searches documents.",
  "Search the governing policy text: forms, endorsements, state bulletins and "
  "Rulebook guidance, by meaning and by keyword, scoped by kind and state. Use to "
  "find the clause or exclusion wording that decides a coverage question. Returns "
  "spans with edition and source. Not for claim facts or policy declarations.",
  ("query", "kind", "state"), ["which clause covers water backing up through a drain",
                               "exclusion wording for surface water"]),
 ("lookup", "Gets a span.",
  "Fetch one span of governing text by its identifier, with edition, effective "
  "dates and source. Use when you already hold a span id from search or a "
  "citation. Not for searching.", ("span_id",), ["open span HO3-2024-06-A-1", "fetch cited span"]),
 ("get_declarations",
  "Returns policy details for a policy number, including coverages, limits, "
  "deductibles and endorsements.",
  "Declarations for the policy term in force on a given date: coverages, limits, "
  "deductibles, endorsements and form editions as of that date. Use for any "
  "coverage question about a loss: pass the date of loss. Not for the current "
  "term, billing or contact details (use get_policy). Not for what happened in "
  "the loss itself (use get_claim).",
  ("policy_number", "as_of_date"), ["what was the policy on the day of the loss",
                                    "did she have the endorsement when it happened"]),
 ("get_claim", "Returns claim information.",
  "The loss record for one claim: what happened, date of loss, cause as "
  "reported, location, status, assigned adjuster and notes. Use to establish "
  "the facts of this claim. Not for other claims on the policy (use "
  "claim_history).",
  ("claim_id",), ["facts of this claim", "date of loss and reported cause"]),
 ("claim_history", "Lists claims.",
  "Earlier claims at Lakeshore on the same policy and property, with dates, "
  "causes and payments. Use to check prior losses on this policy. Not for losses at other "
  "insurers (use prior_loss_search).", ("policy_number",),
  ["previous claims on this policy", "has she claimed before with us"]),
 ("query_measure", "Runs a query.",
  "Computed money measures for one claim from the semantic layer: amount paid "
  "to date, reserves, deductible applied, limit remaining, each with unit and "
  "query. Use for any question about how much money has been paid, reserved or "
  "is left. Not for raw rows or for what the policy covers.",
  ("measure", "claim_id"), ["how much has been paid on this claim", "limit remaining"]),
 ("rulebook_rule", "Gets a rule.",
  "One Rulebook rule by identifier or topic, with owner and version. Use when "
  "the determination needs Lakeshore's own guidance on how to handle a kind of "
  "claim. Not for policy wording.", ("rule_or_topic",), ["guidance on sewer back-up handling", "rule R-17"]),
 ("form_edition", "Form versions.",
  "Every edition of a policy form or endorsement with its effective dates and "
  "what superseded it. Use to check which edition was current on a date, "
  "whether wording has changed or whether an edition is superseded. Not for "
  "which endorsements one policy carried (use get_declarations).",
  ("form_id",), ["which edition of HO 04 95 was current in 2025", "is this edition superseded"]),
]
NEW = [
 ("flood_zone", "FEMA flood zone.",
  "Flood zone for an address from the FEMA flood map in force on a date: zone "
  "code, whether it is a special flood hazard area, map panel and its effective "
  "date. Use for surface-water and flood questions on a loss. Not for current "
  "underwriting risk scores.", ("address", "as_of_date"),
  ["is the house in a floodplain", "special flood hazard area on the date of loss"], "partner"),
 ("storm_events", "Weather data.",
  "Recorded storm events near a location on a date from NOAA storm event records: "
  "hail size, wind gust, time and distance from the property. Use to check a "
  "reported hail or wind cause against the record. Not for current alerts.",
  ("location", "date"), ["was there hail that day", "wind gust near the property on the loss date"], "partner"),
 ("estimate_lines", "Estimate.",
  "Line items from a contractor estimate that passed the document gate: each "
  "item with quantity, unit price, total and page. Use to check what an "
  "estimate includes. Not for estimate workflow status, or for money paid or "
  "reserved on the claim (use query_measure).",
  ("claim_id", "document_id"), ["what is on the contractor quote", "line items for drywall"], "internal"),
 ("prior_loss_search", "Prior losses.",
  "Prior property losses for this address or policyholder reported by other "
  "insurers to the industry database, with dates, causes and paid amounts. Use "
  "to check losses at another insurer, including before the policyholder came "
  "to Lakeshore. Not for Lakeshore's own claim history.",
  ("address", "policyholder"), ["did she claim with another insurer", "losses at other carriers"], "partner"),
]
# Platform tools exposed by other teams' MCP servers (terse; rewrite = terse + use line).
PLATFORM = [
 ("get_policy", "billing", "Returns policy details for a policy number, including coverages, limits and endorsements.",
  "Current policy term summary for customer service: status, renewal date, premium, agent and contact. "
  "Not for the term in force on a past date or for coverage on a date of loss (use get_declarations)."),
 ("property_risk_profile", "underwriting", "Property risk data including flood and wildfire.",
  "Current underwriting risk scores for a property: flood score, wildfire score, roof age. Use for quoting. Not for a past date or a claim."),
 ("weather_alerts", "comms", "Weather alerts for an address.",
  "Active weather alerts now for an address, used to warn customers. Not for past storms or claims."),
 ("estimate_status", "vendors", "Estimate status.",
  "Where a contractor estimate is in the vendor workflow: requested, received, approved. Not its contents."),
 ("get_claim_summary", "portal", "Returns claim information for the customer portal.",
  "Customer-facing claim summary in plain language for the portal: status and next steps. Not for adjusters."),
 ("claims_list", "portal", "Lists claims for a customer.",
  "All claims a customer can see in the portal, newest first. Not for determinations."),
 ("payment_history", "billing", "Payment history.",
  "Premium payments made by the policyholder, with dates and methods. Not claim payments."),
 ("policy_documents", "documents", "Policy documents.",
  "PDF files issued to the policyholder: renewal packs and letters. Not searchable governing text."),
 ("fraud_score", "siu", "Fraud score.", "Special investigations score for a claim, with reasons. Use only for referral."),
 ("siu_referral", "siu", "Refers a claim.", "Open a special investigations referral. An action, not a lookup."),
 ("get_quote", "underwriting", "Gets a quote.", "Premium quote for a new policy. Not for claims."),
 ("inspection_report", "underwriting", "Inspection report.", "Underwriting inspection report from before the policy was written."),
 ("roof_age", "underwriting", "Roof age.", "Estimated roof age from aerial imagery for underwriting."),
 ("wildfire_score", "underwriting", "Wildfire score.", "Wildfire exposure score for a property."),
 ("replacement_cost", "underwriting", "Replacement cost.", "Estimated rebuild cost of a home for setting Coverage A."),
 ("rate_table", "underwriting", "Rates.", "Rate table factors by territory for pricing."),
 ("underwriting_notes", "underwriting", "Notes.", "Underwriter notes on a policy file. Not governing text."),
 ("get_invoice", "billing", "Invoice.", "Premium invoice for a billing period."),
 ("billing_status", "billing", "Billing status.", "Whether a policy is paid up, in grace or cancelled for non-payment."),
 ("refund_premium", "billing", "Refunds premium.", "Issue a premium refund. An action."),
 ("autopay_settings", "billing", "Autopay.", "Direct debit settings for premium."),
 ("billing_plan", "billing", "Billing plan.", "Instalment plan for premium."),
 ("late_fee", "billing", "Late fee.", "Late fee charged on a premium instalment."),
 ("get_customer", "customer", "Customer details.", "Name, address, phone and email for a policyholder."),
 ("contact_preferences", "customer", "Preferences.", "How the customer prefers to be contacted."),
 ("household_members", "customer", "Household.", "Named members of the household."),
 ("loyalty_status", "customer", "Loyalty.", "Years with Lakeshore and loyalty discount."),
 ("customer_notes", "customer", "Customer notes.", "Service notes from customer calls."),
 ("complaints", "customer", "Complaints.", "Complaints logged by a customer and their status."),
 ("agent_of_record", "customer", "Agent.", "The independent agent who wrote the policy."),
 ("language_preference", "customer", "Language.", "Preferred language for letters."),
 ("get_document", "documents", "Gets a document.", "One stored document file by id: a PDF, photo or letter."),
 ("list_documents", "documents", "Lists documents.", "Every file attached to a claim or policy."),
 ("ocr_text", "documents", "OCR.", "Raw OCR text of a scanned file, before the document gate."),
 ("upload_document", "documents", "Upload.", "Attach a file to a claim. An action."),
 ("redact_document", "documents", "Redact.", "Produce a redacted copy of a file."),
 ("find_contractor", "vendors", "Finds contractors.", "Contractors in the preferred network near an address."),
 ("schedule_inspection", "vendors", "Schedules inspection.", "Book a field inspection. An action."),
 ("vendor_rates", "vendors", "Vendor rates.", "Agreed labour and material rates in the network."),
 ("mitigation_job", "vendors", "Mitigation.", "Water mitigation job status and drying logs."),
 ("vendor_invoice", "vendors", "Vendor invoice.", "Invoice from a network vendor for payment."),
 ("subrogation_status", "legal", "Subrogation.", "Recovery from a third party: status and amounts."),
 ("legal_hold", "legal", "Legal hold.", "Whether a claim file is under legal hold."),
 ("litigation_status", "legal", "Litigation.", "Lawsuits linked to a claim."),
 ("send_letter", "comms", "Sends a letter.", "Send a templated letter to a policyholder. An action."),
 ("letter_templates", "comms", "Templates.", "Approved letter templates by state."),
 ("sms_history", "comms", "SMS.", "Text messages sent to a policyholder."),
 ("call_recordings", "comms", "Calls.", "Recordings of service calls."),
]


RETURNS = {
 "search": ("span_id", "text", "form", "edition", "effective_from", "effective_to", "state", "kind", "score"),
 "lookup": ("span_id", "text", "form", "edition", "effective_from", "effective_to", "supersedes", "source_uri"),
 "get_declarations": ("policy_number", "term_start", "term_end", "form", "form_edition", "coverage_a",
                      "coverage_b", "coverage_c", "coverage_d", "deductible", "endorsements", "as_of_date"),
 "get_claim": ("claim_id", "policy_number", "date_of_loss", "date_reported", "cause_reported", "location",
               "status", "adjuster", "notes"),
 "claim_history": ("claim_id", "date_of_loss", "cause", "paid", "status"),
 "query_measure": ("measure", "value", "unit", "as_of", "query"),
 "rulebook_rule": ("rule_id", "text", "owner", "version", "effective_from"),
 "form_edition": ("form_id", "edition", "effective_from", "effective_to", "superseded_by"),
 "flood_zone": ("zone", "sfha", "panel", "panel_effective", "source"),
 "storm_events": ("event_type", "magnitude", "time", "distance_km", "source"),
 "estimate_lines": ("line", "description", "quantity", "unit", "unit_price", "total", "page"),
 "prior_loss_search": ("carrier", "date_of_loss", "cause", "paid", "address_match"),
}
PLATFORM_FIELDS = {
 "billing": ("policy_number", "status", "term_start", "term_end", "premium", "balance", "due_date",
             "agent", "contact", "payment_method", "coverages", "limits", "endorsements", "renewal_date"),
 "underwriting": ("property_id", "address", "score", "band", "model_version", "flood_score",
                  "wildfire_score", "roof_age", "construction", "year_built", "as_of"),
 "comms": ("id", "recipient", "channel", "template", "sent_at", "status", "body"),
 "vendors": ("job_id", "vendor", "status", "requested_at", "received_at", "approved_at", "amount"),
 "portal": ("claim_id", "status", "next_step", "message", "updated_at", "documents_needed"),
 "documents": ("document_id", "title", "kind", "pages", "uploaded_at", "uploaded_by", "uri", "hash"),
 "siu": ("claim_id", "score", "reasons", "model_version", "referral_id", "status"),
 "customer": ("customer_id", "name", "address", "phone", "email", "preference", "since", "notes"),
 "legal": ("claim_id", "status", "counsel", "amount", "opened_at", "closed_at"),
}


def build_tools(rewritten: bool, include_new: bool = True, platform: bool = False):
    out = []
    for name, terse, good, params, ex in CORE:
        out.append(ToolSpec(name, good if rewritten else terse, params, "lakeshore-claims",
                            "internal", tuple(ex) if rewritten else (), RETURNS[name]))
    if include_new:
        for name, terse, good, params, ex, trust in NEW:
            out.append(ToolSpec(name, good if rewritten else terse, params,
                                "lakeshore-external" if trust == "partner" else "lakeshore-claims",
                                trust, tuple(ex) if rewritten else (), RETURNS[name]))
    if platform:
        for name, team, terse, good in PLATFORM:
            params = ("policy_number",) if team == "billing" else ("claim_id",) if team in (
                "portal", "siu", "legal", "vendors") else ("id", "as_of")
            out.append(ToolSpec(name, good if rewritten else terse, params + ("page", "page_size"),
                                f"platform-{team}", "internal", (), PLATFORM_FIELDS[team]))
    return out


# Steps: the requests the assistant's plan makes, in two vocabularies.
STEPS = {
 "search": (["find the exclusion wording for water that backs up through a sewer or drain",
             "search the governing clause on surface water for {state}",
             "find the policy text that decides seepage"],
            ["what does the policy actually say about the drain backing up",
             "is there wording that excludes this kind of water",
             "what clause covers her basement"]),
 "lookup": (["fetch span {span} with its edition", "open the cited span {span}",
             "retrieve span id {span}"],
            ["pull up {span} again", "show me {span}", "open that one, {span}"]),
 "get_declarations": (["declarations for policy {pol} in force on {date}",
                       "coverages, limits and endorsements on {pol} as of the date of loss {date}",
                       "which endorsements were on {pol} when the loss happened"],
                      ["what was she covered for when the basement flooded",
                       "did she have the back-up endorsement on the day it happened",
                       "what deductible was on the policy on {date}"]),
 "get_claim": (["loss record for claim {clm}: date of loss and reported cause",
                "facts of claim {clm}", "status and location for claim {clm}"],
               ["what happened on {clm}", "when did the loss happen on this claim",
                "what did she tell us about the loss"]),
 "claim_history": (["earlier claims on policy {pol}", "prior claims on this policy at Lakeshore",
                    "claim history for {pol}"],
                   ["has she claimed with us before", "any previous water claims on this house with us",
                    "what else has this policy claimed"]),
 "query_measure": (["amount paid to date on this claim", "reserve and limit remaining on this claim",
                    "deductible amount applied so far"],
                   ["how much have we paid out so far", "how much is left on the limit",
                    "what have we reserved"]),
 "rulebook_rule": (["Rulebook rule on sewer back-up handling", "Rulebook guidance for {state} water claims",
                    "rule R-17 text and owner"],
                   ["what is our guidance on this", "what does the Rulebook tell me to do",
                    "how do we handle these here"]),
 "form_edition": (["editions of HO 04 95 and their effective dates", "which edition of the form was current on {date}",
                   "is edition HO-3 (2019-01) superseded"],
                  ["which version of the endorsement applied back then", "was that form still the current one",
                   "has this wording changed since"]),
 "flood_zone": (["FEMA flood zone for {addr} on {date}", "is {addr} in a special flood hazard area",
                 "flood map panel and zone for the property"],
                ["is the house in a floodplain", "was it a flood area when this happened",
                 "what zone is she in"]),
 "storm_events": (["NOAA storm reports near {addr} on {date}", "recorded hail size near the property on {date}",
                   "wind gust recorded near {addr} on the loss date"],
                  ["was there really hail that day", "did a storm come through on the day she gave",
                   "how strong was the wind out there"]),
 "estimate_lines": (["line items on the contractor estimate for {clm}", "itemised estimate for drywall on {clm}",
                     "estimate quantities and unit prices"],
                    ["what is on the contractor's quote", "did the estimate include the carpet",
                     "what are they charging for the repair"]),
 "prior_loss_search": (["prior losses at other insurers for {addr}", "industry database losses for the policyholder",
                        "losses reported by other carriers"],
                       ["did she claim with another insurer", "any losses before she came to us",
                        "has this house had water claims elsewhere"]),
}
STATES = ["Wisconsin", "Illinois", "Iowa", "Minnesota", "Michigan"]


def build_steps(n: int = 600, seed: int = 26, tools=None):
    rng = random.Random(seed)
    names = tools or list(STEPS)
    out = []
    for i in range(n):
        gold = names[i % len(names)]
        style = "forms" if rng.random() < 0.5 else "adjuster"
        tpl = rng.choice(STEPS[gold][0 if style == "forms" else 1])
        text = tpl.format(state=rng.choice(STATES), span=f"HO3-2024-06-{rng.randint(1, 40)}",
                          pol=f"WI-HO-{rng.randint(100000, 999999)}", date="14 March 2026",
                          clm=f"CLM-2026-{rng.randint(1000, 9999):05d}",
                          addr=f"{rng.randint(100, 9999)} Elm Street")
        out.append({"id": f"S{i:04d}", "gold": gold, "style": style, "text": text})
    return out
