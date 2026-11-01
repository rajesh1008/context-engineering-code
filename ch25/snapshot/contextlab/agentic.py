"""contextlab v3.9: agentic retrieval with a budget, a sufficiency check and a trace.

The loop itself is small on purpose. What makes agentic retrieval safe to run
is around it: a budget that ends every run, a sufficiency check that names the
slots still open (so the planner knows what to ask next and the trace shows
why it stopped), an isolation option that keeps search debris out of the lead
context, and a router that sends a question to the loop only when a fixed
pipeline is expected to fail it. The planner is anything with `plan(state)`
and `read(state, tool, arg, hits)`; Chapter 18's stand-in model is one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PLAN_PROMPT, JUDGE_PROMPT, ANSWER_PROMPT, SUB_PROMPT = 300, 250, 450, 200
ANSWER_OUT, ACTION_OUT, CONDENSED = 350, 60, 25


@dataclass
class Budget:
    max_rounds: int = 6
    max_tokens: int = 20_000        # model tokens for the whole run, all calls
    max_calls: int = 16             # tool calls


@dataclass
class Slot:
    name: str                       # e.g. "ref:LK-WB->SL", "bulletin:MN:WB", "clause:HO-EXC-WATER"
    filled: bool
    evidence: str | None = None     # span id, or a recorded absence such as "not attached"


@dataclass
class State:
    question: str
    context: dict                   # structured facts the loop may read (policy record, codes)
    packet: list = field(default_factory=list)     # admitted span ids, in order
    absences: dict = field(default_factory=dict)   # slot name -> recorded absence
    seen: list = field(default_factory=list)       # every span id any tool returned
    slots: list = field(default_factory=list)      # open slots from the last check
    debris: int = 0                 # tokens of tool output the lead has had to read
    rounds: int = 0
    calls: int = 0


@dataclass
class Trace:
    packet: list
    rounds: int
    calls: int
    stop: str                       # sufficient | planner-done | round-cap | token-cap | call-cap
    tokens_in: dict
    tokens_out: int
    log: list
    lead: list = field(default_factory=list)      # lead-context tokens at each plan call
    snapshots: list = field(default_factory=list) # packet after each round
    model_calls: int = 1

    @property
    def tokens(self) -> int:
        return sum(self.tokens_in.values()) + self.tokens_out


def run(state, planner, tools, check, budget=Budget(), isolate=True,
        model_judge=False, span_tokens=lambda sid: 0):
    """Plan, call tools, admit, check; stop on sufficiency or on any budget."""
    tin = {"plan": 0, "read": 0, "judge": 0, "answer": 0}
    tout, log, stop, leads, snaps, mcalls = 0, [], "round-cap", [], [], 1
    while state.rounds < budget.max_rounds:
        state.rounds += 1
        lead = (CONDENSED * (len(state.packet) + len(state.slots)) if isolate
                else state.debris)
        actions = planner.plan(state)
        leads.append(lead); mcalls += 1
        tin["plan"] += PLAN_PROMPT + len(state.question.split()) * 2 + lead
        tout += ACTION_OUT * max(1, len(actions))
        if not actions:
            stop = "planner-done"; break
        for tool, arg in actions:
            if state.calls >= budget.max_calls:
                stop = "call-cap"; break
            state.calls += 1
            hits = tools[tool](arg, state)
            state.seen += hits
            got = sum(span_tokens(h) for h in hits)
            tin["read"] += (SUB_PROMPT + got) if isolate else got
            tout += 40 if isolate else 0
            mcalls += 1 if isolate else 0
            state.debris += got
            planner.read(state, tool, arg, hits)
            log.append((state.rounds, tool, arg, list(hits)))
        ok, state.slots = check(state)
        snaps.append(list(state.packet))
        if model_judge:
            tin["judge"] += JUDGE_PROMPT + 30 * len(state.packet); tout += 20; mcalls += 1
        if stop == "call-cap":
            break
        if ok:
            stop = "sufficient"; break
        if sum(tin.values()) + tout >= budget.max_tokens:
            stop = "token-cap"; break
    tin["answer"] = ANSWER_PROMPT + 60 + sum(span_tokens(s) for s in state.packet)
    tout += ANSWER_OUT
    return Trace(list(state.packet), state.rounds, state.calls, stop, tin, tout, log,
                 leads, snaps, mcalls)


def fixed_tokens(packet, span_tokens):
    """One answer call over a packet assembled without a loop."""
    tin = {"plan": 0, "read": 0, "judge": 0,
           "answer": ANSWER_PROMPT + 60 + sum(span_tokens(s) for s in packet)}
    return Trace(list(packet), 0, 1, "fixed", tin, ANSWER_OUT, [])


def cost_usd(trace, usd_in_per_m=3.00, usd_out_per_m=12.00):
    return sum(trace.tokens_in.values()) * usd_in_per_m / 1e6 + trace.tokens_out * usd_out_per_m / 1e6


@dataclass
class Route:
    name: str                       # "fixed" or "agentic"
    reason: str


def route(features: dict, threshold: int = 2) -> Route:
    """Send a question to the loop only when it is expected to need more than one hop."""
    score = (2 * (features.get("perils", 1) > 1) + features.get("onward_refs", 0)
             + (features.get("free_text_cause", False))
             + 2 * (features.get("untrusted_edge", False)))
    if score >= threshold:
        return Route("agentic", f"score {score}: perils={features.get('perils')}, "
                                f"onward_refs={features.get('onward_refs')}")
    return Route("fixed", f"score {score}")
