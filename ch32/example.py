"""Pack DET-0001 with the Budgeter and print the trace (Listing 32.4)."""
import lakeshore32 as L
from contextlab.budgeter import Budgeter, cost_hier

det = L.full_mix()[0]                      # DET-0001, Emily Carter
budgeter = Budgeter(budget=1_000, hard_cap=2_400, floor=0.60)
packet, trace = budgeter.pack(det.cands)
for span_id, why in trace:
    print(f"{span_id:<14} {why}")
print(f"{len(packet)} spans, {cost_hier(packet):,} tokens of evidence")
missing = L.read(det, packet)["missing"]   # completeness check, on the packed packet
print("withhold:" if missing else "complete", *missing)
