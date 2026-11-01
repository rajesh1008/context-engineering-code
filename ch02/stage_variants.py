"""Stage variants: the baseline with query enrichment switched off (Section 2.4, Stage 1).

`query_facts=False` drops the claim facts from the retrieval query but keeps the header in the
packet, so the row isolates what discovery contributes to retrieval."""
from __future__ import annotations

import lab
from contextlab import scoreboard as sb

dets = lab.corpus.eval_set(200)
print(sb.row("baseline query_facts=False", sb.score(lab.run_all(lab.base.with_config(query_facts=False), dets))))
