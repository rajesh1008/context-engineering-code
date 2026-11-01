"""Listing 2.11: one fixed-pipeline trace, DET-0008, starting from the stale claim facts."""
import lab

det = next(d for d in lab.corpus.eval_set(200) if d.determination_id == "DET-0008")
fixed, note = lab.reconcile(lab.stale_view(det))
t = lab.base.with_config(policy_filter=True).run(fixed)

print(f"{t.det.determination_id}  {t.det.state}  form {t.det.form}  reconcile: {note}")
print(f"retrieved (top 12 of {len(t.retrieved)}):")
for rank, sid in enumerate(t.retrieved[:12], 1):
    print(f"  {rank:>2}. {sid:<38} {t.dropped.get(sid, 'kept')}")
print(f"packet: {len(t.packet.span_ids())} spans, {t.rendered.tokens:,} tokens in, "
      f"{t.answer.tokens_out} out, version checked: {t.answer.version_checked}")
print(f"answer: {t.answer.text}")
print(f"cited : [{t.answer.cited_label}] {t.answer.citation}")
print(f"gold  : {t.det.gold_span_id}  match: {t.answer.cited_span_id == t.det.gold_span_id}")
