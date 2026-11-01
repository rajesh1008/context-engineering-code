"""Listing 2.12: the pilot's routing rule, which reads the policy record and fails closed."""
from collections import Counter

import lab


def route(det, record) -> tuple[bool, str]:
    """Fail closed: route only when every fact the slice depends on is known and agrees."""
    if record is None:
        return False, "no policy record"
    if record.get("endorsements") is None:
        return False, "endorsements unknown"
    if record["endorsements"]:
        return False, "policy has endorsements"
    if sorted(det.endorsements) != sorted(record["endorsements"]):
        return False, "claim and policy disagree on endorsements"
    return True, "eligible"


def main() -> None:
    corpus = lab.corpus
    rec = corpus.policies
    every = corpus.eval_set(200) + corpus.holdout()
    print("all 500 determinations:", dict(Counter(route(d, rec.get(d.policy_number))[1] for d in every)))
    for name, dets in (("working 200", corpus.eval_set(200)), ("held-out 300", corpus.holdout())):
        n = sum(route(d, rec.get(d.policy_number))[0] for d in dets)
        print(f"{name:<13} eligible {n:>3} of {len(dets)} ({100 * n / len(dets):.1f}%)")
    d1 = every[0]
    unreadable = {**rec[d1.policy_number], "endorsements": None}
    print(f"{d1.determination_id} with endorsements unreadable: {route(d1, unreadable)}")
    print(f"{d1.determination_id} as recorded:             {route(d1, rec[d1.policy_number])}")


if __name__ == "__main__":
    main()
