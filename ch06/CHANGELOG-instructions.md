# Changelog: claims assistant instructions (contextlab/instructions.py)

Newest first. Each entry records what changed, why, and the measured effect.

```text
v2.3  BLOCKED by gate. Removed rule precedence as redundant.
      eval 3.5% -> 6.5% (large), 4.5% -> 8.0% (small); hold-out 0.0% -> 12.1%, 3.0% -> 18.2%.
      Reason recorded on rule precedence: only this rule states the order for every topic.
v2.2  Restored Example C (DET-0307, cites WI-OCI-2025-03). Lint added to release.
      Fingerprint 6bc1f97b940b = v2.0. Known error: DET-0061 (Example A date boundary).
v2.1  Replaced Example C with an archived answer citing HO-3 Conditions D.5.
      Withdrawn: WI roof bulletin claims 66.7% wrong (large), 83.3% (small).
v2.0  Added Examples A, B, C and five topic skills, progressive disclosure.
      eval 4.0% -> 3.5% (large), 4.0% -> 4.5% (small). Skills: no measured effect; untested.
v1.2  Added rule quote_only. Motivated by DET-0042 (added a 30-day reporting limit).
      Unsupported answers 6 -> 1 (large), 18 -> 5 (small).
v1.1  Added rules precedence, applicability, cite_exact.
      Motivated by DET-0049, DET-0145, DET-0185: form cited where an endorsement governs.
      eval 6.0% -> 4.0% (large), 8.0% -> 4.0% (small).
v1.0  Role and task only. eval 6.0% (large), 8.0% (small).
```
