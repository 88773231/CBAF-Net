# Bounded Backbone Module Diagnostics

This round follows the completed timing control documented in
`BOTDMM_TIMING_RESULTS.md`. It is not a new final method.

## Fixed Design

- Preserve datasets, labels, profile splits and old result bytes.
- Seeds 42, 43 and 44; all three held-out scheduling policies; both views.
- Two conditions: `botdmm_without_structure` and `botdmm_without_orthogonal`.
- Total: 36 planned fits. Reuse all 18 full-timing backbone results per view.
- Same raw inputs, fitting-only transforms, graph context and checkpoint rule.
- Same seeded initialization and parameter layout.
- No test input access, outer-score tuning or selective result deletion.
- No additional weight or architecture search on these outer scores.

The existing `no_structure` mode masks the structure output after its
computation. Inactive parameters remain serialized and computation is not
claimed reduced. Context rows still enter training BatchNorm; this is not a
fully graph-free model. The existing `no_orthogonal` mode keeps separate
branch normalization but skips mutual projection.

When the structure output is zero, orthogonal processing is algebraically
redundant. A combined removal would not add an independent diagnostic, so
it is not run. Unit tests verify this equivalence, seed-matched initial
parameters and the blocked structure-branch classification gradients.

## Interpretation

Better ablations would indicate an unsupported architectural component,
not an automatically successful replacement model. Worse ablations would
support that component only within this adapted implementation and exposed
development scope. Neither outcome guarantees improvements on both views,
state-of-the-art results or paper acceptance.

These challenges and previous timing outcomes have already been inspected.
This is an explicit post-hoc diagnostic. Independent evaluation is still
needed before final scientific claims.

`analysis/run_botdmm_modules.py` records and replays all selected checkpoints.
`analysis/summarize_botdmm_modules.py` verifies complete records and reports
descriptive paired profile intervals against the full-timing control.
Raw trajectories and per-account predictions remain private.

## Completed Diagnostic

All 36 fits and 36 checkpoint replays completed on October 13, 2026.
The complete table and interpretation are in `BOTDMM_MODULE_RESULTS.md`.
Neither removal repaired the burst-policy failure or established the
requested one-percentage-point improvement. No further architecture or
weight search on these exposed outer challenges follows from this round.
