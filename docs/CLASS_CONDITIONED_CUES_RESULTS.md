# Class-Conditioned Cues: Complete Development Results

Completed 2026-10-15 03:40:22 Asia/Shanghai. Runtime was approximately
11 minutes 41 seconds, including extraction, fitting, checkpoint replay,
prediction audit and evidence packaging. No GPU was used.

## Scope

These are previously exposed, overlapping known-policy inner-selection
folds from the original Quadbot training population, seed 42. Their union
contains 185 profiles / 740 distinct trajectories, with 1,480 row events
across three selections. They are not original validation/test results,
independent replications, real-world benchmark results or SOTA evidence.

The existing 5,031/6,708 dataset views, source records, labels, identities,
partitions, original master and working manuscript were preserved. Only
permitted fitting/selection rows were transformed in each fold. The
excluded-policy population and original validation/test were not scored.
Twibot22 was not newly fitted in this screen.

## Complete Macro-F1 Table

Values are percentages. A fold name identifies the policy excluded from
that fold's fitting/inner-selection population, not a score on that policy.

| Condition | Burst excluded | Cyclic excluded | Uniform excluded | Mean |
|---|---:|---:|---:|---:|
| Class routing + paired margin | 98.3718 | 99.3902 | 98.9894 | 98.9171 |
| Without paired margin | 98.3718 | 99.3902 | 98.9894 | 98.9171 |
| Without class routing | 98.3718 | 99.3902 | 98.9894 | 98.9171 |
| Ordinary scalar-gate control | 98.3718 | 99.3902 | 98.9894 | 98.9171 |
| Cached clean cue backbone | 98.5713 | 98.9812 | 98.9884 | 98.8470 |
| Cached same-input MLP | 98.3628 | 98.3651 | 97.9770 | 98.2350 |

Full minus paired-margin removal: 0.0000 percentage points in every fold.
Full minus routing removal: 0.0000 points in every fold. Full minus the
scalar gate: 0.0000 points in every fold. The predeclared continuation
gate failed; neither intervention has demonstrated classification value.

The full candidate is descriptively 0.0701 points above the cached clean
backbone and 0.6821 above the cached MLP. These differences do not establish
the proposed mechanisms: all newly fitted removals and the scalar control
have the same classifications. Cached and new fits also consume random
initialization/dropout streams differently after module allocation.
Architecture/compute matching is not claimed. Legacy-input BotDMM scores
are not matched comparisons and cannot establish superiority.

## Intervention Activity

Saved predictions were independently paired and compared. Probability
changes are nonzero, but all nine full-versus-new-control comparisons
have zero changed class predictions.

- Maximum probability difference without paired margin: 0.03135-0.03381.
- Maximum probability difference without routing: 0.02802-0.06849.
- Maximum probability difference versus scalar gating: 0.02449-0.06460.
- Selected residual multipliers: 0.10439, 0.10628 and 0.10301; initial 0.1.

Together with the retained-component gradient tests and exact checkpoint
replays, these observations show that the interventions were active.
Activity is not evidence of classification gain. No formal calibration
improvement is established, and the failed primary outcome is not replaced
post hoc with a confidence claim.

## Verification

- 12 new research fits; 6 exact-partition cached control replays.
- 18 retained checkpoint/prediction cells, each checkpoint replayed.
- 36 member hashes checked; every reported metric recomputed.
- Local full regression: 228 tests passed.
- Remote targeted regression: 9 tests passed.
- Independent local prediction/hash audit: 18 cells passed.
- Read-only post hoc cue activity check: no new fits or feature forwards.
- Local/remote working manuscript and original remote master unchanged.
- No canonical remote model replacement, public GitHub push or submission.

Evidence hashes:

- Summary: `2acf8e80752d89ab6b18bc5ad8db5fd60257a8ba5fa430087fde4c0ef9ceb837`
- Evidence archive: `918faf94f2bbd374f03a4f39ae9439ba3b412a8406eacacc2ff84d9225f3c0a1`
- Executed source archive: `c8c2bac68c3259ccb9c64e3eb4bd24e2aa6b5da8526f83cf25aca9ae30aeea4f`
- Working Word: `af7f16a306aa06711de9bed422f00455c30fd67899cf347f4806aab665c3f685`

## Decision

Do not promote, coefficient-tune, extend seeds or rename this two-part
branch into a confirmed contribution using its exposed scores. Preserve
the full table and executed source archive. Do not enlarge or regenerate
the main datasets to rescue its result.

The next hypothesis should address the representation bottleneck rather
than add another fusion weight. Text features remain signed lexical
hashes, not pretrained semantic embeddings. A bounded semantic-encoding
study would require verified encoder/license and availability, fitting-only
adaptation, matched-encoder controls, fixed development selection and a
separate genuinely unexposed confirmation plan. Those experiments have
not started; resource/cost requirements must be stated before GPU work.
No CCF-B readiness percentage or acceptance probability follows from this
screen or from its high selection Macro-F1.
