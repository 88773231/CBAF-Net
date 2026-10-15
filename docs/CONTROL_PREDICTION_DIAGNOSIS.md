# Control Prediction Diagnosis

## Scope and Verification

Completed October 14, 2026 using already exposed, original-training-only
policy predictions. This read-only audit performed zero research fits and
opened no raw feature tensors or checkpoint weights. It neither searched
thresholds nor chose an ensemble, recipe, seed, or checkpoint.

`analysis/audit_control_predictions.py` verified all 138 exported private
evidence members against the export receipt. For both views it also checked:

- Identical candidate/control input-file digests, prior partitions and configuration.
- Nine matching fitting-normalization receipts per view.
- Frozen source-file digests and ordered sample/profile/class identities.
- All 36 preserved control replay records and per-run prediction digests.
- The recorded checkpoint rule, excluding the held-out policy from inner selection.
- All 18 pooled seed metrics against their retained aggregate summaries.
- Shared sample identities between the nested three- and four-class views.

All 160 local regression tests passed, including eight new audit tests.
The synthetic tests are software checks, not new research experiments.
Private identifiers and prediction rows are not included in this document.
The private diagnostic JSON digest is
`653badb4282cf5fec6f449ae8b04207b5ed8ad2887404ed72aeefaeb569ab426`.

These checks found no sample-pairing or fitting-normalization mismatch in
the archived comparison. They do not verify every upstream feature,
eliminate simulation shortcuts, or establish causal model explanations.

## Policy Localization

Macro-F1 below is the mean of three policy-specific scores, in percent.
These diagnostic means do not replace the primary metric, which first
pools policies within each seed.

| View | Excluded Policy | SAM + Contrast | BotSSCL Mechanism | Same Encoder, Supervised Only |
|---|---|---:|---:|---:|
| Twibot22 | Burst | 83.52 | 96.94 | 91.68 |
| Twibot22 | Cyclic | 98.11 | 99.38 | 99.11 |
| Twibot22 | Uniform | 96.94 | 99.16 | 98.80 |
| Quadbot | Burst | 94.09 | 87.78 | 85.14 |
| Quadbot | Cyclic | 97.03 | 97.78 | 98.24 |
| Quadbot | Uniform | 96.86 | 98.05 | 97.92 |

Twibot22 candidate errors are concentrated in Burst: 521 of its 687
error events, or 75.84%. Burst also accounts for 419 of the 536 excess
candidate error events relative to the pretrained control, or 78.17%.
This is a decomposition of error counts, not of nonlinear Macro-F1.
The candidate still loses Cyclic and Uniform means; fixing Burst alone
would not demonstrate a uniformly stronger method.

On Burst, candidate Traditional Bot F1 is 71.98% in Twibot22 versus
96.14% for the pretrained control. The corresponding Quadbot values
are 90.82% and 71.92%. The reversal is not a single dataset-wide
input-identity difference.

## Confusion and Paired Disagreement

All counts below sum prediction events across three already selected
models. They are not independent new samples: each view has the same
1,118 development profile groups repeated across seeds.

| View | Condition | Errors | Human/Traditional Mutual Errors | Traditional Predicted Human |
|---|---|---:|---:|---:|
| Twibot22 | SAM + Contrast | 687 | 475 | 444 |
| Twibot22 | BotSSCL Mechanism | 151 | 99 | 65 |
| Twibot22 | Supervised Only | 342 | 297 | 265 |
| Quadbot | SAM + Contrast | 532 | 162 | 123 |
| Quadbot | BotSSCL Mechanism | 689 | 440 | 412 |
| Quadbot | Supervised Only | 753 | 512 | 488 |

On Twibot22 Burst, the pretrained control alone is correct on 454
events; the candidate alone is correct on 35. On Quadbot Burst,
the direction reverses: the candidate alone is correct on 366,
and the pretrained control alone on 124.

These disagreements locate errors. They do not license choosing a
router, ensemble weight, per-taxonomy recipe or threshold on these
exposed labels. No such combined predictor was constructed or scored.

## Selection and Nested Views

For candidate Twibot22 Burst, known-policy inner-selection Macro-F1
averages 98.29%, compared with 83.52% on the excluded policy.
Quadbot has 97.46% versus 94.09%. The populations differ, so these
are descriptive discrepancies, not paired generalization estimates.
They show that near-perfect known-policy checkpoint selection does
not establish unseen-policy performance.

The four-class candidate makes fewer mistakes than the three-class
candidate on their shared rows in all three runs. Shared-row events
correct only in four-class training are 197/100/105; events correct
only in three-class training are 91/82/64.

Both models were trained separately with different class populations
and heads. This is not an intervention isolating an Agent head,
nor proof that adding an Agent class causes better representations.
The pretrained control shows the opposite shared-row trend in two of
three runs. The nested views must not be called independent replications.

## Architecture Confound and Decision

The candidate encoder is Linear/LayerNorm/GELU with dropout, followed
by a classifier. The control encoder is Linear/ReLU/Linear without
dropout, followed by a classifier. It uses Xavier initialization;
pretraining additionally introduces a training-only projection and
extra updates. The inference parameter counts differ by about 1.5%.
Input matching did not match these architecture or optimization choices.

The supervised-only control already exceeds the three-class candidate
by 3.50 pooled points. Therefore the 5.41-point pretrained-control
advantage cannot be attributed solely to self-supervision. Equally,
this comparison alone cannot identify SAM, normalization, dropout,
depth or initialization as the cause. No hidden representations,
feature attributions or causal sensitivity results were measured here.

Retire the frozen SAM/contrast recipe as a supported unified two-view
paper replacement. Preserve its promising four-class development
record and all negative controls. Do not enlarge or regenerate the
datasets to make this model look better.

Before implementing another claimed innovation, use a bounded
same-encoder objective audit defined in
`ENCODER_OBJECTIVE_DIAGNOSTIC_PLAN.md`. It separates architecture
from training-objective changes without reopening challenge scoring.
No new research training was launched in this turn.

## Recompute

```bash
python -m analysis.audit_control_predictions \
  --private-evidence-root /private/verified_export \
  --export-receipt /private/export_receipt.json \
  --output /private/new_diagnostic.json
```

The paths are placeholders. Existing outputs are never overwritten.
The private export and raw source material are not public repository
artifacts. The working Word and immutable released results remain
unchanged; this document does not assert remote or GitHub synchronization.
