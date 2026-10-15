# Encoder Objective Diagnostic Plan

## Purpose and Status

On 2026-10-14 the user authorized continued model work before manuscript
replacement or submission, with four-class Quadbot as the primary task.
This is a bounded diagnostic, not a final architecture or a submission
gate pass. The completed input-matched control also changed its encoder;
first separate that confound before adding another module.

Execution completed at 19:47:51 Asia/Shanghai on 2026-10-14. Nine new fits,
three reused references and 12 known-policy checkpoint replays completed.
See `ENCODER_OBJECTIVE_DIAGNOSTIC_RESULTS.md`; the encoder swap and combined
objectives are not promoted.

The same supervised-only control encoder becomes a diagnostic reference:
two Linear layers with ReLU, no dropout or LayerNorm, Xavier initialization,
and a linear multiclass head. It is established baseline machinery,
not an original contribution and not renamed as CBAF-Net's final method.

## Fixed Factorial

Use the same inference encoder in all four conditions:

| Condition | Cross-Policy Contrast | SAM |
|---|---|---|
| Supervised Reference | Off | Off |
| Contrast Only | On | Off |
| SAM Only | Off | On |
| SAM + Contrast | On | On |

Both removals concern training objectives. No new inference expert,
hierarchy, gate or metadata input is introduced. Use the encoder's
last hidden representation for contrast in the active conditions.
Fix the previously recorded contrast weight 0.1, temperature 0.2
and SAM radius 0.05 without any coefficient search.

All conditions use seed 42, unchanged Quadbot and all three prior policy
folds. This gives 12 cells. The three exact supervised-only reference
checkpoints are reused after identity checks, leaving nine new fits and
12 selected-checkpoint inner-selection replays. The earlier two-view
proposal is superseded, not an instruction to run 18 additional fits.
One seed is deliberately a mechanism audit, not evidence of stable
superiority or a replacement for the three-seed final comparison.

Freeze the existing 30-epoch ceiling, patience 6, learning rate 0.001,
weight decay 0.001, hidden dimension 128, batch size 128, complete
profile batching and worst-known-policy/overall/MCC checkpoint rule.
Initialize shared parameters identically for each cell's paired seed.
No corruption pretraining occurs in these four objective conditions.

## Strict Access Boundary

Reuse the existing fitting and inner-selection profile indices exactly.
Load only their training rows; do not forward, score or write predictions
for excluded-policy rows, original validation or test material.
Verify source tensor digests, normalization receipts and disjoint profile
groups before fitting. Retain unchanged core accounts, labels and splits.

The current challenges are exposed development data. This fitting-only
diagnostic does not make them independent again. Do not compare new
diagnostic selections with old challenge scores or use the diagnostic
to select a favorable challenge seed, epoch, router or ensemble.

Report known-policy per-class scores, CE/contrast gradient norms and
alignment on fixed fitting batches, selected epochs, completed updates,
parameter counts, replay differences and resource use. Restore parameters
exactly after any sensitivity probe. Retain every planned condition.
Selection metrics measure the known policies only; they cannot establish
that an objective improves the excluded Burst policy.

## Stop and Next Decision

If the objective has no consistent fitting/selection justification or
dominates the classification gradient without benefit, remove/redesign
that objective as a research hypothesis. Do not compensate by enlarging
the datasets, tuning on challenges or hiding an adverse condition.
The outcome cannot by itself justify any unseen-policy claim.

After this audit, specify at most one task-specific candidate, its actual
removals, stronger recent controls and a confirmation boundary before
another performance screen. A pretraining mechanism, SAM or ordinary
supervised contrast must not be presented as a new general algorithm.
Neither a favorable diagnostic nor a better development mean establishes
SOTA, two necessary modules or CCF-B submission readiness.

## Resources

The proposed audit is CPU-only. GPU rental is not needed. A container
may still incur platform charges in CPU mode. Obtain authorization for
the bounded run before reopening it; estimate runtime from a fitting-only
preflight under its actual CPU/memory limits, not from earlier GPU jobs.
Do not promise a completion time before that preflight or imply work
continues after an active turn without a real job or automation.
