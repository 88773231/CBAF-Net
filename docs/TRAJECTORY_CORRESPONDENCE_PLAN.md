# Bounded Trajectory Correspondence Hypothesis

## Scope

Freeze this plan before executing the architecture on research data. This is
one feasibility screen, not a new final method or a submission-ready claim.
Use the existing Quadbot training files and the exact seed-42 known-policy
fitting/inner-selection groups of the frozen BotSSCL mechanism control.
Do not regenerate data, score excluded-policy rows, open original validation
or test files, or change widths, coefficients, seeds or partitions after
seeing the result.

## Hypothesis And Limitations

The candidate explicitly compares projected post blocks with current/past
action blocks within the same account trajectory. A second path represents
adjacent-stage differences and products. This asks whether local post/action
correspondence and changes over the five exported blocks are useful beyond
their concatenated representations.

This is not the earlier class-factor residual, pair margin or conditional
content bridge. It has no hierarchy, graph, expert probability routing,
SAM, contrast loss or source/policy/profile identifiers as prediction input.
Attention, differences and products are established operations; composing
them here does not establish algorithmic novelty.

Raw exports are signed hashes, not pretrained semantic embeddings. Each
block uses a distinct input projection because its hash namespace differs.
The action serialization contains text, so this is overlapping evidence,
not independent modalities. The five blocks are coarse trajectory slices.
Historical attention masks only the correspondence path; the final readout
uses all five blocks. Neither the mask nor attention weights establish
causality, real agent planning, interpretability or online deployment.

## Fixed Comparison

`configs/trajectory_correspondence_screen.json` fixes width 64, readout 128,
dropout 0.2, AdamW learning rate/weight decay 0.001, batch 128, maximum
30 epochs, patience 6 and seed 42. The four conditions are:

- Correspondence plus transition.
- Without correspondence.
- Without transition.
- Without both.

Allocate modules before removing paths so all shared parameters have identical
seeded initialization. Missing discrepancy slots and the missing transition
slot are zeros. The serialized readout retains inactive weight columns; log
these and parameter/update/time counts. This controls readout shape, not exact
capacity or compute. Reuse the control's normalization fitted on fitting rows
only. Select checkpoints by worst known-policy Macro-F1, then overall
Macro-F1, then MCC, exactly as in prior screens.

Twelve new fits and twelve exact known-policy checkpoint replays are planned.
Train serially with two CPU threads and no GPU. Preserve every condition,
normalization receipt, epoch history, prediction, checkpoint and input/source
hash. Refuse to overwrite an existing output. Reference evidence must remain
unchanged.

## Decision Boundary

This can proceed to a separately frozen, repeated-run comparison only if
the full condition has a positive mean inner-selection Macro-F1 difference
against each of the three removals, wins at least two of the three paired
folds against each removal, and has a mean no lower than the existing
supervised control's known-policy selections. Report individual fold values,
parameter counts and time, not only an average.

Failure means do not automatically search dimensions, add losses, run more
seeds until favorable, evaluate excluded-policy scores to rescue it, or
rewrite the paper with this architecture. Retain the negative development
record and reconsider the diagnosed limitation.

Even passing this gate establishes only within-known-policy feasibility.
Three folds at one seed are not three independent replications. It does not
establish excluded-policy robustness, superiority on the three-class view,
SOTA, a new contribution, an acceptance probability or CCF-B eligibility.
Independent confirmation, fair recent baselines, generator provenance,
permissions/ethics and a coherent manuscript remain separate requirements.
