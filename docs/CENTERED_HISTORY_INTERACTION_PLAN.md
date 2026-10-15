# Centered History Interaction Development Check

Frozen before fitting on 2026-10-15. The latest user request prioritizes
model improvement toward CCF-B over final alignment. No acceptance probability
or universal CCF-B score threshold is inferred.

## Diagnosis

The fitting-only nine-checkpoint dependence probe completed at 00:43:22
Asia/Shanghai. Replacing learned history attention with uniform pooling
changed 3 of 22,392 fitting-row prediction passes, not independent samples.
Mean normalized attention entropy was 0.9727/0.9906 in the two directions.
Zeroing the discrepancies changed 40 predictions. Exact baseline replay,
history masking and row isolation passed. These post-hoc fitting probes do
not prove out-of-sample equivalence, causality or retrained removal effects.

The old linear discrepancy expansion repeats direct current-block terms.
This establishes an algebraic redundancy, not that all discrepancy information
is redundant. The next hypothesis removes learned attention and replaces
additive discrepancies with nonlinear within-row history interactions.

## Fixed Hypothesis

Retain the original separate position-specific hash-coordinate encoders,
profile/statistics encoders, five-block readout and input contract.
Let U be fixed uniform current/past pooling, and T/A be the encoded streams.
Use centered streams Tc = T - UT and Ac = A - UA. The two relation channels
are Tc * (UAc) and Ac * (UTc), where * is elementwise multiplication.
The direct T/A inputs remain. Products cannot be folded into the previous
linear direct weights. Constant temporal offsets cancel only within the
relation channels; the complete classifier is not invariant to them.

This is a generic second-order inductive-bias hypothesis, not demonstrated
algorithmic novelty or a causal measure of autonomous planning. The first
step has no nontrivial centered history. Action includes hashed text, so
do not claim independent modalities. The tensors are signed hashes, not
semantic sentence embeddings.

## Budget and Controls

Use exactly the earlier seed-42 fitting/selection partitions and training
seed 42 across three overlapping known-policy folds. Six new fits:
complete centered interactions and the same-capacity uncentered interaction
removal. Reuse nine exact-partition retained checkpoints: direct-only removal,
supervised encoder and BotSSCL mechanism control. Replay all fifteen cells.
The uncentered removal isolates centering with identical parameter capacity.
The direct-only removal has 8,192 fewer block-input weights; disclose that
difference. Neither control is claimed as an exact compute match.

Keep width 64, readout 128, dropout 0.2, AdamW 0.001/0.001, batch 128,
maximum 30 epochs, patience 6 and the earlier worst-known-policy selection
rule. No radius, scale, loss-weight, encoder-width or seed search.
Data and timing hashes and fitting-only standardizers must match.

The continuation gate requires each relation/centering removal to lose at
least 0.25 Macro-F1 percentage points on average and favor the complete
condition in at least two folds. Require positive mean and at least two
positive folds against each stronger control. This is a frozen development
screen, not statistical significance or a CCF-B submission criterion.
Do not tune the recipe on this round if it fails. If it passes, separate
training-seed confirmation, fair comparisons, novelty and evaluation-boundary
checks still precede paper replacement.

## Boundaries

Only fitting and known-policy inner selection may be forwarded. Excluded
policy rows, original validation/test and challenge predictions are not
used. Do not rebuild or enlarge either core dataset. All previous negative
development evidence remains intact. Keep manuscript and public/canonical
code unchanged. CPU only; no GPU is needed.
