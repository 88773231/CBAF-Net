# Policy Contrast and Flatness Development Hypothesis

## Fitting-Only Diagnostic

The preceding cosine readout condition has only a 0.29-point mean gain
over matched ERM in Quadbot and unfavorable three-class removals.
Inspect 36 preserved linear-contrast/cosine-contrast checkpoints using
two deterministic fitting minibatches per checkpoint. Do not forward
challenge profiles. Disable dropout during this probe, perturb weights
by radius 0.05 in the classification-gradient direction, and restore
every parameter exactly. No new fit or checkpoint rewrite is involved.

The diagnostic JSON SHA-256 is
`7781433d0f5489f90014ce4a07e09b11867a12366d166d785644c6e9ea329bda`.
Each table row summarizes 18 fitting minibatch probes:

| View | Head | Fitting CE | Fitting CE Increase | Inner Selection CE Increase | Conflicting Gradients |
|---|---|---:|---:|---:|---:|
| Twibot22 | cosine with contrast | 0.00425 | 0.01502 | 0.00852 | 0/18 |
| Twibot22 | linear with contrast | 0.03404 | 0.01188 | 0.00478 | 1/18 |
| Quadbot | cosine with contrast | 0.01708 | 0.02258 | 0.00372 | 0/18 |
| Quadbot | linear with contrast | 0.04786 | 0.02392 | 0.01363 | 0/18 |

Only one of 72 probes has a negative classification/weighted-contrast
gradient cosine. The samples do not support prioritizing a gradient
conflict correction as the next intervention. Nonzero local sensitivity
motivates a flatness-training hypothesis, but does not prove that
sharpness causes the excluded-policy errors. Classification-only probes
are not exact measurements of the proposed joint objective's sharpness.
They also do not establish that cosine is always more sensitive.

## Fixed Recipe and Literature Boundary

Return explicitly to the original ordinary MLP head, with its same
initialization and parameter count. The rejected cosine head and pair
margin are not retained. Do not promote or rename their removals into a
final successful method.

Use standard, non-adaptive Sharpness-Aware Minimization (SAM):
compute the fitting objective gradient, perturb all active parameters by
`0.05 * gradient / norm(gradient)`, evaluate the same objective using
the same dropout realization at the perturbed weights, restore the
weights, and perform one AdamW update from the second gradient.
Gradient clipping occurs before that update, not before computing the
perturbation. Use the existing cross-policy class contrast with weight
0.1 and temperature 0.2. Inference remains an ordinary matched MLP.

SAM is established technology: Foret et al., "Sharpness-Aware
Minimization for Efficiently Improving Generalization," ICLR 2021,
OpenReview identifier `6Tm1mposlrM`. Class-supervised contrast follows
Khosla et al., NeurIPS 2020. Neither is our new general algorithm.
This is a compatibility/stability experiment, not an established
CCF-B-ready contribution.

Recent bot work provides context, not reproduced baselines:
Zheng et al., "Bot Meets Shortcut," AAAI 2026, motivates robustness
checks; Han et al., "Exploring and Distilling Multi-Dimensional Clues
for Interpretable Social Bot Detection," ACL 2026, uses teacher/clue
training that is absent here. Hash inputs are not semantic clue
representations, and adding their names to a comparison table without
faithful implementations would not fill the recent-baseline gap.

## Controls and Compute Disclosure

New main fits:

- `flat_policy`: ordinary matched MLP, cross-policy contrast and SAM.
- `without_policy_contrast`: the same MLP and SAM, without contrast.
- `two_pass_erm`: two ordinary AdamW fitting updates on each same batch.

Reuse the exact ordinary linear-plus-contrast condition from the
preceding paired-policy screen as `without_flatness`; retain its
original `without_pair_margin` name and source provenance. Reuse
ordinary one-pass ERM as removing both. HGB and adapted BotDMM remain
controls. The latter has extra feature-kNN information and is not a
faithful reproduction of its original paper setting.

SAM has two gradient evaluations and one optimizer update per batch.
The extra ERM control has two gradient evaluations and two updates.
They match per-batch forward/backward counts, not optimizer updates,
actual selected epochs, wall time, or exact total compute. Do not claim
equal compute from equal epoch limits. Log evaluation/update counts,
selected epochs and elapsed time for every new run.

## Frozen Main and Axis Probes

Retain the exact original fitting/selection folds and training seeds
42/43/44 for main comparisons, all three excluded policies and both
views. There are 54 new main fits and 72 verified cached fits.
Primary Quadbot scores pool all three policy challenges once per
profile/class within each seed. Twibot22 is fully reported transfer
evidence. One-point gains over all controls and a positive difference
in at least two of three seed/fold pairs are desired, not promised.
Neither per-class nor per-policy universal improvement is required.

Add two fixed axes for full versus one-pass ERM:

- Fixed partition 42, vary training seed 42/43/44.
- Fixed training seed 42, vary partition seed 42/43/44.

Reuse the common 42/42 full and ERM results; fit the other axis cells.
This requires 48 new fits. Combined study: 102 new fits and 102
selected-checkpoint replays. Training seed changes initialization,
dropout and minibatch order together. These axes are not a complete
factorial study, do not isolate pure initialization effects or measure
all interactions, and are not fresh independent confirmation.
Do not replace an unfavorable primary result with the best axis result.

Use the same features, normalization, optimizer settings, epoch limit,
patience and known-policy selection rule as before. No scale/radius/
contrast search, generation, relabeling, account/timing augmentation,
discarded conditions or original validation/test evaluation is allowed.
Every new result is development evidence because the challenges have
prior exposure. Preserve old studies, datasets, Word, release results,
GitHub main and released tags. Revision-branch work alone is allowed.
