# Fixed Factor Screen Replication

Protocol recorded on October 13, 2026, after viewing the seed-42 screen.
This is post-hoc development replication, not untouched confirmation.

## Fixed Scope

Reuse the 36 preserved seed-42 fits and checkpoint replays. Add seeds
43 and 44 for all six existing methods, both nested views and all three
training-only policy challenges: 72 new fits, 108 fits in the aggregate.
Do not rerun, overwrite, omit or replace seed-42 results.

The original recipe, implementation, inputs, taxonomy mapping, auxiliary
weight, transforms, optimizer, checkpoint-selection rule and epoch budget
are fixed. The seed controls both optimization and the deterministic
inner fitting/selection profile partition. The held-out policy profiles
are unchanged. Original validation/test tensors are not accessed.

`analysis/replicate_factor_training_screen.py` is a separate entrypoint.
The original runner and model source remain unchanged. The extension
imports their preparation, fold, model, fitting and replay helpers.
It verifies original source and input hashes and freezes its own sources.
`configs/factor_training_replication.json` declares the added seeds.

## Complete Reporting

Retain factor+residual, flat-fusion+residual, factors without residual,
ERM MLP, HGB statistics and adapted BotDMM under every policy and seed.
Replay all 72 new checkpoints and verify the original 36 replay receipts.
Report every seed, mean Macro-F1, sample seed SD, per-class F1, parameter
counts, and paired seed differences. No p-value or independent-sample
claim is made. Equal-policy means are descriptive, not a replacement
for the complete policy tables or the manuscript's original main metrics.

The two views share profiles. Three seeds do not create three independent
datasets, and reseeding previously used profiles does not provide external
validation. Adapted BotDMM has extra feature-kNN context, HGB uses global
statistics, and model capacities differ. They are not exact information-
or capacity-matched comparisons or faithful recent-method reproductions.

## Decision Boundary

The full candidate and the no-residual direction are both examined, with
all negative results retained. The extension does not authorize automatic
best-variant promotion, tuning on challenge scores, manuscript replacement,
or a SOTA claim. Independent evidence and recent fair comparisons remain
required before a new final method can be presented.
