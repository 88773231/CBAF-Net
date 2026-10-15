# Compact Correspondence Stability Check

Frozen on 2026-10-14 before new training. This is a development-informed
follow-up to the completed twelve-cell screen, not independent confirmation.
No original dataset, public release or manuscript result is replaced.

## Architecture

Keep the five independent hash-coordinate input projections, profile and
statistics encoders, and historical within-row bidirectional discrepancies.
Remove the adjacent transition path and its unused readout columns.
The removal control also removes the correspondence parameters and the
zero discrepancy input columns. Do not add a second module to rescue a claim.
The operations use signed hash features, not semantic language embeddings.
The action input contains text; the towers are not independent modalities.
Historical masking is not causal evidence or evidence of agent planning.

Allocate the original model in its existing order and retain only active
weight columns without consuming extra RNG. This preserves shared seeded
initialization and dropout RNG; small floating-point differences from
matrix multiplication shape changes are possible.

## Fixed Matrix

Use the existing seed-42 fitting/selection partitions for all three training
seeds, 42/43/44. Compare compact correspondence, its removal, the existing
supervised encoder, and the matched-input BotSSCL mechanism. Three policy
exclusion folds give 36 cells and exact selected-checkpoint replays.
Reuse only the six seed-42 control checkpoints with the exact same partitions.
Refit all other cells: 30 new fits. Older seed-43/44 controls have different
partitions and must not be reused.

Freeze widths 64/128, dropout 0.2, AdamW learning rate/decay 0.001,
batch 128, ceiling 30 epochs, patience 6. Keep the reference supervised
encoder width 128. BotSSCL uses its frozen 30-epoch pretraining, corruption
0.6 and temperature 1.0. Its extra compute and different architecture are
disclosed. This is not a faithful original-paper reproduction.
All selection uses worst known-policy Macro-F1, then overall Macro-F1,
then MCC. No coefficient, seed, budget or split search is allowed.

## Access and Decision

Load serialized original training tensors, slice to known fitting/selection
rows before transforms, and fit normalization on fitting rows only.
Excluded rows, original validation/test and challenge predictions receive
no transforms, model forwards or scoring in this round. Cached controls
contain earlier challenge fields; these are not used for selection.

Report every cell, class, policy and seed. Average fold selection Macro-F1
within each training seed; continue only if the candidate has a strictly
positive mean difference and at least two positive seed means against
each of the removal, supervised encoder and BotSSCL mechanism.
This continuation gate is not a statistical significance or submission gate.
The three folds overlap, and repeated training seeds are not new data.
Near-ceiling known-policy scores are not held-out generalization results.

If the gate fails, stop this design without tuning on exposed challenges.
If it passes, independent evaluation and compatible recent baselines still
must be designed before a submission-model replacement. Dataset provenance,
source permission and the novelty argument remain separate unresolved gates.
The Word manuscript and public/canonical GitHub remain unchanged.
