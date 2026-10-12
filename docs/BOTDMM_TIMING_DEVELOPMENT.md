# Fixed BotDMM Timing Development Control

This comparison is a development check, not a replacement final method.
Its validation policy challenges have already been inspected during earlier
development. A positive result would not establish untouched confirmation,
state-of-the-art performance, or publication readiness.

## Frozen Comparison

- Original account archives, labels and profile splits remain unchanged.
- Two nested views: Twibot22 and Quadbot.
- Seeds: 42, 43 and 44.
- Three held-out traditional scheduling policies per view.
- Two conditions: local BotDMM backbone with and without relative timing.
- Total: 36 planned fits, with every result retained.
- No test tensors or test metadata are opened.
- Selection uses only a seeded inner partition of the remaining training
  profiles, ranked by worst-known-policy Macro-F1, overall Macro-F1 and MCC.
- No hyperparameter search is performed on the outer challenge results.

The fixed optimizer, dropout, epoch limit, patience and batch size match the
earlier ERM MLP development control. Both BotDMM conditions rebuild the same
cosine feature-kNN graph separately in each fold, excluding same-profile
neighbors. Graphs do not use labels or policy tags for neighbor selection.
These are feature graphs, not real social edges. Inference within each
challenge is unlabelled transductive rather than inductive.

## Reproduction Boundary

This uses the repository's local release BotDMM architecture with adapted
input dimensions and a shared development training schedule. Auxiliary
prototype, domain and contrastive losses remain disabled as in release
training. It is not a full reproduction of the original BotDMM paper.

Timing adds seven global numerical inputs and seven per-step action inputs.
The two conditions therefore have different input projection sizes, which
must be reported. The MLP has the same individual information and profile
folds but does not receive BotDMM's graph-neighbor channel. Model capacities
and graph information are not matched across architectures.

## Verification

`analysis/run_botdmm_timing.py` freezes input and implementation hashes,
normalization masks/statistics, selected checkpoints, inner histories and
row-level challenge probabilities. Output directories cannot be overwritten.

`analysis/audit_botdmm_timing.py` reconstructs all fold transforms and graphs,
checks checkpoint identities and replays all 36 selected checkpoints.

`analysis/summarize_botdmm_timing.py` verifies complete prediction records
and creates paired profile-cluster intervals against the ERM MLP control and
the no-timing condition. These intervals are descriptive under fixed training
seeds; they cannot remove prior challenge exposure or multiple development
attempts.

Raw trajectories and account-level predictions remain private artifacts.
Old released results and the working manuscript are not replaced by this
development round. No completed scores are asserted in this protocol note.
