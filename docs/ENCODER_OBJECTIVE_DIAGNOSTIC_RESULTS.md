# Four Class Encoder Objective Diagnostic

## Completion and Boundary

The user-authorized Quadbot diagnostic completed on 2026-10-14 at
19:47:51 Asia/Shanghai, about 6 minutes 45 seconds after deployment began.
Nine new fits, three reused supervised references and 12 known-policy
selection replays completed. Five targeted tests passed remotely and
the full 165-test local regression passed. The downloaded summary
matched its remote SHA-256:

`17a3ae504faba62a52717c7aca45a6e484054af78bdc7e75c350819bdcc8da4b`.

All conditions use the same two-linear-layer control encoder, seed 42,
the same existing fitting/selection partitions, fixed coefficients,
and no pretraining. Initial shared parameters match. The supervised
reference was replayed from its existing checkpoint and reproduced its
stored inner-selection metrics exactly. All new selected checkpoints
were replayed with a maximum probability difference of zero.

The serialized original-training tensor files contain both known and
excluded rows. Files were read and known rows were sliced before
feature transforms or model forwards. Excluded-policy trajectories were
not forwarded, scored or exported. Cached control metrics contain old
challenge fields, but only checkpoint identities and known-policy
histories were used. Original validation/test inputs were not opened.
This is not an independent test or a new unseen-policy comparison.

## Known Policy Selections

Macro-F1 percentages on each fold's known-policy inner selection:

| Excluded Fold Label | Supervised Reference | Contrast Only | SAM Only | SAM Plus Contrast |
|---|---:|---:|---:|---:|
| Burst | 97.76 | 98.37 | 98.15 | 97.96 |
| Cyclic | 98.78 | 97.55 | 97.75 | 97.75 |
| Uniform | 98.39 | 97.78 | 97.77 | 97.98 |

The fold label identifies the partition, not the evaluated policy.
These overlapping inner-selection sets are not independent replicates.
Their descriptive arithmetic means are 98.31, 97.90, 97.89 and 97.89%;
they are not the pooled excluded-policy metric used in the prior study.
SAM plus contrast exceeds the supervised reference in only one of three
known-policy selections, with paired differences +0.20, -1.02 and
-0.41 percentage points.

For the combination, the selected fixed-fitting-batch gradient cosine
between CE and the weighted contrast term is +0.63, +0.53 and +0.45.
The weighted-contrast/CE norm ratios are 0.30, 0.39 and 0.60. These
observations do not support a universal gradient-conflict explanation.
Large ratios in some converged single-objective references reflect very
small CE gradients; they are not evidence that an unused objective
actually harmed that model. Probes preserve parameters and RNG state.

## Decision

Do not promote an encoder swap plus the same two objectives as an
established improvement. The narrow diagnostic does not establish that
either objective is universally harmful, nor does it measure a new
challenge score. The old MLP development result remains unchanged.
No further radius, contrast-weight, seed or epoch search is authorized
by these selections.

The next model change needs a task-specific mechanism and fitting-side
justification, rather than relabeling SAM and supervised contrast as
new architecture. Before a new performance screen, fix its real module
removals, matched controls and confirmation protocol. Recent relevant
comparisons and genuinely unexposed confirmation remain open CCF-B
preparation gaps. A universal conference acceptance threshold does not
exist, and no acceptance or SOTA claim follows.

Core views, canonical remote checkout, original method and Word are
unchanged. The diagnostic ran in an isolated remote code stage;
the source manifest records its exact added files. GitHub main and the
released tag were not updated. No GPU was used.
