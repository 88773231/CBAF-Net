# Cosine Readout and Cross-Policy Contrast Hypothesis

## Why Change the Second Mechanism

The prior pair margin only acts on the Human/Traditional logit gap.
At many selected epochs its recorded loss is already close to zero.
This observation is not proof that saturation causes the failed overall
ablation. It does show that the objective is not a multiclass readout
mechanism and does not support retaining it as a broadly useful module.
The rejected pair-margin implementation and results remain unchanged.

The contrast objective operates on normalized directions, whereas the
previous classification head uses unconstrained vector magnitudes and
class biases. Test one concrete alternative: make the readout compatible
with the directional geometry used for contrast. No class-specific branch
or dataset re-generation is involved.

## Mechanisms and Literature Boundary

Retain the exact cross-policy class contrast with weight 0.1 and
temperature 0.2. Replace the flat linear head with
`16 * dot(normalize(h), normalize(w_class))`, without a class bias.
The scale is fixed before fitting, not learned or searched after scoring.
The dropout input to the classifier is normalized; the contrast input
uses the pre-dropout representation as before.

Cosine-normalized classifiers are established techniques, for example
Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face Recognition,"
CVPR 2018, pages 5265-5274. This screen does not use CosFace's additive
margin and is not its reproduction. Class-supervised contrast follows the
established idea in Khosla et al., "Supervised Contrastive Learning,"
NeurIPS 2020. Neither component is a new general algorithm.

Recent bot work such as Zheng et al., "Bot Meets Shortcut: How Can LLMs
Aid in Handling Unknown Invariance OOD Scenarios?," AAAI 2026, motivates
testing robustness to changing stylistic contexts. It does not prove that
our classifier fixes shortcut learning. The archive's hashing inputs
must not be called semantic embeddings or faithful human/agent intentions.

This is a task-specific compatibility hypothesis. Novelty, effectiveness
and a sufficiently strong submission contribution are not established by
combining these components.

## Fixed 2x2 Test

New fits: normalized head with and without contrast, for all three
existing seed/fold pairs, all three excluded policies and both views.
This requires 36 fits and 36 selected-checkpoint replays.

Removing the normalized head is exactly the preceding screen's
`without_pair_margin` condition: an ordinary flat MLP with cross-policy
contrast and no margin. Reuse its 18 verified fits, explicitly recording
their original names and source artifacts. This is a mathematical
equivalence of the removal, not promotion or renaming of that model.
Removing both is the existing ERM MLP. Reuse all 54 ERM/HGB/adapted-BotDMM
control fits. Total matching cached fits: 72.

Encoder initialization, input features, optimizer, batch sequence, inner
selection rule and maximum epochs stay unchanged. Normalized readout
removes the C class biases, so the parameter count is smaller by C;
do not claim exact capacity equality. The prototype weights start from
the ordinary head's same seeded weight values.

## Primary and Transfer Reporting

Following the clarified paper scope, Quadbot is the core four-class task;
Twibot22 is three-class transfer evidence. Freeze this reporting scope
before the new fits, not after selecting favorable results.

Primary score: pool all three excluded-policy predictions once per
profile/class within each seed, then report mean and sample SD across
42/43/44. The full Quadbot condition should improve on all three controls
and both individual removals; a one-point gain over controls is desired,
not promised. Per-class or every-policy gains are not required.

Report every three-class transfer outcome, condition and seed, even if a
removal is better. Such a reversal narrows transfer claims; it is not
silently suppressed. A successful four-class development score alone is
not independent validation, SOTA or proof that the model generalizes
to real accounts.

Keep original validation/test tensors excluded, existing datasets and
splits unchanged, and all previous evidence intact. Do not update the
Word manuscript or GitHub main/release until supported results and
independent evaluation are available. No scale, coefficient, seed or
fold search is allowed in this fixed screen.
