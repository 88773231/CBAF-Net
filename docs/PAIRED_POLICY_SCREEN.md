# Literature-Informed Paired Policy Development

## Recent Papers and Limits of Transfer

1. Zheng et al., "Bot Meets Shortcut: How Can LLMs Aid in Handling Unknown
   Invariance OOD Scenarios?," AAAI 2026, volume 40, issue 2,
   pages 1605-1613. DOI: `10.1609/aaai.v40i2.37137`.
   This work motivates checking stylistic
   shortcuts instead of equating high in-distribution scores with robustness.
   We do not implement its text augmentation or claim to reproduce it.
   Rewriting content may change our construction-defined Traditional/LLM
   taxonomy, so augmentation is not used in this screen.
2. Han et al., "Exploring and Distilling Multi-Dimensional Clues for
   Interpretable Social Bot Detection," ACL 2026, pages 28971-28992.
   DOI: `10.18653/v1/2026.acl-long.1338`. It trains clue inspectors with
   outcome-reward reinforcement learning and combines four clue dimensions.
   We do not have
   equivalent clue annotations or a teacher, and do not rename the archive's
   hash features as distilled semantic clues.
3. Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020,
   volume 33, pages 18661-18673. This is the established mathematical
   basis of supervised class contrast, not a recent bot-detection result.
   Our additional positive-pair restriction uses known fitting policies.

Publication metadata were checked against the publishers on October 13,
2026. None of these papers establishes this candidate's effectiveness. This
screen is a task-specific training hypothesis, not a claimed reproduction
or a novel general contrastive-learning algorithm. Recent fair baselines
still require separate implementation and information-budget checks.

## Actual Diagnosis

Read-only fitting-mean input replacements on 18 existing excluded-burst
checkpoints confirm strong timing sensitivity. For the full conditional
model, known-policy selection mean Macro-F1 changes from 99.37% to 58.39%
in Twibot22 and from 98.98% to 66.57% in Quadbot when timing inputs are
replaced. This is a diagnostic distribution change, not causal proof.
Removing all timing is not a supported solution.

Keep the existing input schema and the exact matched MLP encoder/head,
which outperformed the rejected conditional structures on burst means.
There is no probability-product bottleneck, additional expert, learned
residual, new account generation or relabeling.

## Two Training Mechanisms

**Cross-policy class contrast.** In a fitting minibatch, positives have
the same account label but different known scheduler policies. Other
classes are negatives. Same-class/same-policy rows are ignored rather
than treated as false negatives. This tests whether a shared class
representation can preserve evidence without fitting policy-specific
directions. A one-policy batch has a differentiable zero contrast loss.

**Matched Human/Traditional boundary margin.** For each complete training
profile, define `g(x) = logit_Traditional(x) - logit_Human(x)` and penalize
`softplus(1 - (g(Traditional_variant) - g(Human_variant)))`. Shared
profile-specific class-logit offsets cancel. The loss does not act on
LLM/Agent logits. This tests a focused classification boundary instead
of imposing the failed hierarchy on every class.

These are objective components, not two new inference modules. Policy
and profile identifiers supervise fitting losses only. Predictions do
not need identifiers or another account's variant. Matched variants
are synthetic construction pairs, not validated real-world counterfactuals.

## Fixed Evaluation

Freeze weights 0.1/0.1, contrast temperature 0.2 and boundary margin 1.
These are fixed development choices, not tuned on the challenge.
The previous optimizer, seed/fold pairs, selection rule, batch budget and
30-epoch maximum remain unchanged.

Train the full condition, without cross-policy contrast, and without
the matched margin for seeds 42/43/44, all three excluded policies and
both views: 54 new fits. The exact both-objectives-removed condition is
the existing verified ERM MLP. Reuse all 54 matching ERM/HGB/adapted-BotDMM
control fits; do not run an unnecessary duplicate ERM. All three new
conditions have exactly ERM's inference parameter count.

Replay all 54 selected checkpoints and preserve every planned condition.
Compare the full condition with both individual removals and ERM,
including per-class F1 and paired seed differences. No condition or
seed is omitted after scoring. Favorable means on one policy are not
universal superiority.

Before scoring, the user clarified the primary target: improve the
overall three-class and four-class outcomes, not every individual class.
The primary metric pools the three excluded-policy prediction sets once
per profile/class within each seed, then reports the mean and sample SD
across seeds. Both views need higher overall means against matched ERM,
adapted BotDMM and both individual removals. A one-point control gain is
a desired target, not a promised result. Class/policy details are
diagnostic, not an additional requirement that every cell improve.
They remain visible and cannot be silently excluded from the pooled score.

All challenges have prior development exposure. These results cannot
be independent confirmation, and the shared-profile views are not two
independent real-world datasets. Independent evaluation and fair recent
comparisons remain necessary before revising the manuscript. The
source data, old results, Word and GitHub main/release remain unchanged.
