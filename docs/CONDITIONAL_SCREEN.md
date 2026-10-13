# Conditional Evidence Development Screen

## Diagnosed Limitation and Hypothesis

The preserved factor screen's burst errors mostly affect Human/Traditional
Bot, not LLM Bot. For example, the seed-44 Twibot22 no-residual model had
Traditional F1 9.18% but LLM F1 99.19%. This is descriptive evidence of the
affected boundary, not proof of a causal mechanism.

The former structure lets the operation head use only its designated
behavior group. Relevant content evidence may therefore be underused
for Human/Traditional discrimination. Its label-score product also
requires a generation score for Human and Agent, although the subtype
decision only separates Traditional Bot from LLM Bot. The new hypothesis
addresses these two structural restrictions, without an unconstrained
classification-logit residual.

These are task-specific architectural hypotheses. Conditional probability
factorization and linear evidence fusion are not claimed to be new general
algorithms. Novelty and effectiveness are not established by implementation.

## One Structure, Two Independent Removals

The operation head predicts Human, automatic-posting bot, and (in Quadbot)
Agent. The automatic-posting leaf is split into Traditional and LLM Bot
by a conditional subtype head. Class probabilities are:

```
p(Human) = p(operation=Human)
p(Traditional) = p(operation=automatic) * p(subtype=Traditional)
p(LLM Bot) = p(operation=automatic) * p(subtype=LLM)
p(Agent) = p(operation=Agent)                 # Quadbot only
```

No undeclared combinations are discarded or renormalized. Changing the
subtype logits cannot change the direct Human/Agent leaf probabilities.
The subtype auxiliary objective uses only Traditional/LLM rows in all
four conditions. Operation auxiliary supervision uses all rows.

An optional zero-initialized linear content projection joins the behavior
projection before operation LayerNorm/GELU. This allows content to inform
the operation decision while keeping the conditional prediction topology.
There is no global class-logit residual or test-dependent routing.

The fixed 2x2 conditions are `conditional_bridge`,
`without_conditional_routing`, `without_content_bridge`, `without_both`.
The bridge removal physically omits its parameterized projection. The
routing removal replaces conditional leaves with a flat joint head,
retaining the same towers and auxiliary supervision. Shared encoder/head
parameters have seed-matched initialization. Capacities and computation
are not identical across removals and will be reported.

## Input and Evaluation Boundary

Preserve the existing datasets, labels, profile splits and tensors.
Use the exact existing original-training-only factor-screen folds, seeds
42/43/44, training transforms, optimizer, checkpoint rule and epoch budget.
The four candidates require 72 new fits. Reuse the 54 matching ERM MLP,
HGB and adapted BotDMM controls after verifying recipes, input hashes,
normalization partitions, metrics and checkpoint/replay provenance.
Do not rerun controls with a new recipe or silently substitute old folds.

Each temporal hash namespace has its own projection. Event hashes contain
text, so content and behavior groups overlap. The conditional axes encode
the archive's synthetic construction, not independently annotated intent.
There is no inference access to labels, policy ids or another account's
counterfactual class. Profile ids only organize training minibatches and
grouped partitions.

The challenge profiles have already participated in development.
This is another explicit post-hoc screen, not independent confirmation.
Original validation/test tensors are not read. Do not change weights or
select epochs using the excluded policy. No data augmentation, new
accounts, label changes or hyperparameter search is part of this run.

Replay all 72 selected checkpoints and retain every condition and seed.
Report means, sample seed SD, per-class F1, parameter counts and paired
seed differences. A better removal rejects or narrows the corresponding
claim; it is not automatically renamed into a successful final model.
Relevant recent fair comparisons and independent evaluation remain needed
before replacing the paper. Published results and GitHub main/release
remain unchanged until a supported replacement exists.
