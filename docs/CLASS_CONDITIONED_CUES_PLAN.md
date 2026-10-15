# Class-Conditioned Cues: Bounded Development Check

Frozen before new fitting on 2026-10-15. This is a feasibility screen,
not a novelty claim, final test result or permission to replace the paper.

## Diagnosis

The audited evolution/distillation branch has 15 error events among
1,480 rows in overlapping known-policy selections; only nine distinct
trajectories are involved. Thirteen events confuse Human and Agent and
eleven error events have confidence at least 0.9.

Replaying the clean cue-separated backbone, without either retired
mechanism, produces 17 error events. Nine have at least one correct cue
classifier; eight distinct erroneous trajectories have a correct cue in
at least one fold. This is a label-using diagnostic, not an oracle score
that a deployable model can attain.

Individual cue classifiers are much weaker than the contextual backbone.
For example, timing has approximately 0.49 overall Macro-F1 but approximately
0.99 Traditional-Bot F1. Behavior has approximately 0.69-0.72 Macro-F1 but
approximately 0.93-0.96 Agent F1. The average probability JSD is 0.267 on
errors and 0.260 on correct predictions. Thus general cue disagreement is
not a demonstrated discriminator of errors. It does not justify blindly
discounting every disagreeing cue or replacing the contextual backbone
with a mixture of weak independent classifiers.

## Tested Hypothesis

Preserve the clean, full-history contextual classifier. Add a residual
from four existing observable cue classifiers, weighted separately for
each candidate class. All classes see all cues. No Human/Agent-specific
route or hand-selected cue preference is hard-coded.

Gate inputs consist of projected cue histories and contextual logits.
Encoder inputs to the gate are detached; the projections and gate remain
trainable. Expert scores still receive gradients through classification.
The residual multiplier starts at 0.1 and is learned. Gate weights are
not calibrated probabilities of reliability or causal explanations.

The second intervention is a training-only paired-profile margin. For
each fitting profile, compare the same two class scores on its two
class-specific trajectories. A soft margin encourages separation within
the matched context. It uses all six class pairs equally, never selects
pairs from detector scores, and never consumes profile identifiers or
labels in inference. It is a loss term, not an architectural module.
Simulation siblings do not establish real causal counterfactuals.

## Relation to Literature

- CDRBot motivates keeping different evidence dimensions visible. Its
  external clue distillation and outcome-reward inspectors are not used.
- BotCHF motivates preserving independent branch decisions and adapting
  fusion to the account. Its binary text/graph weighting is not a
  reproduction of this four-class residual.
- ECML and DBF motivate treating agreement and confidence carefully.
  The new model does not implement their Dirichlet opinions, uncertainty
  mass, conflict-discounting rule or an independent-modality assumption.
- Bot Meets Shortcut motivates accounting for context shared across
  strategy variants. This loss does not implement its LLM counterfactual
  augmentation or prove invariance.

Generic routing plus a margin is not established algorithmic novelty.
See RECENT_METHOD_TRANSFER_20261015.md for verified titles, primary
references, compatibility boundaries and exclusions.

## Frozen Matrix and Boundaries

Retain the 5,031/6,708 views and every existing identity and partition.
Primary screening is Quadbot only, seed 42, on the same three retained
known-policy fitting/inner-selection folds. These selections have already
been exposed, overlap and are not independent confirmation.

Four new fits per fold:

1. Class-conditioned routing plus paired margin.
2. Without paired margin: identical inference architecture.
3. Without routing: exact contextual forward, paired margin retained.
4. Ordinary scalar-gate control: same score network, class-shared weights.

Replay two exact-partition controls per fold from the observation study:
the clean cue backbone and the same-input MLP. Total budget is 12 new fits,
6 cached control replays and 18 saved prediction/checkpoint cells.
Keep hidden width, optimizer, selection rule, epochs, patience and batch
size fixed. Pair weight is 0.1; margin is 1.0. No coefficient search.

Both interventions must each yield at least 0.25 percentage points mean
Macro-F1 gain over their respective removals, positive in at least two
folds. Also require a positive mean gain in at least two folds against
the scalar gate, contextual backbone and MLP. This project stopping rule
is not a CCF-B acceptance standard.

Use fitting-only normalization and replay every checkpoint. Extract only
the permitted fitting/selection identities in each fold. Do not transform
or forward its excluded-policy rows, original validation or original test.
Do not regenerate accounts, change source labels, omit negative removals
or compare legacy-input BotDMM scores as matched controls. If the fixed
gate fails, retain the complete record without tuning this branch on its
now-exposed numbers.

## Resources

CUDA remains disabled. Maximum remote screen runtime is 30 minutes,
including extraction and checkpoint replays. No rented GPU is required.
No public push, manuscript replacement, final-score claim or submission
is authorized by this development screen.
