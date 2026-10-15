# Model Revision Decision Boundary

## Development Records Versus Submission Evidence

Preserving unsuccessful experiments does not mean presenting unsuccessful
modules as effective contributions. The full factor+residual candidate
is not a replacement submission model. A harmful residual must be removed
or redesigned before a new method can be claimed.

The full table is an internal development record. It prevents repeating
failed designs and keeps any eventual model selection traceable. A paper
must describe the modules in its actual final architecture and report
their relevant, honestly measured removals. It must not rename a better
removal into a new full model while retaining the old claims, delete
unfavorable runs, or change splits until the ablation looks favorable.

## Current Decision

Latest instruction, 2026-10-14: continue bounded optimization before
manuscript replacement or submission, with Quadbot as primary and
four-class ablations in the main text. Do not impose the earlier
two-view superiority requirement as the current user requirement.
Retain three-class results and do not make unsupported transfer claims.

The fixed three-seed conditional-routing/content-bridge screen failed
the replacement gate. Its bridge lowered excluded-burst mean Macro-F1 by
5.77 points in Twibot22 and 10.49 points in Quadbot.
See `CONDITIONAL_SCREEN_RESULTS.md` for every condition and seed.

The subsequent paired-policy screen also fails the user-confirmed
overall criterion. Overall Twibot22 Macro-F1 is 92.73% versus ERM's
91.73%, but Quadbot is 94.70% versus ERM's 94.96%. Both views exceed
adapted BotDMM, which is not sufficient when a stronger control exists.
Twibot22's pair-margin removal is better by 1.89 points overall.
See `PAIRED_POLICY_RESULTS.md` for the full development evidence.

The normalized-readout/cross-policy-contrast screen meets its frozen
Quadbot mean gate, but does not establish the desired robust improvement.
Full pooled Macro-F1 is 95.25% versus ERM's 94.96%, a 0.29-point gain
with paired SD 3.76; only one of three seed/fold pairs favors the full
condition. Quadbot means favor both module removals less than the full
condition, but the differences are small relative to their paired
variability. Twibot22 transfer favors removing contrast by 0.56 points
and removing cosine readout by 1.06 points. The full model is 93.57%.
See `COSINE_POLICY_RESULTS.md`; the candidate is not a submission
replacement, and the scale/weight must not be tuned on exposed scores.

The subsequent SAM/contrast screen is a promising four-class development
candidate. Quadbot Macro-F1 is 96.01% (run SD 0.17), versus one-pass
ERM's 94.96%, a +1.04-point mean gain; both seed/partition axes favor
the full model in 3/3 pairs. Two-pass ERM narrows the gain to +0.66
points. Contrast's mean contribution is +0.94 points but is positive
in only 1/3 paired main runs, so the two-component claim still needs
confirmation. Twibot22 is 93.09%; removing SAM is better by 1.53
points in all three main pairs. See `FLAT_POLICY_RESULTS.md` for
every condition, both axes and explicitly conditional profile intervals.
Retain the recipe for further evidence, not as a submission replacement.

The matched-input BotSSCL mechanism control subsequently reached 98.50%
in Twibot22 versus the candidate's 93.09%; its supervised-only encoder
reached 96.59%. Quadbot remains favorable for the candidate at 96.01%
versus 94.81% and 94.30%. Retire the frozen recipe as a supported
unified two-view replacement. The read-only prediction audit found
matching sample and normalization identities, but the control encoder's
depth, normalization, activation, dropout and initialization differ.
See `CONTROL_PREDICTION_DIAGNOSIS.md`; these data do not isolate a
single objective as the cause. The same-encoder fitting-only diagnostic
has completed nine new fits and twelve checkpoint replays. Its combined
objectives beat the supervised reference in only one of three known-policy
selections; do not promote that encoder swap. See
`ENCODER_OBJECTIVE_DIAGNOSTIC_RESULTS.md`.

The earlier user requirement was overall three/four-class improvement, not an
improvement in every class or every policy. Retain class/policy
breakdowns as diagnostics, but do not confuse them with that primary
success requirement. A favorable class cannot rescue a failed overall
score; a worse class alone does not invalidate an improved overall score.

Near-perfect known-policy inner selection does not establish robust
Human/Traditional discrimination under an excluded scheduler. Before
training another design, diagnose the dependency with the existing
checkpoints. Any input masking or fitting-reference replacement is
post-hoc diagnostic evidence, not a new model result or causal proof.

## Next Candidate

The five-block historical correspondence/adjacent transition candidate now has
an implementation and eight focused tests; the full local suite passes 173
tests. `TRAJECTORY_CORRESPONDENCE_PLAN.md` freezes one four-class twelve-fit
known-policy feasibility screen. All twelve fits and twelve exact checkpoint
replays have now completed and been locally verified. The complete
architecture fails the transition-removal gate: known-policy means are
98.71% complete, 98.30% without correspondence, 98.85% without transition
and 98.50% without both. Stop expanding this full architecture; see
`TRAJECTORY_CORRESPONDENCE_RESULTS.md`. Correspondence has a positive
development signal, not final-model or novelty confirmation. Generic
attention/difference/product operations do not establish originality.

A transition-free compact follow-up is now frozen in
`COMPACT_CORRESPONDENCE_PLAN.md`. The 188-test local suite and 11-test
remote regression pass. Training began at 23:18:06 Asia/Shanghai on
2026-10-14, using CPU only. All three training seeds share the existing
seed-42 fitting/selection partitions. The 36-cell matrix includes 30
new fits and six exact-partition cached controls; older seed-43/44
controls must not be reused because their partitions differ.
All 36 cells have completed and passed the local receipt audit; see
`COMPACT_CORRESPONDENCE_RESULTS.md`. The weak directional development
gate passes, but module contribution is only +0.047 points and one seed
mean reverses. This is not consistent module improvement, significance,
independent confirmation or a supported final-model replacement.
The latest instruction, 2026-10-15, restores the CCF-B target and authorizes
diagnosis-led bounded optimization before alignment. The earlier finish-first
freeze is superseded; see the current override in `SUBMISSION_SPRINT.md`.
There is no measurable universal "70% submission standard". Inspect fitting-only
checkpoint dependence before choosing a new recipe. Keep the manuscript
and canonical/public release unchanged until all gates are addressed.

The fitting-only dependence diagnosis and centered-interaction redesign have
now completed. All 197 local tests and five remote targeted tests pass; the
fifteen-cell prediction receipt audit also passes. The redesign does not pass
its scientific continuation gate: full/removal means are 98.5730%/98.5042%,
and the BotSSCL-mechanism control is beaten in only one fold. Same-seed compact
correspondence was 98.8488%, so this is not an improvement over that architecture.
See `CENTERED_HISTORY_INTERACTION_RESULTS.md`. Stop expanding or tuning this
branch. A new recipe requires a specific additional prediction diagnosis;
keep the CCF-B target without implying a numerical readiness or acceptance
probability. No final manuscript or public model replacement is justified.

Treat no-residual, no-bridge and single-objective models as development references,
not automatically promoted final methods. Improve a specifically
diagnosed prediction limitation with a justified architectural or
training mechanism. Freeze the recipe and define its controls before
evaluating it. Do not add modules merely to obtain two named contributions
or repeat bridge/residual-weight searches on exposed challenge scores.
The paired margin is not a supported second contribution. Reconsider
its mechanism before another fixed round instead of coefficient-tuning
the same exposed challenge. Training losses must be described as losses,
not presented as new inference architecture modules.

The cosine candidate has now replaced that margin in a fixed test.
Although its four-class means improve, there is no established stable
one-point improvement over the stronger control. Its high known-policy
selection scores still coexist with substantial excluded-policy variation.
Saved histories do not isolate initialization, inner-partition or
early-stopping effects. Diagnose representation or prediction stability
using fitting/inner-selection information before adding another component.
Do not change epochs, seeds or scales using challenge scores, or treat
three-class transfer failures as grounds to suppress their reporting.

The fixed flatness round now provides better four-class development
stability. Freeze it without searching its radius or contrast weight.
Confirm module effects and comparable information/compute budgets;
do not equate a stable full-model score with stable paired module effects.
Its unchanged MLP inference and established SAM/contrast components
still need substantive task-specific novelty before a method contribution
can support a CCF-B submission. Fair recent controls and independent
confirmation remain separate requirements.

Use equal folds, input features, optimization budgets and evaluation
rules where possible. Disclose extra graph information, representation
differences and parameter counts where matching is not possible.

## Submission Gate

Support claims with repeated-seed comparisons against relevant baselines
and removals. Report uncertainty and all planned conditions, including
those on which a mechanism does not help. A small stochastic reversal
does not by itself invalidate a module, but a substantial or repeated
reversal requires narrowing the claim, removing the module, or revising
the method. A better score on one policy is not evidence of universal
superiority.

Recently published, fairly implemented controls and independent
evaluation remain gaps. The shared-profile synthetic views are not two
independent real-world validations. Previously exposed development
profiles cannot be made independent simply by changing seeds or folds.

Keep original datasets, released results, submitted manuscript and
GitHub main/release unchanged until a genuinely supported replacement
is ready. Archive the development work separately. Do not promise a
particular gain, SOTA or acceptance before the evidence exists.

Do not require a universal one-percentage-point gain where a comparator
already exceeds 99%; the mathematical ceiling may preclude it. Set
meaningful criteria for the affected boundary and disclose unchanged
or worse conditions rather than suppressing them.
