# CCF-B Conference Revision Plan

## Goal and Current Boundary

The latest instruction on 2026-10-15 restores CCF-B as the target and requests
model improvement before final alignment. Follow the current override in
`SUBMISSION_SPRINT.md`; the previous finish-first work order is historical.
There is no objective numerical "70% CCF-B submission standard" or supported
acceptance probability. Do not pursue open-ended searches or withhold results.

Aim for a regular/full main-conference paper with a supported bot-detection
method contribution. No acceptance, SOTA or one-point improvement is
promised. No venue-specific submission is authorized by this plan.
The current working paper is not ready for another submission.

The latest 2026-10-14 user instruction keeps Quadbot as the primary task,
allows four-class main-text ablations, and requests continued model work
before the paper is rewritten. Three-class auxiliary weaknesses still
limit cross-taxonomy claims. There is no universal CCF-B score threshold.

Keep the 5,031-row Twibot22 and 6,708-row Quadbot controlled views unchanged.
They share 1,677 base profiles and are not independent real-world
collections. The local Word, released main branch and release tag remain
unchanged while model development continues on the revision branch.

## Work Checklist

- Consolidation: `SUBMISSION_CONSOLIDATION.md` is the current work-order and
  method/figure boundary. Keep one working Word and unchanged core views.
- Completed: 36 complete-history BotSSCL-mechanism/supervised-only control
  fits and 36 checkpoint replays on the existing training-only folds.
  Twibot22 control/candidate Macro-F1 is 98.50%/93.09%; Quadbot is
  94.81%/96.01%. The current candidate fails the two-view superiority
  requirement. This is a mechanism adaptation, not a faithful reproduction
  or a completed recent-baseline table. See `BOTSSCL_POLICY_CONTROL_RESULTS.md`.
- Completed: record the failed cosine/pair-margin screens without
  promoting unsupported modules.
- Completed: fitting/inner-selection sensitivity diagnostic on 36
  preserved checkpoints, with exact parameter restoration.
- Completed: fixed SAM/contrast development screen, two objective
  removals, one-pass ERM, per-batch compute-control ERM, HGB and adapted
  BotDMM; 102 new fits including training-seed/partition-axis probes.
- Completed: all planned results, per-class/policy evidence, both axes
  and a conditional profile-bootstrap analysis are retained in
  `FLAT_POLICY_RESULTS.md`. Quadbot full is 96.01% versus ordinary
  MLP's 94.96%; two-pass ERM is 95.35%. Module and transfer limitations
  remain explicit. No final paper replacement is authorized.
- Pending: confirm module effects and a substantive task-specific
  method contribution. SAM plus established supervised contrast alone
  is not demonstrated algorithmic novelty.
- Completed: read-only candidate/control prediction diagnosis with
  138 exported member digests, same ordered rows and fitting-normalization
  identities, and 18 pooled seed metrics verified. All 160 local tests
  passed. Burst explains 75.84% of the three-class candidate error
  events; architecture changes remain confounded. See
  `CONTROL_PREDICTION_DIAGNOSIS.md`.
- Completed: bounded four-class same-encoder objective diagnostic,
  nine new fits, three reused references and 12 known-policy checkpoint
  replays. The combination beats its supervised reference in only one
  of three inner selections and is not promoted. No challenge scoring
  occurred. See `ENCODER_OBJECTIVE_DIAGNOSTIC_RESULTS.md`; the full local
  regression passed 165 tests and five targeted remote tests passed.
- Completed: historical within-account correspondence/adjacent-transition
  screen, twelve fits and twelve exact checkpoint replays. Eight remote
  tests and a twelve-cell read-only local receipt audit passed. The
  complete condition fails the fixed transition-removal gate, 98.71%
  versus 98.85% without transition on known-policy inner selections.
  Stop expanding the full architecture. See
  `TRAJECTORY_CORRESPONDENCE_RESULTS.md`; no test result or final-model
  claim is authorized.
- Completed: the compact three-seed matrix and receipt audit. Module mean
  contribution is only +0.047 percentage points with one seed reversal.
  Do not promote it into a stable contribution.
- Completed: fitting-only dependence probes on nine retained checkpoints,
  zero new fits and no selection/challenge/test forwards. Uniform history
  replacement changed only 3/22,392 prediction passes. See
  `CORRESPONDENCE_DEPENDENCE_RESULTS.md`.
- Completed, not promoted: centered nonlinear historical interactions,
  six new fits and nine exact-partition references. The fifteen-cell receipt
  audit and 197 local tests pass, but the continuation gate fails. Relation
  gain is only +0.0689 points and BotSSCL is beaten in one fold only.
  The prior compact condition is also better at the same training seed.
  See `CENTERED_HISTORY_INTERACTION_RESULTS.md`; no seed expansion or
  coefficient search is authorized for this failed branch.
- Pending: identify and implement recent bot-specific comparisons that
  use compatible inputs, or label adaptations and teacher/graph access
  explicitly. Do not use paper names without actual reproductions.
- Pending: genuinely unexposed confirmation after the model recipe is
  frozen. Changing folds or seeds on already inspected profiles is not
  independent evidence. Reserve external evaluation records and
  provenance before scoring; do not rebuild the two core datasets.
- Completed metadata disclosure, not end-to-end reproduction:
  `GENERATION_DISCLOSURE.md` records class-wide generator settings and
  explicitly unverified weight, prompt/archive, and source-permission gaps.
- Pending: articulate a task-specific, technically substantive method
  contribution. SAM, cosine heads and supervised contrast are
  established methods; a favorable combination alone does not establish
  algorithmic novelty.
- Pending: update the actual manuscript architecture, equations,
  comparisons, ablations, limitations and figures from supported
  evidence; align local, remote and public release with hashed records.
- Pending: confirm the intended conference's current CCF category,
  full-paper track, deadline, format, anonymization, submission policies
  and whether any other submission is still active.

## Evidence Gates

Evaluate aggregate three/four-class scores against the stronger matched
controls, not just adapted BotDMM. Report every seed, policy, class,
removal and compute control. Analyze prediction uncertainty using profile
groups and report paired differences; small positive mean differences
with large variability are not stable effects.

Four-class ablations can be the main-text focus, but complete three-class
transfer removals remain available and limit any cross-task claim.
If a module consistently hurts, revise/remove it and change the actual
method claim. Do not conceal the result or retain a discarded module
in the architecture figure.

Model selection uses fitting/inner-selection evidence only. The current
policy challenges are development data. Independent confirmation,
novelty and recent comparisons remain separate gates even if all
development means improve. No paper update or submission follows
automatically from a passing development score.

## Resource Policy

This bounded screen uses CPU and disables CUDA. Stop the container
after collection and remote/GitHub synchronization if no other user jobs
need it. A running container may still incur platform charges despite
CUDA being unused. Do not imply that work continues after the active
turn ends unless an actual remote job or authorized automation remains.
