# CCF-B Conference Revision Plan

## Goal and Current Boundary

Aim for a regular/full main-conference paper with a supported bot-detection
method contribution. No acceptance, SOTA or one-point improvement is
promised. No venue-specific submission is authorized by this plan.
The current working paper is not ready for another submission.

Keep the 5,031-row Twibot22 and 6,708-row Quadbot controlled views unchanged.
They share 1,677 base profiles and are not independent real-world
collections. The local Word, released main branch and release tag remain
unchanged while model development continues on the revision branch.

## Work Checklist

- Consolidation: `SUBMISSION_CONSOLIDATION.md` is the current work-order and
  method/figure boundary. Keep one working Word and unchanged core views.
- Prepared, not run: a complete-history BotSSCL-mechanism comparator and an
  identical-architecture supervised-only control, scheduled for 36 fits and
  36 replays on the existing training-only folds. This is an adaptation,
  not a faithful reproduction or a completed recent-baseline table.
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
- Pending: identify and implement recent bot-specific comparisons that
  use compatible inputs, or label adaptations and teacher/graph access
  explicitly. Do not use paper names without actual reproductions.
- Pending: genuinely unexposed confirmation after the model recipe is
  frozen. Changing folds or seeds on already inspected profiles is not
  independent evidence. Reserve external evaluation records and
  provenance before scoring; do not rebuild the two core datasets.
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
