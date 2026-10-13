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

## Next Candidate

Treat a no-residual factor model as a development reference, not an
automatically promoted final method. Improve a specific, diagnosed
prediction limitation with a justified architectural mechanism. Freeze
the recipe and define removals of that mechanism before evaluating it.
Do not add modules merely to obtain two named contributions.

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
