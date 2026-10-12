# Development Results and Unresolved Evidence

These results are development diagnostics, not confirmatory test results or
replacements for the immutable `v1.1.0` paper tables. No main-view account,
label, text, grouped split, or old tensor was regenerated. Only training and
validation timing sidecars were exported. The existing test split was already
inspected repeatedly before this work and is not an untouched final test.

## Representation Repair

In each view, 370 of 1,118 training profiles and 83 of 279 validation profiles
had identical old Human and Traditional Bot direct inputs and selected
neighbors despite different labels. Their relative timestamps were absent
from the model inputs. Adding relative, whole-second timing features reduced
these direct-input collisions to zero in both development splits.

This repairs a representation defect. It does not establish that a complex
architecture is needed or that the controlled strategies match real accounts.

## Known-Policy Development Validation

Each entry is the Macro-F1 mean and sample standard deviation across seeds
42, 43, and 44. There are 54 completed validation runs: nine methods, three
seeds, and two views. Checkpoints were selected using these validation data;
the resulting scores are not independent confirmation.

| Method | Twibot22 | Quadbot |
|---|---:|---:|
| HGB statistics | 1.0000 +/- 0.0000 | 0.9919 +/- 0.0000 |
| Pooled logistic regression | 0.9964 +/- 0.0000 | 0.9847 +/- 0.0000 |
| Matched flattened-input MLP | 0.9909 +/- 0.0007 | 0.9829 +/- 0.0009 |
| Temporal evidence candidate | 0.9928 +/- 0.0021 | 0.9871 +/- 0.0022 |
| Without text/action interaction | 0.9936 +/- 0.0038 | 0.9880 +/- 0.0026 |
| Without classwise gate | 0.9940 +/- 0.0021 | 0.9883 +/- 0.0016 |
| Without style features | 0.9944 +/- 0.0007 | 0.9895 +/- 0.0029 |
| Without timing features | 0.7615 +/- 0.0013 | 0.8086 +/- 0.0034 |
| BotSSCL-inspired adapter | 0.9928 +/- 0.0012 | 0.9818 +/- 0.0014 |

HGB receives seven global timing statistics. The temporal candidate and matched
MLP also receive seven per-step statistics. Pooled comparators receive temporal
means. The underlying information source is shared, but the encodings are not
identical. The independently implemented BotSSCL-inspired adapter is not a
faithful reproduction of the original representation, binary task, or budget.

The full candidate is not the strongest comparator, and removing interaction,
gate, or style improves the mean in both views. These observations do not
support module necessity, architectural superiority, or SOTA claims.
The nearly perfect simple-model results indicate that the known controlled
policies are readily distinguishable once timing is included.

## Unseen Traditional Policy Screen

This post-hoc, one-seed screen comprises 18 completed runs: three held-out
policies, three methods, and two views. The held-out policy is excluded from
all fitting profile variants. Checkpoints are selected using an inner grouped
split of the remaining original training profiles. Scaling is fitted on the
inner fitting profiles only. Challenge scores are not used for checkpoint
selection. The challenge uses the held-out-policy original validation
profiles, not a new independent dataset.

| View | Held-out policy | HGB | Matched MLP | Temporal candidate |
|---|---|---:|---:|---:|
| Twibot22 | burst_scheduler | 0.5556 | 0.8160 | 0.5464 |
| Twibot22 | cyclic_reposter | 1.0000 | 0.9901 | 0.9934 |
| Twibot22 | uniform_scheduler | 1.0000 | 0.9798 | 0.9880 |
| Quadbot | burst_scheduler | 0.6628 | 0.8488 | 0.6465 |
| Quadbot | cyclic_reposter | 0.9926 | 0.9678 | 0.9852 |
| Quadbot | uniform_scheduler | 0.9940 | 0.9818 | 0.9817 |

The burst-policy failure is an actionable target, but this screen does not
establish cross-generator or real-platform generalization. The current archive
contains only one generated-model family. Multiple profile splits cannot
substitute for an unseen generator.

## Reproduction and Next Gates

Use the public source with authorized private inputs:

```sh
python -m unittest discover -s tests -v
python -m analysis.summarize_evidence_repair --help
python -m analysis.screen_unseen_policy --help
```

Run-level records include input hashes, selected epochs, predictions, and
checkpoints in the private development archive. The public summary contains
no raw source text, account identifiers, or credentials. Incomplete early
explorations are not counted as completed runs. Seed standard deviations
describe training variability, not confidence intervals for generalization.

Before replacing the paper method or preparing a submission, require fair
same-input backbone comparisons, three-seed unseen-policy evaluation, honest
module removals, independent confirmation after protocol freeze, and actual
author/source-permission approval. `configs/next_evidence_gates.json` records
unproven hypotheses and stopping rules; it does not authorize submission.
