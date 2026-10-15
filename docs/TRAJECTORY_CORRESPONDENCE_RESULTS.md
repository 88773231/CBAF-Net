# Trajectory Correspondence Feasibility Results

## Decision

The fixed two-module candidate fails its predeclared feasibility gate.
Correspondence has a positive development signal, but the adjacent-transition
path does not have consistent support. Do not promote the complete architecture,
automatically enlarge its seed search, or rewrite the manuscript around it.

This is not an execution failure. All twelve planned research fits and twelve
checkpoint replays completed on October 14, 2026, at 22:21:41 Asia/Shanghai.
The isolated deployment took 7 minutes 43 seconds including regression,
loading/hashing, training, replays and evidence packaging. It used CPU only.
There was no dataset generation or original validation/test access.

## Scores And Scope

The table reports **known-policy inner-selection Macro-F1 in percent**.
For example, the Burst-excluded fitting fold scores only Cyclic and Uniform
inner-selection profiles. It does not score the excluded Burst policy.
These numbers are neither main test results nor substitutes for the older
development challenge results.

| Condition | Burst-Excluded Fold | Cyclic-Excluded Fold | Uniform-Excluded Fold | Arithmetic Fold Mean |
|---|---:|---:|---:|---:|
| Correspondence + transition | 98.36 | 98.98 | 98.79 | 98.71 |
| Without correspondence | 97.75 | 98.77 | 98.38 | 98.30 |
| Without transition | 98.78 | 99.18 | 98.59 | 98.85 |
| Without both | 98.16 | 98.57 | 98.78 | 98.50 |
| Preserved supervised encoder control | 97.76 | 98.78 | 98.39 | 98.31 |

The selection predictions cover 185 distinct base profiles. Profiles appear
in two known-policy folds, so these are overlapping development comparisons,
not three independent replications. This is one training seed, 42.

The complete condition's paired differences, in percentage points:

| Compared With | Burst Fold | Cyclic Fold | Uniform Fold | Mean | Positive Folds |
|---|---:|---:|---:|---:|---:|
| Without correspondence | +0.615 | +0.206 | +0.412 | +0.411 | 3/3 |
| Without transition | -0.416 | -0.204 | +0.203 | -0.139 | 1/3 |
| Without both | +0.204 | +0.408 | +0.004 | +0.205 | 3/3 |

The full condition exceeds the supervised control mean by 0.403 points,
but fails the transition-removal requirement. The tiny Uniform difference
against double removal does not establish a practically important gain.

The correspondence-only removal has a +0.345-point mean against double
removal and wins 2/3 paired folds. That is a secondary descriptive comparison,
not a newly promoted final method. Any simplified architecture must be
explicitly specified and separately confirmed; it must not inherit the
failed full model's claims or be selected using excluded-policy scores.

## Capacity And Compute

| Condition | Serialized Parameters | Inactive Serialized Weights |
|---|---:|---:|
| Correspondence + transition | 662,734 | 0 |
| Without correspondence | 646,340 | 8,192 |
| Without transition | 646,158 | 8,192 |
| Without both | 629,764 | 16,384 |

Shared parameters had identical seeded initialization in every paired
condition. Removed feature slots remained zero in the same-shaped readout.
This is not exact capacity matching; inactive columns and removed modules
change effective capacity. Selected epochs and actual compute also differ.
The separate supervised control changes architecture and has 1,106,052
inference parameters, so matching inputs does not match model or compute.

The twelve fits used 129 epochs and 2,580 optimizer updates in total.
Their logged fitting time sums to 368.924 seconds. Keep every epoch history,
not only the selected checkpoints.

## Verification And Preservation

- Eight targeted tests passed remotely before research execution.
- Twelve selected checkpoints replayed with zero probability differences.
- A read-only local audit verified all twelve checkpoint/prediction hashes,
  envelopes, ordered identities, metrics, best-epoch rules and shared
  initialization digests. It added no research fits and opened no challenge
  predictions or research tensors.
- Four audit tests passed, including a complete fixture receipt and rejection
  of a score invented consistently in the run record and summary.
- The final full local regression passed all 177 tests in 313.942 seconds.
- Core data, frozen control evidence, canonical remote checkout and Word
  were unchanged. Public GitHub main/release were not updated.

Private summary SHA256:
`a573fbe1496f55aa8edfc5e606fb6f77862623715c03e90318557d3c59e5777c`.

Private evidence archive SHA256:
`923c4f24a6ff4cbd7970ced336c2bb3f5d97afc2a842521b465fed2bf1a043f3`.

The fixed plan, code archive and source manifest predate the research fits.
The post-run local auditor is separate from the frozen training sources.

## Next Boundary

Do not keep the transition path merely to advertise a second innovation.
Stop expanding this full architecture. Correspondence remains a hypothesis
for a simpler model, not proof of algorithmic originality, robustness or SOTA.
Any next recipe needs a frozen specification, fair controls and repeated-run
support, followed by evaluation not used for design. The current data
permissions/generator-archive and independent-evaluation gaps remain.

No manuscript replacement or submission follows from this screen. The paper,
diagram, equations and repository must eventually describe the actual
supported final architecture, not this rejected two-module candidate.
