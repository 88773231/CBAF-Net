# Centered History Interaction Results

Completed 2026-10-15 at 01:11:04 Asia/Shanghai. Six new fits, nine retained
exact-partition controls and fifteen freshly instantiated checkpoint
replays completed. Five targeted remote tests and the full 197-test local
suite passed. The local receipt audit recomputed all fifteen cells from
saved probabilities and verified parameter, source, identity and selection
records. The core dataset/manuscript digests and canonical source were
unchanged. CUDA was disabled throughout.

## Complete Comparison

Every number below is known-policy inner-selection Macro-F1, not final test
performance. All conditions use training seed 42 on the existing seed-42
partitions. The three folds overlap and contain 185 distinct selection
profiles collectively; they are not independent replications.

| Condition | Burst-Excluded Fold | Cyclic-Excluded Fold | Uniform-Excluded Fold | Mean |
|---|---:|---:|---:|---:|
| Centered history interaction | 98.16 | 98.78 | 98.79 | 98.57 |
| Without centering | 97.96 | 98.57 | 98.38 | 98.30 |
| Direct-only removal | 98.16 | 98.57 | 98.78 | 98.50 |
| Supervised encoder | 97.76 | 98.78 | 98.39 | 98.31 |
| BotSSCL mechanism control | 98.16 | 99.18 | 97.78 | 98.37 |

The fold names identify which scheduler was excluded from fitting and inner
selection. No excluded-scheduler predictions were made. Rounding makes some
slightly different scores look equal; the gate uses exact saved values.

| Paired Full-Minus-Control Difference | Mean Percentage Points | Positive Folds | Frozen Gate |
|---|---:|---:|---|
| Without centering | +0.2727 | 3/3 | Pass |
| Direct-only removal | +0.0689 | 2/3, one exact tie | Fail: module gain below 0.25 points |
| Supervised encoder | +0.2670 | 2/3, one exact tie | Pass |
| BotSSCL mechanism control | +0.1989 | 1/3 | Fail: positive in only one fold |

The previous compact correspondence condition on the same seed/partitions
averaged 98.8488%. The new condition averaged 98.5730%, a decrease of
0.2758 percentage points. Do not describe this round as an overall improvement
over that retained architecture or compare one new seed to the earlier
three-seed aggregate as if the populations were matched.

## Decision

The frozen continuation gate failed. Centering has a positive result within
this particular interaction design, but the whole relation branch adds only
0.0689 points over its removal, and the stronger pretrained control wins in
two folds. Do not start seed expansion, tune this recipe repeatedly, replace
the manuscript, redraw a final-model figure or update GitHub as though this
were a supported new method.

Retain the implementation and complete evidence as an unsuccessful development
branch. A next design requires a newly identified prediction limitation, not
simply another generic named module. Current results do not establish stable
module improvement, final-test superiority, algorithmic novelty, a "70%"
CCF-B quality level or an acceptance probability.

The attention-dependence diagnosis remains useful despite this failed redesign.
Known-policy selection is already near its ceiling and may not measure the
desired real-world mechanism distinction. Fair recent comparisons, exposure
history, construction provenance and unexposed confirmation remain separate
gaps, recorded in `CCFB_READINESS_GATES.md`.

## Retained Digests

| Artifact | SHA-256 |
|---|---|
| Summary | `147cb611efa65dbb5c72fc920d71336a81285ae2892882ae451a4d783ff1b0b2` |
| Evidence archive | `3e69f1a348852f91097e2ba8ede845ffb9dd03936dcfe06a7c0810c0b5897f34` |
| Frozen source archive | `3b1c85ac0f08b1775938243669bf0e4005680c8134c44d5fce59b676d3f7864c` |

The private prediction/checkpoint archive remains outside the public paper
table. No remote analysis task from this round remains running.
