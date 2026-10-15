# Final Representation Trial Results

## Decision

The bounded CPU study finished on 2026-10-15 at 04:17:42 UTC
(12:17:42 Asia/Shanghai), after approximately 24 minutes. All 12
predeclared fits and 12 checkpoint replays completed. Independent metric
recomputation and 27 evidence-member hash checks passed.

Semantic features help the ordinary MLP, but do not help the matched
cue-encoder architecture on average. Both the original 0.25-percentage-point
gate and the prospectively relaxed small-gain rule fail for the complete
study. Close this semantic-led method branch. Do not start the proposed
conditional-residual experiment, search another encoder or promote the
favorable MLP pair as a new original architecture.

## Matched Results

All numbers below are Quadbot known-policy inner-selection Macro-F1 in
percent. Each fold excludes the named scheduler policy from both fitting
and selection. The excluded-policy accounts themselves were not encoded
or scored. Fold populations overlap; these are not three independent test
replications. Only training seed 42 was fitted.

| Condition | Burst Fold | Cyclic Fold | Uniform Fold | Mean |
|---|---:|---:|---:|---:|
| Lexical hash MLP | 98.3628 | 98.1603 | 98.7878 | 98.4370 |
| Frozen semantic MLP | 98.3628 | 98.9812 | 99.3943 | 98.9128 |
| Lexical hash cue encoders | 98.5713 | 99.1868 | 98.9884 | 98.9155 |
| Frozen semantic cue encoders | 98.3690 | 98.9833 | 99.1894 | 98.8472 |

| Semantic Minus Hash | Burst | Cyclic | Uniform | Mean | Positive Folds |
|---|---:|---:|---:|---:|---:|
| MLP, percentage points | 0.0000 | +0.8208 | +0.6065 | +0.4758 | 2/3 |
| Cue encoders, percentage points | -0.2024 | -0.2035 | +0.2010 | -0.0683 | 1/3 |

The semantic MLP does not exceed the hash cue-encoder mean either.
That descriptive cross-architecture comparison does not isolate a text
representation effect. The condition named `backbone` in the machine
records is this small graph-free cue-encoder control, not the released
BotDMM graph/LSTM backbone.

## Controls and Verification

Text representations are dimension-matched at 384. Each architecture pair
has identical initial parameter values and parameter counts: 327,428 for
the MLP and 116,804 for the cue encoders. Nontext features, fitting-only
standardizers, partition identities, training objective and selection rule
match within every representation pair. Each saved checkpoint reproduces
its probabilities with maximum absolute difference zero.

The encoder is frozen `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Length bucketing changes
padding only, restores text order and passed the prior 1e-6 equivalence
check. Frozen-text memoization occurs only after an allowed text request.

The executed-source archive contains 141 files and has SHA-256
`e21b20cef6f2d1592319e70fface8c56615988c45037b3638072a842ecd543cc`.
The private study summary SHA-256 is
`2cd19bf71d34fd2dec36842b286783e8437f4d019e7673076c89f688d209e1a8`.
The private evidence archive SHA-256 is
`6f19015142137d0c4d426e5caf284773310c60a54656268d3727262f10a7cc47`.

The final full local regression passed 239 tests. Three additional export
tests passed separately; they check privacy-oriented field selection,
preservation of existing exports and separation of the original and
relaxed decisions. Software tests do not establish model effectiveness.

## Release and Manuscript Boundary

`results/semantic_representation/selection_summary.json` contains audited
aggregate evidence only, with no raw text, embeddings, account identifiers
or private machine paths. Its SHA-256 is
`0ff1abb98092adc2e09e2faa7a8acd3d836e78fb9c3bf34b43a210e242e5c145`.
The executed source and full private evidence remain preserved separately.

No new accounts were generated. The original training files, labels,
partitions and working manuscript passed their preservation checks.
No original validation/test or excluded-policy account forward was run.
No remote task or paid GPU was used, and no training process remains.

This is neither a three-class result nor an original-test improvement,
module ablation, novel architecture, significance result or SOTA claim.
The old manuscript and immutable `v1.1.0` test table remain associated only
with the released BotDMM/BSE probability mixture. They have not been
replaced by this development experiment.

Honor the finish-first instruction: stop method expansion and proceed to
evidence-backed manuscript/release closeout. No supported new model
replacement has been established. A CCF-C fallback still needs a suitable
contribution, compatible venue, accurate data permissions and author
approval; changing the venue does not establish those requirements.
Remote/public synchronization and a rendered final submission manuscript
are subsequent tasks, not completed by this local evidence export.
