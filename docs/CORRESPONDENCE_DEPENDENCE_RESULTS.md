# Correspondence Dependence Diagnosis

Completed 2026-10-15 00:43:22 Asia/Shanghai, CPU only. Nine retained compact
checkpoints, zero new fits. Four focused local and remote regression tests
passed. The full local suite before the later interaction extension passed
192 tests. No checkpoint, core dataset, manuscript or canonical/public source
was changed by the diagnostic.

## Measurements

Only fitting rows were forwarded. The nine checkpoints account for 22,392
prediction passes across overlapping fitting populations, not 22,392 distinct
profiles or independent observations.

| Fixed forward replacement | Changed classifications | Mean fitting cross-entropy change | Mean absolute logit change |
|---|---:|---:|---:|
| Zero discrepancies | 40 | +0.010703 | 0.319497 |
| Uniform current/past pooling | 3 | -0.000035 | 0.012987 |
| Current-block pairing only | 9 | +0.001304 | 0.114569 |

Mean normalized entropy across nontrivial history steps/checkpoints is
0.972683 text-to-action and 0.990583 action-to-text; 1 denotes uniform history.
Adaptive alignment still differs from uniform in representation space, so
do not call all attention weights exactly uniform or unused. The stronger
observation is that fitted predictions barely depend on that adaptation.

Baseline replay error was exactly zero. Maximum row-isolation difference
was 0.0000019074 and maximum expanded-linear decomposition difference was
0.0000030994, within tested floating-point tolerances. Both future-attention
masses were zero. No implementation failure was found by these checks.

## Interpretation and Limits

The learned alignment is close to uniform and has little fitted predictive
dependence in this diagnostic. The old discrepancy projection also repeats
direct current-block terms algebraically. This supports replacing the
redundant additive design, not merely renaming it.

These are post-hoc forward replacements, not retrained module removals.
Fitting cross-entropy/accuracy are not out-of-sample performance; the near
perfect fit itself limits how much a classification-change probe can reveal.
The findings do not prove that uniform pooling will match after retraining,
that correspondence cannot help another task, or that attention causes
generalization. No statistical significance or novelty is established.

The next frozen hypothesis is `CENTERED_HISTORY_INTERACTION_PLAN.md`.
Do not change its configuration based on this round's future scores.

## Private Receipt

Summary SHA-256:
`2300c2c25c232e7f29fe039e58512d2b9b1307554a70c8c6c1e34056a9c8a422`.
The downloaded receipt and diagnostic source manifest match this digest.
All nine original checkpoint hashes were unchanged. The source archive and
raw private machine paths remain outside the public result table.
