# Fixed Three-Seed Factor Replication

Seed 42 is reused byte-for-byte; seeds 43/44 are new fits of the same recipe.
These are post-hoc, original-training-only policy challenges, not independent confirmation.
Macro-F1 values below are percentages; SD is sample SD across three seeds.
Each seed also changes the deterministic inner fitting/selection profile partition.
The two views share profiles and are not independent datasets.

| View | Policy | Method | Seed 42 | Seed 43 | Seed 44 | Mean | Seed SD |
|---|---|---|---:|---:|---:|---:|---:|
| Twibot22 | burst_scheduler | factor_residual | 79.20 | 55.51 | 55.88 | 63.53 | 13.57 |
| Twibot22 | burst_scheduler | without_factor_fusion | 55.49 | 55.60 | 55.21 | 55.43 | 0.20 |
| Twibot22 | burst_scheduler | without_interaction_residual | 90.78 | 84.48 | 58.57 | 77.94 | 17.07 |
| Twibot22 | burst_scheduler | erm_mlp | 75.86 | 71.25 | 90.69 | 79.27 | 10.16 |
| Twibot22 | burst_scheduler | hgb_statistics | 55.45 | 55.45 | 55.45 | 55.45 | 0.00 |
| Twibot22 | burst_scheduler | botdmm_timing | 55.15 | 55.23 | 55.23 | 55.20 | 0.05 |
| Twibot22 | cyclic_reposter | factor_residual | 99.38 | 99.29 | 99.64 | 99.44 | 0.19 |
| Twibot22 | cyclic_reposter | without_factor_fusion | 99.20 | 99.38 | 99.64 | 99.41 | 0.22 |
| Twibot22 | cyclic_reposter | without_interaction_residual | 99.02 | 97.78 | 98.76 | 98.52 | 0.66 |
| Twibot22 | cyclic_reposter | erm_mlp | 97.79 | 98.31 | 98.13 | 98.07 | 0.27 |
| Twibot22 | cyclic_reposter | hgb_statistics | 99.82 | 99.82 | 99.82 | 99.82 | 0.00 |
| Twibot22 | cyclic_reposter | botdmm_timing | 99.82 | 99.47 | 99.82 | 99.70 | 0.21 |
| Twibot22 | uniform_scheduler | factor_residual | 99.55 | 99.37 | 99.19 | 99.37 | 0.18 |
| Twibot22 | uniform_scheduler | without_factor_fusion | 99.37 | 99.55 | 99.37 | 99.43 | 0.10 |
| Twibot22 | uniform_scheduler | without_interaction_residual | 98.47 | 98.47 | 98.56 | 98.50 | 0.05 |
| Twibot22 | uniform_scheduler | erm_mlp | 95.40 | 96.58 | 96.30 | 96.09 | 0.62 |
| Twibot22 | uniform_scheduler | hgb_statistics | 99.55 | 99.55 | 99.55 | 99.55 | 0.00 |
| Twibot22 | uniform_scheduler | botdmm_timing | 99.37 | 99.46 | 99.37 | 99.40 | 0.05 |
| Quadbot | burst_scheduler | factor_residual | 67.48 | 71.03 | 72.48 | 70.33 | 2.57 |
| Quadbot | burst_scheduler | without_factor_fusion | 70.32 | 93.98 | 66.29 | 76.86 | 14.96 |
| Quadbot | burst_scheduler | without_interaction_residual | 94.34 | 80.17 | 78.86 | 84.46 | 8.59 |
| Quadbot | burst_scheduler | erm_mlp | 90.78 | 95.40 | 85.30 | 90.49 | 5.05 |
| Quadbot | burst_scheduler | hgb_statistics | 65.78 | 65.74 | 66.00 | 65.84 | 0.14 |
| Quadbot | burst_scheduler | botdmm_timing | 65.35 | 65.97 | 65.71 | 65.68 | 0.31 |
| Quadbot | cyclic_reposter | factor_residual | 97.46 | 98.40 | 98.06 | 97.97 | 0.48 |
| Quadbot | cyclic_reposter | without_factor_fusion | 97.86 | 98.86 | 98.39 | 98.37 | 0.50 |
| Quadbot | cyclic_reposter | without_interaction_residual | 95.85 | 97.99 | 95.98 | 96.61 | 1.20 |
| Quadbot | cyclic_reposter | erm_mlp | 96.99 | 97.85 | 96.57 | 97.14 | 0.65 |
| Quadbot | cyclic_reposter | hgb_statistics | 98.73 | 98.73 | 98.60 | 98.69 | 0.08 |
| Quadbot | cyclic_reposter | botdmm_timing | 98.80 | 99.20 | 98.93 | 98.98 | 0.20 |
| Quadbot | uniform_scheduler | factor_residual | 98.30 | 98.64 | 98.78 | 98.57 | 0.25 |
| Quadbot | uniform_scheduler | without_factor_fusion | 98.44 | 98.71 | 98.98 | 98.71 | 0.27 |
| Quadbot | uniform_scheduler | without_interaction_residual | 97.16 | 96.80 | 96.28 | 96.75 | 0.44 |
| Quadbot | uniform_scheduler | erm_mlp | 97.00 | 97.42 | 96.42 | 96.94 | 0.50 |
| Quadbot | uniform_scheduler | hgb_statistics | 98.30 | 98.03 | 98.71 | 98.35 | 0.34 |
| Quadbot | uniform_scheduler | botdmm_timing | 98.85 | 98.72 | 98.99 | 98.85 | 0.13 |

## Paired Differences

Mean differences are candidate minus comparator, in percentage points.
Counts are positive paired seed differences, not independent significance tests.

| View | Policy | Candidate | Comparator | Mean Delta | Delta SD | Positive Seeds |
|---|---|---|---|---:|---:|---:|
| Twibot22 | burst_scheduler | factor_residual | without_factor_fusion | +8.10 | 13.53 | 2/3 |
| Twibot22 | burst_scheduler | factor_residual | without_interaction_residual | -14.41 | 13.36 | 0/3 |
| Twibot22 | burst_scheduler | factor_residual | erm_mlp | -15.73 | 19.07 | 1/3 |
| Twibot22 | burst_scheduler | factor_residual | hgb_statistics | +8.09 | 13.57 | 3/3 |
| Twibot22 | burst_scheduler | factor_residual | botdmm_timing | +8.33 | 13.62 | 3/3 |
| Twibot22 | burst_scheduler | without_interaction_residual | factor_residual | +14.41 | 13.36 | 3/3 |
| Twibot22 | burst_scheduler | without_interaction_residual | without_factor_fusion | +22.51 | 16.89 | 3/3 |
| Twibot22 | burst_scheduler | without_interaction_residual | erm_mlp | -1.32 | 26.69 | 2/3 |
| Twibot22 | burst_scheduler | without_interaction_residual | hgb_statistics | +22.50 | 17.07 | 3/3 |
| Twibot22 | burst_scheduler | without_interaction_residual | botdmm_timing | +22.74 | 17.10 | 3/3 |
| Twibot22 | cyclic_reposter | factor_residual | without_factor_fusion | +0.03 | 0.14 | 1/3 |
| Twibot22 | cyclic_reposter | factor_residual | without_interaction_residual | +0.92 | 0.58 | 3/3 |
| Twibot22 | cyclic_reposter | factor_residual | erm_mlp | +1.36 | 0.34 | 3/3 |
| Twibot22 | cyclic_reposter | factor_residual | hgb_statistics | -0.39 | 0.19 | 0/3 |
| Twibot22 | cyclic_reposter | factor_residual | botdmm_timing | -0.27 | 0.15 | 0/3 |
| Twibot22 | cyclic_reposter | without_interaction_residual | factor_residual | -0.92 | 0.58 | 0/3 |
| Twibot22 | cyclic_reposter | without_interaction_residual | without_factor_fusion | -0.89 | 0.71 | 0/3 |
| Twibot22 | cyclic_reposter | without_interaction_residual | erm_mlp | +0.44 | 0.90 | 2/3 |
| Twibot22 | cyclic_reposter | without_interaction_residual | hgb_statistics | -1.30 | 0.66 | 0/3 |
| Twibot22 | cyclic_reposter | without_interaction_residual | botdmm_timing | -1.18 | 0.46 | 0/3 |
| Twibot22 | uniform_scheduler | factor_residual | without_factor_fusion | -0.06 | 0.21 | 1/3 |
| Twibot22 | uniform_scheduler | factor_residual | without_interaction_residual | +0.87 | 0.23 | 3/3 |
| Twibot22 | uniform_scheduler | factor_residual | erm_mlp | +3.28 | 0.76 | 3/3 |
| Twibot22 | uniform_scheduler | factor_residual | hgb_statistics | -0.18 | 0.18 | 1/3 |
| Twibot22 | uniform_scheduler | factor_residual | botdmm_timing | -0.03 | 0.19 | 1/3 |
| Twibot22 | uniform_scheduler | without_interaction_residual | factor_residual | -0.87 | 0.23 | 0/3 |
| Twibot22 | uniform_scheduler | without_interaction_residual | without_factor_fusion | -0.93 | 0.14 | 0/3 |
| Twibot22 | uniform_scheduler | without_interaction_residual | erm_mlp | +2.41 | 0.60 | 3/3 |
| Twibot22 | uniform_scheduler | without_interaction_residual | hgb_statistics | -1.05 | 0.05 | 0/3 |
| Twibot22 | uniform_scheduler | without_interaction_residual | botdmm_timing | -0.90 | 0.09 | 0/3 |
| Quadbot | burst_scheduler | factor_residual | without_factor_fusion | -6.54 | 14.92 | 1/3 |
| Quadbot | burst_scheduler | factor_residual | without_interaction_residual | -14.13 | 11.12 | 0/3 |
| Quadbot | burst_scheduler | factor_residual | erm_mlp | -20.16 | 6.38 | 0/3 |
| Quadbot | burst_scheduler | factor_residual | hgb_statistics | +4.49 | 2.49 | 3/3 |
| Quadbot | burst_scheduler | factor_residual | botdmm_timing | +4.65 | 2.35 | 3/3 |
| Quadbot | burst_scheduler | without_interaction_residual | factor_residual | +14.13 | 11.12 | 3/3 |
| Quadbot | burst_scheduler | without_interaction_residual | without_factor_fusion | +7.59 | 19.40 | 2/3 |
| Quadbot | burst_scheduler | without_interaction_residual | erm_mlp | -6.04 | 9.40 | 1/3 |
| Quadbot | burst_scheduler | without_interaction_residual | hgb_statistics | +18.62 | 8.65 | 3/3 |
| Quadbot | burst_scheduler | without_interaction_residual | botdmm_timing | +18.78 | 8.86 | 3/3 |
| Quadbot | cyclic_reposter | factor_residual | without_factor_fusion | -0.40 | 0.07 | 0/3 |
| Quadbot | cyclic_reposter | factor_residual | without_interaction_residual | +1.37 | 0.87 | 3/3 |
| Quadbot | cyclic_reposter | factor_residual | erm_mlp | +0.84 | 0.57 | 3/3 |
| Quadbot | cyclic_reposter | factor_residual | hgb_statistics | -0.72 | 0.50 | 0/3 |
| Quadbot | cyclic_reposter | factor_residual | botdmm_timing | -1.00 | 0.29 | 0/3 |
| Quadbot | cyclic_reposter | without_interaction_residual | factor_residual | -1.37 | 0.87 | 0/3 |
| Quadbot | cyclic_reposter | without_interaction_residual | without_factor_fusion | -1.76 | 0.80 | 0/3 |
| Quadbot | cyclic_reposter | without_interaction_residual | erm_mlp | -0.53 | 0.64 | 1/3 |
| Quadbot | cyclic_reposter | without_interaction_residual | hgb_statistics | -2.08 | 1.17 | 0/3 |
| Quadbot | cyclic_reposter | without_interaction_residual | botdmm_timing | -2.37 | 1.01 | 0/3 |
| Quadbot | uniform_scheduler | factor_residual | without_factor_fusion | -0.14 | 0.07 | 0/3 |
| Quadbot | uniform_scheduler | factor_residual | without_interaction_residual | +1.83 | 0.68 | 3/3 |
| Quadbot | uniform_scheduler | factor_residual | erm_mlp | +1.63 | 0.63 | 3/3 |
| Quadbot | uniform_scheduler | factor_residual | hgb_statistics | +0.22 | 0.34 | 2/3 |
| Quadbot | uniform_scheduler | factor_residual | botdmm_timing | -0.28 | 0.24 | 0/3 |
| Quadbot | uniform_scheduler | without_interaction_residual | factor_residual | -1.83 | 0.68 | 0/3 |
| Quadbot | uniform_scheduler | without_interaction_residual | without_factor_fusion | -1.97 | 0.71 | 0/3 |
| Quadbot | uniform_scheduler | without_interaction_residual | erm_mlp | -0.20 | 0.39 | 1/3 |
| Quadbot | uniform_scheduler | without_interaction_residual | hgb_statistics | -1.61 | 0.72 | 0/3 |
| Quadbot | uniform_scheduler | without_interaction_residual | botdmm_timing | -2.10 | 0.53 | 0/3 |

## Boundary

All six methods and all three policies are retained, including negative ablations.
No new architecture, hyperparameter search, seed exclusion or automatic promotion is used.
Original validation/test tensors are not model inputs. Old seed-42 outputs remain preserved.
HGB receives global summaries, neural methods receive step inputs, and adapted BotDMM
has additional unlabelled transductive feature-kNN context. Capacity and information differ.
Factor targets encode synthetic construction, not independently annotated account intent.
More seeds quantify recipe sensitivity; they do not establish external validity or SOTA.
This is development evidence and does not authorize replacing manuscript claims.

## Diagnostic Decision

The fixed full candidate failed as a supported replacement method. On the
burst condition, every seed in each view favored removing the residual.
The mean full-minus-removal differences were -14.41 points for Twibot22
and -14.13 points for Quadbot. The residual cannot be presented as a
consistently beneficial contribution.

Removing the residual is not by itself a successful final redesign.
Burst Macro-F1 seed SD was 17.07 points in Twibot22 and 8.59 points in
Quadbot. Its burst means, 77.94% and 84.46%, were below the ERM MLP means
of 79.27% and 90.49%. It also trailed adapted BotDMM on the cyclic and
uniform policy means in both views.

The large seed-42 burst result did not reproduce as a stable advantage
over every control. Do not present only that seed, promote the better
removal by renaming it, or retain two unsupported module claims.
The next architectural hypothesis must address a specific prediction
limitation and define its controls before training. It is not authorized
to repeat a residual-weight search on these exposed challenge scores.
See `MODEL_REVISION_GATE.md` for the distinction between preserved
development records and supported final submission evidence.

The 72 new fits and replays completed on October 13, 2026, at
19:41:38 Asia/Shanghai. The new-job duration was about 7 minutes 52 seconds.
The original 36 fits and replays were reused unchanged, giving 108
verified fits. All 97 regression tests passed. Dataset records, old
published results and the working Word manuscript were preserved.
