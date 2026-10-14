# Matched Input BotSSCL Mechanism Control Results

Completed 36 fitting-only control fits and 36 selected-checkpoint replays.
Both local and remote 150-test regressions passed. These are original-training
policy-exclusion development results, not independent final-test results.
No core account dataset, label, original split, released score or Word changed.

The adapter consumes the same flattened input blocks as the current MLP.
It is not a faithful original BotSSCL reproduction: the hash representation,
three/four-class head, end-to-end fine-tuning, training budget and grouped
selection differ. The original core corruption/NT-Xent mechanism is retained.
Source repository commit: `ed3b17588fc2c271947aade266bb1357fc835f61`.

Input equality is not architecture or total-compute equality. The adapter has
two encoder layers, no dropout and a training-only projection. Its corruption
donors come only from fitting profiles, not selection or challenge profiles.

All tables use Macro-F1 percentage units and sample SD across seeds 42/43/44.
Each seed pools its three excluded-policy predictions before metric calculation.

Control fits ran serially with CUDA disabled and two requested CPU threads.
The collected container resource limits were `cpu.max=50000 100000` and `memory.max=2147483648`.
Wall-clock times are operational records, not a speed comparison with older
candidate runs on different resources. The epoch ceilings also do not match
total optimization budgets because pretraining adds updates.

The initial replay process exited with signal 9 after fitting completed.
All 36 fits were retained. A single-fold-cache resource adapter replayed the
same frozen function without changing its model, prediction, normalization
or metric checks. Two synthetic equivalence/tampering tests passed locally
and remotely; the original failure record remains in the private evidence
records. Recovery performed zero additional research-data fits.

## Twibot22

### Pooled Comparison

| Method | Seed 42 | Seed 43 | Seed 44 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| SAM + policy contrast | 91.60 | 93.92 | 93.75 | 93.09 | 1.29 |
| SAM only | 95.63 | 93.89 | 90.38 | 93.30 | 2.67 |
| Policy contrast only | 94.00 | 93.99 | 95.88 | 94.62 | 1.09 |
| Ordinary matched MLP | 90.32 | 89.78 | 95.08 | 91.73 | 2.92 |
| Two-pass ERM | 94.42 | 93.11 | 93.88 | 93.80 | 0.66 |
| HGB statistics | 88.33 | 88.33 | 88.33 | 88.33 | 0.00 |
| Adapted BotDMM | 88.15 | 88.11 | 88.20 | 88.15 | 0.04 |
| BotSSCL-mechanism control | 99.07 | 99.01 | 97.40 | 98.50 | 0.95 |
| Same encoder without pretraining | 95.48 | 98.93 | 95.37 | 96.59 | 2.02 |

### Paired Full Minus New Controls

| Comparator | Seed 42 | Seed 43 | Seed 44 | Mean | Paired SD | Positive pairs |
|---|---:|---:|---:|---:|---:|---:|
| BotSSCL-mechanism control | -7.48 | -5.09 | -3.65 | -5.41 | 1.93 | 0/3 |
| Same encoder without pretraining | -3.88 | -5.00 | -1.62 | -3.50 | 1.72 | 0/3 |

### Same-Architecture Pretraining Difference

BotSSCL-mechanism control minus its supervised-only counterpart,
using the complete matched input and the same fitting/selection rules.
Additional pretraining compute remains part of this difference.

| Seed 42 | Seed 43 | Seed 44 | Mean | Paired SD | Positive pairs |
|---:|---:|---:|---:|---:|---:|
| 3.60 | 0.09 | 2.03 | 1.90 | 1.76 | 3/3 |

### Excluded Policy Breakdown

| Excluded policy | Condition | Seed 42 | Seed 43 | Seed 44 | Mean | Sample SD |
|---|---|---:|---:|---:|---:|---:|
| burst_scheduler | BotSSCL-mechanism control | 98.39 | 98.93 | 93.49 | 96.94 | 3.00 |
| burst_scheduler | Same encoder without pretraining | 87.47 | 99.02 | 88.55 | 91.68 | 6.38 |
| cyclic_reposter | BotSSCL-mechanism control | 99.47 | 99.02 | 99.64 | 99.38 | 0.32 |
| cyclic_reposter | Same encoder without pretraining | 99.64 | 98.93 | 98.76 | 99.11 | 0.47 |
| uniform_scheduler | BotSSCL-mechanism control | 99.37 | 99.10 | 99.01 | 99.16 | 0.19 |
| uniform_scheduler | Same encoder without pretraining | 99.01 | 98.83 | 98.56 | 98.80 | 0.23 |

### Class Breakdown

| Condition | Class | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---|---:|---:|---:|---:|
| SAM + policy contrast | Human | 90.46 | 92.69 | 91.79 | 91.65 |
| SAM + policy contrast | Traditional Bot | 88.45 | 92.15 | 91.42 | 90.67 |
| SAM + policy contrast | LLM Bot | 95.88 | 96.92 | 98.03 | 96.94 |
| BotSSCL-mechanism control | Human | 98.87 | 98.56 | 96.58 | 98.00 |
| BotSSCL-mechanism control | Traditional Bot | 99.20 | 99.07 | 96.50 | 98.26 |
| BotSSCL-mechanism control | LLM Bot | 99.16 | 99.42 | 99.11 | 99.23 |
| Same encoder without pretraining | Human | 93.85 | 98.52 | 93.46 | 95.28 |
| Same encoder without pretraining | Traditional Bot | 93.34 | 98.88 | 93.28 | 95.17 |
| Same encoder without pretraining | LLM Bot | 99.24 | 99.38 | 99.38 | 99.33 |

### Actual Control Training Budget

| Condition | Median selected epoch | Mean pretraining updates | Mean supervised updates | Mean seconds | Inference parameters | Training parameters |
|---|---:|---:|---:|---:|---:|---:|
| BotSSCL-mechanism control | 7 | 450.0 | 201.7 | 168.0 | 1105923 | 1122435 |
| Same encoder without pretraining | 7 | 0.0 | 203.3 | 13.2 | 1105923 | 1122435 |

### Conditional Profile Intervals

| Full minus comparator | Point difference | Conditional 95% interval |
|---|---:|---|
| BotSSCL-mechanism control | -5.41 | [-6.09, -4.72] |
| Same encoder without pretraining | -3.50 | [-4.08, -2.92] |
## Quadbot

### Pooled Comparison

| Method | Seed 42 | Seed 43 | Seed 44 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| SAM + policy contrast | 96.08 | 95.81 | 96.13 | 96.01 | 0.17 |
| SAM only | 92.41 | 96.54 | 96.24 | 95.06 | 2.30 |
| Policy contrast only | 94.71 | 96.26 | 92.58 | 94.52 | 1.85 |
| Ordinary matched MLP | 94.98 | 96.90 | 93.01 | 94.96 | 1.95 |
| Two-pass ERM | 94.68 | 95.48 | 95.89 | 95.35 | 0.61 |
| HGB statistics | 90.18 | 90.10 | 90.34 | 90.21 | 0.12 |
| Adapted BotDMM | 90.27 | 90.54 | 90.43 | 90.41 | 0.14 |
| BotSSCL-mechanism control | 94.21 | 92.62 | 97.60 | 94.81 | 2.54 |
| Same encoder without pretraining | 93.57 | 97.96 | 91.37 | 94.30 | 3.36 |

### Paired Full Minus New Controls

| Comparator | Seed 42 | Seed 43 | Seed 44 | Mean | Paired SD | Positive pairs |
|---|---:|---:|---:|---:|---:|---:|
| BotSSCL-mechanism control | 1.87 | 3.19 | -1.48 | 1.19 | 2.40 | 2/3 |
| Same encoder without pretraining | 2.52 | -2.15 | 4.76 | 1.71 | 3.53 | 2/3 |

### Same-Architecture Pretraining Difference

BotSSCL-mechanism control minus its supervised-only counterpart,
using the complete matched input and the same fitting/selection rules.
Additional pretraining compute remains part of this difference.

| Seed 42 | Seed 43 | Seed 44 | Mean | Paired SD | Positive pairs |
|---:|---:|---:|---:|---:|---:|
| 0.65 | -5.34 | 6.24 | 0.51 | 5.79 | 2/3 |

### Excluded Policy Breakdown

| Excluded policy | Condition | Seed 42 | Seed 43 | Seed 44 | Mean | Sample SD |
|---|---|---:|---:|---:|---:|---:|
| burst_scheduler | BotSSCL-mechanism control | 85.43 | 80.81 | 97.10 | 87.78 | 8.40 |
| burst_scheduler | Same encoder without pretraining | 82.82 | 97.31 | 75.29 | 85.14 | 11.19 |
| cyclic_reposter | BotSSCL-mechanism control | 98.06 | 97.60 | 97.67 | 97.78 | 0.25 |
| cyclic_reposter | Same encoder without pretraining | 98.33 | 98.27 | 98.14 | 98.24 | 0.10 |
| uniform_scheduler | BotSSCL-mechanism control | 98.30 | 97.81 | 98.03 | 98.05 | 0.25 |
| uniform_scheduler | Same encoder without pretraining | 98.24 | 98.30 | 97.23 | 97.92 | 0.60 |

### Class Breakdown

| Condition | Class | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---|---:|---:|---:|---:|
| SAM + policy contrast | Human | 94.33 | 93.39 | 93.51 | 93.74 |
| SAM + policy contrast | Traditional Bot | 96.67 | 95.22 | 95.04 | 95.64 |
| SAM + policy contrast | LLM Bot | 96.96 | 98.16 | 98.59 | 97.90 |
| SAM + policy contrast | Full-stack Agent | 96.37 | 96.46 | 97.38 | 96.73 |
| BotSSCL-mechanism control | Human | 90.01 | 87.67 | 96.16 | 91.28 |
| BotSSCL-mechanism control | Traditional Bot | 90.15 | 87.06 | 97.64 | 91.62 |
| BotSSCL-mechanism control | LLM Bot | 99.33 | 98.37 | 99.29 | 99.00 |
| BotSSCL-mechanism control | Full-stack Agent | 97.36 | 97.37 | 97.32 | 97.35 |
| Same encoder without pretraining | Human | 89.20 | 96.90 | 86.55 | 90.88 |
| Same encoder without pretraining | Traditional Bot | 88.20 | 98.28 | 83.06 | 89.84 |
| Same encoder without pretraining | LLM Bot | 99.07 | 99.16 | 99.20 | 99.14 |
| Same encoder without pretraining | Full-stack Agent | 97.80 | 97.51 | 96.66 | 97.32 |

### Actual Control Training Budget

| Condition | Median selected epoch | Mean pretraining updates | Mean supervised updates | Mean seconds | Inference parameters | Training parameters |
|---|---:|---:|---:|---:|---:|---:|
| BotSSCL-mechanism control | 7 | 600.0 | 264.4 | 215.3 | 1106052 | 1122564 |
| Same encoder without pretraining | 10 | 0.0 | 288.9 | 17.2 | 1106052 | 1122564 |

### Conditional Profile Intervals

| Full minus comparator | Point difference | Conditional 95% interval |
|---|---:|---|
| BotSSCL-mechanism control | 1.19 | [0.77, 1.61] |
| Same encoder without pretraining | 1.71 | [1.25, 2.17] |

## Interpretation Boundary

The frozen full candidate fails the requested overall two-view superiority
criterion. Twibot22 is 5.41 points below the pretrained control and 3.50 below
the same-encoder supervised-only control, with all three paired differences
negative in each comparison. Quadbot is 1.19 points above the pretrained
control and 1.71 above supervised-only, with two of three pairs positive.
Do not promote the current full recipe as the final submission method.

Pretraining adds 1.90 points on Twibot22 in 3/3 pairs and 0.51 points on
Quadbot in 2/3. This difference includes additional compute; it is not a
new original contribution or proof that pretraining universally helps.
The current candidate's existing SAM/contrast removal weaknesses remain.
No further recipe search is automatically authorized by these scores.

These comparisons address one compatible mechanism and its same-architecture
supervised-only control. They do not reproduce an original binary detector,
complete the recent bot-baseline table, resolve SAM/contrast module necessity,
or establish task-specific novelty. Profile intervals condition on the three
selected models, exposed development profiles and frozen recipes; they do not
override paired-seed instability or account for selecting the candidate.
No SOTA or CCF-B submission readiness follows from these scores.

## Evidence Digests

Retained master SHA-256:
`b5ce0d14eed32b053f5ca78409cc07f0d8ff595d9511e762a127e7a5c2e7b16b`.

Control summary SHA-256:
`fdbf988d717885a8edbb478d25f6124a7d3bf682d3c7bc37aa561c6f98dc507e`.

Privately archived verification evidence SHA-256:
`95beb6667b6a6cc4caf2898d51847422a274f5193df5a649f2a236efce6af764`.

The private archive is not a raw-data or checkpoint release. All selected
checkpoints were replayed remotely; downloaded prediction artifacts were
independently checked and used to recompute every pooled seed metric.
