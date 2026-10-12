# Policy Robustness Development Results

This is a post-hoc development screen on previously inspected validation
challenges. It is not an untouched test, an established final method, or
evidence of SOTA. The main account data and old paper results are unchanged.

## Twibot22

### burst_scheduler

| Method | Macro-F1 mean +/- sample seed SD |
|---|---:|
| hgb_statistics | 0.5556 +/- 0.0000 |
| erm_mlp | 0.7869 +/- 0.0258 |
| group_dro_mlp | 0.7962 +/- 0.0079 |
| rdrop_mlp | 0.7634 +/- 0.0326 |
| group_dro_rdrop_mlp | 0.7225 +/- 0.0089 |

- burst_scheduler: combined-minus-ERM descriptive Macro-F1 difference -0.0644, profile-resampling 95% interval [-0.0887, -0.0432].

### cyclic_reposter

| Method | Macro-F1 mean +/- sample seed SD |
|---|---:|
| hgb_statistics | 1.0000 +/- 0.0000 |
| erm_mlp | 0.9967 +/- 0.0033 |
| group_dro_mlp | 0.9967 +/- 0.0033 |
| rdrop_mlp | 0.9945 +/- 0.0019 |
| group_dro_rdrop_mlp | 0.9967 +/- 0.0033 |

- cyclic_reposter: combined-minus-ERM descriptive Macro-F1 difference 0.0000, profile-resampling 95% interval [-0.0033, 0.0033].

### uniform_scheduler

| Method | Macro-F1 mean +/- sample seed SD |
|---|---:|
| hgb_statistics | 1.0000 +/- 0.0000 |
| erm_mlp | 0.9770 +/- 0.0125 |
| group_dro_mlp | 0.9717 +/- 0.0145 |
| rdrop_mlp | 0.9675 +/- 0.0148 |
| group_dro_rdrop_mlp | 0.9718 +/- 0.0106 |

- uniform_scheduler: combined-minus-ERM descriptive Macro-F1 difference -0.0052, profile-resampling 95% interval [-0.0175, 0.0070].

## Quadbot

### burst_scheduler

| Method | Macro-F1 mean +/- sample seed SD |
|---|---:|
| hgb_statistics | 0.6628 +/- 0.0019 |
| erm_mlp | 0.8534 +/- 0.0351 |
| group_dro_mlp | 0.8136 +/- 0.1129 |
| rdrop_mlp | 0.8325 +/- 0.0236 |
| group_dro_rdrop_mlp | 0.8730 +/- 0.0314 |

- burst_scheduler: combined-minus-ERM descriptive Macro-F1 difference 0.0196, profile-resampling 95% interval [0.0033, 0.0367].

### cyclic_reposter

| Method | Macro-F1 mean +/- sample seed SD |
|---|---:|
| hgb_statistics | 0.9926 +/- 0.0025 |
| erm_mlp | 0.9730 +/- 0.0026 |
| group_dro_mlp | 0.9762 +/- 0.0037 |
| rdrop_mlp | 0.9730 +/- 0.0024 |
| group_dro_rdrop_mlp | 0.9738 +/- 0.0052 |

- cyclic_reposter: combined-minus-ERM descriptive Macro-F1 difference 0.0009, profile-resampling 95% interval [-0.0024, 0.0042].

### uniform_scheduler

| Method | Macro-F1 mean +/- sample seed SD |
|---|---:|
| hgb_statistics | 0.9960 +/- 0.0017 |
| erm_mlp | 0.9817 +/- 0.0054 |
| group_dro_mlp | 0.9818 +/- 0.0052 |
| rdrop_mlp | 0.9817 +/- 0.0054 |
| group_dro_rdrop_mlp | 0.9838 +/- 0.0047 |

- uniform_scheduler: combined-minus-ERM descriptive Macro-F1 difference 0.0020, profile-resampling 95% interval [-0.0031, 0.0083].

## Interpretation Boundary

All four neural methods have the same inference architecture and inputs.
The comparison changes training objectives, not architectural modules.
R-Drop uses two training forward passes, so epoch budgets are equal but
computational work is not. HGB uses global statistics, not identical
flattened encoding. The current ERM control also uses the newly frozen
weight decay, profile batching, and worst-known-policy checkpoint rule;
its differences from the earlier MLP screen are not objective-only gains.

Intervals resample whole paired profiles synchronously across fixed
training seeds. Prior challenge exposure and multiple explored mechanisms
prevent treating them as confirmatory significance or external robustness.

No result is filtered because an ablation is better. A failed combination
must not be relabelled as a successful final method.

