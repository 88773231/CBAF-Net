# Paired standalone-branch audit

This audit uses the immutable `v1.1.0` test prediction files, three fixed
seeds (42, 43, 44), and 5,000 synchronized bootstrap draws of complete
`base_profile_id` groups. It is an additional analysis of the released model,
not a new model result or a replacement for the original strict rerun.

| View | Contrast | Macro-F1 difference | 95% profile-cluster interval | MCC difference | 95% interval |
| --- | --- | ---: | ---: | ---: | ---: |
| Twibot22 | CBAF-Net - BotDMM | 0.001441 | [-0.002599, 0.005344] | 0.013153 | [0.006773, 0.019515] |
| Twibot22 | CBAF-Net - BSE-only | 0.001552 | [-0.004518, 0.007560] | -0.016614 | [-0.027692, -0.005726] |
| Quadbot | CBAF-Net - BotDMM | 0.007128 | [-0.002127, 0.016504] | 0.032112 | [0.021088, 0.043001] |
| Quadbot | CBAF-Net - BSE-only | 0.000423 | [-0.003680, 0.004776] | -0.005071 | [-0.012113, 0.002113] |

Both Macro-F1 contrasts include zero on both views. BSE-only has higher
Accuracy and MCC than the fusion; the Twibot22 MCC regression has a
negative paired interval. The interval conditions on these three seeds and
the controlled profile population, not on new source domains or generators.

Recompute with the command in `README.md`. The generated
`results/paired_profile_cluster_bootstrap.json` is separate from the frozen
`results/strict_postprocessed/` package. No test prediction was modified or
used for model selection in this audit.
