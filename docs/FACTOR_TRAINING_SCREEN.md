# Training Only Factor Fusion Screen

This is one bounded architecture hypothesis, not the released CBAF-Net method,
an established novel contribution or a claim of SOTA. The original datasets,
labels, splits, tensors, Word manuscript and published results remain unchanged.

## Two Mechanisms

The first mechanism combines taxonomic generation and operation head scores.
The fixed label mappings are Human=(0,0), Traditional Bot=(0,1),
LLM Bot=(1,1), and Full-stack Agent=(1,2). Only declared class combinations
enter the multiclass softmax. These axes describe the construction of this
controlled archive; they are not independently annotated intent or universal
attributes of real accounts. No factor independence is assumed.

The second mechanism adds a cross-tower residual from content, behavior,
their elementwise product and absolute difference. It can represent evidence
that a factor combination misses. Its usefulness must be measured, not assumed.

Each temporal hashing namespace has its own projection. The content tower
receives descriptions, step text and nine surface statistics. The behavior
tower receives event hashes with relative timing, six numerical fields, two
action statistics and seven global timing fields. Event hashes already
contain post text, so these are overlapping input groups, not clean causal
modalities or semantic pretrained language embeddings.

## Architectural Removals

- Without factor fusion: replace factor score combination by a flat joint
  classification head, retaining the same auxiliary supervision and residual.
  This isolates fusion topology, not the entire taxonomy learning objective.
- Without interaction residual: physically omit that module, keeping factors
  and auxiliary supervision.

Common encoder and auxiliary-head parameters use the same initialization
seeds. Removal variants can be better. Parameter counts and compute differ
and must be disclosed. Do not call these capacity-matched controls.

## Fixed Evaluation Boundary

`configs/factor_training_screen.json` fixes one seed (42), three held-out
traditional scheduling policies, two views and six methods: 36 fits.
Comparators are flattened-input ERM MLP, global-statistics HGB, and the
local timing-adapted BotDMM with fold-local feature kNN. This is not the
original BotDMM paper reproduction. The HGB input is a global summary;
the neural models receive all five text/event steps.

Only original training tensors and existing training timing sidecars are read.
Each held-out policy's original training profiles form an exploratory
challenge. Fitting and checkpoint selection exclude those profiles and use
only the remaining policies. The inner split selects one sixth of profiles
within each remaining policy, ensuring both are represented. This split is
new and is not paired with the previous BotDMM module study; all controls
are retrained on these same new internal folds. Standardization is fitted inside the fitting
partition. No original validation or test paths are opened.

The model runner never reopens the raw master archive. Alignment guards may
hash the immutable archive, manuscript and old result files to verify their
bytes, without supplying them to training, prediction or model selection.

These training profiles participated in earlier development. Changing their
role cannot make them an untouched independent evaluation. This boundary
avoids additional original validation/test tuning; it does not restore
independence or authorize final scientific claims.

All methods, seeds, checkpoints and outcomes are retained, followed by
checkpoint replay. No hyperparameter search, automatic promotion or renewed
architecture search on previously exposed original-validation policy scores
is part of this screen. A one-seed result cannot establish reproducibility.

The run is a technical feasibility check. Real novelty, a reproducible
advantage against recent fair comparators and independent evaluation remain
separate requirements before replacing the paper or preparing submission.
