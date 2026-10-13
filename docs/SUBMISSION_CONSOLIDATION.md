# Submission Consolidation

## Decision

Retain the existing datasets and freeze the current SAM/cross-policy-contrast
recipe. It is a development candidate, not the manuscript's final method.
Do not submit the unchanged fusion paper or replace its test scores with the
development scores. Keep one working Word document; do not create more
submission-shaped versions before the scientific replacement is supported.

The current four-class result is promising, but the contrast removal is better
in two of three paired main runs. In the three-class view, removing SAM is
better in all three runs. These are separate limitations from the favorable
whole-model mean and cannot be solved by rearranging a figure.

The existing four-cell objective comparison also permits a descriptive
interaction contrast: full minus SAM-only minus contrast-only plus ERM.
Recomputed from the frozen pooled scores, in percentage points:

| View | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---:|---:|---:|---:|
| Quadbot | 3.94 | -0.10 | 0.32 | 1.39 |
| Twibot22 | -7.71 | -4.18 | 2.57 | -3.11 |

This comparison concerns whole trained-and-selected recipes, not a causal
interaction between independently manipulated prediction modules. Macro-F1
is nonlinear, the checkpoints are selected separately, and the same exposed
development profiles are reused. The positive four-class mean is mostly
driven by one seed; it does not rescue the inconsistent component effects or
support an interaction claim across both taxonomies. Do not tune the recipe
using these post-hoc contrasts.

## Method and Figure Contract

For the frozen candidate, inference concatenates description, five text blocks,
five action blocks including relative timing, and tabular statistics, then uses
the ordinary matched MLP. It does not contain four experts, a graph encoder,
a cosine head, the rejected pair-margin objective, a learned fusion gate, or
the released BotDMM/HGB mixture.

Cross-policy contrast supervises fitting representations. SAM changes the
training update. Neither is an additional prediction-time branch. A future
candidate schematic must separate prediction-time modules from training-only
objectives and label illustrative account icons separately from measured
inputs. Until a supported final method is fixed, the existing manuscript
figure continues to depict its older, honestly reported release method.

SAM and supervised contrast are established mechanisms. A task-specific
contribution and evidence for it are still required; two names attached to
existing losses do not establish two algorithmic innovations.

## Next Bounded Comparison

`configs/botsscl_policy_control.json` schedules 36 additional fits and 36
checkpoint replays: two graph-free conditions, three excluded policies, three
seeds, and both unchanged views. There is no data generation or label change.

The BotSSCL-mechanism condition uses fitting-only feature donors and NT-Xent
pretraining, then supervised multiclass fine-tuning. Its paired control uses
the identical encoder and classifier without pretraining. Both consume the
complete flattened history used by the current MLP, not the old adapter's
temporal means. Every fold must match the preserved flat-policy study's
normalization and row partition.

The primary source inspected is `code4thispaper/BotSSCL` at
`ed3b17588fc2c271947aade266bb1357fc835f61`, particularly
`_4_BotSSCL/scarf/model_botsscl.py`. The independent adapter retains the
feature-corruption/contrast mechanism, not the original input representation,
binary downstream classifier, estimator, or original benchmark scores.
The new outputs must say "matched-input mechanism control", not "faithful
BotSSCL reproduction". This additional control alone does not complete the
2025/2026 bot-specific-baseline requirement.

Freeze 30 pretraining epochs, corruption rate 0.6, temperature 1.0 and the
existing 30-epoch supervised ceiling and grouped inner-selection rule.
No tuning uses excluded-policy scores. Pretraining has extra compute;
identical input and supervised ceilings do not imply identical model,
optimizer, selected epochs or total compute. The adapter has no dropout;
the configuration's dropout field remains present only to preserve the
reference configuration and is not an active adapter setting.

Run using the preserved private training inputs and study paths:

```bash
python -m analysis.run_botsscl_policy_control run \
  --dataset Quadbot \
  --data-dir /private/strict/views/Quadbot \
  --temporal-dir /private/relative_timing/Quadbot \
  --partition-root /private/original_policy_partitions/Quadbot/seed42 \
  --reference-root /private/flat_policy_screen \
  --study-root /private/botsscl_policy_control

python -m analysis.run_botsscl_policy_control replay \
  --dataset Quadbot \
  --data-dir /private/strict/views/Quadbot \
  --temporal-dir /private/relative_timing/Quadbot \
  --partition-root /private/original_policy_partitions/Quadbot/seed42 \
  --study-root /private/botsscl_policy_control
```

Repeat both operations for Twibot22, then run:

```bash
python -m analysis.run_botsscl_policy_control summarize \
  --study-root /private/botsscl_policy_control
```

Paths are placeholders, not a command to download or rebuild data. The study
refuses overwrites and incomplete/tampered prediction or checkpoint evidence.
Its aggregate pools all excluded-policy predictions per seed before computing
Macro-F1 and keeps every class, policy, condition, seed and compute record.

## Paper Assembly Boundary

| Component | Supported now | Still needed |
|---|---|---|
| Data | 5,031 and 6,708 controlled trajectories sharing 1,677 profiles | Exact source permission and generation disclosures |
| Candidate | Ordinary MLP with contrast and SAM training | Task-specific technical contribution and confirmation |
| Main comparison | Three-seed development means and compute controls | New input-matched bot controls and independent evidence |
| Module removals | Complete two-component results on both views | Stable support for any module retained as a contribution |
| Evaluation | Excluded-policy development on original training profiles | Evaluation not used for design or selection |
| Word | One retained fusion manuscript, consistent with its release | Method, equations, tables and figures rewritten together |
| Public code | Frozen main/tag plus separately recorded revision work | Verified replacement release after evidence is accepted |

The three- and four-class views are nested; do not call them independent
real-world replications. New seeds on already examined profiles do not produce
an untouched test. If unexposed evaluation material cannot be obtained without
expanding the core datasets, state the limitation and narrow the claim rather
than presenting exposed results as confirmatory.

## Work Order

1. Finish local control implementation and regression checks.
2. Run the fixed matched-input controls on CPU when the private inputs are
   accessible. Do not keep a GPU container billed while authoring locally.
3. Review all controls and paired module differences. Revise/remove an
   unsupported mechanism; do not hide its development record or rename its
   removal as the old full method.
4. Establish task-specific novelty and reserve an independent evaluation
   protocol before scoring it. Do not promise SOTA.
5. Rewrite the one working manuscript and its actual final schematic, then
   verify code/data/checkpoint/prediction/table identities end to end.
6. Confirm authors' approvals, submission exclusivity and a currently open
   CCF-B full-paper track using that venue's official rules.

This is a local preparation step. Actual completion of runs, synchronization
and independent validation must be recorded separately, never inferred from
passing synthetic regression fixtures.
