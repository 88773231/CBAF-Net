# Submission Consolidation

## Latest User Scope

The subsequent explicit instruction is CCF-C manuscript/release closeout,
not another model search. See `CLOSEOUT_20261015.md` for the current
evidence boundary and completed checks. Earlier optimization priorities
below are retained as history, not the current work order.

The latest 2026-10-15 instruction permits one final limited improvement
attempt and a CCF-C fallback if unsupported. See
`FINAL_BOUNDED_TRIAL_20261015.md` and the latest override in
`SUBMISSION_SPRINT.md`; earlier optimization-first text below is historical.
The local semantic screen completed all 12 fits/replays and its audit
against the archived source. Both complete-study continuation rules fail;
close the branch and stop additional method searches. Full results are
in `SEMANTIC_REPRESENTATION_RESULTS.md`. No core data regeneration,
manuscript result replacement or public model promotion has occurred.
No training process or remote GPU job remains. Local evidence export is
not remote/GitHub synchronization or a rendered final manuscript.

## Historical Decisions

An earlier instruction on 2026-10-15 restored the CCF-B target and requested
diagnosis-led bounded model work before manuscript alignment. See the current
override in `SUBMISSION_SPRINT.md`. The prior finish-first instruction is
historical. No new data generation, automatic final-model claim or formal
submission is authorized. "70%" is not a measurable venue quality threshold.

The user instruction on 2026-10-14 was to continue bounded model
improvement before rewriting or submitting the manuscript. Quadbot is
primary; main-text module removals may focus on four classes. Twibot22
remains auxiliary and its negative results are retained. The former
two-view superiority requirement was superseded.
No optimization-until-acceptance promise or new submission is authorized.

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

## Completed Bounded Comparison

`configs/botsscl_policy_control.json` completed 36 additional fits and 36
checkpoint replays: two graph-free conditions, three excluded policies, three
seeds, and both unchanged views. There was no data generation or label change.
All original fitting files and the failed initial replay record were preserved.
The initial replay exited with signal 9 under a 2 GiB memory ceiling. A
single-fold-cache adapter passed two equivalence/tampering tests locally and
remotely and completed the same frozen checks without additional research fits.
`analysis/bounded_botsscl_replay.py` preserves the hashed training source.

The new control's pooled means are 98.50% in Twibot22 and 94.81% in Quadbot.
The current candidate is 93.09% and 96.01%, respectively. The candidate loses
all three paired three-class comparisons; it wins two of three four-class
pairs. The same-encoder supervised-only control is 96.59% and 94.30%.
This completes the input-matched diagnostic, not the user's unified two-view
superiority requirement. Do not tune toward these exposed scores or present
the stronger three-class comparator as a new original contribution.
Full seed, class, policy, compute and conditional uncertainty records are in
`BOTSSCL_POLICY_CONTROL_RESULTS.md`.

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

The preserved execution commands are shown below. Completed directories refuse
overwrites; these are documentation, not an instruction to rerun the study:

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

1. Completed: control implementation, 150-test local/remote regressions,
   36 fits and 36 replays, plus two local/remote resource-adapter tests.
2. Completed: local prediction/identity verification and pooled-metric
   recomputation. Retain all controls and the initial replay failure record.
3. Completed: the four-class same-encoder objective diagnostic, nine new
   fits, three reused references and 12 known-policy checkpoint replays.
   The encoder swap with combined objectives is not promoted; see
   `ENCODER_OBJECTIVE_DIAGNOSTIC_RESULTS.md`. Do not hide development
   records or rename a removal as the old full method.
4. Establish task-specific novelty and reserve an independent evaluation
   protocol before scoring it. Do not promise SOTA.
5. Rewrite the one working manuscript and its actual final schematic, then
   verify code/data/checkpoint/prediction/table identities end to end.
6. Confirm authors' approvals, submission exclusivity and a currently open
   CCF-B full-paper track using that venue's official rules.

This is a local preparation step. Actual completion of runs, synchronization
and independent validation must be recorded separately, never inferred from
passing synthetic regression fixtures.

The 2026-10-14 metadata audit is summarized in `GENERATION_DISCLOSURE.md`.
It does not close the source-permission or generator-archive requirements.

## Completed Prediction Diagnosis

`CONTROL_PREDICTION_DIAGNOSIS.md` records a local, read-only diagnosis of
the verified private export. All 138 member digests and 18 pooled seed
metrics matched. Candidate/control sample identities and nine fitting
normalization pairs per view matched; all 160 local tests passed.
No research fit, raw-tensor access or checkpoint-weight access occurred.

Twibot22 candidate Burst accounts for 75.84% of its error events.
Its Human/Traditional boundary is weaker, but the control also changes
encoder depth, activation, normalization, dropout and initialization.
The current evidence cannot isolate SAM or pretraining as the cause.
Quadbot's advantage is also concentrated in Burst; Cyclic and Uniform
remain stronger for the new controls.

Retire the frozen recipe as a unified final-model candidate, preserving
all development results. The bounded objective audit has completed; see
`ENCODER_OBJECTIVE_DIAGNOSTIC_RESULTS.md`. Continued four-class method work
now has a fixed twelve-fit architecture hypothesis in
`TRAJECTORY_CORRESPONDENCE_PLAN.md`, not a new two-view search or
permission to call known-policy selections test results. All twelve
architecture fits and checkpoint replays completed and were verified, but
the full condition fails its transition-removal gate. See
`TRAJECTORY_CORRESPONDENCE_RESULTS.md`. Stop expanding that full
architecture; its best removal is not automatically a new final method.
Do not construct a challenge-tuned
router or ensemble from the observed prediction disagreements.
Word, released results and GitHub main/tag remain unchanged.
