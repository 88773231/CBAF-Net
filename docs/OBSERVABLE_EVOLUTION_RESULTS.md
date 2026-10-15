# Observable Evolution Results

Completed on 2026-10-15 at 02:19:28 Asia/Shanghai. The bounded remote run
lasted approximately 12 minutes 46 seconds including extraction, regression,
training, replay and auditing. CUDA was disabled. No generation, dataset
enlargement, original validation/test access or excluded-policy forwards
occurred.

## Scope

This is a single-seed, three-fold development screen on the existing Quadbot
training population. Only known-policy fitting and inner-selection rows
were re-encoded. The selection folds share profiles; their union contains
185 profiles. They are not independent replications or submission test
sets. Account identities and grouped partitions were retained.

The inputs changed from the legacy action/text hashes. All five conditions
in this screen use the new observation features. Earlier BotDMM, compact
correspondence and BotSSCL scores are not matched-input comparisons.

## Complete Comparison

Macro-F1 percentages on the known-policy inner-selection populations:

| Condition | Burst-exclusion fold | Cyclic-exclusion fold | Uniform-exclusion fold | Mean |
|---|---:|---:|---:|---:|
| Evolution + self-distillation | 98.7758 | 99.1869 | 98.9884 | 98.9837 |
| Without evolution | 98.5713 | 99.1869 | 98.9884 | 98.9155 |
| Without self-distillation | 98.3690 | 99.1869 | 98.9894 | 98.8484 |
| Without both | 98.5713 | 98.9812 | 98.9884 | 98.8470 |
| Same-input MLP | 98.3628 | 98.3651 | 97.9770 | 98.2350 |

The fold names indicate the policy excluded from all feature extraction
and model forwards, not the population evaluated in this table.

Paired full-minus-removal differences, in percentage points:

| Comparison | Burst | Cyclic | Uniform | Mean | Positive folds | Frozen gate |
|---|---:|---:|---:|---:|---:|---|
| Evolution contribution | +0.2044 | 0.0000 | 0.0000 | +0.0681 | 1/3 | Failed |
| Distillation contribution | +0.4068 | 0.0000 | -0.0010 | +0.1353 | 1/3 | Failed |
| Both components jointly | +0.2044 | +0.2058 | 0.0000 | +0.1367 | 2/3 | Failed |
| Against same-input MLP | +0.4130 | +0.8219 | +1.0114 | +0.7487 | 3/3 | Passed |

## Decision

The whole continuation gate failed. A higher score than the MLP does not
establish either component, superiority over BotDMM, SOTA or algorithmic
novelty. Most of the observed full-minus-MLP difference is already present
in the cue-separated backbone without the two added components.

Do not promote the two-component claim, expand its seeds as though it
passed, or tune its coefficients using these inspected scores. Retain
the cleaner observable input contract and the complete development record
as possible foundations for a specifically justified future hypothesis.
A new hypothesis needs an explicit prediction limitation and a bounded
protocol; these results alone do not authorize another broad search.

The single working Word, canonical remote code and public GitHub model
were not replaced. No formal paper result was changed.

## Verification

- 15 new fits and 15 exact selected-checkpoint replays completed.
- Remote targeted regression: 12 tests passed.
- Local full regression before launch: 209 tests passed.
- Three additional local evidence-audit tests passed after launch.
- Independently recomputed all 15 cells locally; 30 prediction/checkpoint
  member hashes checked. This is an evidence audit, not new evaluation.
- The original master and working manuscript hashes stayed unchanged.
- No coefficient search or favorable-result filtering took place.

Private evidence resides in `work/ccfc_submission_build/observable_evolution/`,
not in the public result tables. Source observations/posts/prompts were
not exported. Relevant hashes:

- Summary: `f884060fa7090d848abd5354fa7eafe4c1fc63f12f2234f246505091506b7c95`
- Evidence archive: `d7989c67b2596f2651353a6132109ce71cc5f07ac63693f5c0f1c8fa5ca5ed54`
- Executed source archive: `68979a4155a4f20a738462cb70237e5f42d89139499c24d8c70182fb0fa00248`

Literature-transfer details and implementation boundaries are recorded in
`RECENT_METHOD_TRANSFER_20261015.md`. None of the adapted mechanisms is
presented as a faithful CDRBot or BotEvo reproduction.
