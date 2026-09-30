# Data Card

## Overview

The study uses a controlled paired-profile design for social-bot account
classification. Each of 1,677 base profile groups contributes one trajectory per
class in the four-class view. Each trajectory contains 25 posts and 25 aligned
actions. Splits are assigned at the profile-group level, preventing a profile group
from appearing in more than one split.

## Dataset views

| View | Classes | Trajectories | Profile groups | Train/val/test rows |
|---|---|---:|---:|---:|
| Twibot22 | human, traditional bot, LLM bot | 5,031 | 1,677 | 3,354 / 837 / 840 |
| Quadbot | human, traditional bot, LLM bot, full-stack agent | 6,708 | 1,677 | 4,472 / 1,116 / 1,120 |

Class counts are exactly balanced: 1,677 trajectories per class. The three-class
view and four-class view use the same 1,677 profile groups.

## Data provenance and construction

The released views are hybrid research datasets rather than wholly synthetic
datasets. They combine source-derived public material with controlled synthetic
simulations. The source-derived portion remains governed by the original
provider terms, while the simulated classes are generated for controlled
experiments and are not verified real-platform labels.

- Human profiles: TwiBot-22-derived historical replay.
- Traditional bots: TwiBot-22-derived profiles with three controlled policies
  (burst scheduler, cyclic reposter, and uniform scheduler; 559 groups each).
- LLM bots: controlled content-only generation without environment observation.
- Full-stack agents: controlled observe-remember-act simulation.

The public-facing class names `Twibot22` and `Quadbot` denote the experimental
views in this study; they do not imply that every class is an original TwiBot-22
annotation. Controlled-generation classes must not be represented as verified
real-platform accounts.

Earlier pipeline records use compatibility schema identifiers such as
`quadbot_3class`, `quadbot_4class`, and `quadbot-v3-*`. These strings are stable
machine-readable provenance identifiers, not additional public dataset names.
New model-facing output directories default to `Twibot22` and `Quadbot`.

## Feature representation

The frozen export records 768-dimensional description, text, and AMR blocks; a
2,304-dimensional graph block; six numerical features; and eleven style features.
The URL feature counts both literal URLs and privacy-redacted `<URL>` tokens.
Because every released trajectory has 25 posts and 25 actions, the numerical
post/action counts and the duplicated style post count are three constant schema
columns. The release backbone retains their positions but maps them to zero using
training-fitted per-column standardization; validation and test never contribute
normalization statistics. The raw-tree BSE cannot split on constant columns.
Graph files contain split-local cosine kNN `(target, selected_neighbor)` pairs
with `k=10`; labels and provenance fields are excluded from graph construction.
The strict loader excludes same-profile pairs and converts them to
`selected_neighbor -> target` messages in one-hop target-neighbor batches.

## Quality controls

- No profile group crosses train, validation, and test splits.
- Every profile group has the expected paired classes.
- No invalid time steps, duplicate within-record post/action IDs, alignment
  mismatches, graph-edge reference errors, or empty post texts were reported by
  the frozen release audit.
- The strict graph export removes all same-profile kNN edges; the audited
  same-profile edge fraction is `0.0` for every split in both views.

## Known limitations

- LLM-bot and full-stack-agent classes are controlled simulations, not evidence of
  real-platform generalization.
- Source/generator differences can create dataset shortcuts. The audited
  behavior/style diagnostics retain substantial predictive signal, so they are
  reported as a limitation rather than evidence of real-platform generalization.
- Removing same-profile graph edges addresses one leakage route only. It does not
  establish cross-source, cross-generator, temporal, or cross-platform robustness.
- The effective independent sample size is 1,677 profile groups, not 6,708 fully
  independent trajectories. Uncertainty analyses therefore resample profiles.
- The balanced class distribution is a research design choice and should not be
  interpreted as platform prevalence.

## Privacy and redistribution

This package contains no raw post text, profile description, original account ID,
or source-platform handle. Prediction artifacts use package-local profile-group
indices. Access to any underlying TwiBot-22-derived material remains subject to
the source dataset's terms and is not granted by this supplement.

## Intended use

The artifacts support audit and reproduction of aggregate research results. They
are not suitable for production moderation, identity inference, user profiling,
or consequential decisions about individual accounts.

## Submission disclosures outside this anonymous package

- The supplement does not redistribute TwiBot-22-derived source records. The final manuscript and submission metadata must cite the exact source terms and permitted access procedure verified by the authors.
- Generator checkpoint identifiers, prompts, decoding settings, simulator version, and filtering or deduplication configuration are not included in this results-only anonymous package. They must accompany any claimed end-to-end public release.
- The package contains no raw text or original account identifiers, but the institutional ethics or exemption wording remains an author-level submission declaration.
- Access to underlying source data is governed by the original provider; this package grants access only to the derived, reindexed audit artifacts it contains.
