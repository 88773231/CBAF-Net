# CBAF-Net Reproducibility Release

This staging directory contains the code and audit materials intended for a public repository accompanying the manuscript “CBAF-Net: Auditable Decision-Level Fusion for Controlled Evaluation of LLM-Driven and Agentic Social Accounts.”

## Scope

The release contains the multimodal baseline and decision-level behavioral-statistics fusion code, deterministic feature-export logic, dataset-view validation scripts, configuration examples, and reviewer-facing audit documentation.

The release does **not** contain raw TwiBot-22 records, original account identifiers, private checkpoints, credentials, API keys, local virtual environments, download caches, or unrestricted generated records. Users must obtain any restricted source material directly under the original provider terms.

## Repository layout

- `src/`: training, evaluation, data loading, baseline, calibration, and fusion code.
- `dataset_pipeline/`: schema validation, feature export, view construction, and audit scripts.
- `configs/`: non-sensitive schema and source-registry examples.
- `docs/`: data card, reproducibility boundary, and supplement notes.
- `results/`: reserved for aggregate metrics and non-identifying audit outputs; raw predictions should be added only after a final privacy and license review.

## Reproduction boundary

The manuscript's reported results are based on frozen controlled views named `Twibot22` and `Quadbot`, group-disjoint splits, three fixed seeds, a split-local label-free transductive kNN compatibility graph, and a validation-selected fusion weight. The generated LLM Bot and Full-stack Agent classes are controlled simulations; the paper does not claim real-platform or cross-generator generalization.

## Before publishing

1. Remove every local path, credential, raw record, private identifier, and temporary file.
2. Confirm the exact TwiBot-22 license and permitted-access wording.
3. Add the verified Python, PyTorch, CUDA, graph-library, operating-system, and hardware versions.
4. Decide which generator checkpoints, prompts, decoding settings, simulator settings, and filtering rules may be released.
5. Choose and add a code license. Do not publish this staging copy until the authors approve the license and the data-access statement.
6. Create an immutable release tag, such as `v1.0.0`, and record the commit URL in the manuscript and FCS submission metadata.

## Local smoke test

The code is organized for local smoke testing with schema examples only. Full training requires the authors' verified environment and legally accessible source data.
