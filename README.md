# CBAF-Net Reproducibility Release

This repository accompanies the manuscript "CBAF-Net: Auditable Decision-Level Fusion for Controlled Evaluation of LLM-Driven and Agentic Social Accounts."

CBAF-Net combines probabilities from a frozen BotDMM checkpoint with an independently fitted HistGradientBoosting behavioral-statistics expert. The scalar mixture weight is selected on the validation split and then frozen before test evaluation.

## Repository layout

- `src/run_cbaf.py`: canonical CBAF-Net evaluator used by the paper.
- `src/botdmm.py`: frozen BotDMM backbone definition required to load the reported checkpoints.
- `src/data_loader.py`: tensor-view loaders and batching logic.
- `src/layers/`: backbone layers used by `src/botdmm.py`.
- `analysis/`: profile-cluster bootstrap and paper-figure recomputation scripts.
- `dataset_pipeline/`: schema validation, feature export, view construction, and release audits.
- `configs/`: non-sensitive schema and source-registry examples.
- `docs/`: data card and reproducibility boundary.

## Install

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The exact environment used for the reported runs is documented separately with the submission artifacts. The version ranges above are intended for code inspection and local reproduction, not as a claim that every dependency combination reproduces bit-identical checkpoints.

## Run CBAF-Net

The evaluator requires a legally accessible tensor view and its matching frozen BotDMM checkpoint. It never searches the test labels for a fusion weight.

```bash
python -m src.run_cbaf \
  --dataset twibot22 \
  --data_dir /path/to/Twibot22 \
  --checkpoint /path/to/Twibot22_seed42/best_model.pt \
  --seed 42 \
  --out_dir results/Twibot22_seed42
```

Use `--dataset quadbot` with the matching four-class checkpoint for Quadbot. Repeat with seeds 42, 43, and 44 to reproduce the paper's aggregation protocol.

The output directory contains:

- `metrics.json`;
- `val_predictions.jsonl`;
- `test_predictions.jsonl`;
- `behavioral_statistics_expert.joblib`.

## Recompute audits and figures

```bash
python analysis/profile_cluster_bootstrap.py \
  --predictions results/Twibot22_seed42/test_predictions.jsonl \
  --out results/Twibot22_seed42/profile_bootstrap.json

python analysis/plot_reliability.py \
  --predictions-root results/predictions

python analysis/plot_confusion.py \
  --predictions-root results/predictions

python analysis/plot_ablation.py
```

The figure scripts are post-hoc diagnostics over frozen prediction files; they
do not retrain a model or tune on the test split. The confusion script uses
the same three-seed, row-normalized aggregation protocol as the manuscript.

`results/predictions` should contain `Twibot22_seed42`, `Twibot22_seed43`, `Twibot22_seed44`, `Quadbot_seed42`, `Quadbot_seed43`, and `Quadbot_seed44` folders.

## Data and claim boundary

`Twibot22` and `Quadbot` are hybrid controlled benchmarks. Human trajectories are historical replays derived from TwiBot-22 material; the automated strategy variants are controlled transformations or simulations. LLM Bot and Full-stack Agent are not verified labels for live platform accounts.

The repository does not contain raw TwiBot-22 records, original account identifiers, private checkpoints, credentials, API keys, or unrestricted generated records. Users must obtain restricted source material under the original provider terms. The reported results do not establish cross-platform, cross-generator, inductive-graph, or real-account generalization.

## License

The MIT license in `LICENSE` covers only code and documentation owned by the authors. It does not relicense third-party source data, pretrained models, or checkpoints.
