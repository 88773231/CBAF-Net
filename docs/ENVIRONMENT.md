# Strict Rerun Environment

This file records the environment used for the audited strict rerun. It is a
reproduction record, not a claim that every later dependency release produces
bit-identical checkpoints.

## Platform

- Operating system: Ubuntu 22.04.3 LTS.
- Kernel: Linux 5.15.0-78-generic, x86_64.
- Python: 3.10.8.
- GPU: NVIDIA GeForce RTX 3080 Ti, 12,288 MiB.
- NVIDIA driver: 595.71.05.
- CUDA reported by PyTorch: 12.1.
- cuDNN reported by PyTorch: 8902.
- Host allocation observed during the run: 80 logical CPUs and 125 GiB RAM.

## Core packages

- PyTorch: 2.1.2+cu121.
- NumPy: 1.26.3.
- SciPy: 1.15.3.
- scikit-learn: 1.7.2.
- joblib: 1.5.3.
- Matplotlib: 3.8.2.

The exact core versions used by the audited run are listed above. A complete
`pip freeze` record is retained in the authors' immutable run archive but is not
needed to verify the distributed predictions and figures. The raw-data
selection utility additionally requires `ijson`; that utility was not part of
the strict tensor-view training run and is declared separately in
`requirements.txt`.

## Integrity records

- Deployed postprocessor code bundle SHA-256:
  `ff292f931503f6ee5b41af60d71ca9d358ab8e865ed87eed4e011b6da98934d9`.
- Downloaded strict-results bundle SHA-256:
  `383f2ac5d2c305e142efbdf1b2a83d4f595797668ea80bb9369eedad4cf20ba8`.
- Environment manifest SHA-256:
  `5196ea6b3a5f48ed4db81fa98239c49c5f1655a08972af93387de876d8ba5572`.
- Twibot22 feature-export audit SHA-256:
  `92c5b383bffd6e4bfa039bf7cdaa70910d55697ae803ef9b4c508c33da0248a5`.
- Quadbot feature-export audit SHA-256:
  `d804096535374858ce50de7ebd956375fc68c1a8aaca7988f0910832ac274a28`.

Each strict-release checkpoint is separately hashed in its adjacent
`training_summary.json`, and the evaluator rejects a checkpoint whose digest
does not match the recorded value.
