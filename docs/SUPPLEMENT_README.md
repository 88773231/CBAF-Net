# Companion Anonymous Supplementary Package for CBAF-Net

This document describes the reviewer-facing, results-only supplement generated
from the single strict publication protocol.

## Scope

- `Twibot22`: 5,031 trajectories, three classes, 1,677 profile groups.
- `Quadbot`: 6,708 trajectories, four classes, 1,677 profile groups.
- Seeds: 42, 43, and 44.
- Group-disjoint splits: 1,118/279/280 profile groups for
  train/validation/test.
- Graph: split-local directed cosine kNN with `k=10`, excluding self-pairs and
  every candidate sharing the target's profile group.

The supplement does not redistribute raw posts, account records, original
platform identifiers, checkpoints, author information, credentials, or
machine-specific paths.

## Directory map

- `strict_rerun_summary.json`: audited per-seed and aggregate metrics.
- `strict_rerun_table.csv`: manuscript-table input.
- `predictions/`: validation/test probabilities with package-local identifiers.
- `figures/`: figures regenerated from the audited prediction files.
- `SHA256SUMS`: integrity hashes for every result artifact under
  `results/strict_postprocessed/`, except the manifest itself.
- `docs/DATA_CARD.md`: dataset composition, intended use, and limitations.
- `docs/REPRODUCIBILITY_MANIFEST.md`: protocol and artifact provenance.

## Prediction-file anonymization

Public prediction artifacts replace internal sample and base-profile identifiers
with deterministic package-local indices. The mapping is not distributed. Row
order, labels, predictions, probability vectors, and profile-cluster membership
are unchanged, so paired analyses remain reproducible without exposing internal
identifiers.

## Integrity check

From the supplement directory, verify a file with PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 .\strict_rerun_summary.json
```

On systems providing `sha256sum`, verify all files with:

```bash
sha256sum -c SHA256SUMS
```

## Availability boundary

The supplement supports recomputation of reported metrics, figures, calibration
statistics, and profile-cluster uncertainty from frozen predictions. It does
not contain restricted training records or checkpoints and therefore is not an
end-to-end data redistribution package. Source-data access, controlled-generator
disclosure, and institutional ethics wording must be reported accurately in the
submission.
