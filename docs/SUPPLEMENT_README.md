# Anonymous Supplementary Package for CBAF-Net

This directory is the anonymous, reviewer-facing supplementary package for the
CBAF-Net manuscript. It contains only frozen formal results, the strict graph-edge
sensitivity analysis, aggregate audits, and reindexed prediction artifacts needed
to verify the reported statistics.

## Scope

- `Twibot22`: 5,031 trajectories, three classes, 1,677 profile groups.
- `Quadbot`: 6,708 trajectories, four classes, 1,677 profile groups.
- Seeds: 42, 43, and 44.
- Group-disjoint splits: 1,118/279/280 profile groups for train/validation/test.
- Strict edge sensitivity: split-local directed cosine kNN with `k=10`, excluding
  every edge whose endpoints share the same profile group.

The package does not redistribute raw posts, account records, original platform
identifiers, model checkpoints, author information, connection credentials, or
machine-specific paths.

## Directory map

- `audits/formal/`: frozen release-contract and shortcut audits.
- `audits/strict/`: strict edge policy and zero-same-profile-edge summary.
- `configs/`: machine-readable frozen training and fusion configuration.
- `reports/`: aggregate fusion controls and exact 5,000-resample comparisons.
- `results/formal/`: formal baseline/CBAF metrics and anonymized probabilities.
- `results/strict/`: strict-edge CBAF metrics and anonymized probabilities.
- `DATA_CARD.md`: dataset composition, intended use, and limitations.
- `REPRODUCIBILITY_MANIFEST.md`: protocol and artifact-level provenance.
- `SHA256SUMS`: integrity hashes for every distributed file except itself.

## Prediction-file anonymization

The source prediction artifacts contained opaque internal `sample_id` and
`base_profile_id` fields. In this package, `sample_id` is removed and
`base_profile_id` is replaced by a package-local sequential `profile_group`.
The mapping is not distributed. Row order, labels, predictions, and probability
vectors are unchanged. The same mapping is used for formal and strict artifacts,
so paired profile-level analyses remain reproducible.

## Integrity check

From this directory, verify a file with PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 .\README.md
```

Compare the reported digest with `SHA256SUMS`. On systems providing `sha256sum`,
run `sha256sum -c SHA256SUMS`.

## Availability boundary

This anonymous package is intentionally results-only. It supports metric and audit recomputation from frozen predictions, but not end-to-end retraining. The authors must separately verify the public code version, formal environment, source-data access terms, generator disclosure, and institutional ethics wording before claiming a complete reproducibility release.
