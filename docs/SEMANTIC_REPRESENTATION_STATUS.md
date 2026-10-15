# Semantic Representation Status

## Completed Final Trial

The final user-authorized local attempt ran on 2026-10-15 from
11:53:26 to 12:17:42 Asia/Shanghai, CPU only. All 12 fits, 12 checkpoint
replays and the independent evidence audit completed within the three-hour
maximum. No training or remote GPU task remains.

Semantic-minus-hash mean Macro-F1 is +0.4758 percentage points for the MLP
and -0.0683 for the cue-encoder control. The original 0.25-point gate and
the prospectively relaxed small-gain rule both fail for the complete
study. Close the branch; do not start residual/encoder searches or promote
the favorable MLP comparison as novel architecture evidence.

See `SEMANTIC_REPRESENTATION_RESULTS.md` for all 12 scores, controls,
hashes and release boundaries. The final full regression passed 239 tests;
three additional export tests passed separately. Raw inputs, partitions
and the working manuscript are unchanged. No public or remote replacement
release, new test result or final submission manuscript was produced.

The preparation history below predates this completed execution.

## Historical Preparation

At the historical preparation checkpoint on 2026-10-15, the screen was
prepared and tested but not fitted. The remainder of this section records
that checkpoint, not the final execution status above.

## Completed

- Confirmed that clean observation text features are lexical hashes,
  not pretrained sentence embeddings. A semantic bottleneck remains a
  hypothesis, not a diagnosed cause of classification errors.
- Fixed the 2x2 representation/control matrix and original partitions in
  `configs/semantic_representation.json` and
  `docs/SEMANTIC_REPRESENTATION_PLAN.md`.
- Implemented local-only frozen inference, shared 384-dimensional
  representations, fresh CE-only controls, allowed-only extraction,
  fitting-only standardization, saved checkpoints, exact prediction
  replays and an independent saved-metric/hash audit.
- Passed all 238 local regression tests, including ten new representation
  and evidence-audit tests. These are software tests, not proof of an
  effective method.
- Downloaded the pinned public encoder to the private build directory.
  All eight small files match the pinned Git blob IDs; model safetensors
  match the predeclared SHA-256. The snapshot records mirror retrieval.
- Installed optional dependencies in an isolated local directory without
  replacing the canonical environment.
- Connected to the latest authorized remote with pinned host-key
  verification. Canonical code was clean at
  `a7605e2af6f2990ce80b15df8305aabb7b3dcaad`; no research processes were
  running. Only frozen original-training source/metadata and three
  existing partition receipts were copied locally. Input hashes match.
  No original validation/test files were requested.

## Measured Resource Gate

Using local CPU PyTorch 2.6.0, two threads and frozen transformers 4.51.3:

| Resource Probe | Texts | Seconds | Texts/Second |
|---|---:|---:|---:|
| Public short fixture | 256 | 2.781 | 92.05 |
| Allowed training text, content-hash selected | 256 | 12.063 | 21.22 |

The real-text probe uses only the burst-excluded fold's permitted known
rows. It does not fit a classifier, select text using labels, export
private text/vectors or calculate a prediction score. There were 8,087
tokens before truncation and no capped sequences among these 256 texts.
This does not establish truncation rates for the entire study.

Each fold has approximately 54,000 unique nonempty normalized texts.
At the measured real-text rate, separate allowed-only extraction for
three folds is roughly 7,673 seconds, or 128 minutes. With a 15-minute
fit/audit reserve the rough estimate is 143 minutes. Real batch lengths
and system load vary; this is not a completion promise.

The fixed 60-minute CPU bound fails. Accordingly, no 12-fit screen was
launched and no paid GPU job was started. Do not substitute the optimistic
short-fixture rate for the measured real-text rate. A GPU alternative
requires user resource approval and its own short benchmark before
claiming time or cost.

## Preserved Boundaries

The 5,031/6,708 views, original raw master, labels and splits are unchanged.
The 256-text resource probe does not make an independent confirmation set.
The working manuscript and public GitHub model have not been replaced.
No novelty, ablation gain, final-test improvement, SOTA or submission
readiness is established. Prior failed screens remain preserved.

All commands from this preparation turn have completed. There is no
background training or remote GPU task to wait for. The rented container
can remain closed until an explicitly budgeted remote run is needed.

Private build evidence:

- `semantic_representation_preflight.json`
- `semantic_representation_real_text_preflight.json`
- `semantic_encoder/all-MiniLM-L6-v2_1110a243/snapshot_manifest.json`
- `semantic_representation_inputs/` (private training-only copy)

These files are outside the public repository and must not be uploaded as
public account data. Preparation receipts and immutable source archive are
recorded by the separate preparation packaging helper.
