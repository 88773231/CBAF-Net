# Frozen Semantic Representation Check

Frozen on 2026-10-15 before any new account encoding or fitting. This is a
representation feasibility study, not another named method or a submission
claim. Prior unsuccessful architecture screens remain recorded.

## Question and Inputs

The existing clean observations use signed word/character hashes. Their
near-ceiling known-policy selections do not establish that a pretrained
encoder will help. Test that hypothesis directly before inventing more
fusion modules.

Keep the original 5,031/6,708 views, source labels, accounts and partitions.
Use only the same three Quadbot known-policy fitting/selection folds, seed
42. Original validation, test and each fold's excluded-policy observations
are not encoded, transformed or forwarded for that fold. The selection
sets are already exposed and overlap; no independent confirmation follows.

The sentence encoder receives only normalized public profile text and
individual public post text. Normalization, post ranking, five windows,
mean window pooling and all nontext cues are shared with a fresh lexical
control. IDs, labels, simulator actions, policy names and provenance never
enter either encoder. No text is sent to an external API.

## Encoder Contract

Use `sentence-transformers/all-MiniLM-L6-v2`, frozen revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The publisher model card declares
English and Apache-2.0; output width is 384. Use attention-masked mean token
pooling, L2 normalization and a 256-token cap. Empty text maps to zero.
Record actual truncation counts and declared profile languages; these are
not automated language identification or evidence of coverage outside
English. Do not select an encoder using these evaluation scores.

Primary artifact host was unreachable locally. On 2026-10-15 its mirrored
publisher model card, pinned file listing and sentence configuration were
retrieved through `hf-mirror.com`. All downloaded small files must match
predeclared Git blob IDs, and safetensor weights must match SHA-256
`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.
This verifies consistency with pinned mirrored metadata, not independent
upstream attestation or source-dataset rights. The snapshot records the
actual retrieval endpoint. Loading is local-only, safetensors-only, with
`trust_remote_code=False`; no encoder finetuning or private-text upload.

Publisher reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Frozen Matrix

Run the two representations with both controls:

1. Signed lexical hashing plus flat two-hidden-layer MLP.
2. Frozen semantic vectors plus the same MLP.
3. Signed lexical hashing plus the clean cue-separated contextual backbone.
4. Frozen semantic vectors plus that same backbone.

Both representations have width 384, so each architecture has exactly
matched parameters and initial weights across the representation pair.
This changes the lexical control from its previous 256 dimensions; old
results are not matched controls. Both architectures use cross-entropy
only. No retired evolution, distillation, routing, pair loss, SAM or
contrastive term is included. The architectures have different parameter
counts; do not claim compute/capacity equality between architectures.

Use width 32, readout 128, dropout 0.2, AdamW learning rate 0.001, weight
decay 0.001, batch size 128, at most 30 epochs and patience 6. Retain the
existing selection key and fitting-only nontext standardizers. No weight,
seed, epoch, encoder or truncation search. Total: 12 new fits and 12 exact
selected-checkpoint replays. Save fold-specific allowed-only feature caches,
normalization scopes, selected identities, predictions and hashes.

## Decision and Resources

Require semantic-minus-lexical mean Macro-F1 at least 0.25 percentage
points, positive in at least two folds, for **both** architectures before
continuing a representation-led branch. This is a project stopping rule,
not a CCF-B requirement. A gain here is not architectural novelty, a
module ablation, final-test improvement, SOTA or acceptance evidence.
Failure stops this fixed encoder branch without tuning on its scores.

Prepare dependencies and verify public-fixture inference on CPU. Benchmark
allowed text encoding before committing to the full screen. Limit the
CPU screen to 60 minutes including encoding, fitting and replay. If this
bound is not feasible, stop before training and give the user a measured
cost/time choice; do not silently start a long rented-GPU job.

The working manuscript, public GitHub model and canonical remote checkout
remain unchanged. A successful feasibility screen would still require a
defensible model contribution, recent matched-input comparisons and
genuinely unexposed confirmation before paper replacement.
