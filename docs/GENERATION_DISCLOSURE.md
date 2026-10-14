# Controlled Generation Disclosure

## Evidence Status

This disclosure summarizes a read-only metadata audit on 2026-10-14. The
master JSONL matched the retained SHA-256:
`b5ce0d14eed32b053f5ca78409cc07f0d8ff595d9511e762a127e7a5c2e7b16b`.
The audit did not run a detector or modify accounts, labels, or partitions.
Recorded settings are not an independent verification of the actual generator
weights, execution history, source permissions, or end-to-end reproducibility.

The two experimental views remain nested: Twibot22 has 5,031 trajectories and
Quadbot has 6,708, built from the same 1,677 base profiles. They are not two
independent real-platform datasets, and Twibot22 is not the original TwiBot-22
benchmark. A trajectory is a class-conditioned account history, not a newly
verified real account.

## Construction Records

| Class | Trajectories | Construction recorded in metadata |
|---|---:|---|
| Human | 1,677 | TwiBot-22-derived historical replay |
| Traditional Bot | 1,677 | Controlled burst, cyclic-reposting, or uniform scheduling |
| LLM Bot | 1,677 | Content-only generation without memory or environment observation |
| Full-stack Agent | 1,677 | Observe-remember-act simulation with memory and environment observation |

The traditional policies each have 559 records. Human replay and controlled
automation are different construction processes. Sharing base profiles reduces
one source-composition difference but does not eliminate simulation shortcuts,
label-correlated construction artifacts, or dependence among class variants.

## Recorded Generator Settings

Every LLM Bot and Full-stack Agent record contains the following values:

| Field | LLM Bot | Full-stack Agent |
|---|---|---|
| Generator family | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| Sampling enabled | True | True |
| Temperature | 0.7 | 0.7 |
| Top-p | 0.9 | 0.9 |
| Maximum new tokens | 1,200 | 1,200 |
| Repetition penalty | 1.05 | 1.05 |
| Generation rounds | 5 | 5 |
| Items per round | 5 | 5 |
| Shared conditioning identifier | twibot-22-shared-seed | twibot-22-shared-seed |
| Memory enabled | False | True |
| Environment observation | False | True |
| Recorded simulator version | Not recorded for this class | OASIS 0.2.5 |
| Pipeline identifier | quadbot-v3-llm-agent-generation-33 | quadbot-v3-llm-agent-generation-33 |

The pipeline identifier is retained provenance, not a public dataset name.
Shared recorded settings do not prove identical effective output distributions.
The values above are class-wide metadata counts, not conclusions from an
independent generator rerun.

## Remaining Reproducibility Gaps

The retained metadata contains checkpoint-fingerprint, generation-seed,
grounding-verifier, prompt-hash, quantization, and agent-audit fields. This
bounded audit inventories their presence; it does not verify their values
against model files, prompt templates, complete source code, or execution logs.
In particular, the current evidence does not establish:

- Exact generator weight revision and independently recomputed weight hashes.
- An archived, runnable generation pipeline with its dependencies and prompts.
- Verified filtering, fallback, deduplication, and grounding-execution records.
- Exact source-data access and redistribution permissions.
- An institutional ethics determination or exemption approved by the authors.

The MIT code license does not grant rights to third-party data or model
weights. Private machine paths, raw posts, prompts, original account
identifiers, and source tweet identifiers are not exported by this disclosure.

Before claiming end-to-end reproduction, retain an author-verified source-term
record and a permission-compatible generator archive, and verify the listed
fingerprints against that archive. These are disclosure and archive tasks;
they do not require enlarging or regenerating the two existing core views.
