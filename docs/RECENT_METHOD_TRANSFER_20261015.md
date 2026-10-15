# Recent Method Transfer

Verified against primary publication pages and papers on 2026-10-15. This
is a feasibility record, not a novelty claim or a faithful reproduction.

## Sources and Boundaries

| Primary source | Relevant mechanism | Compatibility with the current controlled views |
|---|---|---|
| Han et al., *Exploring and Distilling Multi-Dimensional Clues for Interpretable Social Bot Detection*, ACL 2026, pp. 28971-28992, DOI 10.18653/v1/2026.acl-long.1338 | CDRBot distills textual clues for cold-start, optimizes dimension-specific inspectors using outcome-reward RL, then trains a fusion predictor. Its four dimensions cover user information, semantics, interactions and behavior. | Transfer only the motivation to expose distinct grounded evidence. Our logit self-distillation is a separate conventional objective: no external clue teacher, textual explanations, RL or real follow graph. It is not a CDRBot reproduction or its clue-distillation algorithm. |
| Zhang et al., *BotEvo: LLM-Driven social bot detection via behavioral evolution modeling and cross-modal fusion*, Journal of King Saud University Computer and Information Sciences 38, article 460 (2026), published May 28, DOI 10.1007/s44443-026-00814-3 | Graph-agnostic post-sequence modeling, dynamic post weighting, implicit text/behavior/sentiment gating and two-stage threshold optimization. Its main task is binary, with an auxiliary three-class formulation. | Test local temporal encoding of observable action ratios, writing statistics and relative timing. We do not implement its post-weighting, sentiment encoder, gating or threshold optimization, nor claim it originally solves four-class detection. |
| Zheng et al., *Bot Meets Shortcut: How Can LLMs Aid in Handling Unknown Invariance OOD Scenarios?*, AAAI 2026, 40(2), pp. 1605-1613, published March 14, DOI 10.1609/aaai.v40i2.37137 | Constructs text-based shortcut scenarios and uses LLM-driven counterfactual augmentation for mitigation. | Supports checking input and distribution boundaries before model comparison. Our schema audit is not its mitigation method, does not prove shortcut freedom and cannot establish real-world validity. |
| Wang et al., *BotCHF: camouflage-heterogeneity-aware fusion for social bot detection*, Journal of King Saud University Computer and Information Sciences 38, article 592 (2026), published July 8, DOI 10.1007/s44443-026-01013-w | Preserves text and graph decisions, learns account-specific fusion weights, and alternates semantic updates with graph learning. | Account-specific decision fusion is compatible as a motivation. The current features lack the real follow graph and a fine-tuned language encoder, so neither its graph module nor its full collaborative optimization can be faithfully reproduced here. |
| Xu et al., *Reliable Conflictive Multi-View Learning*, AAAI 2024, DOI 10.1609/aaai.v38i14.29546 | ECML learns view-specific evidential opinions and combines decisions and reliability using conflictive opinion aggregation. Metadata and abstract were checked against publisher-deposited Crossref records; the author repository was also located. Publisher PDF/full-text requests failed. | Relevant to avoiding forced agreement. No Dirichlet fusion formula is imported without verifying full equations. Current text/style/behavior/timing cues can be dependent; they are not four independently sampled modalities. |
| Bezirganyan et al., *Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion*, AISTATS 2025, PMLR 258, pp. 3142-3150 | Order-invariant evidence fusion with conflict-dependent reassignment to uncertainty mass; verified against the official PMLR abstract and bibliography. Full PDF retrieval failed. | Motivates checking confidence and conflict, not treating maximum softmax as calibrated reliability. The candidate uses learned residual gates, not DBF, and makes no uncertainty-quantification or conditional-independence claim. |

Primary references:

- https://aclanthology.org/2026.acl-long.1338/
- https://aclanthology.org/2026.acl-long.1338.pdf
- https://link.springer.com/article/10.1007/s44443-026-00814-3
- https://ojs.aaai.org/index.php/AAAI/article/view/37137
- https://link.springer.com/article/10.1007/s44443-026-01013-w
- https://ojs.aaai.org/index.php/AAAI/article/view/29546
- https://api.crossref.org/works/10.1609/aaai.v38i14.29546
- https://github.com/jiajunsi/RCML
- https://proceedings.mlr.press/v258/bezirganyan25a.html

### Exclusions and Verification Limits

*Certainly Bot Or Not? Trustworthy Social Bot Detection via Robust
Multi-Modal Neural Processes*, arXiv:2503.09626, is withdrawn. The current
abstract page, checked on 2026-10-15, records a withdrawal on 2025-12-26
due to an experimental-setting/evaluation-protocol error. It is excluded
as empirical support or a validated recent baseline; the accessible v1
HTML does not override the withdrawal.

- https://arxiv.org/abs/2503.09626

BotBR publication metadata was found for SIGIR 2025, DOI
10.1145/3726302.3729908, but the publisher page/full mechanism was not
retrieved. TRUST, TRJD and UEFCS were not verified to an exact primary
paper in this work. Do not attribute mechanisms or venue status to them
or place them in a comparison table from name-only search hits.

## Diagnosis-Led Follow-Up

The observation-evolution screen failed its frozen module gate. Its
coefficients are not retuned. The subsequent selection-only cue replay
finds weak cue classifiers with different class preferences, not a strong
generic conflict signal. See CLASS_CONDITIONED_CUES_PLAN.md for a bounded
class-conditioned residual and paired-profile margin check on unchanged
inputs, with actual removals, an ordinary scalar gate and exact-partition
cached controls. These mechanisms remain adaptations, not demonstrated
novelty or improvements.

## Input Audit

The original model-facing Quadbot training view contains 4,472 trajectories
and 1,118 class-complete profiles. Each trajectory has 25 posts and 25
actions. `environment_triggered` is absent in every class. The legacy
exporter still serializes a constant `triggered=0`; the audit does not
support the suspicion that this particular field separates classes.

The legacy action vector repeats post text. It is an event serialization,
not AMR and not an independent modality. Step-specific hash namespaces
also do not align corresponding words across time windows.

The new development extractor uses only public post text/type, target
counts, rounded relative timestamps and public profile fields. It never
reads actions, graph, simulator step numbers, IDs, labels or provenance
as features. Identifiers/labels remain available only to select and check
the frozen training records. Text features remain signed lexical hashes,
not pretrained semantic representations.

## Frozen Candidate Screen

- Keep the existing 5,031/6,708 views, account identities and partitions.
- Build five chronological rank windows without simulator `time_step`.
- Use shared lexical hashing and explicit behavior/style/timing cues.
- Candidate A: shared cue encoders with a local causal convolution.
  The classifier still consumes the complete observed history; this is
  not an online classifier or proof of causal relationships.
- Candidate B: training-only cue self-distillation from detached fused
  logits. Individual cue supervision stays enabled in the distillation
  removal, isolating the additional alignment objective.
- Fit full, without A, without B, without both, and same-input MLP
  on three retained known-policy folds: 15 new fits, seed 42.
- Slice fitting/inner-selection identities before feature extraction.
  Fit normalization on fitting rows only. Do not open original validation
  or test files or transform/forward excluded-policy records.
- Replay every selected checkpoint and recompute every prediction metric.
- Continue only with the predeclared module contribution gate and a
  positive comparison against the same-input control. The 0.25-point
  development gate is not a CCF-B standard.
- Do not use legacy-input scores as matched comparisons. A surviving
  branch would still need same-input BotDMM/recent-method controls,
  additional seeds and genuinely unexposed confirmation.

No coefficient search, synthetic-account regeneration, manuscript
replacement, public push or submission is authorized by this screen.
