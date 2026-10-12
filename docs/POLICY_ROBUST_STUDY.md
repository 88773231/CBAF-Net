# Bounded Policy Robustness Study

This study retains the two canonical account views and their grouped splits.
It does not change the reported CBAF-Net method or the immutable `v1.1.0`
results. The existing outer validation challenges have already been inspected;
this is a post-hoc development study, not an independent final test.

The earlier temporal candidate failed on the unseen burst-scheduler policy.
The ordinary flattened-input MLP was stronger on that challenge. This round
therefore holds its architecture and inputs fixed and tests training objectives
before adding another architectural component.

## Two Existing Mechanisms

Group-DRO uses class-by-traditional-policy groups in the fitting data and
detached exponentiated loss weights. Policy metadata is only used for fitting
groups and auditing, never as a prediction feature.

R-Drop averages supervised losses from two stochastic dropout passes and
adds their symmetric KL divergence. Inputs, timestamps and labels are not
perturbed. No claim that timing changes preserve labels is needed.

Both mechanisms are established prior methods, not original contributions:

- Sagawa et al. Distributionally Robust Neural Networks for Group Shifts
  [primary paper](https://arxiv.org/abs/1911.08731),
  [official code](https://github.com/kohpangwei/group_DRO).
- Liang et al. R-Drop Regularized Dropout for Neural Networks
  [primary paper](https://arxiv.org/abs/2106.14448).

The implementations are independent adaptations, not full reproductions of
the original datasets, architectures or budgets.

## Fixed Comparison

`configs/policy_robust_protocol.json` freezes the protocol before this round.
ERM, Group-DRO, R-Drop and their combination use the same MLP parameter count,
training-fitted normalization, class-complete profile batching, optimizer,
epoch budget and inner checkpoint-selection rule. R-Drop uses two forward
passes per training batch; runtime and epoch count are reported rather than
claiming equal computational work. HGB is a fixed global-statistics control,
not an identical-encoding neural comparator.

Each held-out traditional policy is excluded from every fitting and inner
selection profile variant. Inner selection uses one sixth of the remaining
original training profiles. The checkpoint rule is worst-known-policy
Macro-F1, then overall inner Macro-F1, then MCC. Outer challenge scores are
computed only after checkpoint selection.

Seeds 42, 43 and 44, three excluded policies, five methods and two views yield
90 scheduled runs. Every result must be retained. Removing a training term
is an objective ablation, not an architectural module-removal experiment.

`analysis/audit_policy_robust.py` reloads each neural checkpoint and verifies
its ordered challenge probabilities. Complete normalization masks are
reconstructed from the frozen inner fitting indices into separate audit
sidecars; original checkpoint bytes are not edited. The selected checkpoints
are development artifacts, not paper deployment releases.

## Decision Boundary

No coefficient search is allowed after inspecting this round's outer scores.
A gain over one weak comparator, a favorable seed or a single policy is not
enough to establish a new final method. Prior challenge exposure prevents
confirmatory interpretation regardless of the scores. Independent behavior
or generator evidence and fair same-input domain backbones remain required.

The working paper's main tables and architecture figure are unchanged until
a defensible method contribution has evidence. Software regression success
does not authorize submission.
