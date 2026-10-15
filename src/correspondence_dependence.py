"""Read-only fitting-row dependence probes, not retrained module removals."""

import math

import torch
from torch.nn import functional as F

from src.evidence_repair import model_inputs


PROBES = ("zero_discrepancy", "uniform_history", "current_block")


def fixed_history(steps, device, dtype):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 2:
        raise ValueError("at least two ordered blocks required")
    valid = torch.arange(steps, device=device)[:, None] >= torch.arange(steps, device=device)
    uniform = valid.to(dtype)
    return uniform / uniform.sum(-1, keepdim=True)


def encoded_blocks(model, inputs):
    description, text, action, tabular = inputs
    return (
        model.profile_encoder(description), model.statistics_encoder(tabular),
        torch.stack([encoder(text[:, step])
                     for step, encoder in enumerate(model.text_encoders)], 1),
        torch.stack([encoder(action[:, step])
                     for step, encoder in enumerate(model.action_encoders)], 1),
    )


def readout(model, encoded, text_difference, action_difference):
    profile, statistics, text, action = encoded
    blocks = model.block_encoder(torch.cat([
        text, action, text_difference, action_difference,
    ], -1))
    return model.readout(torch.cat([profile, statistics, blocks.flatten(1)], 1))


def attention_summary(weights):
    if (weights.ndim != 3 or weights.shape[1] != weights.shape[2]
            or weights.shape[0] < 1 or weights.shape[1] < 2
            or not torch.isfinite(weights).all() or (weights < 0).any()
            or not torch.allclose(weights.sum(-1), torch.ones_like(weights[:, :, 0]), atol=1e-6)):
        raise ValueError("finite normalized history attention required")
    steps = weights.shape[1]
    uniform = fixed_history(steps, weights.device, weights.dtype)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
    normalizer = torch.arange(1, steps + 1, device=weights.device, dtype=weights.dtype).log()
    lag = torch.arange(steps, device=weights.device)[:, None] - torch.arange(steps, device=weights.device)
    return {
        "diagonal_mass_by_step": weights.diagonal(dim1=1, dim2=2).mean(0).tolist(),
        "normalized_entropy_by_nontrivial_step": (entropy[:, 1:] / normalizer[1:]).mean(0).tolist(),
        "future_mass": float(weights[:, lag < 0].sum() / weights.shape[0]),
        "mean_absolute_uniform_difference": float((weights - uniform).abs().mean()),
        "lag_mass_mean_over_steps": [
            float(weights[:, lag == offset].sum() / (weights.shape[0] * steps))
            for offset in range(steps)
        ],
    }


def output_summary(logits, baseline, labels):
    difference = logits - baseline
    return {
        "fitting_cross_entropy": float(F.cross_entropy(logits, labels)),
        "fitting_accuracy": float((logits.argmax(1) == labels).float().mean()),
        "prediction_changes_from_baseline": int((logits.argmax(1) != baseline.argmax(1)).sum()),
        "mean_absolute_logit_change": float(difference.abs().mean()),
        "rms_logit_change": float(difference.square().mean().sqrt()),
        "maximum_probability_change": float(
            (logits.softmax(1) - baseline.softmax(1)).abs().max()),
    }


def diagnose(model, population, batch_size=128):
    if (model.training or model.correspondence is None or model.transition is not None
            or model.block_encoder[0].in_features != 4 * model.hidden_dim):
        raise ValueError("evaluation-mode complete compact checkpoint required")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("positive batch size required")
    inputs = model_inputs(population, "mlp_matched")
    labels = population["labels"]
    if (labels.ndim != 1 or len(labels) < 2 or labels.dtype != torch.long
            or (labels < 0).any() or (labels >= 4).any()
            or any(value.shape[0] != len(labels) for value in inputs)):
        raise ValueError("class-aligned fitting rows required")
    outputs = {name: [] for name in ("baseline", *PROBES)}
    attention = {"text_to_action": [], "action_to_text": []}
    alignment_norms = {"text_to_action": [], "action_to_text": []}
    representation_norms = {"text": [], "action": [], "text_difference": [], "action_difference": []}
    replay_error, decomposition_error = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            batch = [value[start:start + batch_size] for value in inputs]
            original = model(*batch)["logits"]
            encoded = encoded_blocks(model, batch)
            _, _, text, action = encoded
            correspondence = model.correspondence(text, action)
            td, ad = correspondence["text_discrepancy"], correspondence["action_discrepancy"]
            baseline = readout(model, encoded, td, ad)
            replay_error = max(replay_error, float((baseline - original).abs().max()))
            outputs["baseline"].append(baseline)
            outputs["zero_discrepancy"].append(readout(model, encoded, torch.zeros_like(td), torch.zeros_like(ad)))
            uniform = fixed_history(model.steps, text.device, text.dtype)
            outputs["uniform_history"].append(readout(
                model, encoded, text - uniform @ action, action - uniform @ text))
            outputs["current_block"].append(readout(model, encoded, text - action, action - text))
            for name, other in (("text_to_action", action), ("action_to_text", text)):
                weights = correspondence[name]
                attention[name].append(weights)
                alignment_norms[name].append(torch.stack([
                    ((weights - uniform) @ other).square().sum(-1),
                    (uniform @ other).square().sum(-1),
                ], -1))
            for name, block in (("text", text), ("action", action),
                                ("text_difference", td), ("action_difference", ad)):
                representation_norms[name].append(block.square().sum(-1))
            # Discrepancy concatenation repeats the direct current-block terms.
            wt, wa, wtd, wad = model.block_encoder[0].weight.split(model.hidden_dim, 1)
            linear = model.block_encoder[0](torch.cat([text, action, td, ad], -1))
            expanded = (
                F.linear(text, wt + wtd) + F.linear(action, wa + wad)
                - F.linear(correspondence["text_to_action"] @ action, wtd)
                - F.linear(correspondence["action_to_text"] @ text, wad)
                + model.block_encoder[0].bias
            )
            decomposition_error = max(decomposition_error, float((linear - expanded).abs().max()))
        baseline = torch.cat(outputs["baseline"])
        output_metrics = {name: output_summary(torch.cat(blocks), baseline, labels)
                          for name, blocks in outputs.items()}
        isolated = model(*(value[:1] for value in inputs))["logits"]
        changed = [value[:2].clone() for value in inputs]
        for value in changed:
            value[1].add_(10.0)
        isolation_error = max(
            float((isolated - baseline[:1]).abs().max()),
            float((model(*changed)["logits"][:1] - isolated).abs().max()),
        )
    if replay_error > 1e-6 or isolation_error > 1e-5 or decomposition_error > 1e-5:
        raise ValueError("diagnostic replay, row isolation or linear decomposition differs")
    norms = {name: float(torch.cat(blocks).mean().sqrt())
             for name, blocks in representation_norms.items()}
    relative_alignment = {}
    for name, blocks in alignment_norms.items():
        squared = torch.cat(blocks).mean((0, 1))
        relative_alignment[name] = float((squared[0] / squared[1].clamp_min(1e-30)).sqrt())
    weight = model.block_encoder[0].weight
    weight_norms = {
        name: float(block.norm()) for name, block in zip(
            ("direct_text", "direct_action", "text_difference", "action_difference"),
            weight.split(model.hidden_dim, 1))
    }
    result = {
        "fitting_rows": len(labels),
        "class_counts": torch.bincount(labels, minlength=4).tolist(),
        "outputs": output_metrics,
        "attention": {name: attention_summary(torch.cat(blocks)) for name, blocks in attention.items()},
        "representation_rms_norm": norms, "block_input_weight_l2_norm": weight_norms,
        "adaptive_alignment_rms_relative_to_uniform": relative_alignment,
        "maximum_baseline_replay_error": replay_error, "maximum_row_isolation_error": isolation_error,
        "maximum_linear_decomposition_error": decomposition_error,
        "probes_are_retrained_ablations": False, "causal_effect_established": False,
        "selection_or_challenge_or_test_forwards": False, "submission_allowed": False,
    }
    if any(not math.isfinite(value) for value in (*norms.values(), *weight_norms.values(),
                                                 *relative_alignment.values())):
        raise ValueError("nonfinite dependence diagnostic")
    return result
