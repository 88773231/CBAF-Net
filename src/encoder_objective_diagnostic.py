"""Known-policy objective audit on one unchanged control encoder."""

import copy
import hashlib
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.botsscl_adapter import CorruptionAdapter
from src.botsscl_policy_control import predict_control
from src.evidence_repair import assert_group_disjoint, flattened_inputs, seed_everything
from src.flat_policy import sam_step
from src.paired_policy import cross_policy_contrast
from src.policy_robust import profile_batches, selection_key, selection_scores


CONDITIONS = ("supervised_reference", "contrast_only", "sam_only", "sam_contrast")


def state_digest(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def validate_populations(fitting, selection, policies, class_count):
    assert_group_disjoint(fitting, selection)
    names = sorted({policies[row["base_profile_id"]] for row in fitting["metadata"]})
    if len(names) != 2 or set(names) != {
        policies[row["base_profile_id"]] for row in selection["metadata"]
    }:
        raise ValueError("only both known policies may enter the objective audit")
    for values in (fitting, selection):
        if values["labels"].tolist() != [int(row["label"]) for row in values["metadata"]]:
            raise ValueError("ordered labels differ")
        if set(values["labels"].tolist()) != set(range(class_count)):
            raise ValueError("incomplete diagnostic taxonomy")
        list(profile_batches(values["metadata"], class_count, class_count,
                             torch.Generator().manual_seed(42)))
    return names


def gradient_probe(model, inputs, labels, policy_ids, regularization):
    """Measure objective gradients without changing parameters or RNG state."""
    before, rng = state_digest(model), torch.get_rng_state().clone()
    was_training = model.training
    model.eval()
    try:
        representation = model.encoder(inputs)
        ce = F.cross_entropy(model.classifier(representation), labels)
        contrast = cross_policy_contrast(
            representation, labels, policy_ids, regularization["temperature"],
        )
        parameters = [*model.encoder.parameters(), *model.classifier.parameters()]
        ce_gradients = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
        pc_gradients = torch.autograd.grad(
            regularization["contrast_weight"] * contrast, parameters, allow_unused=True,
        )
        left = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                          for p, g in zip(parameters, ce_gradients)])
        right = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                           for p, g in zip(parameters, pc_gradients)])
        left_norm, right_norm = float(left.norm()), float(right.norm())
        alignment = (
            float(torch.dot(left, right)) / (left_norm * right_norm)
            if left_norm > 0 and right_norm > 0 else None
        )
        result = {
            "cross_entropy": float(ce.detach()), "contrast_loss": float(contrast.detach()),
            "ce_gradient_norm": left_norm, "weighted_contrast_gradient_norm": right_norm,
            "weighted_contrast_to_ce_norm_ratio": right_norm / left_norm if left_norm else None,
            "gradient_cosine": alignment,
        }
    finally:
        model.train(was_training)
    if state_digest(model) != before or not torch.equal(torch.get_rng_state(), rng):
        raise ValueError("gradient diagnostic changed model or RNG")
    return {**result, "parameters_and_rng_unchanged": True}


def train_objective(fitting, selection, policies, condition, class_count, seed, config,
                    regularization, progress=None):
    if condition not in CONDITIONS or class_count != 4:
        raise ValueError("this bounded diagnostic is four-class only")
    names = validate_populations(fitting, selection, policies, class_count)
    if min(config["max_epochs"], config["patience"], config["mlp_hidden_dim"]) < 1:
        raise ValueError("positive training budget required")
    if config["batch_size"] < class_count:
        raise ValueError("class-complete batch required")
    for key in ("contrast_weight", "temperature", "sam_radius"):
        value = regularization[key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid fixed diagnostic regularization")
    train_x = torch.from_numpy(flattened_inputs(fitting))
    if not torch.isfinite(train_x).all():
        raise ValueError("finite known-policy inputs required")
    policy_ids = torch.tensor([
        names.index(policies[row["base_profile_id"]]) for row in fitting["metadata"]
    ], dtype=torch.long)
    seed_everything(seed)
    model = CorruptionAdapter(train_x.shape[1], class_count, config["mlp_hidden_dim"])
    # Keep the control's initialization order, but never train its unused projector.
    model.projection.requires_grad_(False)
    initialization_digest = state_digest(model)
    seed_everything(seed)
    parameters = [*model.encoder.parameters(), *model.classifier.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    probe_indices = next(profile_batches(
        fitting["metadata"], class_count, config["batch_size"],
        torch.Generator().manual_seed(seed),
    ))
    initial_probe = gradient_probe(
        model, train_x[probe_indices], fitting["labels"][probe_indices],
        policy_ids[probe_indices], regularization,
    )
    use_contrast = condition in ("contrast_only", "sam_contrast")
    use_sam = condition in ("sam_only", "sam_contrast")
    generator = torch.Generator().manual_seed(seed)
    history, best_key, best_state, best_epoch, stale = [], None, None, None, 0
    updates, evaluations, started = 0, 0, time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        for indices in profile_batches(fitting["metadata"], class_count, config["batch_size"], generator):
            def closure():
                representation = model.encoder(train_x[indices])
                logits = model.classifier(representation)
                loss = F.cross_entropy(logits, fitting["labels"][indices])
                if use_contrast:
                    loss = loss + regularization["contrast_weight"] * cross_policy_contrast(
                        representation, fitting["labels"][indices],
                        policy_ids[indices], regularization["temperature"],
                    )
                return loss

            if use_sam:
                record = sam_step(model, optimizer, closure, regularization["sam_radius"])
                losses.append(record["base_loss"])
                evaluations += 2
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = closure()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                evaluations += 1
            updates += 1
        scores = selection_scores(
            selection, predict_control(model, selection, config["batch_size"]), policies, class_count,
        )
        key = selection_key(scores)
        history.append({
            "epoch": epoch, "training_loss": float(np.mean(losses)),
            "inner_selection": scores, "optimizer_updates": updates,
            "gradient_evaluations": evaluations,
        })
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch, stale = key, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if progress:
            progress({"epoch": epoch, "best_epoch": best_epoch, "stale_epochs": stale})
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    selected_probe = gradient_probe(
        model, train_x[probe_indices], fitting["labels"][probe_indices],
        policy_ids[probe_indices], regularization,
    )
    return model, {
        "condition": condition, "history": history, "best_epoch": best_epoch,
        "initialization_digest": initialization_digest, "initial_probe": initial_probe,
        "selected_probe": selected_probe, "optimizer_updates": updates,
        "gradient_evaluations": evaluations, "elapsed_seconds": time.monotonic() - started,
        "input_dimension": train_x.shape[1],
        "inference_parameters": sum(p.numel() for p in parameters),
        "active_contrast": use_contrast, "active_sam": use_sam,
        "pretraining_used": False, "projector_used": False, "graph_used": False,
        "challenge_forward_or_scoring": False,
    }
