"""Fixed development recipe: policy contrast with standard SAM training."""

import copy
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.evidence_repair import assert_group_disjoint, model_inputs, seed_everything
from src.paired_policy import cross_policy_contrast, make_paired_model, predict_paired
from src.policy_robust import profile_batches, selection_key, selection_scores


METHODS = ("flat_policy", "without_policy_contrast", "two_pass_erm")


def sam_step(model, optimizer, closure, radius):
    """Minimize loss at one gradient-defined L2 perturbation, then restore."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("SAM radius must be finite and positive")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("this frozen recipe is CPU-only")
    optimizer.zero_grad(set_to_none=True)
    initial_rng_state = torch.get_rng_state()
    first_loss = closure()
    if first_loss.ndim != 0 or not torch.isfinite(first_loss):
        raise ValueError("SAM closure must return a finite scalar")
    first_loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters or any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise ValueError("SAM requires finite nonempty gradients")
    norm = torch.sqrt(sum(parameter.grad.square().sum() for parameter in parameters))
    references = [parameter.detach().clone() for parameter in parameters]
    # Reuse the dropout realization so the perturbation targets the same loss.
    rng_state = torch.get_rng_state()
    try:
        with torch.no_grad():
            for parameter, reference in zip(parameters, references):
                parameter.copy_(reference + radius * parameter.grad / norm.clamp_min(1e-12))
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(initial_rng_state)
        second_loss = closure()
        if second_loss.ndim != 0 or not torch.isfinite(second_loss):
            raise ValueError("SAM perturbed closure must return a finite scalar")
        second_loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise ValueError("SAM perturbed gradient is nonfinite")
    finally:
        with torch.no_grad():
            for parameter, reference in zip(parameters, references):
                parameter.copy_(reference)
        torch.set_rng_state(rng_state)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"base_loss": float(first_loss.detach()), "perturbed_loss": float(second_loss.detach()),
            "base_gradient_norm": float(norm.detach()), "loss_increase": float(
                (second_loss - first_loss).detach()),
            "gradient_evaluations": 2, "optimizer_updates": 1}


def train_flat(fitting, selection, policies, method, class_count, seed, config,
               regularization, progress=None):
    if method not in METHODS:
        raise ValueError("unknown fixed flat-policy condition")
    assert_group_disjoint(fitting, selection)
    names = sorted({policies[row["base_profile_id"]] for row in fitting["metadata"]})
    if len(names) != 2 or set(names) != {
        policies[row["base_profile_id"]] for row in selection["metadata"]
    }:
        raise ValueError("selection must cover only both fitting policies")
    if config["max_epochs"] < 1 or config["patience"] < 1:
        raise ValueError("positive training budget required")
    for key in ("contrast_weight", "temperature", "sam_radius"):
        value = regularization[key]
        if not math.isfinite(value) or value < 0 or (key != "contrast_weight" and value == 0):
            raise ValueError("invalid fixed flat-policy regularization")
    if fitting["labels"].tolist() != [int(row["label"]) for row in fitting["metadata"]]:
        raise ValueError("ordered fitting labels differ")
    weight = regularization["contrast_weight"] if method == "flat_policy" else 0.0
    policy_ids = torch.tensor([
        names.index(policies[row["base_profile_id"]]) for row in fitting["metadata"]
    ], dtype=torch.long)
    seed_everything(seed)
    model = make_paired_model(fitting, class_count, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    inputs = model_inputs(fitting, "mlp_matched")
    generator = torch.Generator().manual_seed(seed)
    history, best_key, best_state, best_epoch, stale = [], None, None, None, 0
    evaluations, updates = 0, 0
    started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses, contrasts, increases = [], [], []
        for indices in profile_batches(fitting["metadata"], class_count, config["batch_size"], generator):
            last_contrast = []

            def closure():
                outputs = model(*(value[indices] for value in inputs))
                contrast = (
                    cross_policy_contrast(outputs["representation"], fitting["labels"][indices],
                                          policy_ids[indices], regularization["temperature"])
                    if weight else outputs["logits"].sum() * 0
                )
                last_contrast.append(float(contrast.detach()))
                return F.cross_entropy(outputs["logits"], fitting["labels"][indices]) + weight * contrast

            if method == "two_pass_erm":
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss = closure()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    evaluations += 1
                    updates += 1
            else:
                record = sam_step(model, optimizer, closure, regularization["sam_radius"])
                losses.append(record["base_loss"])
                increases.append(record["loss_increase"])
                evaluations += record["gradient_evaluations"]
                updates += record["optimizer_updates"]
            contrasts.append(last_contrast[0])
        scores = selection_scores(
            selection, predict_paired(model, selection, config["batch_size"]), policies, class_count,
        )
        key = selection_key(scores)
        history.append({
            "epoch": epoch, "training_loss": float(np.mean(losses)),
            "cross_policy_contrast": float(np.mean(contrasts)),
            "mean_sam_loss_increase": float(np.mean(increases)) if increases else None,
            "inner_selection": scores, "gradient_evaluations_so_far": evaluations,
            "optimizer_updates_so_far": updates,
        })
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch, stale = key, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if progress is not None:
            progress({"epoch": epoch, "best_epoch": best_epoch, "stale_epochs": stale})
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {
        "history": history, "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.monotonic() - started, "regularization": regularization,
        "active_contrast_weight": weight, "sam_used": method != "two_pass_erm",
        "sam_radius": regularization["sam_radius"] if method != "two_pass_erm" else None,
        "gradient_evaluations": evaluations, "optimizer_updates": updates,
        "gradient_evaluations_per_batch": 2,
        "optimizer_updates_per_batch": 2 if method == "two_pass_erm" else 1,
        "dropout_realization_reused_for_sam": method != "two_pass_erm",
        "pair_margin_used": False, "cosine_head_used": False,
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "policy_or_profile_metadata_used_as_prediction_input": False,
        "label_timing_or_account_augmentation": False, "graph_input": False,
    }
