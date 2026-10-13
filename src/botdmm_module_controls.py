"""Bounded module removals; preserve the earlier timing experiment source bytes."""

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.botdmm_development import (
    forward_seed_logits, make_backbone, predict_backbone,
)
from src.data_loader import StrictOneHopNeighborCollator
from src.evidence_repair import seed_everything
from src.policy_robust import profile_batches, selection_key, selection_scores


METHOD_MODES = {
    "botdmm_without_structure": "no_structure",
    "botdmm_without_orthogonal": "no_orthogonal",
}


def make_module_control(view, class_count, config, method):
    if method not in METHOD_MODES:
        raise ValueError("unknown fixed module removal")
    model = make_backbone(view, class_count, config)
    model.ablation_mode = METHOD_MODES[method]
    return model


def train_module_control(fitting, selection, selection_values, policies,
                         class_count, seed, config, method, progress=None):
    if set(fitting.base_profile_ids) & set(selection.base_profile_ids):
        raise ValueError("fitting and selection profiles overlap")
    seed_everything(seed)
    model = make_module_control(fitting, class_count, config, method)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    collator = StrictOneHopNeighborCollator(fitting, expected_k=config["knn_k"])
    generator = torch.Generator().manual_seed(seed)
    history, best_state, best_key, best_epoch, stale = [], None, None, None, 0
    started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        for indices in profile_batches(fitting.metadata, class_count, config["batch_size"], generator):
            batch = collator([{"_index": index} for index in indices.tolist()])
            loss = F.cross_entropy(forward_seed_logits(model, batch), batch["seed_labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        probabilities = predict_backbone(model, selection, config["batch_size"], config["knn_k"])
        scores = selection_scores(selection_values, probabilities, policies, class_count)
        key = selection_key(scores)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "inner_selection": scores})
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if progress is not None:
            progress({"epoch": epoch, "best_epoch": best_epoch, "stale_epochs": stale,
                      "elapsed_seconds": time.monotonic() - started})
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {
        "history": history, "best_epoch": best_epoch,
        "elapsed_seconds": time.monotonic() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "ablation_mode": METHOD_MODES[method],
        "objective": "seed_row_cross_entropy_auxiliary_losses_disabled",
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "graph_fit": fitting.graph_receipt, "graph_inner_selection": selection.graph_receipt,
        "parameter_layout_preserved": True, "inactive_modules_not_claimed_removed_from_checkpoint": True,
        "context_batch_and_batchnorm_unchanged": True,
        "policy_metadata_used_as_prediction_input": False,
    }
