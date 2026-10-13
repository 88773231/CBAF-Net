"""Fixed development hypothesis: normalized readout plus cross-policy contrast."""

import copy
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evidence_repair import assert_group_disjoint, model_inputs, seed_everything
from src.paired_policy import PairedPolicyModel, cross_policy_contrast
from src.policy_robust import profile_batches, selection_key, selection_scores


METHODS = ("cosine_policy", "without_policy_contrast")


class CosinePolicyModel(PairedPolicyModel):
    """Preserve the encoder; replace the ordinary linear readout explicitly."""

    def __init__(self, input_dim, class_count, hidden_dim=128, dropout=0.2, scale=16.0):
        if (class_count not in (3, 4) or min(input_dim, hidden_dim) < 1
                or not 0 <= dropout < 1 or not math.isfinite(scale) or scale <= 0):
            raise ValueError("invalid fixed cosine classifier dimensions or scale")
        super().__init__(input_dim, class_count, hidden_dim, dropout)
        # Keep the same seeded weight initialization, without a redundant bias.
        self.prototype_weight = nn.Parameter(self.network[4].weight.detach().clone())
        self.network = self.network[:4]
        self.register_buffer("logit_scale", torch.tensor(float(scale)))

    def forward(self, description, text, action, tabular):
        values = torch.cat([description, text.flatten(1), action.flatten(1), tabular], dim=1)
        representation = self.network[:3](values)
        classifier_input = self.network[3](representation)
        logits = self.logit_scale * F.linear(
            F.normalize(classifier_input, dim=1), F.normalize(self.prototype_weight, dim=1),
        )
        return {"logits": logits, "representation": representation}


def make_cosine_model(values, class_count, config, scale):
    inputs = model_inputs(values, "mlp_matched")
    size = sum(value[0].numel() for value in inputs)
    return CosinePolicyModel(size, class_count, config["mlp_hidden_dim"], config["dropout"], scale)


def predict_cosine(model, values, batch_size):
    if batch_size < 1:
        raise ValueError("positive batch size required")
    model.eval()
    inputs, blocks = model_inputs(values, "mlp_matched"), []
    with torch.no_grad():
        for start in range(0, len(values["labels"]), batch_size):
            blocks.append(model(*(value[start:start + batch_size] for value in inputs))["logits"].softmax(1).numpy())
    return np.concatenate(blocks)


def train_cosine(fitting, selection, policies, method, class_count, seed, config,
                 regularization, progress=None):
    if method not in METHODS:
        raise ValueError("unknown fixed cosine condition")
    assert_group_disjoint(fitting, selection)
    names = sorted({policies[row["base_profile_id"]] for row in fitting["metadata"]})
    if len(names) != 2 or set(names) != {
        policies[row["base_profile_id"]] for row in selection["metadata"]
    }:
        raise ValueError("selection must contain only both fitting policies")
    if config["max_epochs"] < 1 or config["patience"] < 1:
        raise ValueError("positive training budget required")
    if (not math.isfinite(regularization["contrast_weight"]) or regularization["contrast_weight"] < 0
            or not math.isfinite(regularization["temperature"]) or regularization["temperature"] <= 0):
        raise ValueError("invalid fixed contrast objective")
    if fitting["labels"].tolist() != [int(row["label"]) for row in fitting["metadata"]]:
        raise ValueError("ordered label mismatch")
    policy_ids = torch.tensor([
        names.index(policies[row["base_profile_id"]]) for row in fitting["metadata"]
    ], dtype=torch.long)
    weight = regularization["contrast_weight"] if method == "cosine_policy" else 0
    seed_everything(seed)
    model = make_cosine_model(fitting, class_count, config, regularization["cosine_scale"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    inputs = model_inputs(fitting, "mlp_matched")
    generator = torch.Generator().manual_seed(seed)
    history, best_key, best_state, best_epoch, stale = [], None, None, None, 0
    started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses, contrasts = [], []
        for indices in profile_batches(fitting["metadata"], class_count, config["batch_size"], generator):
            outputs = model(*(value[indices] for value in inputs))
            contrast = (
                cross_policy_contrast(outputs["representation"], fitting["labels"][indices],
                                      policy_ids[indices], regularization["temperature"])
                if weight else outputs["logits"].sum() * 0
            )
            loss = F.cross_entropy(outputs["logits"], fitting["labels"][indices])
            if weight:
                loss = loss + weight * contrast
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            contrasts.append(float(contrast.detach()))
        scores = selection_scores(
            selection, predict_cosine(model, selection, config["batch_size"]), policies, class_count,
        )
        key = selection_key(scores)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "cross_policy_contrast": float(np.mean(contrasts)), "inner_selection": scores})
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
        "elapsed_seconds": time.monotonic() - started,
        "regularization": regularization, "active_contrast_weight": weight,
        "pair_margin_used": False, "head": "fixed_scale_normalized_linear_prototypes_without_bias",
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "policy_or_profile_metadata_used_as_prediction_input": False,
        "class_or_timing_augmentation": False, "graph_input": False,
    }
