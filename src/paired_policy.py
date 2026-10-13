"""Development-only cross-policy contrast and matched-profile boundary losses."""

import copy
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.evidence_model import MatchedFeatureMLP
from src.evidence_repair import assert_group_disjoint, model_inputs, seed_everything
from src.policy_robust import profile_batches, selection_key, selection_scores


METHODS = ("paired_policy", "without_policy_contrast", "without_pair_margin")


class PairedPolicyModel(MatchedFeatureMLP):
    """Keep the matched MLP's exact parameters and expose its representation."""

    def forward(self, description, text, action, tabular):
        values = torch.cat([description, text.flatten(1), action.flatten(1), tabular], dim=1)
        representation = self.network[:3](values)
        return {
            "logits": self.network[3:](representation),
            "representation": representation,
        }


def cross_policy_contrast(representation, labels, policy_ids, temperature):
    if (representation.ndim != 2 or not len(representation)
            or labels.shape != (len(representation),) or policy_ids.shape != labels.shape
            or labels.dtype != torch.long or policy_ids.dtype != torch.long
            or not torch.isfinite(representation).all()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("invalid cross-policy contrast inputs")
    same_class = labels[:, None] == labels[None, :]
    positives = same_class & (policy_ids[:, None] != policy_ids[None, :])
    # Same-class/same-policy rows are neither false negatives nor new positives.
    included = positives | ~same_class
    valid = positives.any(1)
    if not valid.any():
        return representation.sum() * 0
    normalized = F.normalize(representation, dim=1)
    similarities = normalized @ normalized.T / temperature
    partition = similarities[valid].masked_fill(~included[valid], float("-inf")).logsumexp(1)
    log_probability = similarities[valid] - partition[:, None]
    return -(
        log_probability.masked_fill(~positives[valid], 0).sum(1)
        / positives[valid].sum(1)
    ).mean()


def matched_boundary_margin(logits, labels, profile_ids, margin):
    if (logits.ndim != 2 or logits.shape[1] not in (3, 4)
            or labels.shape != (len(logits),) or labels.dtype != torch.long
            or len(profile_ids) != len(logits) or not torch.isfinite(logits).all()
            or not math.isfinite(margin) or margin < 0):
        raise ValueError("invalid matched-boundary inputs")
    groups = {}
    for index, (profile, label) in enumerate(zip(profile_ids, labels.tolist())):
        if label < 0 or label >= logits.shape[1]:
            raise ValueError("label outside taxonomy")
        members = groups.setdefault(str(profile), {})
        if label in members:
            raise ValueError("duplicate profile/class row")
        members[label] = index
    if not groups or any(0 not in members or 1 not in members for members in groups.values()):
        raise ValueError("every matched profile requires Human and Traditional Bot")
    human = torch.tensor([members[0] for members in groups.values()], dtype=torch.long)
    traditional = torch.tensor([members[1] for members in groups.values()], dtype=torch.long)
    boundary = logits[:, 1] - logits[:, 0]
    return F.softplus(margin - (boundary[traditional] - boundary[human])).mean()


def make_paired_model(values, class_count, config):
    if class_count not in (3, 4):
        raise ValueError("fixed three/four-class taxonomy required")
    inputs = model_inputs(values, "mlp_matched")
    size = sum(value[0].numel() for value in inputs)
    return PairedPolicyModel(size, class_count, config["mlp_hidden_dim"], config["dropout"])


def predict_paired(model, values, batch_size):
    if batch_size < 1:
        raise ValueError("positive prediction batch size required")
    model.eval()
    inputs, blocks = model_inputs(values, "mlp_matched"), []
    with torch.no_grad():
        for start in range(0, len(values["labels"]), batch_size):
            output = model(*(value[start:start + batch_size] for value in inputs))
            blocks.append(output["logits"].softmax(1).numpy())
    return np.concatenate(blocks)


def train_paired(fitting, selection, policies, method, class_count, seed, config,
                 regularization, progress=None):
    if method not in METHODS:
        raise ValueError("unknown paired-policy method")
    assert_group_disjoint(fitting, selection)
    fitting_policies = sorted({policies[row["base_profile_id"]] for row in fitting["metadata"]})
    if (len(fitting_policies) != 2 or set(fitting_policies) != {
        policies[row["base_profile_id"]] for row in selection["metadata"]
    }):
        raise ValueError("only the two fitting policies may supervise training")
    if config["max_epochs"] < 1 or config["patience"] < 1:
        raise ValueError("positive epoch/patience budget required")
    for name in ("contrast_weight", "pair_weight", "margin"):
        if not math.isfinite(regularization[name]) or regularization[name] < 0:
            raise ValueError("invalid regularization weight or margin")
    if not math.isfinite(regularization["temperature"]) or regularization["temperature"] <= 0:
        raise ValueError("invalid contrast temperature")
    labels = fitting["labels"]
    if labels.tolist() != [int(row["label"]) for row in fitting["metadata"]]:
        raise ValueError("ordered training labels differ")
    policy_ids = torch.tensor([
        fitting_policies.index(policies[row["base_profile_id"]]) for row in fitting["metadata"]
    ], dtype=torch.long)
    profiles = [row["base_profile_id"] for row in fitting["metadata"]]
    contrast_weight = (
        regularization["contrast_weight"] if method != "without_policy_contrast" else 0
    )
    pair_weight = regularization["pair_weight"] if method != "without_pair_margin" else 0
    seed_everything(seed)
    model = make_paired_model(fitting, class_count, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    inputs = model_inputs(fitting, "mlp_matched")
    generator = torch.Generator().manual_seed(seed)
    history, best_state, best_key, best_epoch, stale = [], None, None, None, 0
    started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses, contrasts, margins = [], [], []
        for indices in profile_batches(fitting["metadata"], class_count, config["batch_size"], generator):
            output = model(*(value[indices] for value in inputs))
            contrast = (
                cross_policy_contrast(
                    output["representation"], labels[indices], policy_ids[indices],
                    regularization["temperature"],
                ) if contrast_weight else output["logits"].sum() * 0
            )
            pair = (
                matched_boundary_margin(
                    output["logits"], labels[indices], [profiles[i] for i in indices.tolist()],
                    regularization["margin"],
                ) if pair_weight else output["logits"].sum() * 0
            )
            loss = F.cross_entropy(output["logits"], labels[indices])
            if contrast_weight:
                loss = loss + contrast_weight * contrast
            if pair_weight:
                loss = loss + pair_weight * pair
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            contrasts.append(float(contrast.detach()))
            margins.append(float(pair.detach()))
        scores = selection_scores(
            selection, predict_paired(model, selection, config["batch_size"]), policies, class_count,
        )
        key = selection_key(scores)
        history.append({
            "epoch": epoch, "training_loss": float(np.mean(losses)),
            "cross_policy_contrast": float(np.mean(contrasts)),
            "matched_boundary_margin": float(np.mean(margins)), "inner_selection": scores,
        })
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
        "regularization": regularization, "active_contrast_weight": contrast_weight,
        "active_pair_weight": pair_weight,
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "policy_or_profile_metadata_used_as_prediction_input": False,
        "class_or_timing_augmentation": False, "graph_input": False,
    }
