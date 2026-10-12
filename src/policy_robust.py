"""Development-only policy-group DRO and dropout-consistency objectives."""

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.evidence_repair import (
    make_model, metrics, model_inputs, predict, seed_everything,
)


NEURAL_METHODS = ("erm_mlp", "group_dro_mlp", "rdrop_mlp", "group_dro_rdrop_mlp")


def symmetric_kl(first, second):
    if first.ndim != 2 or first.shape != second.shape or first.shape[1] < 2:
        raise ValueError("aligned multiclass logits are required")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("logits must be finite")
    left, right = first.log_softmax(1), second.log_softmax(1)
    return 0.5 * (
        (left.exp() * (left - right)).sum(1)
        + (right.exp() * (right - left)).sum(1)
    )


class GroupDRO:
    """Detached exponentiated group weights, without metadata prediction inputs."""

    def __init__(self, group_count, step_size):
        if group_count < 1 or not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("invalid robust objective configuration")
        self.log_weights = torch.zeros(group_count)
        self.step_size = step_size

    def loss(self, row_losses, group_ids):
        if row_losses.ndim != 1 or group_ids.shape != row_losses.shape:
            raise ValueError("row loss/group shapes differ")
        if group_ids.dtype != torch.long or not row_losses.numel():
            raise ValueError("nonempty integer group ids are required")
        if not torch.isfinite(row_losses).all() or (row_losses < -1e-6).any():
            raise ValueError("group losses must be finite and nonnegative")
        if group_ids.min() < 0 or group_ids.max() >= self.log_weights.numel():
            raise ValueError("group id outside the fitting vocabulary")
        means = torch.stack([
            row_losses[group_ids == index].mean()
            if (group_ids == index).any() else row_losses.sum() * 0
            for index in range(self.log_weights.numel())
        ])
        present = torch.bincount(group_ids, minlength=len(means)) > 0
        with torch.no_grad():
            self.log_weights += self.step_size * means.detach()
            self.log_weights -= torch.logsumexp(self.log_weights, dim=0)
        weights = self.log_weights.softmax(0) * present
        weights = weights / weights.sum()
        return (weights * means).sum()


def fitting_group_ids(rows, policies, class_count):
    vocabulary = sorted({
        (policies[row["base_profile_id"]], int(row["label"])) for row in rows
    })
    policy_names = sorted({name for name, _ in vocabulary})
    if set(vocabulary) != {
        (policy, label) for policy in policy_names for label in range(class_count)
    }:
        raise ValueError("every fitting policy needs all classes")
    positions = {key: index for index, key in enumerate(vocabulary)}
    return torch.tensor([
        positions[(policies[row["base_profile_id"]], int(row["label"]))]
        for row in rows
    ], dtype=torch.long), vocabulary


def profile_batches(rows, class_count, batch_size, generator):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["base_profile_id"], {})[int(row["label"])] = index
    if len(rows) != len(groups) * class_count:
        raise ValueError("each profile requires one distinct row per class")
    if any(set(indices) != set(range(class_count)) for indices in groups.values()):
        raise ValueError("profile batches require class-complete profiles")
    if batch_size < class_count:
        raise ValueError("batch cannot hold a complete profile")
    names = sorted(groups)
    per_batch = batch_size // class_count
    for block in torch.randperm(len(names), generator=generator).split(per_batch):
        yield torch.tensor([
            groups[names[position]][label]
            for position in block.tolist() for label in range(class_count)
        ], dtype=torch.long)


def selection_scores(values, probabilities, policies, class_count):
    names = sorted({policies[row["base_profile_id"]] for row in values["metadata"]})
    by_policy = {}
    for name in names:
        indices = [index for index, row in enumerate(values["metadata"])
                   if policies[row["base_profile_id"]] == name]
        by_policy[name] = metrics(
            values["labels"][indices].numpy(), probabilities[indices], class_count
        )
    aggregate = metrics(values["labels"].numpy(), probabilities, class_count)
    return {
        "all_profiles": aggregate, "by_fitting_policy": by_policy,
        "worst_policy_macro_f1": min(row["macro_f1"] for row in by_policy.values()),
    }


def selection_key(scores):
    return (scores["worst_policy_macro_f1"], scores["all_profiles"]["macro_f1"],
            scores["all_profiles"]["mcc"])


def train_policy_model(fitting, selection, policies, method, class_count, seed, config):
    if method not in NEURAL_METHODS:
        raise ValueError("unknown policy method")
    fitting_policies = {policies[row["base_profile_id"]] for row in fitting["metadata"]}
    selection_policies = {policies[row["base_profile_id"]] for row in selection["metadata"]}
    if fitting_policies != selection_policies:
        raise ValueError("inner selection must cover every fitting policy, and no others")
    seed_everything(seed)
    model = make_model(fitting, "mlp_matched", class_count, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    group_ids, vocabulary = fitting_group_ids(fitting["metadata"], policies, class_count)
    use_dro, use_rdrop = "group_dro" in method, "rdrop" in method
    objective = GroupDRO(len(vocabulary), config["dro_step_size"]) if use_dro else None
    generator = torch.Generator().manual_seed(seed)
    inputs = model_inputs(fitting, "mlp_matched")
    history, best_state, best_key, best_epoch, stale = [], None, None, None, 0
    started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses, consistency_losses = [], []
        for indices in profile_batches(
            fitting["metadata"], class_count, config["batch_size"], generator
        ):
            batch = [value[indices] for value in inputs]
            labels = fitting["labels"][indices]
            first = model(*batch)["logits"]
            row_loss = F.cross_entropy(first, labels, reduction="none")
            if use_rdrop:
                second = model(*batch)["logits"]
                consistency = symmetric_kl(first, second)
                row_loss = (
                    0.5 * (row_loss + F.cross_entropy(second, labels, reduction="none"))
                    + config["consistency_weight"] * consistency
                )
                consistency_losses.append(float(consistency.detach().mean()))
            loss = objective.loss(row_loss, group_ids[indices]) if objective else row_loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        probabilities, _ = predict(model, selection, "mlp_matched", config["batch_size"])
        scores = selection_scores(selection, probabilities, policies, class_count)
        key = selection_key(scores)
        history.append({
            "epoch": epoch, "training_loss": float(np.mean(losses)),
            "mean_consistency_kl": float(np.mean(consistency_losses)) if use_rdrop else None,
            "inner_selection": scores,
            "group_weights": objective.log_weights.softmax(0).tolist() if objective else None,
        })
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {
        "method": method, "seed": seed, "best_epoch": best_epoch, "history": history,
        "elapsed_seconds": time.monotonic() - started,
        "fitting_group_vocabulary": [list(key) for key in vocabulary],
        "group_metadata_used_as_prediction_input": False,
        "label_or_timing_augmentation": False,
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "inference_model": "MatchedFeatureMLP",
    }
