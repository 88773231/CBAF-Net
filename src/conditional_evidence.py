"""Development-only conditional taxonomy with an optional content bridge."""

import copy
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evidence_repair import assert_group_disjoint, model_inputs, seed_everything
from src.policy_robust import profile_batches, selection_key, selection_scores


METHODS = (
    "conditional_bridge", "without_conditional_routing",
    "without_content_bridge", "without_both",
)


def operation_targets(class_count):
    if class_count not in (3, 4):
        raise ValueError("only the frozen three/four-class taxonomy is supported")
    return torch.tensor([0, 1, 1, 2][:class_count], dtype=torch.long)


def conditional_log_probs(generation, operation, class_count):
    """Only the automatic-posting branch needs the Traditional/LLM decision."""
    operation_targets(class_count)
    if (generation.ndim != 2 or operation.ndim != 2
            or generation.shape != (operation.shape[0], 2)
            or operation.shape[1] != class_count - 1
            or not torch.isfinite(generation).all() or not torch.isfinite(operation).all()):
        raise ValueError("conditional head shapes or values differ")
    generated, operated = generation.log_softmax(1), operation.log_softmax(1)
    leaves = [
        operated[:, 0],
        operated[:, 1] + generated[:, 0],
        operated[:, 1] + generated[:, 1],
    ]
    if class_count == 4:
        leaves.append(operated[:, 2])
    return torch.stack(leaves, dim=1)


class ConditionalEvidenceModel(nn.Module):
    """Content/behavior encoders, conditional leaves and a removable bridge."""

    def __init__(self, description_dim, text_dim, event_dim, class_count, steps=5,
                 hidden_dim=128, dropout=0.2, method="conditional_bridge"):
        super().__init__()
        if method not in METHODS:
            raise ValueError("unsupported conditional evidence method")
        if min(description_dim, text_dim, event_dim, steps, hidden_dim) < 1:
            raise ValueError("positive model dimensions are required")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        targets = operation_targets(class_count)
        self.steps, self.class_count, self.method = steps, class_count, method
        self.register_buffer("operation_targets", targets)

        def encoder(dim):
            return nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
            )

        self.profile_encoder = encoder(description_dim)
        self.text_encoders = nn.ModuleList([encoder(text_dim) for _ in range(steps)])
        self.event_encoders = nn.ModuleList([encoder(event_dim) for _ in range(steps)])
        self.content_encoder = encoder(hidden_dim * (steps + 1) + 9)
        self.behavior_encoder = encoder(hidden_dim * steps + 15)
        self.operation_behavior = nn.Linear(hidden_dim, hidden_dim)
        self.operation_transform = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.generation_head = nn.Linear(hidden_dim, 2)
        self.operation_head = nn.Linear(hidden_dim, class_count - 1)
        self.flat_head = (
            nn.Linear(hidden_dim * 2, class_count)
            if method in ("without_conditional_routing", "without_both") else None
        )
        self.content_bridge = (
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            if method not in ("without_content_bridge", "without_both") else None
        )
        if self.content_bridge is not None:
            # Start with behavior evidence, then learn any useful content correction.
            nn.init.zeros_(self.content_bridge.weight)

    def forward(self, description, text, action, tabular):
        if (description.ndim != 2 or text.ndim != 3 or action.ndim != 3
                or tabular.ndim != 2 or text.shape[1] != self.steps
                or action.shape[1] != self.steps or tabular.shape[1] != 24
                or action.shape[2] < 8
                or len({description.shape[0], text.shape[0], action.shape[0],
                        tabular.shape[0]}) != 1):
            raise ValueError("fixed timing-aware input schema differs")
        content = self.content_encoder(torch.cat([
            self.profile_encoder(description),
            *[encoder(text[:, step]) for step, encoder in enumerate(self.text_encoders)],
            tabular[:, 6:15],
        ], dim=1))
        behavior = self.behavior_encoder(torch.cat([
            *[encoder(action[:, step]) for step, encoder in enumerate(self.event_encoders)],
            tabular[:, :6], tabular[:, 15:],
        ], dim=1))
        operation_value = self.operation_behavior(behavior)
        if self.content_bridge is not None:
            operation_value = operation_value + self.content_bridge(content)
        operation_context = self.operation_transform(operation_value)
        generation = self.generation_head(content)
        operation = self.operation_head(operation_context)
        logits = (
            self.flat_head(torch.cat([content, operation_context], dim=1))
            if self.flat_head is not None
            else conditional_log_probs(generation, operation, self.class_count)
        )
        return {
            "logits": logits, "generation_logits": generation,
            "operation_logits": operation,
        }


def conditional_loss(outputs, labels, class_count, auxiliary_weight):
    targets = operation_targets(class_count).to(labels.device)
    if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError("invalid auxiliary weight")
    if (labels.ndim != 1 or labels.dtype != torch.long or not len(labels)
            or labels.min() < 0 or labels.max() >= class_count):
        raise ValueError("invalid fixed-taxonomy labels")
    # Human and Agent are direct leaves, not targets of the subtype head.
    subtype_rows = (labels == 1) | (labels == 2)
    generation_loss = (
        F.cross_entropy(outputs["generation_logits"][subtype_rows], labels[subtype_rows] - 1)
        if subtype_rows.any() else outputs["generation_logits"].sum() * 0
    )
    operation_loss = F.cross_entropy(outputs["operation_logits"], targets[labels])
    auxiliary = 0.5 * (generation_loss + operation_loss)
    classification = F.cross_entropy(outputs["logits"], labels)
    return classification + auxiliary_weight * auxiliary, {
        "generation_loss": generation_loss, "operation_loss": operation_loss,
    }


def make_conditional_model(values, class_count, config, method):
    description, text, action, _ = model_inputs(values, "mlp_matched")
    return ConditionalEvidenceModel(
        description.shape[1], text.shape[2], action.shape[2], class_count,
        steps=text.shape[1], hidden_dim=config["hidden_dim"],
        dropout=config["dropout"], method=method,
    )


def predict_conditional(model, values, batch_size):
    if batch_size < 1:
        raise ValueError("positive prediction batch size required")
    model.eval()
    inputs, blocks = model_inputs(values, "mlp_matched"), []
    with torch.no_grad():
        for start in range(0, len(values["labels"]), batch_size):
            result = model(*(value[start:start + batch_size] for value in inputs))
            blocks.append(result["logits"].softmax(1).numpy())
    return np.concatenate(blocks)


def train_conditional(fitting, selection, policies, method, class_count, seed, config, progress=None):
    assert_group_disjoint(fitting, selection)
    if {policies[row["base_profile_id"]] for row in fitting["metadata"]} != {
        policies[row["base_profile_id"]] for row in selection["metadata"]
    }:
        raise ValueError("inner selection must cover only the fitting policy set")
    if config["max_epochs"] < 1 or config["patience"] < 1:
        raise ValueError("positive epoch and patience budget required")
    seed_everything(seed)
    model = make_conditional_model(fitting, class_count, config, method)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    inputs = model_inputs(fitting, "mlp_matched")
    generator = torch.Generator().manual_seed(seed)
    history, best_state, best_key, best_epoch, stale = [], None, None, None, 0
    started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses, generation_losses, operation_losses = [], [], []
        for indices in profile_batches(fitting["metadata"], class_count, config["batch_size"], generator):
            outputs = model(*(value[indices] for value in inputs))
            loss, auxiliary = conditional_loss(
                outputs, fitting["labels"][indices], class_count, config["auxiliary_weight"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            generation_losses.append(float(auxiliary["generation_loss"].detach()))
            operation_losses.append(float(auxiliary["operation_loss"].detach()))
        scores = selection_scores(
            selection, predict_conditional(model, selection, config["batch_size"]),
            policies, class_count,
        )
        key = selection_key(scores)
        history.append({
            "epoch": epoch, "training_loss": float(np.mean(losses)),
            "generation_auxiliary_loss": float(np.mean(generation_losses)),
            "operation_auxiliary_loss": float(np.mean(operation_losses)),
            "inner_selection": scores,
        })
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if progress is not None:
            progress({
                "epoch": epoch, "best_epoch": best_epoch, "stale_epochs": stale,
                "elapsed_seconds": time.monotonic() - started,
            })
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {
        "history": history, "best_epoch": best_epoch,
        "elapsed_seconds": time.monotonic() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "objective": "multiclass_ce_plus_operation_and_masked_subtype_auxiliary",
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "policy_metadata_used_as_prediction_input": False,
        "generation_supervision_scope": "Traditional_and_LLM_Bot_only",
        "graph_input": False,
    }
