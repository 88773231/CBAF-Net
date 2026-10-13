"""Training-only taxonomic fusion hypothesis, not a released paper method."""

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evidence_repair import assert_group_disjoint, model_inputs, seed_everything
from src.policy_robust import profile_batches, selection_key, selection_scores


METHODS = ("factor_residual", "without_factor_fusion", "without_interaction_residual")


def taxonomy(class_count):
    """The axes encode this controlled archive's construction, not real intent."""
    if class_count not in (3, 4):
        raise ValueError("only the fixed three/four-class taxonomy is supported")
    return (
        torch.tensor([0, 0, 1, 1][:class_count], dtype=torch.long),
        torch.tensor([0, 1, 1, 2][:class_count], dtype=torch.long),
    )


def factor_logits(generation_logits, operation_logits, class_count):
    generated, operation = taxonomy(class_count)
    if (generation_logits.ndim != 2 or operation_logits.ndim != 2
            or generation_logits.shape != (operation_logits.shape[0], 2)
            or operation_logits.shape[1] != (2 if class_count == 3 else 3)):
        raise ValueError("taxonomic head shapes differ")
    return (
        generation_logits.log_softmax(1)[:, generated.to(generation_logits.device)]
        + operation_logits.log_softmax(1)[:, operation.to(operation_logits.device)]
    )


class FactorEvidenceModel(nn.Module):
    """Step-specific towers with optional taxonomic fusion and joint residual."""

    def __init__(self, description_dim, text_dim, event_dim, class_count, steps=5,
                 hidden_dim=128, dropout=0.2, method="factor_residual"):
        super().__init__()
        if method not in METHODS:
            raise ValueError("unsupported factor evidence method")
        if min(description_dim, text_dim, event_dim, steps, hidden_dim) < 1:
            raise ValueError("positive model dimensions are required")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.steps, self.class_count, self.method = steps, class_count, method
        generation, operation = taxonomy(class_count)
        self.register_buffer("generation_targets", generation)
        self.register_buffer("operation_targets", operation)

        def encoder(dim):
            return nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
            )

        self.profile_encoder = encoder(description_dim)
        # Exported step hashes use distinct coordinate systems.
        self.text_encoders = nn.ModuleList([encoder(text_dim) for _ in range(steps)])
        self.event_encoders = nn.ModuleList([encoder(event_dim) for _ in range(steps)])
        self.content_encoder = encoder(hidden_dim * (steps + 1) + 9)
        self.behavior_encoder = encoder(hidden_dim * steps + 15)
        self.generation_head = nn.Linear(hidden_dim, 2)
        self.operation_head = nn.Linear(hidden_dim, 2 if class_count == 3 else 3)
        self.residual = (
            nn.Sequential(encoder(hidden_dim * 4), nn.Linear(hidden_dim, class_count))
            if method != "without_interaction_residual" else None
        )
        self.flat_head = (
            nn.Linear(hidden_dim * 2, class_count)
            if method == "without_factor_fusion" else None
        )

    def forward(self, description, text, action, tabular):
        if (text.ndim != 3 or action.ndim != 3 or tabular.ndim != 2
                or text.shape[1] != self.steps or action.shape[1] != self.steps
                or tabular.shape[1] != 24 or action.shape[2] < 8
                or len({description.shape[0], text.shape[0], action.shape[0],
                        tabular.shape[0]}) != 1):
            raise ValueError("fixed timing-aware modality schema differs")
        content = self.content_encoder(torch.cat([
            self.profile_encoder(description),
            *[encoder(text[:, step]) for step, encoder in enumerate(self.text_encoders)],
            tabular[:, 6:15],
        ], dim=1))
        behavior = self.behavior_encoder(torch.cat([
            *[encoder(action[:, step]) for step, encoder in enumerate(self.event_encoders)],
            tabular[:, :6], tabular[:, 15:],
        ], dim=1))
        generated, operation = self.generation_head(content), self.operation_head(behavior)
        base = (
            self.flat_head(torch.cat([content, behavior], dim=1)) if self.flat_head is not None
            else factor_logits(generated, operation, self.class_count)
        )
        residual = (
            self.residual(torch.cat([
                content, behavior, content * behavior, torch.abs(content - behavior),
            ], dim=1)) if self.residual is not None else torch.zeros_like(base)
        )
        return {
            "logits": base + residual, "generation_logits": generated,
            "operation_logits": operation, "base_logits": base,
            "residual_logits": residual,
        }


def make_factor_model(values, class_count, config, method):
    description, text, action, _ = model_inputs(values, "mlp_matched")
    return FactorEvidenceModel(
        description.shape[1], text.shape[2], action.shape[2], class_count,
        steps=text.shape[1], hidden_dim=config["hidden_dim"],
        dropout=config["dropout"], method=method,
    )


def factor_loss(outputs, labels, class_count, auxiliary_weight):
    if auxiliary_weight < 0 or not np.isfinite(auxiliary_weight):
        raise ValueError("invalid auxiliary weight")
    generation, operation = taxonomy(class_count)
    classification = F.cross_entropy(outputs["logits"], labels)
    auxiliary = 0.5 * (
        F.cross_entropy(outputs["generation_logits"], generation.to(labels.device)[labels])
        + F.cross_entropy(outputs["operation_logits"], operation.to(labels.device)[labels])
    )
    return classification + auxiliary_weight * auxiliary, auxiliary


def predict_factor(model, values, batch_size):
    model.eval()
    inputs = model_inputs(values, "mlp_matched")
    blocks = []
    with torch.no_grad():
        for start in range(0, len(values["labels"]), batch_size):
            result = model(*(value[start:start + batch_size] for value in inputs))
            blocks.append(result["logits"].softmax(1).numpy())
    return np.concatenate(blocks)


def train_factor_model(fitting, selection, policies, method, class_count, seed, config,
                       progress=None):
    assert_group_disjoint(fitting, selection)
    if {policies[row["base_profile_id"]] for row in fitting["metadata"]} != {
        policies[row["base_profile_id"]] for row in selection["metadata"]
    }:
        raise ValueError("inner selection must cover only the fitting policy set")
    seed_everything(seed)
    model = make_factor_model(fitting, class_count, config, method)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    inputs = model_inputs(fitting, "mlp_matched")
    generator = torch.Generator().manual_seed(seed)
    history, best_state, best_key, best_epoch, stale = [], None, None, None, 0
    started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses, auxiliaries = [], []
        for indices in profile_batches(
            fitting["metadata"], class_count, config["batch_size"], generator,
        ):
            result = model(*(value[indices] for value in inputs))
            loss, auxiliary = factor_loss(
                result, fitting["labels"][indices], class_count, config["auxiliary_weight"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            auxiliaries.append(float(auxiliary.detach()))
        scores = selection_scores(
            selection, predict_factor(model, selection, config["batch_size"]),
            policies, class_count,
        )
        key = selection_key(scores)
        history.append({
            "epoch": epoch, "training_loss": float(np.mean(losses)),
            "auxiliary_loss": float(np.mean(auxiliaries)), "inner_selection": scores,
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
        "objective": "multiclass_cross_entropy_plus_fixed_taxonomy_auxiliary",
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "policy_metadata_used_as_prediction_input": False,
        "class_targets_used_only_during_training": True, "graph_input": False,
    }
