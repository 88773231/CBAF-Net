"""Transition-free correspondence candidate with no zero-input weight columns."""

import copy
import hashlib
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.encoder_objective_diagnostic import validate_populations
from src.evidence_repair import model_inputs, seed_everything
from src.policy_robust import profile_batches, selection_key, selection_scores
from src.trajectory_correspondence import (
    TrajectoryCorrespondenceModel, input_dimensions, predict,
)


CONDITIONS = ("compact_correspondence", "without_correspondence")


def prune_columns(linear, width):
    if width < 1 or width > linear.in_features:
        raise ValueError("invalid compact projection width")
    linear.weight = nn.Parameter(linear.weight.detach()[:, :width].clone())
    linear.in_features = width


class CompactCorrespondenceModel(TrajectoryCorrespondenceModel):
    def __init__(self, description_dim, text_dim, action_dim, tabular_dim,
                 hidden_dim=64, readout_dim=128, steps=5, dropout=.2,
                 condition="compact_correspondence"):
        if condition not in CONDITIONS:
            raise ValueError("unknown compact condition")
        legacy = "without_transition" if condition == CONDITIONS[0] else "without_both"
        # Keep legacy allocation/RNG order, then copy only columns used at inference.
        super().__init__(description_dim, text_dim, action_dim, tabular_dim,
                         hidden_dim, readout_dim, steps, dropout, legacy)
        self.condition = condition
        prune_columns(self.readout[0], hidden_dim * (steps + 2))
        if self.correspondence is None:
            prune_columns(self.block_encoder[0], hidden_dim * 2)

    def forward(self, description, text, action, tabular):
        description_dim, text_dim, action_dim, tabular_dim = self.dimensions
        count = description.shape[0] if description.ndim else 0
        if (count < 1 or description.shape != (count, description_dim)
                or text.shape != (count, self.steps, text_dim)
                or action.shape != (count, self.steps, action_dim)
                or tabular.shape != (count, tabular_dim)
                or any(not torch.isfinite(value).all()
                       for value in (description, text, action, tabular))):
            raise ValueError("trajectory inputs do not match the fitted schema")
        text_blocks = torch.stack([
            encoder(text[:, step]) for step, encoder in enumerate(self.text_encoders)
        ], dim=1)
        action_blocks = torch.stack([
            encoder(action[:, step]) for step, encoder in enumerate(self.action_encoders)
        ], dim=1)
        components = [text_blocks, action_blocks]
        correspondence = None
        if self.correspondence is not None:
            correspondence = self.correspondence(text_blocks, action_blocks)
            components.extend([
                correspondence["text_discrepancy"], correspondence["action_discrepancy"],
            ])
        blocks = self.block_encoder(torch.cat(components, dim=-1))
        representation = torch.cat([
            self.profile_encoder(description), self.statistics_encoder(tabular),
            blocks.flatten(1),
        ], dim=1)
        return {
            "logits": self.readout(representation), "blocks": blocks,
            "correspondence": correspondence,
        }


def shared_state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("correspondence."):
            continue
        if name == "block_encoder.0.weight":
            value = value[:, :model.hidden_dim * 2]
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_model(values, config, condition):
    return CompactCorrespondenceModel(
        *input_dimensions(values), config["block_dim"], config["readout_dim"],
        5, config["dropout"], condition,
    )


def train(fitting, selection, policies, condition, seed, config, progress=None):
    validate_populations(fitting, selection, policies, 4)
    if condition not in CONDITIONS:
        raise ValueError("unknown compact condition")
    if input_dimensions(fitting) != input_dimensions(selection):
        raise ValueError("fitting and selection feature dimensions differ")
    for key in ("max_epochs", "patience", "batch_size", "block_dim", "readout_dim"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError("positive integer optimization budget required")
    if config["batch_size"] < 4:
        raise ValueError("class-complete optimization batch required")
    if (not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0
            or not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0
            or not math.isfinite(config["dropout"]) or not 0 <= config["dropout"] < 1):
        raise ValueError("finite valid optimization configuration required")
    seed_everything(seed)
    model = make_model(fitting, config, condition)
    initialization = shared_state_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    inputs = model_inputs(fitting, "mlp_matched")
    generator = torch.Generator().manual_seed(seed)
    history, best_key, best_state, best_epoch, stale = [], None, None, None, 0
    started, updates = time.monotonic(), 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        for indices in profile_batches(fitting["metadata"], 4, config["batch_size"], generator):
            loss = F.cross_entropy(
                model(*(value[indices] for value in inputs))["logits"], fitting["labels"][indices],
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite compact objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
        scores = selection_scores(
            selection, predict(model, selection, config["batch_size"]), policies, 4,
        )
        key = selection_key(scores)
        history.append({
            "epoch": epoch, "training_loss": float(np.mean(losses)),
            "inner_selection": scores, "optimizer_updates": updates,
        })
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch, stale = key, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if progress:
            progress({"stage": "supervised", "epoch": epoch, "best_epoch": best_epoch})
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {
        "condition": condition, "best_epoch": best_epoch, "history": history,
        "shared_initialization_digest": initialization,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "inactive_serialized_weight_count": 0,
        "optimizer_updates": updates, "elapsed_seconds": time.monotonic() - started,
        "sam_or_contrast_used": False, "graph_input": False,
        "profile_policy_or_generator_metadata_as_prediction_input": False,
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "architecture_novelty_established": False, "submission_allowed": False,
    }
