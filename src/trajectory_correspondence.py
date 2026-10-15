"""Graph-free block correspondence hypothesis, not a released paper method."""

import copy
import hashlib
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evidence_repair import model_inputs, seed_everything
from src.encoder_objective_diagnostic import validate_populations
from src.policy_robust import profile_batches, selection_key, selection_scores


CONDITIONS = (
    "correspondence_transition", "without_correspondence",
    "without_transition", "without_both",
)


class HistoricalCorrespondence(nn.Module):
    """Align each block to current/past opposite-modality blocks within one row."""

    def __init__(self, hidden_dim, steps):
        super().__init__()
        if hidden_dim < 1 or steps < 2:
            raise ValueError("positive width and at least two blocks required")
        self.hidden_dim, self.steps = hidden_dim, steps
        self.text_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.action_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.action_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.text_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.text_lag = nn.Parameter(torch.zeros(steps))
        self.action_lag = nn.Parameter(torch.zeros(steps))
        lag = torch.arange(steps)[:, None] - torch.arange(steps)[None, :]
        self.register_buffer("valid_history", lag >= 0)
        self.register_buffer("lag_index", lag.clamp_min(0))

    def forward(self, text, action):
        if (text.ndim != 3 or text.shape != action.shape
                or text.shape[1:] != (self.steps, self.hidden_dim)
                or not torch.isfinite(text).all() or not torch.isfinite(action).all()):
            raise ValueError("aligned finite block representations required")
        text_scores = self.text_query(text) @ self.action_key(action).transpose(1, 2)
        action_scores = self.action_query(action) @ self.text_key(text).transpose(1, 2)
        text_weights = (
            text_scores / math.sqrt(self.hidden_dim) + self.text_lag[self.lag_index]
        ).masked_fill(~self.valid_history, -torch.inf).softmax(-1)
        action_weights = (
            action_scores / math.sqrt(self.hidden_dim) + self.action_lag[self.lag_index]
        ).masked_fill(~self.valid_history, -torch.inf).softmax(-1)
        return {
            "text_discrepancy": text - text_weights @ action,
            "action_discrepancy": action - action_weights @ text,
            "text_to_action": text_weights, "action_to_text": action_weights,
        }


class AdjacentTransition(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("positive transition width required")
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

    def forward(self, blocks):
        if (blocks.ndim != 3 or blocks.shape[1] < 2
                or blocks.shape[2] != self.hidden_dim
                or not torch.isfinite(blocks).all()):
            raise ValueError("finite, width-matched ordered blocks required")
        before, after = blocks[:, :-1], blocks[:, 1:]
        return self.network(torch.cat([
            before, after, after - before, before * after,
        ], dim=-1)).mean(1)


class TrajectoryCorrespondenceModel(nn.Module):
    """Aligned hash-coordinate towers with optional lag/transition paths."""

    def __init__(self, description_dim, text_dim, action_dim, tabular_dim,
                 hidden_dim=64, readout_dim=128, steps=5, dropout=.2,
                 condition="correspondence_transition"):
        super().__init__()
        dimensions = (description_dim, text_dim, action_dim, hidden_dim, readout_dim)
        if (condition not in CONDITIONS
                or any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 1
                       for dim in dimensions)
                or tabular_dim != 24 or steps != 5 or not 0 <= dropout < 1):
            raise ValueError("fixed five-block four-class schema required")
        self.condition, self.steps, self.hidden_dim = condition, steps, hidden_dim
        self.dimensions = description_dim, text_dim, action_dim, tabular_dim

        def encoder(dim):
            return nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            )

        self.profile_encoder = encoder(description_dim)
        self.statistics_encoder = encoder(tabular_dim)
        # Each export block has a different hash namespace; never share its input projection.
        self.text_encoders = nn.ModuleList([encoder(text_dim) for _ in range(steps)])
        self.action_encoders = nn.ModuleList([encoder(action_dim) for _ in range(steps)])
        self.correspondence = HistoricalCorrespondence(hidden_dim, steps)
        self.block_encoder = encoder(hidden_dim * 4)
        self.transition = AdjacentTransition(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * (steps + 3), readout_dim), nn.LayerNorm(readout_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(readout_dim, 4),
        )
        # Allocate before removal so shared parameters have identical seeded initialization.
        if condition in ("without_correspondence", "without_both"):
            self.correspondence = None
        if condition in ("without_transition", "without_both"):
            self.transition = None

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
        correspondence = (
            self.correspondence(text_blocks, action_blocks)
            if self.correspondence is not None else None
        )
        discrepancies = (
            [correspondence["text_discrepancy"], correspondence["action_discrepancy"]]
            if correspondence is not None else [torch.zeros_like(text_blocks), torch.zeros_like(action_blocks)]
        )
        blocks = self.block_encoder(torch.cat([text_blocks, action_blocks, *discrepancies], dim=-1))
        transition = (
            self.transition(blocks) if self.transition is not None
            else torch.zeros_like(blocks[:, 0])
        )
        representation = torch.cat([
            self.profile_encoder(description), self.statistics_encoder(tabular),
            blocks.flatten(1), transition,
        ], dim=1)
        return {
            "logits": self.readout(representation), "blocks": blocks,
            "transition_representation": transition,
            "correspondence": correspondence,
        }


def input_dimensions(values):
    inputs = model_inputs(values, "mlp_matched")
    description, text, action, tabular = inputs
    count = len(values["labels"])
    if (count < 1 or description.ndim != 2 or text.ndim != 3 or action.ndim != 3
            or tabular.shape != (count, 24) or text.shape[1] != 5 or action.shape[1] != 5
            or min(description.shape[1], text.shape[2], action.shape[2]) < 1
            or any(value.shape[0] != count or not value.is_floating_point()
                   or not torch.isfinite(value).all() for value in inputs)):
        raise ValueError("finite class-aligned five-block feature schema required")
    return description.shape[1], text.shape[2], action.shape[2], tabular.shape[1]


def shared_state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(("correspondence.", "transition.")):
            continue
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_model(values, config, condition):
    dimensions = input_dimensions(values)
    return TrajectoryCorrespondenceModel(
        *dimensions, config["block_dim"], config["readout_dim"], 5, config["dropout"], condition,
    )


def predict(model, values, batch_size):
    if batch_size < 1:
        raise ValueError("positive prediction batch size required")
    model.eval()
    inputs, blocks = model_inputs(values, "mlp_matched"), []
    with torch.no_grad():
        for start in range(0, len(values["labels"]), batch_size):
            output = model(*(value[start:start + batch_size] for value in inputs))
            blocks.append(output["logits"].softmax(1).numpy())
    return np.concatenate(blocks)


def train(fitting, selection, policies, condition, seed, config, progress=None):
    validate_populations(fitting, selection, policies, 4)
    if condition not in CONDITIONS:
        raise ValueError("unknown architecture condition")
    if input_dimensions(fitting) != input_dimensions(selection):
        raise ValueError("fitting and selection feature dimensions differ")
    for name in ("max_epochs", "patience", "batch_size", "block_dim", "readout_dim"):
        if (not isinstance(config[name], int) or isinstance(config[name], bool)
                or config[name] < 1):
            raise ValueError("positive integer optimization budget required")
    if config["batch_size"] < 4:
        raise ValueError("positive optimization budget required")
    if (not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0
            or not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0
            or not math.isfinite(config["dropout"]) or not 0 <= config["dropout"] < 1):
        raise ValueError("finite valid optimization configuration required")
    seed_everything(seed)
    model = make_model(fitting, config, condition)
    initialization_digest = shared_state_digest(model)
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
            output = model(*(value[indices] for value in inputs))
            loss = F.cross_entropy(output["logits"], fitting["labels"][indices])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite trajectory objective")
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
            progress({"epoch": epoch, "best_epoch": best_epoch, "stale_epochs": stale})
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {
        "condition": condition, "best_epoch": best_epoch, "history": history,
        "shared_initialization_digest": initialization_digest,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "inactive_serialized_weight_count": (
            (2 * config["block_dim"] ** 2 if model.correspondence is None else 0)
            + (config["readout_dim"] * config["block_dim"] if model.transition is None else 0)
        ),
        "optimizer_updates": updates, "elapsed_seconds": time.monotonic() - started,
        "sam_or_contrast_used": False, "graph_input": False,
        "profile_policy_or_generator_metadata_as_prediction_input": False,
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "architecture_novelty_established": False, "submission_allowed": False,
    }
