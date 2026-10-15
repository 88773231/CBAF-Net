"""Within-row nonlinear history interactions; an unconfirmed development hypothesis."""

import copy
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.compact_correspondence import CompactCorrespondenceModel, shared_state_digest
from src.correspondence_dependence import fixed_history
from src.encoder_objective_diagnostic import validate_populations
from src.evidence_repair import model_inputs, seed_everything
from src.policy_robust import profile_batches, selection_key, selection_scores
from src.trajectory_correspondence import input_dimensions, predict


CONDITIONS = ("centered_history_interaction", "without_centering")


def interaction_features(text, action, centered=True):
    if (text.ndim != 3 or text.shape != action.shape or text.shape[1] != 5
            or not text.is_floating_point() or not action.is_floating_point()
            or text.device != action.device or text.dtype != action.dtype
            or not torch.isfinite(text).all() or not torch.isfinite(action).all()):
        raise ValueError("finite aligned five-block encoder outputs required")
    uniform = fixed_history(5, text.device, text.dtype)
    if centered:
        text, action = text - uniform @ text, action - uniform @ action
    return text * (uniform @ action), action * (uniform @ text)


class CenteredHistoryInteractionModel(CompactCorrespondenceModel):
    def __init__(self, description_dim, text_dim, action_dim, tabular_dim,
                 hidden_dim=64, readout_dim=128, dropout=.2,
                 condition="centered_history_interaction"):
        if condition not in CONDITIONS:
            raise ValueError("unknown interaction hypothesis condition")
        # Preserve the previous shared allocation order, not its attention mechanism.
        super().__init__(description_dim, text_dim, action_dim, tabular_dim,
                         hidden_dim, readout_dim, 5, dropout, "compact_correspondence")
        self.correspondence = None
        self.condition = condition

    def forward(self, description, text, action, tabular):
        count = description.shape[0] if description.ndim else 0
        description_dim, text_dim, action_dim, tabular_dim = self.dimensions
        if (count < 1 or description.shape != (count, description_dim)
                or text.shape != (count, 5, text_dim) or action.shape != (count, 5, action_dim)
                or tabular.shape != (count, tabular_dim)
                or any(not torch.isfinite(value).all()
                       for value in (description, text, action, tabular))):
            raise ValueError("fixed finite trajectory input contract required")
        text_blocks = torch.stack([
            encoder(text[:, step]) for step, encoder in enumerate(self.text_encoders)], 1)
        action_blocks = torch.stack([
            encoder(action[:, step]) for step, encoder in enumerate(self.action_encoders)], 1)
        interactions = interaction_features(
            text_blocks, action_blocks, self.condition == CONDITIONS[0])
        blocks = self.block_encoder(torch.cat([text_blocks, action_blocks, *interactions], -1))
        representation = torch.cat([
            self.profile_encoder(description), self.statistics_encoder(tabular), blocks.flatten(1),
        ], 1)
        return {"logits": self.readout(representation), "blocks": blocks,
                "interaction_features": interactions}


def make_model(values, config, condition):
    return CenteredHistoryInteractionModel(
        *input_dimensions(values), config["block_dim"], config["readout_dim"],
        config["dropout"], condition)


def train(fitting, selection, policies, condition, seed, config, progress=None):
    validate_populations(fitting, selection, policies, 4)
    if condition not in CONDITIONS:
        raise ValueError("unknown interaction condition")
    if input_dimensions(fitting) != input_dimensions(selection):
        raise ValueError("fitting and selection dimensions differ")
    for name in ("block_dim", "readout_dim", "batch_size", "max_epochs", "patience"):
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] < 1:
            raise ValueError("positive integer fitting budget required")
    if (config["batch_size"] < 4 or not math.isfinite(config["learning_rate"])
            or config["learning_rate"] <= 0 or not math.isfinite(config["weight_decay"])
            or config["weight_decay"] < 0 or not math.isfinite(config["dropout"])
            or not 0 <= config["dropout"] < 1):
        raise ValueError("finite valid optimizer configuration required")
    seed_everything(seed)
    model = make_model(fitting, config, condition)
    initialization = shared_state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    inputs = model_inputs(fitting, "mlp_matched")
    generator = torch.Generator().manual_seed(seed)
    history, best_key, best_state, best_epoch, stale = [], None, None, None, 0
    started, updates = time.monotonic(), 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        for indices in profile_batches(fitting["metadata"], 4, config["batch_size"], generator):
            loss = F.cross_entropy(
                model(*(value[indices] for value in inputs))["logits"], fitting["labels"][indices])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite interaction fitting loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
        scores = selection_scores(selection, predict(model, selection, config["batch_size"]), policies, 4)
        key = selection_key(scores)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "inner_selection": scores, "optimizer_updates": updates})
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch, stale = key, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if progress:
            progress({"epoch": epoch, "best_epoch": best_epoch})
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {
        "condition": condition, "new_fit": True, "best_epoch": best_epoch, "history": history,
        "shared_initialization_digest": initialization,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "inactive_serialized_weight_count": 0, "optimizer_updates": updates,
        "elapsed_seconds": time.monotonic() - started, "learned_attention": False,
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "profile_policy_or_generator_metadata_as_prediction_input": False,
        "algorithmic_novelty_established": False, "submission_allowed": False,
    }
