"""Development-only class-conditioned residuals on the observable backbone."""

import copy
import hashlib
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset_pipeline.observable_cues import STEPS
from src.encoder_objective_diagnostic import validate_populations
from src.evidence_repair import seed_everything
from src.observable_evolution import CUES, INPUTS, ObservableEvolutionModel, predict
from src.policy_robust import profile_batches, selection_key, selection_scores


CONDITIONS = ("class_routing_pair_margin", "without_pair_margin",
              "without_class_routing", "scalar_routing_control")


class ClassConditionedCueModel(nn.Module):
    def __init__(self, width=32, readout=128, dropout=.2, condition=CONDITIONS[0]):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError("unknown class-conditioned cue condition")
        self.condition = condition
        self.backbone = ObservableEvolutionModel(width, readout, dropout, "without_both")
        self.projections = nn.ModuleDict({
            cue: nn.Sequential(nn.Linear(STEPS * width, width), nn.LayerNorm(width), nn.ReLU())
            for cue in CUES
        })
        self.routing = nn.Sequential(
            nn.Linear(len(CUES) * width + 4, width), nn.ReLU(),
            nn.Linear(width, len(CUES) * 4),
        )
        self.residual_log_scale = nn.Parameter(torch.tensor(math.log(.1 / .9)))
        if condition == "without_class_routing":
            self.projections = None
            self.routing = None
            self.register_parameter("residual_log_scale", None)

    def forward(self, description, numeric, text, behavior, style, timing):
        output = self.backbone(description, numeric, text, behavior, style, timing)
        if self.routing is None:
            return {**output, "routing_weights": None, "residual_scale": None}
        # Detach gate inputs, not expert scores: gate fitting must not rewrite cue states.
        context = torch.cat([
            *[self.projections[cue](output["states"][cue].flatten(1).detach()) for cue in CUES],
            output["logits"].detach(),
        ], dim=1)
        scores = self.routing(context).reshape(-1, len(CUES), 4)
        if self.condition == "scalar_routing_control":
            scores = scores.mean(-1, keepdim=True).expand_as(scores)
        weights = scores.softmax(1)
        cue_logits = torch.stack([output["cue_logits"][cue] for cue in CUES], dim=1)
        centered = cue_logits - cue_logits.mean(-1, keepdim=True)
        scale = self.residual_log_scale.sigmoid()
        return {
            **output, "context_logits": output["logits"],
            "logits": output["logits"] + scale * (weights * centered).sum(1),
            "routing_weights": weights, "residual_scale": scale,
        }


def paired_profile_margin(logits, labels, profile_ids, margin):
    if (logits.ndim != 2 or logits.shape[1] != 4 or logits.shape[0] != len(labels)
            or len(profile_ids) != len(labels) or not math.isfinite(margin) or margin < 0
            or not torch.isfinite(logits).all()):
        raise ValueError("finite aligned four-class paired logits required")
    groups = {}
    for index, (profile, label) in enumerate(zip(profile_ids, labels.tolist())):
        if label in groups.setdefault(profile, {}):
            raise ValueError("duplicate paired profile class")
        groups[profile][label] = index
    if not groups or any(set(group) != set(range(4)) for group in groups.values()):
        raise ValueError("class-complete paired profiles required")
    ordered = torch.tensor([[group[label] for label in range(4)] for group in groups.values()],
                           dtype=torch.long, device=logits.device)
    left, right = torch.triu_indices(4, 4, offset=1, device=logits.device)
    paired = logits[ordered]
    gap = paired[:, left, left] - paired[:, left, right] - paired[:, right, left] + paired[:, right, right]
    return F.softplus(margin - gap).mean()


def objective(output, labels, profile_ids, condition, config):
    if condition not in CONDITIONS:
        raise ValueError("unknown paired objective condition")
    loss = F.cross_entropy(output["logits"], labels)
    auxiliary = torch.stack([
        F.cross_entropy(output["cue_logits"][cue], labels) for cue in CUES
    ]).mean()
    loss = loss + config["cue_supervision_weight"] * auxiliary
    if condition != "without_pair_margin":
        loss = loss + config["pair_margin_weight"] * paired_profile_margin(
            output["logits"], labels, profile_ids, config["pair_margin"])
    return loss


def shared_digest(model):
    digest = hashlib.sha256()
    for name, value in model.backbone.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_model(config, condition):
    return ClassConditionedCueModel(config["width"], config["readout"], config["dropout"], condition)


def train(fitting, selection, policies, condition, config, progress=None):
    validate_populations(fitting, selection, policies, 4)
    seed_everything(config["seed"])
    model = make_model(config, condition)
    initialization = shared_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"])
    best_key, best_state, best_epoch, stale = None, None, None, 0
    history, started = [], time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        for indices in profile_batches(fitting["metadata"], 4, config["batch_size"], generator):
            output = model(*(fitting[name][indices] for name in INPUTS))
            profiles = [fitting["metadata"][index]["base_profile_id"] for index in indices.tolist()]
            loss = objective(output, fitting["labels"][indices], profiles, condition, config)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite class-conditioned cue objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        scores = selection_scores(selection, predict(model, selection, config["batch_size"]), policies, 4)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "inner_selection": scores})
        key = selection_key(scores)
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
        "best_epoch": best_epoch, "history": history, "elapsed_seconds": time.monotonic() - started,
        "shared_backbone_initialization_digest": initialization,
        "training_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "profile_ids_and_labels_are_not_forward_inputs": True,
        "pair_margin_is_training_objective_not_architectural_module": True,
        "routing_weights_are_not_calibrated_reliabilities_or_causal_explanations": True,
        "algorithmic_novelty_established": False, "submission_allowed": False,
    }
