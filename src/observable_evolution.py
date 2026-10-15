"""Development candidate: causal cue evolution and training-only distillation."""

import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset_pipeline.observable_cues import FEATURE_DIMENSIONS, STEPS
from src.encoder_objective_diagnostic import validate_populations
from src.evidence_repair import seed_everything
from src.policy_robust import profile_batches, selection_key, selection_scores


CUES = ("text", "behavior", "style", "timing")
CONDITIONS = ("evolution_distillation", "without_evolution",
              "without_distillation", "without_both", "same_input_mlp")
INPUTS = ("description", "numeric", *CUES)


class CausalCueEvolution(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.convolution = nn.Conv1d(width, width, kernel_size=2)
        self.norm = nn.LayerNorm(width)

    def forward(self, states):
        if states.ndim != 3:
            raise ValueError("cue states must have three dimensions")
        history = F.pad(states.transpose(1, 2), (1, 0))
        return self.norm(states + F.gelu(self.convolution(history).transpose(1, 2)))


class ObservableEvolutionModel(nn.Module):
    def __init__(self, width=32, readout=128, dropout=.2, condition=CONDITIONS[0]):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError("unknown observable candidate condition")
        self.condition = condition
        if condition == "same_input_mlp":
            size = FEATURE_DIMENSIONS["description"] + FEATURE_DIMENSIONS["numeric"]
            size += STEPS * sum(FEATURE_DIMENSIONS[cue] for cue in CUES)
            self.flat = nn.Sequential(
                nn.Linear(size, readout), nn.ReLU(), nn.Linear(readout, readout),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(readout, 4),
            )
            return
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(FEATURE_DIMENSIONS[name], width),
                                nn.LayerNorm(width), nn.ReLU(), nn.Dropout(dropout))
            for name in INPUTS
        })
        self.evolution = nn.ModuleDict({name: CausalCueEvolution(width) for name in CUES})
        self.cue_heads = nn.ModuleDict({name: nn.Linear(STEPS * width, 4) for name in CUES})
        self.readout = nn.Sequential(
            nn.Linear(width * (2 + len(CUES) * STEPS), readout), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(readout, 4),
        )
        # Allocate before removal to preserve active shared initialization.
        if condition in ("without_evolution", "without_both"):
            self.evolution = None

    def forward(self, description, numeric, text, behavior, style, timing):
        values = dict(zip(INPUTS, (description, numeric, text, behavior, style, timing)))
        count = description.shape[0]
        for name, value in values.items():
            shape = (count, FEATURE_DIMENSIONS[name])
            if name in CUES:
                shape = (count, STEPS, FEATURE_DIMENSIONS[name])
            if count < 1 or value.shape != shape or not torch.isfinite(value).all():
                raise ValueError("observation inputs differ from the fixed schema")
        if self.condition == "same_input_mlp":
            flat = torch.cat([values[name].flatten(1) for name in INPUTS], dim=1)
            return {"logits": self.flat(flat), "cue_logits": {}, "states": {}}
        states = {name: self.encoders[name](values[name]) for name in CUES}
        if self.evolution is not None:
            states = {name: self.evolution[name](value) for name, value in states.items()}
        representation = torch.cat([
            self.encoders["description"](description), self.encoders["numeric"](numeric),
            *[states[name].flatten(1) for name in CUES],
        ], dim=1)
        return {
            "logits": self.readout(representation),
            "cue_logits": {name: self.cue_heads[name](states[name].flatten(1)) for name in CUES},
            "states": states,
        }


def cue_distillation(student, teacher, temperature):
    if temperature <= 0 or student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("aligned logits and positive temperature required")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("finite distillation logits required")
    return F.kl_div((student / temperature).log_softmax(1),
                    (teacher.detach() / temperature).softmax(1), reduction="batchmean") * temperature ** 2


def objective(output, labels, condition, epoch, config):
    loss = F.cross_entropy(output["logits"], labels)
    if condition == "same_input_mlp":
        return loss
    auxiliary = torch.stack([F.cross_entropy(logits, labels) for logits in output["cue_logits"].values()]).mean()
    loss = loss + config["cue_supervision_weight"] * auxiliary
    if condition not in ("without_distillation", "without_both") and epoch > config["warmup_epochs"]:
        alignment = torch.stack([
            cue_distillation(logits, output["logits"], config["temperature"])
            for logits in output["cue_logits"].values()
        ]).mean()
        loss = loss + config["distillation_weight"] * alignment
    return loss


def shared_digest(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("evolution."):
            continue
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def make_model(config, condition):
    return ObservableEvolutionModel(config["width"], config["readout"], config["dropout"], condition)


def predict(model, values, batch_size):
    model.eval()
    with torch.no_grad():
        return torch.cat([
            model(*(values[name][indices] for name in INPUTS))["logits"].softmax(1)
            for indices in torch.arange(len(values["labels"])).split(batch_size)
        ]).numpy()


def train(fitting, selection, policies, condition, config, progress=None):
    validate_populations(fitting, selection, policies, 4)
    seed_everything(config["seed"])
    model = make_model(config, condition)
    digest = shared_digest(model)
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
            loss = objective(output, fitting["labels"][indices], condition, epoch, config)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite observation objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        scores = selection_scores(selection, predict(model, selection, config["batch_size"]), policies, 4)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "inner_selection": scores})
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
        "best_epoch": best_epoch, "history": history,
        "elapsed_seconds": time.monotonic() - started, "shared_initialization_digest": digest,
        "training_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "cue_heads_used_only_in_training_objective": condition != "same_input_mlp",
        "fused_teacher_stop_gradient": True, "external_teacher_or_model_used": False,
        "algorithmic_novelty_established": False, "submission_allowed": False,
    }
