"""Dimension-matched representation controls, not a new architecture claim."""

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset_pipeline.frozen_semantic import DIMENSIONS
from dataset_pipeline.observable_cues import STEPS
from src.encoder_objective_diagnostic import state_digest, validate_populations
from src.evidence_repair import seed_everything
from src.observable_evolution import CUES, INPUTS
from src.policy_robust import profile_batches, selection_key, selection_scores


CONDITIONS = ("hash_mlp", "semantic_mlp", "hash_backbone", "semantic_backbone")


class RepresentationControl(nn.Module):
    def __init__(self, architecture, width=32, readout=128, dropout=.2):
        super().__init__()
        if architecture not in ("mlp", "backbone"):
            raise ValueError("unknown representation control")
        self.architecture = architecture
        if architecture == "mlp":
            size = DIMENSIONS["description"] + DIMENSIONS["numeric"]
            size += STEPS * sum(DIMENSIONS[name] for name in CUES)
            self.flat = nn.Sequential(
                nn.Linear(size, readout), nn.ReLU(), nn.Linear(readout, readout),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(readout, 4),
            )
        else:
            self.encoders = nn.ModuleDict({
                name: nn.Sequential(nn.Linear(DIMENSIONS[name], width), nn.LayerNorm(width),
                                    nn.ReLU(), nn.Dropout(dropout))
                for name in INPUTS
            })
            self.readout = nn.Sequential(
                nn.Linear(width * (2 + len(CUES) * STEPS), readout), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(readout, 4),
            )

    def forward(self, *inputs):
        if len(inputs) != len(INPUTS):
            raise ValueError("complete observation tuple required")
        values = dict(zip(INPUTS, inputs))
        count = inputs[0].shape[0]
        for name, value in values.items():
            shape = (count, STEPS, DIMENSIONS[name]) if name in CUES else (count, DIMENSIONS[name])
            if count < 1 or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError("dimension-matched observations required")
        if self.architecture == "mlp":
            return self.flat(torch.cat([values[name].flatten(1) for name in INPUTS], 1))
        return self.readout(torch.cat([
            self.encoders[name](values[name]).flatten(1) for name in INPUTS
        ], 1))


def make_model(configuration, architecture):
    return RepresentationControl(
        architecture, configuration["width"], configuration["readout"], configuration["dropout"],
    )


def predict(model, values, batch_size):
    model.eval()
    with torch.inference_mode():
        return torch.cat([
            model(*(values[name][indices] for name in INPUTS)).softmax(1)
            for indices in torch.arange(len(values["labels"])).split(batch_size)
        ]).numpy()


def train(fitting, selection, policies, architecture, config, seed=42, progress=None):
    validate_populations(fitting, selection, policies, 4)
    seed_everything(seed)
    model = make_model(config, architecture)
    digest = state_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    generator = torch.Generator().manual_seed(seed)
    best_key, best_state, best_epoch, stale = None, None, None, 0
    history, started = [], time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        for indices in profile_batches(fitting["metadata"], 4, config["batch_size"], generator):
            loss = F.cross_entropy(
                model(*(fitting[name][indices] for name in INPUTS)), fitting["labels"][indices],
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite representation-control loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
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
            progress({"event": "epoch_complete", "epoch": epoch, "best_epoch": best_epoch})
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch, "history": history, "initialization_digest": digest,
        "elapsed_seconds": time.monotonic() - started,
        "training_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "objective": "cross_entropy_only", "encoder_finetuned": False,
        "new_module_or_novelty_claim": False, "submission_allowed": False,
    }
