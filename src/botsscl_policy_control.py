"""Matched-input BotSSCL mechanism control, not an original-paper reproduction."""

import copy
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.botsscl_adapter import CorruptionAdapter, corrupt_features, nt_xent
from src.evidence_repair import assert_group_disjoint, flattened_inputs, seed_everything
from src.policy_robust import profile_batches, selection_key, selection_scores


METHODS = ("botsscl_flattened", "botsscl_supervised_only")
SOURCE_COMMIT = "ed3b17588fc2c271947aade266bb1357fc835f61"


def predict_control(model, values, batch_size):
    if batch_size < 1:
        raise ValueError("positive prediction batch size required")
    inputs = torch.from_numpy(flattened_inputs(values))
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch).softmax(1)
                          for batch in inputs.split(batch_size)]).numpy()


def train_control(fitting, selection, policies, method, class_count, seed, config,
                  pretraining, progress=None):
    if method not in METHODS or class_count not in (3, 4):
        raise ValueError("unknown matched-input BotSSCL condition or taxonomy")
    assert_group_disjoint(fitting, selection)
    names = {policies[row["base_profile_id"]] for row in fitting["metadata"]}
    if len(names) != 2 or names != {
        policies[row["base_profile_id"]] for row in selection["metadata"]
    }:
        raise ValueError("inner selection must cover only both fitting policies")
    for values in (fitting, selection):
        if values["labels"].tolist() != [int(row["label"]) for row in values["metadata"]]:
            raise ValueError("ordered labels differ from metadata")
        if set(values["labels"].tolist()) != set(range(class_count)):
            raise ValueError("each population must contain the fixed taxonomy")
    for key in ("max_epochs", "patience", "mlp_hidden_dim"):
        if config[key] < 1:
            raise ValueError("positive supervised budget and dimensions required")
    if config["batch_size"] < class_count or pretraining["epochs"] < 1:
        raise ValueError("positive pretraining and class-complete batch required")
    if (not math.isfinite(pretraining["corruption_rate"])
            or not 0 < pretraining["corruption_rate"] < 1
            or not math.isfinite(pretraining["temperature"])
            or pretraining["temperature"] <= 0):
        raise ValueError("invalid corruption rate or temperature")
    train_x = torch.from_numpy(flattened_inputs(fitting))
    if not torch.isfinite(train_x).all() or train_x.ndim != 2:
        raise ValueError("finite flattened fitting inputs required")
    # Validate complete profiles before allocating a model or taking any step.
    list(profile_batches(fitting["metadata"], class_count, config["batch_size"],
                         torch.Generator().manual_seed(seed)))
    list(profile_batches(selection["metadata"], class_count, config["batch_size"],
                         torch.Generator().manual_seed(seed)))
    seed_everything(seed)
    model = CorruptionAdapter(train_x.shape[1], class_count, config["mlp_hidden_dim"])
    started = time.monotonic()
    pretraining_history, pretraining_updates, pretraining_forward_evaluations = [], 0, 0
    if method == "botsscl_flattened":
        optimizer = torch.optim.Adam(
            [*model.encoder.parameters(), *model.projection.parameters()],
            lr=config["learning_rate"],
        )
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(1, pretraining["epochs"] + 1):
            model.train()
            losses = []
            for indices in torch.randperm(len(train_x), generator=generator).split(config["batch_size"]):
                if len(indices) < 2:
                    continue
                anchor = train_x[indices]
                donors = train_x[torch.randint(len(train_x), (len(indices),), generator=generator)]
                positive = corrupt_features(anchor, donors, pretraining["corruption_rate"], generator)
                loss = nt_xent(
                    model.projection(model.encoder(anchor)),
                    model.projection(model.encoder(positive)), pretraining["temperature"],
                )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite pretraining objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                pretraining_updates += 1
                pretraining_forward_evaluations += 2
                losses.append(float(loss.detach()))
            if not losses:
                raise ValueError("pretraining population cannot form a paired batch")
            pretraining_history.append({"epoch": epoch, "loss": float(np.mean(losses))})
            if progress:
                progress({"stage": "pretraining", "epoch": epoch})
    # Both conditions use the same supervised batch schedule after pretraining.
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(
        [*model.encoder.parameters(), *model.classifier.parameters()],
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    history, best_key, best_state, best_epoch, stale = [], None, None, None, 0
    supervised_updates = 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        for indices in profile_batches(fitting["metadata"], class_count, config["batch_size"], generator):
            loss = F.cross_entropy(model(train_x[indices]), fitting["labels"][indices])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite supervised objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*model.encoder.parameters(), *model.classifier.parameters()], 1.0,
            )
            optimizer.step()
            supervised_updates += 1
            losses.append(float(loss.detach()))
        scores = selection_scores(
            selection, predict_control(model, selection, config["batch_size"]), policies, class_count,
        )
        key = selection_key(scores)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "inner_selection": scores})
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
        "method": method, "best_epoch": best_epoch, "history": history,
        "pretraining_history": pretraining_history,
        "pretraining_optimizer_updates": pretraining_updates,
        "pretraining_gradient_evaluations": pretraining_updates,
        "pretraining_encoder_forward_evaluations": pretraining_forward_evaluations,
        "supervised_optimizer_updates": supervised_updates,
        "supervised_gradient_evaluations": supervised_updates,
        "elapsed_seconds": time.monotonic() - started,
        "training_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "inference_parameter_count": sum(parameter.numel() for module in (
            model.encoder, model.classifier,
        ) for parameter in module.parameters()),
        "input_dimension": train_x.shape[1],
        "input_aggregation": "complete_flattened_history_not_temporal_mean",
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "donor_population": "fitting_only" if pretraining_updates else None,
        "policy_or_profile_metadata_used_as_prediction_input": False,
        "graph_input": False, "source_commit": SOURCE_COMMIT,
        "adaptation_status": "matched_input_mechanism_control_not_faithful_reproduction",
    }
