"""Train/validation-only BotDMM input adapter with fold-local strict kNN."""

import copy
import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset_pipeline.export_quadbot_v3_features import build_edge_index
from src.botdmm import BotDMM
from src.data_loader import StrictOneHopNeighborCollator, validate_knn_pairs
from src.evidence_repair import seed_everything
from src.policy_robust import profile_batches, selection_key, selection_scores


METHODS = ("botdmm_timing", "botdmm_without_timing")


class DevelopmentGraphView:
    """In-memory fold view compatible with the canonical one-hop collator."""

    def __init__(self, values, graph_features, method, k=10):
        if method not in METHODS:
            raise ValueError("unknown development backbone method")
        self.metadata = values["metadata"]
        self.base_profile_ids = [str(row["base_profile_id"]) for row in self.metadata]
        self.labels = values["labels"]
        self.des_tensor = values["description"]
        self.tweets_list = list(values["text"].unbind(1))
        action = values["action"] if method == "botdmm_timing" else values["action"][:, :, :-7]
        self.amrs_list = list(action.unbind(1))
        self.num_prop = values["numerical"] if method == "botdmm_timing" else values["numerical"][:, :-7]
        self.llm_features = values["tabular"][:, 6:17]
        if self.llm_features.shape[1] != 11 or values["numerical"].shape[1] != 13:
            raise ValueError("development timing/statistics schema differs")
        if graph_features.ndim != 2 or graph_features.shape[0] != len(self.metadata):
            raise ValueError("fold graph features do not align")
        if not torch.isfinite(graph_features).all():
            raise ValueError("fold graph features must be finite")
        self.edge_pairs = build_edge_index(
            graph_features.numpy(), k=k, base_profile_ids=self.base_profile_ids
        )
        self.edge_report = validate_knn_pairs(
            self.edge_pairs, len(self), self.base_profile_ids, expected_k=k,
        )
        self.edge_indices = [self.edge_pairs] * len(self.tweets_list)
        self.graph_receipt = {
            **self.edge_report, "scope": "this_fold_only_no_other_profile_pool",
            "construction": "cosine_knn_on_original_hash_graph_features",
            "true_social_edges": False, "uses_labels_or_policy": False,
            "edge_pairs_sha256": hashlib.sha256(self.edge_pairs.numpy().tobytes()).hexdigest(),
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"_index": index}


def make_backbone(dataset, class_count, config):
    return BotDMM(
        des_size=dataset.des_tensor.shape[1], tweet_size=dataset.tweets_list[0].shape[1],
        amr_size=dataset.amrs_list[0].shape[1], num_prop_size=dataset.num_prop.shape[1],
        llm_features_size=dataset.llm_features.shape[1],
        embedding_dimension=config["embedding_dimension"],
        feature_dim=config["feature_dim"], num_temporal_steps=len(dataset.tweets_list),
        dropout=config["dropout"], temperature=0.1, alpha=0.5,
        num_classes=class_count, ablation_mode="base",
    )


def forward_seed_logits(model, batch):
    result = model(
        batch["des"], batch["tweets"], batch["amrs"], batch["num_prop"],
        batch["llm_features"], batch["edge_indices"], compute_auxiliary_losses=False,
    )
    return result["logits"][batch["seed_positions"]]


def predict_backbone(model, dataset, batch_size, k=10):
    model.eval()
    collator = StrictOneHopNeighborCollator(dataset, expected_k=k)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = collator([{"_index": index}
                              for index in range(start, min(start + batch_size, len(dataset)))])
            probabilities.append(forward_seed_logits(model, batch).softmax(1).numpy())
    return np.concatenate(probabilities)


def train_backbone(fitting, selection, selection_values, policies, class_count, seed, config,
                   progress=None):
    if set(fitting.base_profile_ids) & set(selection.base_profile_ids):
        raise ValueError("fitting and inner selection profiles overlap")
    seed_everything(seed)
    model = make_backbone(fitting, class_count, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    collator = StrictOneHopNeighborCollator(fitting, expected_k=config["knn_k"])
    generator = torch.Generator().manual_seed(seed)
    history, best_state, best_key, best_epoch, stale = [], None, None, None, 0
    started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        for indices in profile_batches(
            fitting.metadata, class_count, config["batch_size"], generator
        ):
            batch = collator([{"_index": index} for index in indices.tolist()])
            logits = forward_seed_logits(model, batch)
            loss = F.cross_entropy(logits, batch["seed_labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        probabilities = predict_backbone(
            model, selection, config["batch_size"], config["knn_k"]
        )
        scores = selection_scores(selection_values, probabilities, policies, class_count)
        key = selection_key(scores)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "inner_selection": scores})
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
        "objective": "seed_row_cross_entropy_auxiliary_losses_disabled",
        "selection_rule": "worst_known_policy_macro_f1_then_overall_macro_f1_then_mcc",
        "graph_fit": fitting.graph_receipt, "graph_inner_selection": selection.graph_receipt,
        "graph_context_training_labels_used": False,
        "policy_metadata_used_as_prediction_input": False,
    }
