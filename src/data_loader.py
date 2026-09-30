import hashlib
import json
import logging
import os

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


TABULAR_PREPROCESSING_VERSION = "train-column-zscore-v1"
TABULAR_STD_EPS = 1e-8
TABULAR_CLAMP = 5.0


def validate_knn_pairs(
    edge_pairs,
    num_nodes,
    base_profile_ids=None,
    expected_k=10,
    require_strict=True,
):
    """Validate split-local ``(target, neighbor)`` kNN selection pairs.

    Exported kNN files record the row whose neighborhood was queried first and
    its selected neighbor second. They are selection pairs, not message-flow
    edges. ``StrictOneHopNeighborCollator`` reverses them to
    ``neighbor -> target`` immediately before graph attention.
    """
    if edge_pairs.ndim != 2 or edge_pairs.shape[0] != 2:
        raise ValueError(f"edge pairs must have shape [2, E], got {tuple(edge_pairs.shape)}")
    if edge_pairs.dtype != torch.long:
        raise TypeError(f"edge pairs must use torch.long, got {edge_pairs.dtype}")
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if edge_pairs.numel() == 0:
        if expected_k not in (None, 0):
            raise ValueError(f"empty edge pairs cannot satisfy expected_k={expected_k}")
        return {"edge_count": 0, "min_degree": 0, "max_degree": 0}

    targets, neighbors = edge_pairs
    invalid = (targets < 0) | (neighbors < 0) | (targets >= num_nodes) | (neighbors >= num_nodes)
    if invalid.any():
        bad = int(invalid.nonzero(as_tuple=False)[0].item())
        raise ValueError(
            "edge pairs are not split-local: "
            f"pair {bad} is ({int(targets[bad])}, {int(neighbors[bad])}) for {num_nodes} rows"
        )
    if (targets == neighbors).any():
        raise ValueError("kNN edge pairs contain self-neighbors")

    encoded = targets * num_nodes + neighbors
    if torch.unique(encoded).numel() != encoded.numel():
        raise ValueError("kNN edge pairs contain duplicate target/neighbor pairs")

    degree = torch.bincount(targets, minlength=num_nodes)
    if expected_k is not None and not torch.all(degree == int(expected_k)):
        values = torch.unique(degree).tolist()
        raise ValueError(f"target degrees {values} differ from expected_k={expected_k}")

    same_profile = 0
    if require_strict:
        if base_profile_ids is None or len(base_profile_ids) != num_nodes:
            raise ValueError("strict kNN validation requires one base_profile_id per row")
        same_profile = sum(
            base_profile_ids[int(target)] == base_profile_ids[int(neighbor)]
            for target, neighbor in zip(targets.tolist(), neighbors.tolist())
        )
        if same_profile:
            raise ValueError(
                f"strict kNN edge pairs contain {same_profile} same-profile connections"
            )

    return {
        "edge_count": int(edge_pairs.shape[1]),
        "min_degree": int(degree.min().item()),
        "max_degree": int(degree.max().item()),
        "same_profile_edge_count": int(same_profile),
    }


def knn_pairs_to_message_edges(edge_pairs):
    """Convert exported ``(target, neighbor)`` pairs to ``neighbor -> target``."""
    return torch.stack([edge_pairs[1], edge_pairs[0]], dim=0)


def preprocess_tensor(tensor, replace_nan=True, replace_inf=True, normalize=True, clamp=True):
    if replace_nan and torch.isnan(tensor).any():
        tensor = torch.nan_to_num(tensor, nan=0.0)
    if replace_inf and torch.isinf(tensor).any():
        tensor = torch.nan_to_num(tensor, posinf=5.0, neginf=-5.0)
    if normalize and tensor.dim() > 1:
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True) + 1e-8
        tensor = (tensor - mean) / std
    if clamp:
        tensor = torch.clamp(tensor, -5.0, 5.0)
    if torch.isnan(tensor).any():
        tensor = torch.nan_to_num(tensor, nan=0.0)
    return tensor


def _clean_tabular_tensor(tensor):
    """Return a finite floating-point copy without changing feature scale."""
    if tensor.dim() != 2:
        raise ValueError(f"tabular tensors must be two-dimensional, got {tuple(tensor.shape)}")
    tensor = tensor.detach().clone().float()
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def fit_column_standardizer(train_tensor, eps=TABULAR_STD_EPS):
    """Fit per-column statistics using the training split only.

    Zero-variance columns retain their exported positions for schema and model
    compatibility, but are mapped to zero so that they cannot act as evidence.
    """
    train_tensor = _clean_tabular_tensor(train_tensor)
    if train_tensor.shape[0] == 0:
        raise ValueError("cannot fit tabular preprocessing on an empty training tensor")
    mean = train_tensor.mean(dim=0)
    std = train_tensor.std(dim=0, unbiased=False)
    variable_mask = std > float(eps)
    safe_std = torch.where(variable_mask, std, torch.ones_like(std))
    return {
        "mean": mean,
        "std": safe_std,
        "observed_std": std,
        "variable_mask": variable_mask,
    }


def apply_column_standardizer(tensor, standardizer, clamp=TABULAR_CLAMP):
    """Apply training-fitted column statistics to any data split."""
    tensor = _clean_tabular_tensor(tensor)
    mean = standardizer["mean"].to(dtype=tensor.dtype, device=tensor.device)
    std = standardizer["std"].to(dtype=tensor.dtype, device=tensor.device)
    variable_mask = standardizer["variable_mask"].to(device=tensor.device)
    if mean.dim() != 1 or tensor.shape[1] != mean.numel():
        raise ValueError(
            f"tabular dimension mismatch: tensor has {tensor.shape[1]} columns, "
            f"standardizer has {mean.numel()}"
        )
    transformed = (tensor - mean) / std
    transformed[:, ~variable_mask] = 0.0
    if clamp is not None:
        transformed = torch.clamp(transformed, -float(clamp), float(clamp))
    return torch.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)


def _standardizer_payload(standardizer):
    return {
        "mean_hex": [float(value).hex() for value in standardizer["mean"].tolist()],
        "std_hex": [float(value).hex() for value in standardizer["std"].tolist()],
        "observed_std_hex": [
            float(value).hex() for value in standardizer["observed_std"].tolist()
        ],
        "variable_mask": [bool(value) for value in standardizer["variable_mask"].tolist()],
    }


def tabular_preprocessing_report(standardizers):
    """Build a JSON-safe, deterministic preprocessing record and signature."""
    canonical = {
        "version": TABULAR_PREPROCESSING_VERSION,
        "fit_split": "train",
        "transform": "per_column_population_zscore",
        "zero_variance_policy": "retain_position_and_set_to_zero",
        "clamp": [-TABULAR_CLAMP, TABULAR_CLAMP],
        "num_prop": _standardizer_payload(standardizers["num_prop"]),
        "llm_features": _standardizer_payload(standardizers["llm_features"]),
    }
    signature = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "version": canonical["version"],
        "fit_split": canonical["fit_split"],
        "transform": canonical["transform"],
        "zero_variance_policy": canonical["zero_variance_policy"],
        "clamp": canonical["clamp"],
        "signature_sha256": signature,
    }
    for name in ("num_prop", "llm_features"):
        mask = standardizers[name]["variable_mask"]
        report[name] = {
            "dimension": int(mask.numel()),
            "train_mean": [float(value) for value in standardizers[name]["mean"].tolist()],
            "train_population_std": [
                float(value) for value in standardizers[name]["observed_std"].tolist()
            ],
            "zero_variance_indices": [
                index for index, is_variable in enumerate(mask.tolist()) if not is_variable
            ],
        }
    return report


def fit_tabular_standardizers(data_dir):
    """Fit both release tabular transforms from raw training tensors only."""
    numerical_dir = os.path.join(data_dir, "numerical")
    num_train = torch.load(
        os.path.join(numerical_dir, "train_num_properties_tensor.pt"),
        map_location="cpu",
        weights_only=True,
    )
    style_train = torch.load(
        os.path.join(numerical_dir, "train_llm_features.pt"),
        map_location="cpu",
        weights_only=True,
    )
    standardizers = {
        "num_prop": fit_column_standardizer(num_train),
        "llm_features": fit_column_standardizer(style_train),
    }
    standardizers["report"] = tabular_preprocessing_report(standardizers)
    return standardizers


class Twibot22Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        split='train',
        num_steps=5,
        remap_llm_label=True,
        preserve_tabular_scale=False,
        tabular_standardizers=None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.num_steps = num_steps
        self.remap_llm_label = remap_llm_label
        self.preserve_tabular_scale = bool(preserve_tabular_scale)
        self.tabular_standardizers = tabular_standardizers
        if not self.preserve_tabular_scale and self.tabular_standardizers is None:
            self.tabular_standardizers = fit_tabular_standardizers(self.data_dir)
        self.tabular_preprocessing = (
            {
                "version": "raw-exported-values-v1",
                "fit_split": None,
                "transform": "finite_values_only",
                "signature_sha256": None,
            }
            if self.preserve_tabular_scale
            else self.tabular_standardizers["report"]
        )
        self._load_features()

    def _load_features(self):
        prefix = self.split
        map_loc = torch.device('cpu')

        des_path = os.path.join(self.data_dir, 'descriptions', f'{prefix}_des_tensor.pt')
        self.des_tensor = torch.load(des_path, map_location=map_loc, weights_only=True)
        self.des_tensor = preprocess_tensor(self.des_tensor)

        num_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_num_properties_tensor.pt')
        self.num_prop = torch.load(num_path, map_location=map_loc, weights_only=True)
        self.num_prop = (
            _clean_tabular_tensor(self.num_prop)
            if self.preserve_tabular_scale
            else apply_column_standardizer(
                self.num_prop,
                self.tabular_standardizers["num_prop"],
            )
        )

        llm_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_llm_features.pt')
        self.llm_features = torch.load(llm_path, map_location=map_loc, weights_only=True)
        self.llm_features = (
            _clean_tabular_tensor(self.llm_features)
            if self.preserve_tabular_scale
            else apply_column_standardizer(
                self.llm_features,
                self.tabular_standardizers["llm_features"],
            )
        )

        label_path = os.path.join(self.data_dir, f'{prefix}_labels.pt')
        self.labels = torch.load(label_path, map_location=map_loc, weights_only=True)
        if self.remap_llm_label:
            self.labels[self.labels == 3] = 2

        self.tweets_list = []
        self.amrs_list = []
        self.edge_indices = []

        for step in range(1, self.num_steps + 1):
            text_path = os.path.join(self.data_dir, f'{prefix}_text_tensor{step}.pt')
            if os.path.exists(text_path):
                text_tensor = torch.load(text_path, map_location=map_loc, weights_only=True)
            else:
                text_tensor = torch.zeros((len(self.labels), 768))
            self.tweets_list.append(preprocess_tensor(text_tensor))

            amr_path = os.path.join(self.data_dir, f'{prefix}_amr_tensor{step}.pt')
            if os.path.exists(amr_path):
                amr_tensor = torch.load(amr_path, map_location=map_loc, weights_only=True)
            else:
                amr_tensor = torch.zeros((len(self.labels), 768))
            self.amrs_list.append(preprocess_tensor(amr_tensor))

            edge_path = os.path.join(self.data_dir, f'{prefix}_edge_index_part{step}.pt')
            if os.path.exists(edge_path):
                edge_index = torch.load(edge_path, map_location=map_loc, weights_only=True)
            else:
                graph_path = os.path.join(self.data_dir, 'graph', f'{prefix}_edge_index.pt')
                if os.path.exists(graph_path):
                    edge_index = torch.load(graph_path, map_location=map_loc, weights_only=True)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_indices.append(edge_index)

        self.num_samples = len(self.labels)
        self.metadata = self._load_metadata()
        self.base_profile_ids = (
            [str(row["base_profile_id"]) for row in self.metadata]
            if self.metadata is not None
            else None
        )
        logger.info(f"Loaded {self.num_samples} samples for {self.split} split")

    def _load_metadata(self):
        path = os.path.join(self.data_dir, f"{self.split}_metadata.jsonl")
        if not os.path.exists(path):
            return None
        rows = []
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
        if len(rows) != self.num_samples:
            raise ValueError(
                f"{path} has {len(rows)} rows but the tensor split has {self.num_samples}"
            )
        for index, row in enumerate(rows):
            if "base_profile_id" not in row:
                raise ValueError(f"{path} row {index} is missing base_profile_id")
            if "row_index" in row and int(row["row_index"]) != index:
                raise ValueError(f"{path} row_index is not aligned at row {index}")
            if "label" in row and int(row["label"]) != int(self.labels[index]):
                raise ValueError(f"{path} label is not aligned at row {index}")
            if "split" in row and str(row["split"]) != self.split:
                raise ValueError(f"{path} contains a row from split={row['split']}")
        return rows

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            '_index': idx,
            'des': self.des_tensor[idx],
            'tweets': [tweet[idx] for tweet in self.tweets_list],
            'amrs': [amr[idx] for amr in self.amrs_list],
            'num_prop': self.num_prop[idx],
            'llm_features': self.llm_features[idx],
            'edge_indices': self.edge_indices,
            'label': self.labels[idx]
        }


class QuadBotDataset(Twibot22Dataset):
    def __init__(
        self,
        data_dir,
        split='train',
        num_steps=5,
        preserve_tabular_scale=False,
        tabular_standardizers=None,
    ):
        super().__init__(data_dir, split=split, num_steps=num_steps, remap_llm_label=False,
                         preserve_tabular_scale=preserve_tabular_scale,
                         tabular_standardizers=tabular_standardizers)


class StrictOneHopNeighborCollator:
    """Expand seed rows with their complete strict one-hop kNN neighborhoods.

    The returned graph contains only messages required by the seed rows. With
    the released one-layer structural attention this is exact for seed-node
    inference, while preserving conventional mini-batch optimization.
    """

    def __init__(self, dataset, expected_k=10, require_strict=True):
        self.dataset = dataset
        self.expected_k = expected_k
        self.require_strict = bool(require_strict)
        self.num_nodes = len(dataset)
        self.base_profile_ids = getattr(dataset, "base_profile_ids", None)
        self.validation_reports = []
        for edge_pairs in dataset.edge_indices:
            self.validation_reports.append(
                validate_knn_pairs(
                    edge_pairs,
                    self.num_nodes,
                    base_profile_ids=self.base_profile_ids,
                    expected_k=self.expected_k,
                    require_strict=self.require_strict,
                )
            )

    def __call__(self, seed_items):
        seed_indices = torch.tensor([int(item["_index"]) for item in seed_items], dtype=torch.long)
        if seed_indices.numel() == 0:
            raise ValueError("cannot collate an empty seed batch")
        if torch.unique(seed_indices).numel() != seed_indices.numel():
            raise ValueError("seed batch contains duplicate row indices")
        if (seed_indices < 0).any() or (seed_indices >= self.num_nodes).any():
            raise ValueError("seed batch contains an out-of-range row index")

        is_seed = torch.zeros(self.num_nodes, dtype=torch.bool)
        is_seed[seed_indices] = True
        selected_pairs = []
        context_parts = []
        for edge_pairs in self.dataset.edge_indices:
            mask = is_seed[edge_pairs[0]]
            pairs = edge_pairs[:, mask]
            degree = torch.bincount(pairs[0], minlength=self.num_nodes)[seed_indices]
            if self.expected_k is not None and not torch.all(degree == int(self.expected_k)):
                raise ValueError(
                    f"seed degrees {torch.unique(degree).tolist()} differ from expected_k={self.expected_k}"
                )
            selected_pairs.append(pairs)
            context_parts.append(pairs[1])

        context_indices = torch.unique(torch.cat(context_parts), sorted=True)
        context_indices = context_indices[~is_seed[context_indices]]
        node_indices = torch.cat([seed_indices, context_indices])
        global_to_local = torch.full((self.num_nodes,), -1, dtype=torch.long)
        global_to_local[node_indices] = torch.arange(node_indices.numel(), dtype=torch.long)

        edge_indices = []
        for pairs in selected_pairs:
            message_edges = knn_pairs_to_message_edges(pairs)
            local_edges = global_to_local[message_edges]
            if (local_edges < 0).any():
                raise RuntimeError("neighbor expansion omitted a required message endpoint")
            edge_indices.append(local_edges)

        tweets = [tensor[node_indices] for tensor in self.dataset.tweets_list]
        amrs = [tensor[node_indices] for tensor in self.dataset.amrs_list]
        seed_count = int(seed_indices.numel())
        return {
            "des": self.dataset.des_tensor[node_indices],
            "tweets": tweets,
            "amrs": amrs,
            "num_prop": self.dataset.num_prop[node_indices],
            "llm_features": self.dataset.llm_features[node_indices],
            "edge_indices": edge_indices,
            "labels": self.dataset.labels[node_indices],
            "seed_positions": torch.arange(seed_count, dtype=torch.long),
            "seed_labels": self.dataset.labels[seed_indices],
            "seed_global_indices": seed_indices,
            "node_global_indices": node_indices,
        }


def build_full_graph_batch(dataset, expected_k=10, require_strict=True):
    """Build the full split graph as a correctness oracle for evaluation."""
    collator = StrictOneHopNeighborCollator(
        dataset,
        expected_k=expected_k,
        require_strict=require_strict,
    )
    seed_items = [{"_index": index} for index in range(len(dataset))]
    return collator(seed_items)

