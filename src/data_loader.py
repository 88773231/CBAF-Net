import torch
import torch.nn.functional as F
import json
import os
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


SOURCE_TO_ID = {
    "twibot22": 0,
    "botsim": 1,
    "gag": 2,
    "graphagent": 2,
}


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


class Twibot20Dataset(Dataset):
    def __init__(self, data_dir, split='train', num_steps=5, remap_llm_label=True, preserve_tabular_scale=False):
        self.data_dir = data_dir
        self.split = split
        self.num_steps = num_steps
        self.remap_llm_label = remap_llm_label
        self.preserve_tabular_scale = bool(preserve_tabular_scale)
        self._load_features()

    def _load_features(self):
        prefix = self.split
        map_loc = torch.device('cpu')

        des_path = os.path.join(self.data_dir, 'descriptions', f'{prefix}_des_tensor.pt')
        self.des_tensor = torch.load(des_path, map_location=map_loc, weights_only=True)
        self.des_tensor = preprocess_tensor(self.des_tensor)

        num_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_num_properties_tensor.pt')
        self.num_prop = torch.load(num_path, map_location=map_loc, weights_only=True)
        self.num_prop = preprocess_tensor(self.num_prop, normalize=not self.preserve_tabular_scale)

        llm_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_llm_features.pt')
        self.llm_features = torch.load(llm_path, map_location=map_loc, weights_only=True)
        self.llm_features = preprocess_tensor(self.llm_features, normalize=not self.preserve_tabular_scale)

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
        self.source_labels = torch.full((self.num_samples,), -1, dtype=torch.long)
        logger.info(f"Loaded {self.num_samples} samples for {self.split} split")

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
            'label': self.labels[idx],
            'source_label': self.source_labels[idx]
        }


# The historical class name is retained for checkpoint compatibility.
Twibot22Dataset = Twibot20Dataset


class QuadBotDataset(Twibot20Dataset):
    def __init__(self, data_dir, split='train', num_steps=5, preserve_tabular_scale=False):
        super().__init__(data_dir, split=split, num_steps=num_steps, remap_llm_label=False,
                         preserve_tabular_scale=preserve_tabular_scale)

    def _load_features(self):
        super()._load_features()
        source_labels = self._load_source_labels()
        if source_labels is None:
            source_labels = torch.zeros_like(self.labels, dtype=torch.long)
            source_labels[self.labels == 2] = 1
            source_labels[self.labels == 3] = 2
        self.source_labels = source_labels.long()

    def _load_source_labels(self):
        manifest_path = os.path.join(self.data_dir, 'raw', 'quadbot_manifest.jsonl')
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join(self.data_dir, 'raw', 'quadbot22_manifest.jsonl')
        if not os.path.exists(manifest_path):
            return None

        rows = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get('split') == self.split:
                    rows.append(item)

        if len(rows) != self.num_samples:
            logger.warning(
                "Source manifest mismatch for %s: %s rows vs %s samples; using label-derived sources",
                self.split,
                len(rows),
                self.num_samples,
            )
            return None

        labels = []
        for item in rows:
            source = str(item.get('source', '')).lower()
            if source not in SOURCE_TO_ID:
                label = int(item.get('label', 0))
                source_id = 1 if label == 2 else 2 if label == 3 else 0
            else:
                source_id = SOURCE_TO_ID[source]
            labels.append(source_id)
        return torch.tensor(labels, dtype=torch.long)


class BotSimDataset(Dataset):
    def __init__(self, data_dir, split='train', num_steps=5):
        self.data_dir = data_dir
        self.split = split
        self.num_steps = num_steps
        self._load_features()

    def _load_features(self):
        prefix = self.split
        map_loc = torch.device('cpu')

        des_path = os.path.join(self.data_dir, 'descriptions', f'{prefix}_des_tensor.pt')
        self.des_tensor = torch.load(des_path, map_location=map_loc, weights_only=True)
        self.des_tensor = preprocess_tensor(self.des_tensor)

        num_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_num_properties_tensor.pt')
        self.num_prop = torch.load(num_path, map_location=map_loc, weights_only=True)
        self.num_prop = preprocess_tensor(self.num_prop)

        llm_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_llm_features.pt')
        self.llm_features = torch.load(llm_path, map_location=map_loc, weights_only=True)
        self.llm_features = preprocess_tensor(self.llm_features)

        label_file = os.path.join(self.data_dir, 'labels', f'{prefix}.json')
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            self.labels = torch.tensor([d.get('label', 0) for d in data.values()], dtype=torch.long)
        else:
            self.labels = torch.tensor([d.get('label', 0) for d in data], dtype=torch.long)

        min_samples = min(self.des_tensor.shape[0], self.num_prop.shape[0],
                          self.llm_features.shape[0], self.labels.shape[0])

        self.des_tensor = self.des_tensor[:min_samples]
        self.num_prop = self.num_prop[:min_samples]
        self.llm_features = self.llm_features[:min_samples]
        self.labels = self.labels[:min_samples]

        self.tweets_list = []
        self.amrs_list = []
        self.edge_indices = []

        for step in range(1, self.num_steps + 1):
            text_path = os.path.join(self.data_dir, f'{prefix}_text_tensor{step}.pt')
            if os.path.exists(text_path):
                text_tensor = torch.load(text_path, map_location=map_loc, weights_only=True)[:min_samples]
            else:
                text_tensor = torch.zeros((min_samples, 768))
            self.tweets_list.append(preprocess_tensor(text_tensor))

            amr_path = os.path.join(self.data_dir, f'{prefix}_amr_tensor{step}.pt')
            if os.path.exists(amr_path):
                amr_tensor = torch.load(amr_path, map_location=map_loc, weights_only=True)[:min_samples]
            else:
                amr_tensor = torch.zeros((min_samples, 768))
            self.amrs_list.append(preprocess_tensor(amr_tensor))

            edge_path = os.path.join(self.data_dir, f'{prefix}_edge_index_part{step}.pt')
            if os.path.exists(edge_path):
                edge_index = torch.load(edge_path, map_location=map_loc, weights_only=True)
                if edge_index.size(1) > 0:
                    valid_edges = (edge_index[0] < min_samples) & (edge_index[1] < min_samples)
                    edge_index = edge_index[:, valid_edges]
            else:
                graph_path = os.path.join(self.data_dir, 'graph', f'{prefix}_edge_index.pt')
                if os.path.exists(graph_path):
                    edge_index = torch.load(graph_path, map_location=map_loc, weights_only=True)
                    if edge_index.size(1) > 0:
                        valid_edges = (edge_index[0] < min_samples) & (edge_index[1] < min_samples)
                        edge_index = edge_index[:, valid_edges]
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_indices.append(edge_index)

        self.num_samples = min_samples
        self.source_labels = torch.full((self.num_samples,), -1, dtype=torch.long)
        logger.info(f"Loaded {self.num_samples} samples for {self.split} split")

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
            'label': self.labels[idx],
            'source_label': self.source_labels[idx]
        }


def collate_fn(batch):
    batch_indices = torch.tensor([item['_index'] for item in batch], dtype=torch.long)
    des = torch.stack([item['des'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    num_prop = torch.stack([item['num_prop'] for item in batch])
    llm_features = torch.stack([item['llm_features'] for item in batch])
    source_labels = torch.stack([item.get('source_label', torch.tensor(-1, dtype=torch.long)) for item in batch])
    tweets = []
    amrs = []
    for step in range(len(batch[0]['tweets'])):
        tweets.append(torch.stack([item['tweets'][step] for item in batch]))
        amrs.append(torch.stack([item['amrs'][step] for item in batch]))
    # Edge files are indexed in split-global coordinates. Remap them to the
    # current mini-batch; otherwise the old loader silently discarded almost
    # every edge after shuffling.
    edge_indices = []
    max_batch_index = int(batch_indices.max().item()) if batch_indices.numel() else -1
    for full_edge_index in batch[0]['edge_indices']:
        if full_edge_index.numel() == 0:
            edge_indices.append(torch.zeros((2, 0), dtype=torch.long))
            continue
        max_edge_index = int(full_edge_index.max().item())
        mapping = torch.full(
            (max(max_batch_index, max_edge_index) + 1,), -1, dtype=torch.long
        )
        mapping[batch_indices] = torch.arange(len(batch), dtype=torch.long)
        src = mapping[full_edge_index[0]]
        dst = mapping[full_edge_index[1]]
        valid = (src >= 0) & (dst >= 0)
        edge_indices.append(torch.stack([src[valid], dst[valid]], dim=0))
    return des, tweets, amrs, num_prop, llm_features, edge_indices, labels, source_labels

