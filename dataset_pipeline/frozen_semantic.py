"""Local-only frozen sentence encoding for the bounded representation study."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset_pipeline.export_quadbot_v3_features import (
    clean_text, hash_embed, normalize_text, profile_text,
)
from dataset_pipeline.export_relative_timing import parse_time
from dataset_pipeline.observable_cues import FEATURE_DIMENSIONS, STEPS, observe


DIMENSION = 384
DIMENSIONS = {**FEATURE_DIMENSIONS, "description": DIMENSION, "text": DIMENSION}
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WEIGHT_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
FILES = (
    "config.json", "sentence_bert_config.json", "tokenizer_config.json",
    "tokenizer.json", "vocab.txt", "special_tokens_map.json",
    "1_Pooling/config.json", "model.safetensors", "README.md",
)


def masked_mean(hidden, mask):
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("aligned token states and attention mask required")
    if not torch.isfinite(hidden).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("finite states and binary mask required")
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return F.normalize(
        (hidden * weights).sum(1) / weights.sum(1).clamp_min(1), p=2, dim=1,
    )


def verify_snapshot(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "snapshot_manifest.json").read_text(encoding="utf-8"))
    if (manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION
            or manifest["license"] != "apache-2.0" or set(manifest["files"]) != set(FILES)):
        raise ValueError("frozen semantic snapshot identity differs")
    for name, record in manifest["files"].items():
        path = folder / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("semantic snapshot member changed: " + name)
    if manifest["files"]["model.safetensors"]["sha256"] != WEIGHT_SHA256:
        raise ValueError("semantic weights do not match the pinned artifact")
    settings = json.loads((folder / "sentence_bert_config.json").read_text())
    pooling = json.loads((folder / "1_Pooling/config.json").read_text())
    config = json.loads((folder / "config.json").read_text())
    if (settings["max_seq_length"] != 256 or config["hidden_size"] != DIMENSION
            or not pooling["pooling_mode_mean_tokens"]
            or pooling["pooling_mode_cls_token"] or pooling["pooling_mode_max_tokens"]
            or pooling["pooling_mode_mean_sqrt_len_tokens"]):
        raise ValueError("semantic encoding contract differs")
    return manifest


class FrozenSentenceEncoder:
    def __init__(self, folder, batch_size=64, cache_embeddings=False):
        if batch_size < 1:
            raise ValueError("positive encoding batch size required")
        self.manifest = verify_snapshot(folder)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).cpu().eval().requires_grad_(False)
        self.batch_size = batch_size
        self.embedding_cache = {} if cache_embeddings else None
        self.statistics = {"encoded_texts": 0, "truncated_texts": 0, "tokens_before_truncation": 0}

    def encode(self, texts):
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("sentence encoder consumes strings only")
        result = np.zeros((len(texts), DIMENSION), dtype=np.float32)
        selected = [index for index, text in enumerate(texts) if text]
        cache = getattr(self, "embedding_cache", None)
        if cache is not None:
            missing = []
            for index in selected:
                if texts[index] in cache:
                    result[index] = cache[texts[index]]
                    self.statistics["frozen_text_cache_hits"] = self.statistics.get("frozen_text_cache_hits", 0) + 1
                else:
                    missing.append(index)
            selected = missing
        if not selected:
            return result
        lengths = self.tokenizer(
            [texts[index] for index in selected], truncation=False,
            return_length=True, verbose=False,
        )["length"]
        self.statistics["encoded_texts"] += len(selected)
        self.statistics["truncated_texts"] += sum(length > 256 for length in lengths)
        self.statistics["tokens_before_truncation"] += sum(lengths)
        # Length sorting changes padding only; outputs return to original text order.
        selected = [index for _, index in sorted(zip(lengths, selected))]
        with torch.inference_mode():
            for start in range(0, len(selected), self.batch_size):
                indices = selected[start:start + self.batch_size]
                batch = [texts[index] for index in indices]
                tokens = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=256, return_tensors="pt",
                )
                result[indices] = masked_mean(
                    self.model(**tokens).last_hidden_state, tokens["attention_mask"],
                ).cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("sentence embeddings must be finite")
        if cache is not None:
            for index in selected:
                cache[texts[index]] = result[index].copy()
        return result


def public_text_groups(row):
    posts = row.get("posts")
    if not isinstance(posts, list) or len(posts) < STEPS:
        raise ValueError("at least five timestamped public posts required")
    times = np.asarray([parse_time(post.get("created_at")) for post in posts])
    rounded = np.round(times - times.min())
    order = sorted(range(len(posts)), key=lambda index: (rounded[index], index))
    windows = np.array_split(order, STEPS)
    return [
        [normalize_text(profile_text(row))],
        *[[normalize_text(clean_text(posts[int(index)].get("text"))) for index in window]
          for window in windows],
    ]


def pool_group(vectors, dimension=DIMENSION):
    if vectors.ndim != 2 or vectors.shape[1] != dimension or len(vectors) < 1:
        raise ValueError("nonempty aligned post vectors required")
    vector = vectors.mean(0).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else vector


def representation_features(rows, encoder, progress=None):
    """Only caller-allowlisted rows enter this function; identities are not encoded."""
    if not rows:
        raise ValueError("selected observation records required")
    groups = [public_text_groups(row) for row in rows]
    unique = sorted({text for account in groups for group in account for text in group})
    embeddings = {}
    # Keep a bounded encoder call so a progress update is possible during CPU extraction.
    for start in range(0, len(unique), 1024):
        batch = unique[start:start + 1024]
        values = encoder.encode(batch)
        if values.shape != (len(batch), DIMENSION) or not np.isfinite(values).all():
            raise ValueError("unexpected frozen encoder output")
        embeddings.update(zip(batch, values))
        if progress:
            progress({"event": "encoded_public_texts", "completed": min(start + 1024, len(unique)),
                      "planned": len(unique)})
    output = {"hash": [], "semantic": []}
    for row, account in zip(rows, groups):
        common = observe(row)
        for kind in output:
            values = []
            for index, texts in enumerate(account):
                if kind == "semantic":
                    vectors = np.stack([embeddings[text] for text in texts])
                else:
                    namespace = "observable-profile" if index == 0 else "observable-post-text"
                    vectors = np.stack([hash_embed(text, namespace, DIMENSION) for text in texts])
                values.append(pool_group(vectors))
            output[kind].append({**common, "description": values[0], "text": np.stack(values[1:])})
    return output, {
        "selected_trajectories": len(rows), "unique_normalized_texts": len(unique),
        "empty_texts": sum(not text for account in groups for group in account for text in group),
        "declared_profile_languages": sorted({
            str((row.get("profile") or {}).get("language") or "unspecified").casefold() for row in rows
        }),
        "language_detection_performed": False,
        "all_identity_label_and_internal_fields_are_excluded_from_encoder_input": True,
        "normalization_shared_between_representations": True,
        "encoder_frozen": True, "gpu_used": False,
    }
