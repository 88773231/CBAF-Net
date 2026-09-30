#!/usr/bin/env python3
"""Export deterministic, source-agnostic QuadBot-v3 feature tensors.

The release manifest is the source of truth.  Every view (profile, post text,
action/AMR, numerical attributes and graph structure) is derived with the same
rules for all four labels.  No label, provenance field or source identifier is
read while constructing features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


TEXT_DIM = 768
GRAPH_DIM = TEXT_DIM * 3
NUM_DIM = 6
LLM_DIM = 11
NUM_STEPS = 5
KNN_K = 10
# Released source text is privacy-sanitized with the literal ``<URL>`` token.
# Count that token as well as unsanitized URLs so the exported URL statistic is
# not accidentally constant after redaction.
URL_RE = re.compile(r"https?://\S+|www\.\S+|<URL>", re.I)
USER_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def normalize_text(value: object) -> str:
    text = clean_text(value).casefold()
    text = URL_RE.sub(" <url> ", text)
    text = USER_RE.sub(" <user> ", text)
    text = HASHTAG_RE.sub(lambda m: " " + m.group(0)[1:] + " ", text)
    return " ".join(text.split())


def _bucket(namespace: str, feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(
        f"quadbot-v3:{namespace}:{feature}".encode("utf-8"), digest_size=8
    ).digest()
    index = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def hash_embed(value: object, namespace: str, dim: int = TEXT_DIM) -> np.ndarray:
    """Shared signed word/character hashing used for every text-bearing class."""
    text = normalize_text(value)
    out = np.zeros(dim, dtype=np.float32)
    if not text:
        return out

    tokens = TOKEN_RE.findall(text)
    features: list[tuple[str, float]] = []
    for token in tokens:
        features.append((f"w:{token}", 1.0))
        if len(token) > 2:
            for n in (3, 4, 5):
                for start in range(max(0, len(token) - n + 1)):
                    features.append((f"c{n}:{token[start:start+n]}", 0.35))
    for left, right in zip(tokens, tokens[1:]):
        features.append((f"b:{left}_{right}", 0.65))

    for feature, weight in features:
        index, sign = _bucket(namespace, feature, dim)
        out[index] += sign * weight
    norm = float(np.linalg.norm(out))
    if norm > 1e-8:
        out /= norm
    return out


def post_texts_by_step(row: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for post in row.get("posts") or []:
        try:
            step = int(post.get("time_step", post.get("step", 1)))
        except (TypeError, ValueError):
            step = 1
        step = min(max(step, 1), NUM_STEPS)
        grouped[step].append(post)
    return grouped


def action_by_step(row: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for action in row.get("actions") or []:
        try:
            step = int(action.get("time_step", action.get("step", 1)))
        except (TypeError, ValueError):
            step = 1
        step = min(max(step, 1), NUM_STEPS)
        grouped[step].append(action)
    return grouped


def profile_text(row: dict) -> str:
    profile = row.get("profile") or {}
    return " | ".join(
        part
        for part in (
            profile.get("description"),
            profile.get("location"),
            profile.get("language"),
        )
        if clean_text(part)
    )


def encode_modalities(row: dict) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    profile_vec = hash_embed(profile_text(row), "profile")
    posts = post_texts_by_step(row)
    actions = action_by_step(row)
    text_steps: list[np.ndarray] = []
    amr_steps: list[np.ndarray] = []
    for step in range(1, NUM_STEPS + 1):
        step_posts = posts.get(step, [])
        step_actions = actions.get(step, [])
        texts = [clean_text(post.get("text")) for post in step_posts]
        texts = [text for text in texts if text]
        if texts:
            vectors = [hash_embed(text, f"text-step-{step}") for text in texts]
            text_vec = np.mean(vectors, axis=0).astype(np.float32)
            norm = float(np.linalg.norm(text_vec))
            if norm > 1e-8:
                text_vec /= norm
        else:
            text_vec = np.zeros(TEXT_DIM, dtype=np.float32)

        # AMR is represented by a shared event/argument serialization.  IDs
        # are intentionally omitted; only action type, target presence and
        # time-step structure are retained.
        event_parts = []
        for post in step_posts:
            action_type = clean_text(post.get("action_type", post.get("type", "post"))).casefold()
            target_count = len(post.get("interaction_targets") or post.get("target_ids") or [])
            event_parts.append(
                f"event type={action_type} target_count={target_count} text={clean_text(post.get('text'))}"
            )
        for action in step_actions:
            action_type = clean_text(action.get("type", action.get("action", "post"))).casefold()
            target_count = len(action.get("target_ids") or [])
            triggered = bool(action.get("environment_triggered", False))
            event_parts.append(
                f"action type={action_type} target_count={target_count} triggered={int(triggered)}"
            )
        event_parts.append(f"step={step} post_count={len(step_posts)} action_count={len(step_actions)}")
        amr_vec = hash_embed(" ; ".join(event_parts), f"amr-step-{step}")
        text_steps.append(text_vec)
        amr_steps.append(amr_vec)
    return profile_vec, text_steps, amr_steps


def numerical_features(row: dict) -> np.ndarray:
    profile = row.get("profile") or {}
    metrics = profile.get("public_metrics") or {}
    posts = row.get("posts") or []
    actions = row.get("actions") or []
    values = [
        math.log1p(max(0, float(metrics.get("followers_count", 0) or 0))),
        math.log1p(max(0, float(metrics.get("following_count", 0) or 0))),
        math.log1p(max(0, float(metrics.get("tweet_count", 0) or 0))),
        math.log1p(max(0, float(metrics.get("listed_count", 0) or 0))),
        math.log1p(len(posts)),
        math.log1p(len(actions)),
    ]
    return np.asarray(values, dtype=np.float32)


def llm_style_features(row: dict) -> np.ndarray:
    posts = row.get("posts") or []
    actions = row.get("actions") or []
    texts = [clean_text(post.get("text")) for post in posts]
    texts = [text for text in texts if text]
    joined = " ".join(texts)
    words = TOKEN_RE.findall(joined)
    chars = max(len(joined), 1)
    unique_ratio = len(set(token.casefold() for token in words)) / max(len(words), 1)
    punctuation = sum(ch in "!?.,;:" for ch in joined) / chars
    urls = len(URL_RE.findall(joined)) / max(len(texts), 1)
    hashtags = len(HASHTAG_RE.findall(joined)) / max(len(texts), 1)
    uppercase = sum(ch.isupper() for ch in joined) / max(sum(ch.isalpha() for ch in joined), 1)
    non_ascii = sum(ord(ch) > 127 for ch in joined) / chars
    mean_len = float(np.mean([len(text) for text in texts])) if texts else 0.0
    std_len = float(np.std([len(text) for text in texts])) if texts else 0.0
    type_counts = Counter(clean_text(action.get("type", action.get("action", "post"))).casefold() for action in actions)
    action_total = max(sum(type_counts.values()), 1)
    repost_comment_ratio = (type_counts.get("repost", 0) + type_counts.get("retweet", 0)) / action_total
    interaction_ratio = sum(bool(action.get("target_ids")) for action in actions) / action_total
    return np.asarray(
        [
            math.log1p(len(texts)),
            math.log1p(mean_len),
            math.log1p(std_len),
            unique_ratio,
            punctuation,
            urls,
            hashtags,
            uppercase,
            non_ascii,
            repost_comment_ratio,
            interaction_ratio,
        ],
        dtype=np.float32,
    )


def graph_features(row: dict, profile_vec: np.ndarray, text_steps: list[np.ndarray], amr_steps: list[np.ndarray]) -> np.ndarray:
    graph = row.get("graph") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_types = Counter(clean_text(node.get("type", "unknown")).casefold() for node in nodes)
    edge_parts = []
    for edge in edges:
        edge_type = clean_text(edge.get("type", "unknown")).casefold()
        step = clean_text(edge.get("time_step", "0"))
        edge_parts.append(f"edge type={edge_type} step={step}")
    node_part = " ".join(f"node type={key} count={value}" for key, value in sorted(node_types.items()))
    structure_vec = hash_embed(node_part + " ; " + " ; ".join(edge_parts), "graph-structure")
    temporal_text = " ".join(
        f"step={index + 1} text_norm={float(np.linalg.norm(vec)):.6f} amr_norm={float(np.linalg.norm(amr_steps[index])):.6f}"
        for index, vec in enumerate(text_steps)
    )
    temporal_vec = hash_embed(temporal_text, "graph-temporal")
    return np.concatenate([profile_vec, structure_vec, temporal_vec]).astype(np.float32)


def build_edge_index(
    graph_matrix: np.ndarray,
    k: int = KNN_K,
    base_profile_ids: list[str] | None = None,
    exclude_same_base_profile: bool = True,
) -> torch.Tensor:
    """Build split-local, label-free ``(target, neighbor)`` kNN pairs.

    The release default excludes paired counterfactual variants sharing a base
    profile. Rows with fewer than ``k`` valid neighbors have a lower selection
    degree. Runtime graph attention reverses each pair to send a message from
    the selected neighbor to its target row.
    """
    n = int(graph_matrix.shape[0])
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    k = min(int(k), n - 1)
    x = torch.from_numpy(graph_matrix).float()
    x = torch.nn.functional.normalize(x, dim=1)
    similarity = x @ x.T
    similarity.fill_diagonal_(-float("inf"))
    if exclude_same_base_profile:
        if base_profile_ids is None or len(base_profile_ids) != n:
            raise ValueError("base_profile_ids are required when excluding paired profiles")
        same = torch.tensor(
            [[base_profile_ids[i] == base_profile_ids[j] for j in range(n)] for i in range(n)],
            dtype=torch.bool,
        )
        similarity.masked_fill_(same, -float("inf"))
    values, neighbors = similarity.topk(k=k, dim=1)
    valid = torch.isfinite(values)
    sources = torch.arange(n, dtype=torch.long).repeat_interleave(valid.sum(dim=1))
    return torch.stack([sources, neighbors[valid].long()], dim=0)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def export_split(
    rows: list[dict],
    split: str,
    out_root: Path,
    knn_k: int = KNN_K,
    exclude_same_base_profile: bool = True,
) -> dict:
    descriptions = []
    texts = [[] for _ in range(NUM_STEPS)]
    amrs = [[] for _ in range(NUM_STEPS)]
    graphs = []
    nums = []
    llms = []
    labels = []
    sample_ids = []
    base_ids = []
    for row in rows:
        profile_vec, text_steps, amr_steps = encode_modalities(row)
        descriptions.append(profile_vec)
        for index in range(NUM_STEPS):
            texts[index].append(text_steps[index])
            amrs[index].append(amr_steps[index])
        graphs.append(graph_features(row, profile_vec, text_steps, amr_steps))
        nums.append(numerical_features(row))
        llms.append(llm_style_features(row))
        labels.append(int(row["label"]))
        sample_ids.append(str(row["sample_id"]))
        base_ids.append(str(row["base_profile_id"]))

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "descriptions").mkdir(exist_ok=True)
    (out_root / "numerical").mkdir(exist_ok=True)
    (out_root / "graph").mkdir(exist_ok=True)
    torch.save(torch.from_numpy(np.stack(descriptions)), out_root / "descriptions" / f"{split}_des_tensor.pt")
    for index in range(NUM_STEPS):
        torch.save(torch.from_numpy(np.stack(texts[index])), out_root / f"{split}_text_tensor{index + 1}.pt")
        torch.save(torch.from_numpy(np.stack(amrs[index])), out_root / f"{split}_amr_tensor{index + 1}.pt")
    graph_matrix = np.stack(graphs)
    edge_index = build_edge_index(
        graph_matrix,
        k=knn_k,
        base_profile_ids=base_ids,
        exclude_same_base_profile=exclude_same_base_profile,
    )
    torch.save(torch.from_numpy(graph_matrix), out_root / "graph" / f"{split}_node_features.pt")
    torch.save(edge_index, out_root / "graph" / f"{split}_edge_index.pt")
    torch.save(torch.from_numpy(np.stack(nums)), out_root / "numerical" / f"{split}_num_properties_tensor.pt")
    torch.save(torch.from_numpy(np.stack(llms)), out_root / "numerical" / f"{split}_llm_features.pt")
    torch.save(torch.tensor(labels, dtype=torch.long), out_root / f"{split}_labels.pt")
    with (out_root / f"{split}_metadata.jsonl").open("w", encoding="utf-8") as stream:
        for row_index, (sample_id, base_id, label) in enumerate(zip(sample_ids, base_ids, labels)):
            stream.write(json.dumps({"row_index": row_index, "sample_id": sample_id, "base_profile_id": base_id, "label": label}) + "\n")
    same_base_edges = 0
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
    if edge_count:
        for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if base_ids[int(source)] == base_ids[int(target)]:
                same_base_edges += 1
    if exclude_same_base_profile and same_base_edges:
        raise RuntimeError(
            f"strict export produced {same_base_edges} same-base-profile edges for {split}"
        )
    return {
        "rows": len(rows),
        "labels": dict(sorted(Counter(labels).items())),
        "sample_ids": sample_ids,
        "base_profile_ids": base_ids,
        "knn": {
            "k": int(min(knn_k, max(0, len(rows) - 1))),
            "edge_count": edge_count,
            "self_loop_count": int(sum(int(source) == int(target) for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()))) if edge_count else 0,
            "same_base_profile_edge_count": same_base_edges,
            "same_base_profile_edge_fraction": (same_base_edges / edge_count) if edge_count else 0.0,
            "exclude_same_base_profile": bool(exclude_same_base_profile),
            "scope": "split_local",
            "direction": "directed",
            "metric": "cosine",
            "uses_labels": False,
            "uses_provenance": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, default=KNN_K)
    parser.add_argument(
        "--exclude-same-base-profile",
        action="store_true",
        default=True,
        help=(
            "Compatibility flag; strict exports always exclude paired "
            "counterfactual variants sharing base_profile_id."
        ),
    )
    args = parser.parse_args()
    if args.knn_k < 1:
        raise SystemExit("--knn-k must be >= 1")

    audit = {
        "schema_version": "quadbot-v3-feature-export-1",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "dimensions": {"description": TEXT_DIM, "text": TEXT_DIM, "amr": TEXT_DIM, "graph": GRAPH_DIM, "num": NUM_DIM, "llm": LLM_DIM},
        "rules": {
            "text_encoder": "shared signed word/character hashing",
            "profile_text_fields": ["description", "location", "language"],
            "graph_identifier_policy": "node and edge identifiers excluded",
            "graph_record_semantics": "interaction graph from each manifest record",
            "graph_feature_semantics": "hashed profile/structure/temporal vector",
            "edge_index_semantics": "split-local cosine kNN (target, selected_neighbor) pairs over graph feature vectors; runtime messages flow selected_neighbor to target",
            "knn_k": int(args.knn_k),
            "knn_scope": "split_local_transductive",
            "knn_cross_split_edges": False,
            "knn_same_base_profile_edges": "always excluded by the strict release export",
            "knn_exclude_same_base_profile": bool(args.exclude_same_base_profile),
            "label_and_provenance_used": False,
        },
        "splits": {},
    }
    for split in ("train", "val", "test"):
        rows = load_rows(args.input_dir / f"{split}.jsonl")
        audit["splits"][split] = export_split(
            rows,
            split,
            args.output_dir,
            knn_k=args.knn_k,
            exclude_same_base_profile=args.exclude_same_base_profile,
        )
    (args.output_dir / "feature_export_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: {s: {"rows": v["rows"], "labels": v["labels"]} for s, v in value.items()} if k == "splits" else value for k, value in audit.items() if k in {"schema_version", "dimensions", "splits"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
