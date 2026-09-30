#!/usr/bin/env python3
"""Audit a QuadBot release against the model/data contract.

The audit is intentionally independent of training code.  It checks the
account-level grouping, paired labels, split isolation, record quality,
provenance completeness, and (when feature tensors are supplied) the exact
split-local kNN edge protocol.  It never edits the input release.

The output is a machine-readable report suitable for archiving beside a
release manifest.  A non-zero exit code is returned only when ``--strict`` is
used and a contract error is found; missing optional artifacts are reported as
warnings so the same script can be used before feature export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Optional


LABELS = {0: "human", 1: "traditional_bot", 2: "llm_bot", 3: "full_stack_agent"}
SPLITS = ("train", "val", "test")
REQUIRED_MASTER_FIELDS = {
    "sample_id",
    "base_profile_id",
    "label",
    "label_name",
    "platform_schema",
    "profile",
    "posts",
    "actions",
    "graph",
    "provenance",
}
MODEL_VIEW_FIELDS = {
    "sample_id",
    "base_profile_id",
    "label",
    "split",
    "profile",
    "posts",
    "actions",
    "graph",
}
PROVENANCE_FIELDS = (
    "source_dataset",
    "conditioning_source",
    "generation_policy",
    "pipeline_version",
    "generator_family",
    "model_path",
    "temperature",
    "top_p",
    "repetition_penalty",
    "max_new_tokens",
    "do_sample",
    "generation_seed",
    "simulation_seed",
    "generation_rounds",
    "items_per_round",
    "oasis_version",
    "observes_environment",
    "memory_enabled",
)
REQUIRED_PROVENANCE_BY_LABEL = {
    "human": (
        "source_dataset",
        "generation_policy",
        "source_profile_id",
        "pilot_stratum",
    ),
    "traditional_bot": (
        "source_dataset",
        "generation_policy",
        "source_profile_id",
        "pilot_stratum",
        "observes_environment",
    ),
    "llm_bot": (
        "source_dataset",
        "conditioning_source",
        "generation_policy",
        "pipeline_version",
        "generator_family",
        "model_path",
        "temperature",
        "top_p",
        "repetition_penalty",
        "max_new_tokens",
        "do_sample",
        "generation_seed",
        "generation_rounds",
        "items_per_round",
        "observes_environment",
        "memory_enabled",
    ),
    "full_stack_agent": (
        "source_dataset",
        "conditioning_source",
        "generation_policy",
        "pipeline_version",
        "generator_family",
        "model_path",
        "temperature",
        "top_p",
        "repetition_penalty",
        "max_new_tokens",
        "do_sample",
        "generation_seed",
        "generation_rounds",
        "items_per_round",
        "oasis_version",
        "observes_environment",
        "memory_enabled",
    ),
}
FORBIDDEN_MODEL_KEYS = re.compile(
    r"(?:provenance|source_dataset|source_id|generator|generation_|simulation_seed|"
    r"pipeline_version|conditioning_source|grounding|oasis|label_name)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True, help="canonical master JSONL")
    parser.add_argument(
        "--views-root",
        type=Path,
        default=None,
        help="optional root containing Twibot22/Quadbot split JSONL views",
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=None,
        help="optional root containing split tensors and graph/edge_index files",
    )
    parser.add_argument(
        "--feature-audit",
        type=Path,
        default=None,
        help="optional feature_export_audit.json; defaults to features-root file",
    )
    parser.add_argument("--out", type=Path, required=True, help="audit report path")
    parser.add_argument("--expected-posts", type=int, default=25)
    parser.add_argument("--expected-actions", type=int, default=25)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            yield value


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def word_count(value: Any) -> int:
    return len(re.findall(r"\b\w+\b", str(value or ""), flags=re.UNICODE))


def add_error(errors: list[str], message: str, limit: int = 100) -> None:
    if len(errors) < limit:
        errors.append(message)


def collect_master(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    sample_ids: set[str] = set()
    base_to_labels: dict[str, set[str]] = defaultdict(set)
    base_to_splits: dict[str, set[str]] = defaultdict(set)
    base_to_profile_digests: dict[str, set[str]] = defaultdict(set)
    base_to_records: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    provenance_missing: Counter[str] = Counter()
    required_provenance_missing: Counter[str] = Counter()
    post_counts: Counter[int] = Counter()
    action_counts: Counter[int] = Counter()
    step_counts: Counter[int] = Counter()
    action_type_counts: Counter[str] = Counter()
    text_lengths: list[int] = []
    empty_texts = 0
    duplicate_post_ids = 0
    duplicate_action_ids = 0
    post_action_mismatches = 0
    graph_reference_errors = 0
    invalid_steps = 0
    graph_edge_counts: list[int] = []
    record_count = 0

    for index, record in enumerate(iter_jsonl(args.master), start=1):
        record_count += 1
        missing = REQUIRED_MASTER_FIELDS - set(record)
        for field in sorted(missing):
            add_error(errors, f"record {index}: missing {field}")

        sample_id = str(record.get("sample_id", ""))
        base_id = str(record.get("base_profile_id", ""))
        label_name = str(record.get("label_name", ""))
        if sample_id in sample_ids:
            add_error(errors, f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        base_to_labels[base_id].add(label_name)
        base_to_records[base_id] += 1
        if record.get("split") in SPLITS:
            base_to_splits[base_id].add(str(record["split"]))
        if isinstance(record.get("profile"), dict):
            base_to_profile_digests[base_id].add(digest(record["profile"]))

        label = record.get("label")
        if label in LABELS and label_name != LABELS[label]:
            add_error(errors, f"record {index}: label/name mismatch {label!r}/{label_name!r}")
        label_counts[label_name] += 1
        schema_counts[str(record.get("platform_schema", "<missing>"))] += 1

        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            add_error(errors, f"record {index}: provenance is not an object")
            provenance = {}
        for field in PROVENANCE_FIELDS:
            if field not in provenance:
                provenance_missing[f"{label_name}:{field}"] += 1
        for field in REQUIRED_PROVENANCE_BY_LABEL.get(label_name, ()):
            if field not in provenance or provenance.get(field) in (None, ""):
                required_provenance_missing[f"{label_name}:{field}"] += 1
        source_counts[f"{label_name}|{provenance.get('source_dataset', '<missing>')}"] += 1
        policy_counts[f"{label_name}|{provenance.get('generation_policy', '<missing>')}"] += 1

        posts = record.get("posts")
        actions = record.get("actions")
        graph = record.get("graph")
        if not isinstance(posts, list):
            posts = []
        if not isinstance(actions, list):
            actions = []
        if not isinstance(graph, dict):
            graph = {}
        post_counts[len(posts)] += 1
        action_counts[len(actions)] += 1
        post_ids = []
        for post in posts:
            if not isinstance(post, dict):
                add_error(errors, f"record {index}: non-object post")
                continue
            post_ids.append(str(post.get("post_id", "")))
            text = str(post.get("text") or "")
            text_lengths.append(len(text))
            if not text.strip():
                empty_texts += 1
            step = int_or_none(post.get("time_step", post.get("step")))
            if step not in range(1, 6):
                invalid_steps += 1
            else:
                step_counts[step] += 1
        duplicate_post_ids += len(post_ids) - len(set(post_ids))

        action_ids = []
        for action in actions:
            if not isinstance(action, dict):
                add_error(errors, f"record {index}: non-object action")
                continue
            action_ids.append(str(action.get("action_id", "")))
            action_type = str(action.get("type", action.get("action", "<missing>"))).casefold()
            action_type_counts[action_type] += 1
            step = int_or_none(action.get("time_step", action.get("step")))
            if step not in range(1, 6):
                invalid_steps += 1
        duplicate_action_ids += len(action_ids) - len(set(action_ids))
        if len(posts) != len(actions):
            post_action_mismatches += abs(len(posts) - len(actions))
        for post, action in zip(posts, actions):
            if not isinstance(post, dict) or not isinstance(action, dict):
                continue
            post_step = int_or_none(post.get("time_step", post.get("step")))
            action_step = int_or_none(action.get("time_step", action.get("step")))
            post_type = str(post.get("action_type", post.get("type", ""))).casefold()
            action_type = str(action.get("type", action.get("action", ""))).casefold()
            post_targets = [str(value) for value in (post.get("interaction_targets") or [])]
            action_targets = [str(value) for value in (action.get("target_ids") or [])]
            if (
                post_step != action_step
                or str(post.get("created_at", "")) != str(action.get("created_at", ""))
                or post_type != action_type
                or post_targets != action_targets
            ):
                post_action_mismatches += 1

        edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
        node_ids = {
            str(node.get("id"))
            for node in (graph.get("nodes") or [])
            if isinstance(node, dict) and node.get("id") is not None
        }
        for edge in edges:
            if not isinstance(edge, dict):
                graph_reference_errors += 1
                continue
            if str(edge.get("source")) not in node_ids or str(edge.get("target")) not in node_ids:
                graph_reference_errors += 1
        graph_edge_counts.append(len(edges))

    incomplete = {
        base: sorted(labels)
        for base, labels in base_to_labels.items()
        if labels != set(LABELS.values())
    }
    mixed_split = {base: sorted(splits) for base, splits in base_to_splits.items() if len(splits) > 1}
    profile_mismatch = {
        base: len(digests) for base, digests in base_to_profile_digests.items() if len(digests) > 1
    }
    if incomplete:
        add_error(errors, f"incomplete paired profiles: {len(incomplete)}")
    if mixed_split:
        add_error(errors, f"base profiles assigned to multiple splits: {len(mixed_split)}")
    if profile_mismatch:
        add_error(errors, f"paired profile fields differ: {len(profile_mismatch)}")
    if duplicate_post_ids:
        add_error(errors, f"duplicate post IDs within records: {duplicate_post_ids}")
    if duplicate_action_ids:
        add_error(errors, f"duplicate action IDs within records: {duplicate_action_ids}")
    if post_action_mismatches:
        add_error(errors, f"post/action alignment mismatches: {post_action_mismatches}")
    if graph_reference_errors:
        add_error(errors, f"graph edges reference missing nodes: {graph_reference_errors}")
    if invalid_steps:
        add_error(errors, f"invalid time_step values: {invalid_steps}")
    if required_provenance_missing:
        add_error(errors, f"required provenance fields missing: {dict(required_provenance_missing)}")

    expected_post_bad = sum(count for count, n in post_counts.items() if count != args.expected_posts for _ in range(n))
    expected_action_bad = sum(count for count, n in action_counts.items() if count != args.expected_actions for _ in range(n))
    if expected_post_bad:
        add_error(errors, f"records with post count != {args.expected_posts}: {expected_post_bad}")
    if expected_action_bad:
        add_error(errors, f"records with action count != {args.expected_actions}: {expected_action_bad}")

    profile_sizes = Counter(base_to_records.values())
    return {
        "record_count": record_count,
        "base_profile_count": len(base_to_records),
        "label_counts": dict(sorted(label_counts.items())),
        "platform_schema_counts": dict(sorted(schema_counts.items())),
        "source_counts_audit_only": dict(sorted(source_counts.items())),
        "generation_policy_counts_audit_only": dict(sorted(policy_counts.items())),
        "missing_provenance_fields_by_label": dict(sorted(provenance_missing.items())),
        "missing_required_provenance_fields_by_label": dict(sorted(required_provenance_missing.items())),
        "paired_profile_check": {
            "expected_labels": list(LABELS.values()),
            "incomplete_profile_count": len(incomplete),
            "examples": dict(list(sorted(incomplete.items()))[:20]),
            "profile_field_mismatch_count": len(profile_mismatch),
            "records_per_profile": dict(sorted((str(k), v) for k, v in profile_sizes.items())),
        },
        "split_check_from_master": {
            "profiles_with_multiple_splits": len(mixed_split),
            "examples": dict(list(sorted(mixed_split.items()))[:20]),
        },
        "record_quality": {
            "post_count_distribution": dict(sorted((str(k), v) for k, v in post_counts.items())),
            "action_count_distribution": dict(sorted((str(k), v) for k, v in action_counts.items())),
            "invalid_time_step_count": invalid_steps,
            "duplicate_post_ids_within_record": duplicate_post_ids,
            "duplicate_action_ids_within_record": duplicate_action_ids,
            "post_action_alignment_mismatches": post_action_mismatches,
            "graph_edge_reference_errors": graph_reference_errors,
            "empty_post_text_count": empty_texts,
            "post_text_length": {
                "min": min(text_lengths) if text_lengths else None,
                "median": median(text_lengths) if text_lengths else None,
                "mean": mean(text_lengths) if text_lengths else None,
                "max": max(text_lengths) if text_lengths else None,
            },
            "time_step_counts": dict(sorted((str(k), v) for k, v in step_counts.items())),
            "action_type_counts": dict(sorted(action_type_counts.items())),
            "graph_edge_count": {
                "min": min(graph_edge_counts) if graph_edge_counts else None,
                "median": median(graph_edge_counts) if graph_edge_counts else None,
                "mean": mean(graph_edge_counts) if graph_edge_counts else None,
                "max": max(graph_edge_counts) if graph_edge_counts else None,
            },
        },
        "errors": errors,
        "warnings": warnings,
    }


def scan_forbidden_keys(value: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if FORBIDDEN_MODEL_KEYS.search(str(key)):
                hits.append(child_path)
            hits.extend(scan_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(scan_forbidden_keys(child, f"{path}[{index}]"))
    return hits


def audit_views(views_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(views_root), "views": {}, "errors": [], "warnings": []}
    # Accept both the canonical final-release names and the lower-case names
    # emitted by older pilot artifacts, without treating them as different data.
    view_specs = {
        "Twibot22": (("Twibot22", "quadbot_3class"), {0, 1, 2}),
        "Quadbot": (("Quadbot", "quadbot_4class"), {0, 1, 2, 3}),
    }
    all_view_bases: dict[str, set[str]] = {}
    for view, (allowed_names, expected_labels) in view_specs.items():
        view_dir = next((views_root / name for name in allowed_names if (views_root / name).exists()), views_root / allowed_names[0])
        profile_splits: dict[str, set[str]] = defaultdict(set)
        profile_labels: dict[str, set[int]] = defaultdict(set)
        rows = 0
        labels: Counter[int] = Counter()
        model_key_hits: list[str] = []
        for split in SPLITS:
            path = view_dir / f"{split}.jsonl"
            if not path.exists():
                result["warnings"].append(f"missing view file: {path}")
                continue
            for row in iter_jsonl(path):
                rows += 1
                label = int_or_none(row.get("label"))
                labels[label] += 1  # type: ignore[index]
                if label not in expected_labels:
                    result["errors"].append(f"{view}/{split}: disallowed label {label}")
                base = str(row.get("base_profile_id", ""))
                profile_splits[base].add(split)
                profile_labels[base].add(label)  # type: ignore[arg-type]
                actual_fields = set(row)
                if actual_fields != MODEL_VIEW_FIELDS:
                    result["errors"].append(
                        f"{view}/{split}: model schema mismatch missing={sorted(MODEL_VIEW_FIELDS - actual_fields)} "
                        f"extra={sorted(actual_fields - MODEL_VIEW_FIELDS)}"
                    )
                if len(model_key_hits) < 20:
                    model_key_hits.extend(scan_forbidden_keys(row))
        all_view_bases[view] = set(profile_splits)
        cross_split = {k: sorted(v) for k, v in profile_splits.items() if len(v) > 1}
        result["views"][view] = {
            "rows": rows,
            "label_counts": dict(sorted((str(k), v) for k, v in labels.items())),
            "profiles": len(profile_splits),
            "cross_split_profile_count": len(cross_split),
            "cross_split_examples": dict(list(sorted(cross_split.items()))[:20]),
            "profiles_with_multiple_labels": sum(len(v) > 1 for v in profile_labels.values()),
            "forbidden_model_key_examples": sorted(set(model_key_hits))[:20],
            "path_used": str(view_dir),
        }
        if cross_split:
            result["errors"].append(f"{view}: {len(cross_split)} profiles cross split")
        if model_key_hits:
            result["errors"].append(f"{view}: model-facing records contain audit/provenance keys")

    if all_view_bases.get("Twibot22") and all_view_bases.get("Quadbot"):
        result["view_nesting"] = {
            "three_class_profiles_subset_of_four_class": all_view_bases["Twibot22"] <= all_view_bases["Quadbot"],
            "three_class_profile_count": len(all_view_bases["Twibot22"]),
            "four_class_profile_count": len(all_view_bases["Quadbot"]),
        }
        if not result["view_nesting"]["three_class_profiles_subset_of_four_class"]:
            result["errors"].append("Twibot22 profiles are not a subset of Quadbot profiles")
    return result


def audit_features(
    features_root: Path,
    feature_audit_path: Optional[Path],
    expected_k: int,
    views_root: Optional[Path] = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": str(features_root),
        "protocol": {
            "edge_semantics": "split-local directed cosine kNN over exported graph vectors",
            "k": expected_k,
            "uses_labels": False,
            "uses_provenance": False,
            "cross_split_edges_allowed": False,
            "same_base_profile_edges_allowed": False,
            "exclude_same_base_profile": True,
        },
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    view_name: Optional[str] = None
    if views_root:
        name = features_root.name.casefold()
        if "3class" in name or name in {"twibot22", "quadbot_3class"}:
            view_name = next(
                (candidate for candidate in ("Twibot22", "quadbot_3class") if (views_root / candidate).exists()),
                "Twibot22",
            )
        elif "4class" in name or name in {"quadbot", "quadbot_4class"}:
            view_name = next(
                (candidate for candidate in ("Quadbot", "quadbot_4class") if (views_root / candidate).exists()),
                "Quadbot",
            )
        else:
            result["warnings"].append(
                "cannot infer the corresponding model view from feature directory name; "
                "feature-to-view identity alignment was skipped"
            )
    if view_name:
        result["view_alignment"] = {"view_name": view_name, "splits": {}}
    audit_path = feature_audit_path or (features_root / "feature_export_audit.json")
    if audit_path.exists():
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            result["feature_export_audit"] = {
                "schema_version": payload.get("schema_version"),
                "dimensions": payload.get("dimensions"),
                "rules": payload.get("rules"),
            }
            rules = payload.get("rules") or {}
            if "knn_k" not in rules and "k" not in rules:
                result["warnings"].append("feature_export_audit.json does not record kNN k")
            declared_exclusion = rules.get("knn_exclude_same_base_profile") is True
            result["protocol"]["exclude_same_base_profile"] = declared_exclusion
            if not declared_exclusion:
                result["errors"].append(
                    "feature_export_audit.json does not declare "
                    "knn_exclude_same_base_profile=true"
                )
        except (OSError, json.JSONDecodeError) as exc:
            result["errors"].append(f"cannot read feature audit: {exc}")
    else:
        result["errors"].append(f"missing feature export audit: {audit_path}")

    try:
        import torch  # type: ignore
    except ImportError:
        result["warnings"].append("torch unavailable; edge-index checks skipped")
        return result

    for split in SPLITS:
        edge_path = features_root / "graph" / f"{split}_edge_index.pt"
        meta_path = features_root / f"{split}_metadata.jsonl"
        if not edge_path.exists():
            result["warnings"].append(f"missing edge index: {edge_path}")
            continue
        try:
            edge = torch.load(edge_path, map_location="cpu", weights_only=True)
            shape = tuple(int(x) for x in edge.shape)
            src = edge[0].tolist() if shape and shape[0] == 2 else []
            dst = edge[1].tolist() if shape and shape[0] == 2 else []
            degrees = Counter(src)
            self_loops = sum(a == b for a, b in zip(src, dst))
            invalid_indices = sum(
                1
                for a, b in zip(src, dst)
                if int(a) < 0 or int(b) < 0
            )
            split_report: dict[str, Any] = {
                "shape": list(shape),
                "node_count_inferred": len(degrees),
                "edge_count": len(src),
                "out_degree_min": min(degrees.values()) if degrees else 0,
                "out_degree_max": max(degrees.values()) if degrees else 0,
                "out_degree_values": sorted(set(degrees.values())),
                "self_loop_count": self_loops,
                "negative_index_count": invalid_indices,
            }
            if meta_path.exists() and shape and shape[0] == 2:
                metadata = list(iter_jsonl(meta_path))
                base_ids = [str(row.get("base_profile_id", "")) for row in metadata]
                out_of_range = sum(
                    1
                    for a, b in zip(src, dst)
                    if int(a) >= len(base_ids) or int(b) >= len(base_ids)
                )
                split_report["out_of_range_index_count"] = out_of_range
                same_base = sum(
                    1 for a, b in zip(src, dst)
                    if 0 <= a < len(base_ids) and 0 <= b < len(base_ids) and base_ids[a] == base_ids[b]
                )
                split_report["metadata_rows"] = len(metadata)
                split_report["same_base_profile_edge_count"] = same_base
                split_report["same_base_profile_edge_fraction"] = (same_base / len(src)) if src else 0.0
                if same_base:
                    result["errors"].append(
                        f"{split}: {same_base} kNN edges connect paired variants sharing base_profile_id"
                    )
            elif not meta_path.exists():
                result["errors"].append(
                    f"{split}: missing metadata required to audit same-base-profile edges: {meta_path}"
                )
            if view_name and views_root:
                view_path = views_root / view_name / f"{split}.jsonl"
                if not view_path.exists():
                    result["errors"].append(f"missing model view for feature alignment: {view_path}")
                elif meta_path.exists():
                    view_rows = list(iter_jsonl(view_path))
                    metadata_rows = list(iter_jsonl(meta_path))
                    view_identity = [
                        (str(row.get("sample_id")), str(row.get("base_profile_id")), int_or_none(row.get("label")))
                        for row in view_rows
                    ]
                    feature_identity = [
                        (str(row.get("sample_id")), str(row.get("base_profile_id")), int_or_none(row.get("label")))
                        for row in metadata_rows
                    ]
                    aligned = view_identity == feature_identity
                    result["view_alignment"]["splits"][split] = {
                        "view_rows": len(view_rows),
                        "feature_rows": len(metadata_rows),
                        "identity_and_order_match": aligned,
                    }
                    if not aligned:
                        result["errors"].append(
                            f"{split}: feature metadata identities/order do not match {view_path}"
                        )
            if split_report["self_loop_count"]:
                result["errors"].append(f"{split}: kNN edge index contains self loops")
            if split_report["negative_index_count"] or split_report.get("out_of_range_index_count", 0):
                result["errors"].append(
                    f"{split}: kNN edge index contains invalid node indices"
                )
            if split_report["out_degree_values"] and split_report["out_degree_values"] != [expected_k]:
                result["warnings"].append(
                    f"{split}: out degree values {split_report['out_degree_values']} differ from k={expected_k}"
                )
            result["splits"][split] = split_report
        except Exception as exc:  # torch versions differ in safe-loading support
            result["errors"].append(f"cannot audit {edge_path}: {exc}")
    return result


def metadata_probe_contract() -> dict[str, Any]:
    return {
        "name": "metadata_only",
        "purpose": "measure label predictability from scalar metadata without text/graph embeddings or provenance",
        "current_release_input": [
            "profile.public_metrics.followers_count",
            "profile.public_metrics.following_count",
            "profile.public_metrics.tweet_count",
            "profile.public_metrics.listed_count",
            "log1p(number_of_posts)",
            "log1p(number_of_actions)",
        ],
        "current_shortcut_audit_input": [
            "numerical tensor (6 dimensions)",
            "llm_style tensor (11 dimensions)",
        ],
        "excluded": [
            "label and label_name",
            "sample_id, base_profile_id, split",
            "provenance, source_dataset, source_id",
            "generator/policy/checkpoint/seed fields",
            "post text, profile description, graph node/edge identifiers",
        ],
        "implementation_reference": "audit_shortcuts_20260924.py::metadata_probe",
        "interpretation": "high scores are evidence of remaining distributional shortcut, not proof of label leakage",
    }


def main() -> None:
    args = parse_args()
    master_report = collect_master(args)
    report: dict[str, Any] = {
        "schema_version": "quadbot-v3-release-contract-audit-1",
        "audit_date": "2026-09-25",
        "master": {"path": str(args.master), **master_report},
        "metadata_only_probe_contract": metadata_probe_contract(),
    }
    if args.views_root:
        report["views"] = audit_views(args.views_root)
    if args.features_root:
        report["features"] = audit_features(
            args.features_root,
            args.feature_audit,
            args.knn_k,
            args.views_root,
        )

    errors: list[str] = list(master_report["errors"])
    errors.extend((report.get("views") or {}).get("errors", []))
    errors.extend((report.get("features") or {}).get("errors", []))
    warnings: list[str] = list(master_report["warnings"])
    warnings.extend((report.get("views") or {}).get("warnings", []))
    warnings.extend((report.get("features") or {}).get("warnings", []))
    report["summary"] = {
        "passed": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "profile_level_effective_n": master_report["base_profile_count"],
        "trajectory_level_n": master_report["record_count"],
        "profile_cluster_bootstrap_required": master_report["record_count"] > master_report["base_profile_count"],
    }
    report["errors"] = errors[:200]
    report["warnings"] = warnings[:200]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
