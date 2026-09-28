#!/usr/bin/env python3
"""Validate a QuadBot-v3 master manifest and derive safe 3/4-class views."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LABELS = {
    0: "human",
    1: "traditional_bot",
    2: "llm_bot",
    3: "full_stack_agent",
}
REQUIRED_FIELDS = {
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

# Model-facing records use an explicit common schema. Master-manifest fields that
# document collection or generation remain available only under audit_only/.
MODEL_FIELDS = (
    "sample_id",
    "base_profile_id",
    "label",
    "split",
    "profile",
    "posts",
    "actions",
    "graph",
)
PROFILE_FIELDS = (
    "display_name",
    "description",
    "location",
    "created_at",
    "language",
    "public_metrics",
)
PUBLIC_METRIC_FIELDS = (
    "followers_count",
    "following_count",
    "tweet_count",
    "listed_count",
)
POST_FIELDS = (
    "post_id",
    "time_step",
    "created_at",
    "text",
    "action_type",
    "interaction_targets",
)
ACTION_FIELDS = (
    "action_id",
    "time_step",
    "created_at",
    "type",
    "target_ids",
)
GRAPH_FIELDS = ("nodes", "edges")
NODE_FIELDS = ("id", "type")
EDGE_FIELDS = ("source", "target", "type", "time_step")
LIST_ITEM_FIELDS = {
    ("posts",): POST_FIELDS,
    ("actions",): ACTION_FIELDS,
    ("graph", "nodes"): NODE_FIELDS,
    ("graph", "edges"): EDGE_FIELDS,
}

VIEW_LABELS = {
    "quadbot_4class": set(LABELS),
    "quadbot_3class": {0, 1, 2},
}
SPLITS = ("train", "val", "test")
OPAQUE_ID_NAMESPACE = "quadbot-v3-release-id-v1"
OPAQUE_ID_PATTERN = re.compile(r"^id_[0-9a-f]{24}$")
KEY_MARKER_PATTERN = re.compile(
    r"label|source|provenance|qwen|oasis|twibot|generator|grounding|environment_triggered",
    re.IGNORECASE,
)
ALLOWED_MARKER_KEY_PATHS = {
    ("label",),
    ("graph", "edges", "[]", "source"),
}
FORBIDDEN_STRING_PATTERNS = (
    ("traditional_bot label", re.compile(r"\btraditional[ _-]?bot\b", re.IGNORECASE)),
    ("llm_bot label", re.compile(r"\bllm[ _-]?bot\b", re.IGNORECASE)),
    (
        "full_stack_agent label",
        re.compile(r"\bfull[ _-]?stack[ _-]?agent\b", re.IGNORECASE),
    ),
    ("Qwen source", re.compile(r"\bqwen(?:\d+(?:\.\d+)?)?\b", re.IGNORECASE)),
    ("OASIS source", re.compile(r"\boasis\b", re.IGNORECASE)),
    ("TwiBot source", re.compile(r"\btwi[ _-]?bot(?:[ _-]?22)?\b", re.IGNORECASE)),
    (
        "source_dataset marker",
        re.compile(r"\bsource[ _-]?dataset\b", re.IGNORECASE),
    ),
    (
        "controlled-generation marker",
        re.compile(r"\bcontrolled[ _-]?generation\b", re.IGNORECASE),
    ),
)
EXACT_LABEL_STRINGS = {
    "human",
    "traditional bot",
    "llm bot",
    "full stack agent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", default="quadbot-v3-split-2026")
    parser.add_argument(
        "--allow-incomplete-groups",
        action="store_true",
        help="Allow base profiles without all four paired labels.",
    )
    parser.add_argument(
        "--three-class-name",
        default="quadbot_3class",
        help="Output directory name for the three-class view.",
    )
    parser.add_argument(
        "--four-class-name",
        default="quadbot_4class",
        help="Output directory name for the four-class view.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        if not all(isinstance(item, dict) for item in parsed):
            raise ValueError(f"JSON array in {path} must contain only objects")
        return parsed

    records = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def stable_group_order(base_profile_id: str, seed: str) -> tuple[bytes, str]:
    """Return a deterministic ordering key for profile-level splitting."""
    digest = hashlib.sha256(f"{seed}:{base_profile_id}".encode()).digest()
    return digest, base_profile_id


def build_split_map(records: list[dict], seed: str) -> dict[str, str]:
    """Assign complete base-profile groups to an exact 2/3, 1/6, 1/6 split."""
    base_ids = sorted(
        {str(record["base_profile_id"]) for record in records},
        key=lambda base_id: stable_group_order(base_id, seed),
    )
    profile_count = len(base_ids)
    train_count = (profile_count * 2) // 3
    val_count = (profile_count - train_count) // 2
    split_map = {}
    for index, base_id in enumerate(base_ids):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        split_map[base_id] = split
    return split_map


def opaque_id(namespace: str, value: object) -> str:
    if value is None or value == "":
        raise ValueError(f"Cannot create an opaque {namespace} ID from {value!r}")
    payload = f"{OPAQUE_ID_NAMESPACE}\0{namespace}\0{value}".encode("utf-8")
    return f"id_{hashlib.sha256(payload).hexdigest()[:24]}"


def validate(records: list[dict], allow_incomplete: bool) -> dict:
    errors = []
    sample_ids = set()
    groups = defaultdict(list)

    for index, record in enumerate(records, start=1):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            errors.append(f"record {index}: missing fields {sorted(missing)}")
            continue

        sample_id = str(record["sample_id"])
        if sample_id in sample_ids:
            errors.append(f"record {index}: duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)

        label = record["label"]
        if label not in LABELS:
            errors.append(f"record {index}: invalid label {label!r}")
        elif record["label_name"] != LABELS[label]:
            errors.append(
                f"record {index}: label_name {record['label_name']!r} does not match {LABELS[label]!r}"
            )

        if not isinstance(record["profile"], dict):
            errors.append(f"record {index}: profile must be an object")
        if not isinstance(record["posts"], list):
            errors.append(f"record {index}: posts must be a list")
        if not isinstance(record["actions"], list):
            errors.append(f"record {index}: actions must be a list")
        if not isinstance(record["graph"], dict):
            errors.append(f"record {index}: graph must be an object")

        groups[str(record["base_profile_id"])].append(record)

    incomplete = {}
    duplicate_labels = {}
    expected = set(LABELS)
    for base_id, group in groups.items():
        labels = [item.get("label") for item in group]
        if set(labels) != expected:
            incomplete[base_id] = sorted(set(labels), key=str)
        repeated = [label for label, count in Counter(labels).items() if count > 1]
        if repeated:
            duplicate_labels[base_id] = repeated

    if duplicate_labels:
        errors.append(f"{len(duplicate_labels)} base profiles contain duplicate labels")
    if incomplete and not allow_incomplete:
        errors.append(f"{len(incomplete)} base profiles do not contain all four labels")
    if errors:
        preview = "\n".join(errors[:25])
        raise ValueError(f"Manifest validation failed with {len(errors)} error(s):\n{preview}")

    return {
        "record_count": len(records),
        "base_profile_count": len(groups),
        "incomplete_group_count": len(incomplete),
        "incomplete_group_examples": dict(list(sorted(incomplete.items()))[:20]),
    }


def normalize_profile(profile: dict) -> dict:
    metrics = profile.get("public_metrics") or {}
    if not isinstance(metrics, dict):
        raise ValueError("profile.public_metrics must be an object")
    return {
        "display_name": profile.get("display_name", ""),
        "description": profile.get("description", ""),
        "location": profile.get("location", ""),
        "created_at": profile.get("created_at", ""),
        "language": profile.get("language", ""),
        "public_metrics": {
            field: metrics.get(field) for field in PUBLIC_METRIC_FIELDS
        },
    }


def normalize_posts(posts: list[dict]) -> list[dict]:
    output = []
    for index, post in enumerate(posts):
        if not isinstance(post, dict):
            raise ValueError(f"posts[{index}] must be an object")
        targets = post.get("interaction_targets") or []
        if not isinstance(targets, list):
            raise ValueError(f"posts[{index}].interaction_targets must be a list")
        output.append(
            {
                "post_id": opaque_id("post", post.get("post_id")),
                "time_step": post.get("time_step"),
                "created_at": post.get("created_at", ""),
                "text": post.get("text", ""),
                "action_type": post.get("action_type", ""),
                "interaction_targets": [
                    opaque_id("entity", target) for target in targets
                ],
            }
        )
    return output


def normalize_actions(actions: list[dict]) -> list[dict]:
    output = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError(f"actions[{index}] must be an object")
        targets = action.get("target_ids") or []
        if not isinstance(targets, list):
            raise ValueError(f"actions[{index}].target_ids must be a list")
        output.append(
            {
                "action_id": opaque_id("action", action.get("action_id")),
                "time_step": action.get("time_step"),
                "created_at": action.get("created_at", ""),
                "type": action.get("type", ""),
                "target_ids": [opaque_id("entity", target) for target in targets],
            }
        )
    return output


def normalize_graph(graph: dict) -> dict:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("graph.nodes and graph.edges must be lists")

    normalized_nodes = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"graph.nodes[{index}] must be an object")
        normalized_nodes.append(
            {
                "id": opaque_id("entity", node.get("id")),
                "type": node.get("type", ""),
            }
        )

    normalized_edges = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ValueError(f"graph.edges[{index}] must be an object")
        normalized_edges.append(
            {
                "source": opaque_id("entity", edge.get("source")),
                "target": opaque_id("entity", edge.get("target")),
                "type": edge.get("type", ""),
                "time_step": edge.get("time_step"),
            }
        )
    return {"nodes": normalized_nodes, "edges": normalized_edges}


def model_record(record: dict, split: str) -> dict:
    return {
        "sample_id": opaque_id("sample", record["sample_id"]),
        "base_profile_id": opaque_id("base_profile", record["base_profile_id"]),
        "label": record["label"],
        "split": split,
        "profile": normalize_profile(record["profile"]),
        "posts": normalize_posts(record["posts"]),
        "actions": normalize_actions(record["actions"]),
        "graph": normalize_graph(record["graph"]),
    }


def write_audit_only_master(
    records: list[dict],
    output_root: Path,
    input_path: Path,
    view_names: dict[str, str] | None = None,
) -> dict:
    view_names = view_names or {
        "quadbot_3class": "quadbot_3class",
        "quadbot_4class": "quadbot_4class",
    }
    audit_dir = output_root / "audit_only"
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = audit_dir / "master_manifest.audit-only.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    notice_path = audit_dir / "AUDIT_ONLY_DO_NOT_TRAIN.json"
    notice = {
        "schema_version": "quadbot-v3-audit-only-notice-1",
        "manifest_role": "audit-only",
        "excluded_from_training": True,
        "contains_provenance_and_source_metadata": True,
        "source_input": str(input_path),
        "model_facing_files": {
            "three_class": f"../{view_names['quadbot_3class']}/{{train,val,test}}.jsonl",
            "four_class": f"../{view_names['quadbot_4class']}/{{train,val,test}}.jsonl",
        },
    }
    notice_path.write_text(
        json.dumps(notice, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "manifest": str(manifest_path),
        "notice": str(notice_path),
        "excluded_from_training": True,
    }


def write_views(
    records: list[dict],
    output_root: Path,
    seed: str,
    view_names: dict[str, str] | None = None,
) -> dict:
    view_names = view_names or {
        "quadbot_3class": "quadbot_3class",
        "quadbot_4class": "quadbot_4class",
    }
    counters = defaultdict(Counter)
    split_map = build_split_map(records, seed)
    streams = {}
    try:
        for view_key in VIEW_LABELS:
            view_name = view_names[view_key]
            view_dir = output_root / view_name
            view_dir.mkdir(parents=True, exist_ok=True)
            for split in SPLITS:
                streams[(view_key, split)] = (view_dir / f"{split}.jsonl").open(
                    "w", encoding="utf-8"
                )

        for record in records:
            split = split_map[str(record["base_profile_id"])]
            for view_key, allowed_labels in VIEW_LABELS.items():
                if record["label"] not in allowed_labels:
                    continue
                output = model_record(record, split)
                streams[(view_key, split)].write(
                    json.dumps(output, ensure_ascii=False) + "\n"
                )
                counters[view_names[view_key]][f"{split}:{record['label_name']}"] += 1
    finally:
        for stream in streams.values():
            stream.close()

    return {view: dict(sorted(counts.items())) for view, counts in counters.items()}


def format_path(path: tuple[str, ...]) -> str:
    output = ""
    for part in path:
        if part == "[]":
            output += "[]"
        else:
            output += ("." if output else "") + part
    return output


def collect_key_paths(value: object, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = path + (str(key),)
            paths.add(key_path)
            paths.update(collect_key_paths(child, key_path))
    elif isinstance(value, list):
        for field in LIST_ITEM_FIELDS.get(path, ()):
            paths.add(path + ("[]", field))
        for child in value:
            paths.update(collect_key_paths(child, path + ("[]",)))
    return paths


def scan_forbidden_markers(
    value: object, path: tuple[str, ...], errors: list[str]
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = path + (str(key),)
            if KEY_MARKER_PATTERN.search(str(key)) and key_path not in ALLOWED_MARKER_KEY_PATHS:
                errors.append(f"forbidden marker in key {format_path(key_path)}")
            scan_forbidden_markers(child, key_path, errors)
        return
    if isinstance(value, list):
        for child in value:
            scan_forbidden_markers(child, path + ("[]",), errors)
        return
    if not isinstance(value, str):
        return

    normalized = re.sub(r"[ _-]+", " ", value.strip().casefold())
    if normalized in EXACT_LABEL_STRINGS:
        errors.append(f"exact class label string at {format_path(path)}")
    for name, pattern in FORBIDDEN_STRING_PATTERNS:
        if pattern.search(value):
            errors.append(f"{name} at {format_path(path)}")


def expect_keys(
    value: object, expected: tuple[str, ...], path: str, errors: list[str]
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return
    actual = set(value)
    if actual != set(expected):
        errors.append(
            f"{path} keys differ: missing={sorted(set(expected) - actual)} "
            f"extra={sorted(actual - set(expected))}"
        )


def check_opaque_id(value: object, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not OPAQUE_ID_PATTERN.fullmatch(value):
        errors.append(f"{path} is not an opaque ID: {value!r}")


def check_model_record(record: dict, expected_split: str, path: str) -> list[str]:
    errors: list[str] = []
    expect_keys(record, MODEL_FIELDS, path, errors)
    if errors:
        return errors

    if record["split"] != expected_split:
        errors.append(f"{path}.split is {record['split']!r}, expected {expected_split!r}")
    if record["label"] not in LABELS:
        errors.append(f"{path}.label is invalid: {record['label']!r}")

    check_opaque_id(record["sample_id"], f"{path}.sample_id", errors)
    check_opaque_id(record["base_profile_id"], f"{path}.base_profile_id", errors)
    expect_keys(record["profile"], PROFILE_FIELDS, f"{path}.profile", errors)
    if isinstance(record["profile"], dict):
        expect_keys(
            record["profile"].get("public_metrics"),
            PUBLIC_METRIC_FIELDS,
            f"{path}.profile.public_metrics",
            errors,
        )

    for index, post in enumerate(record["posts"]):
        item_path = f"{path}.posts[{index}]"
        expect_keys(post, POST_FIELDS, item_path, errors)
        if not isinstance(post, dict):
            continue
        check_opaque_id(post.get("post_id"), f"{item_path}.post_id", errors)
        targets = post.get("interaction_targets")
        if not isinstance(targets, list):
            errors.append(f"{item_path}.interaction_targets must be a list")
        else:
            for target_index, target in enumerate(targets):
                check_opaque_id(
                    target,
                    f"{item_path}.interaction_targets[{target_index}]",
                    errors,
                )

    for index, action in enumerate(record["actions"]):
        item_path = f"{path}.actions[{index}]"
        expect_keys(action, ACTION_FIELDS, item_path, errors)
        if not isinstance(action, dict):
            continue
        check_opaque_id(action.get("action_id"), f"{item_path}.action_id", errors)
        targets = action.get("target_ids")
        if not isinstance(targets, list):
            errors.append(f"{item_path}.target_ids must be a list")
        else:
            for target_index, target in enumerate(targets):
                check_opaque_id(
                    target, f"{item_path}.target_ids[{target_index}]", errors
                )

    expect_keys(record["graph"], GRAPH_FIELDS, f"{path}.graph", errors)
    if isinstance(record["graph"], dict):
        for index, node in enumerate(record["graph"].get("nodes", [])):
            item_path = f"{path}.graph.nodes[{index}]"
            expect_keys(node, NODE_FIELDS, item_path, errors)
            if isinstance(node, dict):
                check_opaque_id(node.get("id"), f"{item_path}.id", errors)
        for index, edge in enumerate(record["graph"].get("edges", [])):
            item_path = f"{path}.graph.edges[{index}]"
            expect_keys(edge, EDGE_FIELDS, item_path, errors)
            if isinstance(edge, dict):
                check_opaque_id(edge.get("source"), f"{item_path}.source", errors)
                check_opaque_id(edge.get("target"), f"{item_path}.target", errors)

    scan_forbidden_markers(record, (), errors)
    return errors


def self_check_views(output_root: Path, view_names: dict[str, str] | None = None) -> dict:
    view_names = view_names or {
        "quadbot_3class": "quadbot_3class",
        "quadbot_4class": "quadbot_4class",
    }
    errors = []
    files = {}
    presence_by_view: dict[str, dict[int, Counter]] = {}
    totals_by_view: dict[str, Counter] = {}

    for view_key, allowed_labels in VIEW_LABELS.items():
        view_name = view_names[view_key]
        presence_by_label: dict[int, Counter] = defaultdict(Counter)
        label_totals: Counter = Counter()
        sample_ids = set()
        for split in SPLITS:
            path = output_root / view_name / f"{split}.jsonl"
            records = load_jsonl(path)
            files[str(path)] = len(records)
            for index, record in enumerate(records, start=1):
                record_path = f"{path}:{index}"
                record_errors = check_model_record(record, split, record_path)
                errors.extend(record_errors)
                label = record.get("label")
                if label not in allowed_labels:
                    errors.append(f"{record_path}: label {label!r} is not allowed in {view_name}")
                    continue
                sample_id = record.get("sample_id")
                if sample_id in sample_ids:
                    errors.append(f"{record_path}: duplicate sample_id {sample_id!r}")
                sample_ids.add(sample_id)
                label_totals[label] += 1
                for key_path in collect_key_paths(record):
                    presence_by_label[label][key_path] += 1
        presence_by_view[view_name] = presence_by_label
        totals_by_view[view_name] = label_totals

    key_presence_rates = {}
    for view_name, presence_by_label in presence_by_view.items():
        label_totals = totals_by_view[view_name]
        paths = sorted(
            {
                path
                for label_presence in presence_by_label.values()
                for path in label_presence
            }
        )
        view_rates = {}
        for key_path in paths:
            rates = {
                LABELS[label]: presence_by_label[label][key_path] / total
                for label, total in sorted(label_totals.items())
                if total
            }
            if len({round(rate, 12) for rate in rates.values()}) > 1:
                errors.append(
                    f"{view_name}: class-dependent key presence for "
                    f"{format_path(key_path)}: {rates}"
                )
            view_rates[format_path(key_path)] = rates
        key_presence_rates[view_name] = view_rates

    if errors:
        preview = "\n".join(errors[:40])
        raise ValueError(
            f"Generated-view leakage self-check failed with {len(errors)} error(s):\n{preview}"
        )
    return {
        "status": "passed",
        "files": files,
        "opaque_id_format": OPAQUE_ID_PATTERN.pattern,
        "forbidden_key_and_string_scan": "passed",
        "class_key_presence_rates": key_presence_rates,
    }


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    validation = validate(records, args.allow_incomplete_groups)
    args.output_root.mkdir(parents=True, exist_ok=True)
    view_names = {
        "quadbot_3class": args.three_class_name,
        "quadbot_4class": args.four_class_name,
    }
    audit_only = write_audit_only_master(records, args.output_root, args.input, view_names)
    counts = write_views(records, args.output_root, args.seed, view_names)
    leakage_self_check = self_check_views(args.output_root, view_names)
    audit = {
        "schema_version": "quadbot-v3-audit-2",
        "input": str(args.input),
        "split_seed": args.seed,
        "validation": validation,
        "counts": counts,
        "split_group_counts": dict(
            sorted(Counter(build_split_map(records, args.seed).values()).items())
        ),
        "leakage_self_check": leakage_self_check,
        "audit_only_master": audit_only,
        "input_contract": {
            "master_manifest_role": "audit-only",
            "provenance_is_written_to_model_files": False,
            "source_metadata_is_written_to_model_files": False,
            "model_fields": list(MODEL_FIELDS),
            "ids_are_stable_opaque_hashes": True,
            "split_is_grouped_by": "base_profile_id",
            "three_class_labels": [LABELS[i] for i in (0, 1, 2)],
            "four_class_labels": [LABELS[i] for i in (0, 1, 2, 3)],
        },
    }
    (args.output_root / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
