"""Training-only aggregate audit of simulator fields in the legacy exporter."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def training_identities(path):
    if path.name != "train_metadata.jsonl":
        raise ValueError("only the frozen original training metadata is permitted")
    identities, profiles = {}, {}
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            identity = str(item["sample_id"])
            profile = str(item["base_profile_id"])
            label = item["label"]
            if (item["row_index"] != index or identity in identities
                    or type(label) is not int or label not in range(4)):
                raise ValueError("invalid or duplicate training identity")
            identities[identity] = (profile, label)
            profiles.setdefault(profile, []).append(label)
    if not profiles or any(sorted(labels) != list(range(4)) for labels in profiles.values()):
        raise ValueError("training profiles must have exactly four labelled trajectories")
    return identities


def new_statistics():
    return {
        "trajectories": 0, "post_counts": Counter(), "action_counts": Counter(),
        "trigger_true_counts": Counter(), "trigger_present_counts": Counter(),
        "trigger_value_kinds": Counter(), "nonempty_post_text_counts": Counter(),
        "post_field_names": set(), "action_field_names": set(),
    }


def update_statistics(statistics, row):
    posts, actions = row.get("posts") or [], row.get("actions") or []
    if not isinstance(posts, list) or not isinstance(actions, list):
        raise ValueError("post and action collections must be lists")
    statistics["trajectories"] += 1
    statistics["post_counts"][len(posts)] += 1
    statistics["action_counts"][len(actions)] += 1
    true, present = 0, 0
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("action must be an object")
        statistics["action_field_names"].update(action)
        if "environment_triggered" in action:
            present += 1
            value = action["environment_triggered"]
            kind = ("boolean_true" if value else "boolean_false") if type(value) is bool else type(value).__name__
            statistics["trigger_value_kinds"][kind] += 1
        true += int(bool(action.get("environment_triggered", False)))
    for post in posts:
        if not isinstance(post, dict):
            raise ValueError("post must be an object")
        statistics["post_field_names"].update(post)
    statistics["trigger_true_counts"][true] += 1
    statistics["trigger_present_counts"][present] += 1
    statistics["nonempty_post_text_counts"][
        sum(bool(" ".join(str(post.get("text") or "").split())) for post in posts)
    ] += 1


def finish_statistics(statistics):
    return {name: (
        [{"value": key, "trajectories" if name != "trigger_value_kinds" else "actions": count}
         for key, count in sorted(value.items())] if isinstance(value, Counter)
        else sorted(value) if isinstance(value, set) else value
    ) for name, value in statistics.items()}


def audit(source, metadata, expected_source_sha256, expected_metadata_sha256):
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != expected_metadata_sha256:
        raise ValueError("frozen training metadata hash differs")
    identities = training_identities(metadata)
    seen, by_class, hasher = set(), {label: new_statistics() for label in range(4)}, hashlib.sha256()
    with source.open("rb") as stream:
        for line in stream:
            hasher.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get("sample_id", ""))
            if identity not in identities:
                continue
            if identity in seen:
                raise ValueError("duplicate selected training row in source")
            profile, label = identities[identity]
            if (str(row["base_profile_id"]) != profile or row["label"] != label):
                raise ValueError("source and training metadata identities differ")
            seen.add(identity)
            update_statistics(by_class[label], row)
    if hasher.hexdigest() != expected_source_sha256:
        raise ValueError("frozen source hash differs")
    if seen != set(identities):
        raise ValueError("training identities missing from source")
    return {
        "schema": "training-observable-schema-audit-v1",
        "scope": "original_training_rows_only_aggregate_schema_no_model_evaluation",
        "training_source_sha256": expected_source_sha256,
        "training_metadata_sha256": expected_metadata_sha256,
        "training_rows": len(seen),
        "training_profiles": len({profile for profile, _ in identities.values()}),
        "classes": {str(label): finish_statistics(values) for label, values in by_class.items()},
        "legacy_action_stream_repeats_post_text": True,
        "legacy_action_stream_hashes_environment_triggered": True,
        "proven_label_leakage": False,
        "source_ids_posts_or_prompts_exported": False,
        "nontraining_row_feature_statistics_or_forwards": False,
        "new_training_fits": 0,
        "submission_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous input audit")
    if args.training_source.name != "train.jsonl":
        raise ValueError("only the original training view can be audited")
    report = audit(args.training_source, args.training_metadata, args.source_sha256, args.metadata_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
