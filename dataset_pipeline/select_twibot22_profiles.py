#!/usr/bin/env python3
"""Select deterministic TwiBot-22 human profile candidates for QuadBot-v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

import ijson


FOLLOWER_BINS = {
    "micro": (10, 99),
    "small": (100, 999),
    "medium": (1000, 9999),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument(
        "--user-json",
        type=Path,
        help="Optional path to user.json when it is stored separately from metadata-root.",
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        help="Optional path to label.csv when it is stored separately from metadata-root.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        help="Optional path to split.csv when it is stored separately from metadata-root.",
    )
    parser.add_argument(
        "--tweet-stats",
        type=Path,
        help="Optional JSONL author statistics produced by discover_tweet_authors.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-bin", type=int, default=200)
    parser.add_argument("--seed", default="quadbot-v3-pilot-candidates-2026")
    return parser.parse_args()


def load_csv_map(path: Path, value_field: str) -> dict[str, str]:
    output = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            output[row["id"]] = row[value_field]
    return output


def follower_bin(count: int) -> str | None:
    for name, (lower, upper) in FOLLOWER_BINS.items():
        if lower <= count <= upper:
            return name
    return None


def stable_rank(seed: str, user_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{user_id}".encode()).digest()[:8], "big"
    )


def profile_is_eligible(profile: dict, labels: dict, splits: dict) -> tuple[bool, str]:
    user_id = str(profile.get("id", ""))
    if user_id and not user_id.startswith("u"):
        user_id = f"u{user_id}"
    if labels.get(user_id) != "human":
        return False, "not_human"
    if splits.get(user_id) != "train":
        return False, "not_train"
    if profile.get("protected"):
        return False, "protected"
    if profile.get("verified"):
        return False, "verified"

    description = " ".join(str(profile.get("description") or "").split())
    if len(description) < 20:
        return False, "short_description"

    metrics = profile.get("public_metrics") or {}
    followers = int(metrics.get("followers_count") or 0)
    following = int(metrics.get("following_count") or 0)
    tweets = int(metrics.get("tweet_count") or 0)
    if follower_bin(followers) is None:
        return False, "followers_out_of_range"
    if not 10 <= following <= 5000:
        return False, "following_out_of_range"
    if tweets < 100:
        return False, "too_few_tweets"
    return True, "eligible"


def push_candidate(heap: list, candidate: dict, limit: int) -> None:
    item = (-candidate["selection_rank"], candidate["source_id"], candidate)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def load_tweet_stats(path: Path | None) -> dict[str, dict[str, int]]:
    if path is None:
        return {}
    output = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            output[str(record["source_id"])] = {
                "tweet_count": int(record.get("tweet_count", 0)),
                "english_tweet_count": int(record.get("english_tweet_count", 0)),
            }
    return output


def main() -> None:
    args = parse_args()
    user_json = args.user_json or (args.metadata_root / "user.json")
    label_csv = args.label_csv or (args.metadata_root / "label.csv")
    split_csv = args.split_csv or (args.metadata_root / "split.csv")
    labels = load_csv_map(label_csv, "label")
    splits = load_csv_map(split_csv, "split")
    tweet_stats = load_tweet_stats(args.tweet_stats)
    heaps = {name: [] for name in FOLLOWER_BINS}
    audit = Counter()

    with user_json.open("rb") as stream:
        for profile in ijson.items(stream, "item"):
            audit["users_seen"] += 1
            user_id = str(profile.get("id", ""))
            if user_id and not user_id.startswith("u"):
                user_id = f"u{user_id}"
            eligible, reason = profile_is_eligible(profile, labels, splits)
            if eligible and tweet_stats and user_id not in tweet_stats:
                eligible, reason = False, "insufficient_tweet_history"
            audit[reason] += 1
            if not eligible:
                continue

            metrics = profile.get("public_metrics") or {}
            bin_name = follower_bin(int(metrics.get("followers_count") or 0))
            candidate = {
                "base_profile_id": f"twibot22:{user_id}",
                "source_id": user_id,
                "source_dataset": "twibot-22",
                "original_split": "train",
                "original_label": "human",
                "follower_bin": bin_name,
                "selection_rank": stable_rank(args.seed, user_id),
                "profile": profile,
            }
            push_candidate(heaps[bin_name], candidate, args.per_bin)

    selected = []
    for bin_name, heap in heaps.items():
        items = [entry[2] for entry in heap]
        items.sort(key=lambda item: (item["selection_rank"], item["source_id"]))
        selected.extend(items)
        audit[f"selected_{bin_name}"] = len(items)
    selected.sort(key=lambda item: (item["follower_bin"], item["selection_rank"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "quadbot-v3-twibot22-profile-audit-1",
        "selection_seed": args.seed,
        "per_bin_limit": args.per_bin,
        "follower_bins": FOLLOWER_BINS,
        "counts": dict(sorted(audit.items())),
        "output": str(args.output),
        "next_gate": "Retain profiles with sufficient original English tweet history.",
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
