"""Measure contradictory-label input collisions on development splits only."""

import argparse
import json
from pathlib import Path

import torch

from src.evidence_repair import write_new_json
from src.provenance import sha256_file


def run_audit(features, temporal, output, dataset):
    report = {"dataset": dataset, "test_paths_opened": False, "splits": {}}
    for split in ("train", "val"):
        with (features / f"{split}_metadata.jsonl").open(encoding="utf-8") as stream:
            metadata = [json.loads(line) for line in stream if line.strip()]
        grouped = {}
        for index, row in enumerate(metadata):
            grouped.setdefault(row["base_profile_id"], {})[int(row["label"])] = index
        left = torch.tensor([group[0] for group in grouped.values()])
        right = torch.tensor([group[1] for group in grouped.values()])
        paths = [
            features / "descriptions" / f"{split}_des_tensor.pt",
            features / "numerical" / f"{split}_num_properties_tensor.pt",
            features / "numerical" / f"{split}_llm_features.pt",
        ]
        paths.extend(features / f"{split}_{kind}_tensor{step}.pt"
                     for kind in ("text", "amr") for step in range(1, 6))
        all_same = torch.ones(len(left), dtype=torch.bool)
        counts, hashes = {}, {}
        for path in paths:
            value = torch.load(path, map_location="cpu", weights_only=True)
            same = (value[left] == value[right]).flatten(1).all(1)
            counts[path.relative_to(features).as_posix()] = int(same.sum())
            hashes[path.relative_to(features).as_posix()] = sha256_file(path)
            all_same &= same
        same_graph = torch.ones(len(left), dtype=torch.bool)
        for step in range(1, 6):
            path = features / f"{split}_edge_index_part{step}.pt"
            if not path.is_file():
                path = features / "graph" / f"{split}_edge_index.pt"
            edges = torch.load(path, map_location="cpu", weights_only=True)
            neighbors = {}
            for target, neighbor in zip(edges[0].tolist(), edges[1].tolist()):
                neighbors.setdefault(target, []).append(neighbor)
            same_graph &= torch.tensor([
                sorted(neighbors[int(a)]) == sorted(neighbors[int(b)])
                for a, b in zip(left, right)
            ])
            hashes[path.relative_to(features).as_posix()] = sha256_file(path)
        timing_same = torch.ones(len(left), dtype=torch.bool)
        timing_hashes = {}
        for kind in ("global", "steps"):
            path = temporal / f"{split}_{kind}.pt"
            timing = torch.load(path, map_location="cpu", weights_only=True)
            timing_same &= (timing[left] == timing[right]).flatten(1).all(1)
            timing_hashes[path.name] = sha256_file(path)
        contradictory = int((all_same & same_graph).sum())
        report["splits"][split] = {
            "profile_groups": len(grouped), "rows": len(metadata),
            "human_traditional_same_by_modality": counts,
            "identical_non_graph_inputs": int(all_same.sum()),
            "identical_non_graph_inputs_and_selected_neighbors": contradictory,
            "old_deterministic_accuracy_upper_bound_from_these_pairs":
                1.0 - contradictory / len(metadata),
            "remaining_non_graph_collisions_with_relative_timing":
                int((all_same & timing_same).sum()),
            "old_feature_sha256": hashes, "timing_sidecar_sha256": timing_hashes,
        }
    write_new_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_audit(args.features, args.temporal, args.out, args.dataset)


if __name__ == "__main__":
    main()
