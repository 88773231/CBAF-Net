"""Fetch only pinned public weights/configuration; never send account text."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

from dataset_pipeline.frozen_semantic import FILES, MODEL_ID, REVISION, WEIGHT_SHA256


BLOBS = {
    "1_Pooling/config.json": "d1514c3162bbe87b343f565fadc62e6c06f04f03",
    "README.md": "44af2e3b0fa3a0b6239e48422972bf755f28fde0",
    "config.json": "72b987fd805cfa2b58c4c8c952b274a11bfd5a00",
    "sentence_bert_config.json": "59d594003bf59880a884c574bf88ef7555bb0202",
    "special_tokens_map.json": "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5",
    "tokenizer.json": "cb202bfe2e3c98645018a6d12f182a434c9d3e02",
    "tokenizer_config.json": "c79f2b6a0cea6f4b564fed1938984bace9d30ff0",
    "vocab.txt": "fb140275c155a9c7c5a3b3e0e77a9e839594a938",
}


def verify_member(name, content):
    digest = hashlib.sha256(content).hexdigest()
    if name == "model.safetensors":
        if digest != WEIGHT_SHA256:
            raise ValueError("pinned weight hash differs")
    else:
        git_blob = hashlib.sha1(
            ("blob " + str(len(content)) + "\0").encode() + content,
        ).hexdigest()
        if git_blob != BLOBS[name]:
            raise ValueError("pinned configuration Git blob differs: " + name)
    return {"sha256": digest, "size_bytes": len(content)}


def fetch(output, endpoint, resume=False):
    if endpoint not in ("https://huggingface.co", "https://hf-mirror.com"):
        raise ValueError("explicit public artifact endpoint required")
    if output.exists() and not resume:
        raise FileExistsError("do not overwrite an existing model snapshot")
    if (output / "snapshot_manifest.json").exists():
        raise FileExistsError("completed snapshots are immutable")
    output.mkdir(parents=True, exist_ok=resume)
    records = {}
    for name in FILES:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            records[name] = verify_member(name, path.read_bytes())
        else:
            for attempt in range(2):
                try:
                    response = requests.get(
                        "{}/{}/resolve/{}/{}".format(endpoint, MODEL_ID, REVISION, name),
                        timeout=(15, 40),
                        headers={"User-Agent": "frozen-representation-study/1"},
                    )
                    response.raise_for_status()
                    content = response.content
                    record = verify_member(name, content)
                    with path.open("xb") as stream:
                        stream.write(content)
                    records[name] = record
                    break
                except requests.RequestException:
                    if attempt:
                        raise
                    time.sleep(1)
        print(json.dumps({"event": "verified_public_artifact", "file": name,
                          "bytes": records[name]["size_bytes"]}), flush=True)
    manifest = {
        "model_id": MODEL_ID, "revision": REVISION, "license": "apache-2.0",
        "retrieval_endpoint": endpoint, "files": records,
        "mirror_metadata_not_independent_upstream_attestation": endpoint.endswith("hf-mirror.com"),
        "private_account_text_sent": False,
    }
    with (output / "snapshot_manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    fetch(args.output, args.endpoint.rstrip("/"), args.resume)


if __name__ == "__main__":
    main()
