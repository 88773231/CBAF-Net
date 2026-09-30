"""Small provenance helpers shared by training and evaluation entry points."""

import hashlib
from pathlib import Path


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the lowercase SHA-256 digest of a file without loading it whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
