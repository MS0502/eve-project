"""EVE v3 round29 — mini 1k subset registration.

Round29 commits the operator-extracted mini subset as a fixture-level artifact.
It validates provenance, checksums, shape, and parent linkage only. It must not
rewrite self_embedding or mark the subset as runtime-used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SEED_NAME,
    FASTTEXT_KOREAN_SHA256,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTORS_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_CHECKSUM,
    SEED_MANIFEST_PATH,
    SEED_STATE_REGISTERED,
    SUBSET_STATE_EXTRACTED,
    external_seed_state,
    fasttext_korean_subset_mini_1k_entry,
    load_manifest_file,
    subset_state,
    validate_manifest,
    validate_subset_entry,
)

SUBSET_DIR = Path(FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH)
VOCAB = SUBSET_DIR / "vocab.txt"
VECTORS = SUBSET_DIR / "vectors.npy"
SUBSET_MANIFEST = SUBSET_DIR / "subset_manifest.json"
SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "SHA256:" + digest.hexdigest()


def _subset_entry_from_manifest():
    manifest = load_manifest_file(SEED_MANIFEST_PATH)
    matches = [entry for entry in manifest["seeds"] if entry.get("name") == FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME]
    assert len(matches) == 1
    return manifest, matches[0]


def test_subset_directory_and_files_exist():
    """The operator-provided mini subset files must be committed under seeds/subsets."""
    assert SUBSET_DIR.is_dir()
    assert VOCAB.is_file()
    assert VECTORS.is_file()
    assert SUBSET_MANIFEST.is_file()


def test_subset_manifest_entry_registered_and_valid():
    """MANIFEST.yaml must include one valid subset entry linked to the parent seed."""
    manifest, entry = _subset_entry_from_manifest()
    validation = validate_manifest(manifest)

    assert validation["valid"] is True
    assert validation["seed_entries_count"] == 1
    assert validation["subset_entries_count"] >= 1
    assert entry == fasttext_korean_subset_mini_1k_entry()
    assert validate_subset_entry(entry, manifest)["valid"] is True


def test_vocab_has_exactly_1000_lines_and_expected_checksum():
    """The mini subset vocab is deterministic and checksum-verified."""
    vocab_lines = VOCAB.read_text(encoding="utf-8").splitlines()

    assert len(vocab_lines) == 1000
    assert vocab_lines[:10] == [".", ",", "</s>", ")", "(", ":", "'", '"', "/", "수"]
    assert _sha256(VOCAB) == FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_CHECKSUM


def test_vectors_shape_dtype_and_checksum():
    """The mini subset vector matrix must be 1000x300 float32."""
    vectors = np.load(VECTORS)

    assert vectors.shape == (1000, 300)
    assert vectors.dtype == np.float32
    assert _sha256(VECTORS) == FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTORS_CHECKSUM


def test_subset_manifest_json_matches_manifest_entry():
    """The embedded subset manifest must match the repo manifest subset entry."""
    manifest, entry = _subset_entry_from_manifest()
    payload = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))

    assert payload["name"] == entry["name"]
    assert payload["parent_seed"] == entry["parent_seed"]
    assert payload["parent_checksum"] == entry["parent_checksum"]
    assert payload["vocab_checksum"] == entry["vocab_checksum"]
    assert payload["vectors_checksum"] == entry["vectors_checksum"]
    assert payload["vocab_size"] == entry["vocab_size"] == 1000
    assert payload["vector_dim"] == entry["vector_dim"] == 300
    assert payload["deterministic"] is True
    assert _sha256(SUBSET_MANIFEST) == entry["subset_manifest_checksum"]


def test_parent_seed_reference_valid_and_state_ladder_unchanged():
    """Subset extraction does not advance parent seed from registered to loaded/used."""
    manifest, entry = _subset_entry_from_manifest()
    parent = [candidate for candidate in manifest["seeds"] if candidate.get("name") == FASTTEXT_KOREAN_SEED_NAME][0]

    assert parent["checksum"] == FASTTEXT_KOREAN_SHA256
    assert entry["parent_seed"] == FASTTEXT_KOREAN_SEED_NAME
    assert entry["parent_checksum"] == FASTTEXT_KOREAN_SHA256
    assert external_seed_state(manifest) == SEED_STATE_REGISTERED
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME) == SUBSET_STATE_EXTRACTED


def test_subset_validation_rejects_parent_checksum_mismatch():
    """A subset cannot silently detach from its registered parent checksum."""
    manifest, entry = _subset_entry_from_manifest()
    broken = dict(entry)
    broken["parent_checksum"] = "SHA256:" + "0" * 64

    result = validate_subset_entry(broken, manifest)

    assert result["valid"] is False
    assert "parent_checksum_mismatch" in result["errors"]


def test_self_embedding_adapter_unchanged_and_subset_not_used():
    """Round29 is extraction registration only, not the embedding rewrite."""
    text = SELF_EMBEDDING.read_text(encoding="utf-8")

    assert "vectors.npy" not in text
    assert "cc.ko.300.subset" not in text
    assert "300d" not in text.lower()
