"""EVE v3 round31 — small 5k subset registration.

Round31 registers the operator-extracted small 5k subset as the first
production lexical seed candidate. It validates provenance, checksums, shape,
purpose, and parent linkage only. It must not rewrite self_embedding or mark the
subset as runtime-used.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SEED_NAME,
    FASTTEXT_KOREAN_SHA256,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PURPOSE,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_MANIFEST_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_PURPOSE,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTORS_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_VOCAB_CHECKSUM,
    SEED_MANIFEST_PATH,
    SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED,
    SUBSET_STATE_EXTRACTED,
    assess_self_embedding_rewrite_readiness,
    audit_subset_artifact,
    external_seed_state,
    fasttext_korean_subset_small_5k_entry,
    load_manifest_file,
    subset_state,
    validate_manifest,
    validate_subset_entry,
)
from main import build_full_engine

SUBSET_DIR = Path(FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH)
VOCAB = SUBSET_DIR / "vocab.txt"
VECTORS = SUBSET_DIR / "vectors.npy"
SUBSET_MANIFEST = SUBSET_DIR / "subset_manifest.json"
SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
REPORT = Path("ROUND_V3_R31_REPORT.md")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "SHA256:" + digest.hexdigest()


def _manifest_and_small_entry():
    manifest = load_manifest_file(SEED_MANIFEST_PATH)
    matches = [entry for entry in manifest["seeds"] if entry.get("name") == FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME]
    assert len(matches) == 1
    return manifest, matches[0]


def test_small_subset_directory_and_files_exist():
    assert SUBSET_DIR.is_dir()
    assert VOCAB.is_file()
    assert VECTORS.is_file()
    assert SUBSET_MANIFEST.is_file()


def test_small_subset_manifest_entry_registered_and_valid():
    manifest, entry = _manifest_and_small_entry()
    validation = validate_manifest(manifest)

    assert validation["valid"] is True
    assert validation["seed_entries_count"] == 1
    # Round50 registers medium 30k, so the historical round31 invariant evolves
    # from mini+small to mini+small+medium while keeping this small entry valid.
    assert validation["subset_entries_count"] == 3
    assert entry == fasttext_korean_subset_small_5k_entry()
    assert validate_subset_entry(entry, manifest)["valid"] is True


def test_small_vocab_has_exactly_5000_lines_utf8_and_expected_checksum():
    vocab_lines = VOCAB.read_text(encoding="utf-8").splitlines()

    assert len(vocab_lines) == 5000
    assert vocab_lines[:10] == [".", ",", "</s>", ")", "(", ":", "'", '"', "/", "수"]
    assert "\ufffd" not in "".join(vocab_lines)
    assert _sha256(VOCAB) == FASTTEXT_KOREAN_SUBSET_SMALL_5K_VOCAB_CHECKSUM


def test_small_vectors_shape_dtype_and_checksum():
    vectors = np.load(VECTORS, mmap_mode="r")

    assert vectors.shape == (5000, 300)
    assert str(vectors.dtype) == "float32"
    assert _sha256(VECTORS) == FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTORS_CHECKSUM


def test_small_subset_manifest_json_matches_manifest_entry():
    manifest, entry = _manifest_and_small_entry()
    payload = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))

    assert payload["name"] == entry["name"]
    assert payload["parent_seed"] == entry["parent_seed"]
    assert payload["parent_checksum"] == entry["parent_checksum"]
    assert payload["vocab_checksum"] == entry["vocab_checksum"]
    assert payload["vectors_checksum"] == entry["vectors_checksum"]
    assert payload["vocab_size"] == entry["vocab_size"] == 5000
    assert payload["vector_dim"] == entry["vector_dim"] == 300
    assert payload["purpose"] == entry["purpose"] == FASTTEXT_KOREAN_SUBSET_SMALL_5K_PURPOSE
    assert payload["deterministic"] is True
    assert _sha256(SUBSET_MANIFEST) == entry["subset_manifest_checksum"] == FASTTEXT_KOREAN_SUBSET_SMALL_5K_MANIFEST_CHECKSUM


def test_small_parent_seed_reference_and_subset_state():
    manifest, entry = _manifest_and_small_entry()
    parent = [candidate for candidate in manifest["seeds"] if candidate.get("name") == FASTTEXT_KOREAN_SEED_NAME][0]

    assert parent["checksum"] == FASTTEXT_KOREAN_SHA256
    assert entry["parent_seed"] == FASTTEXT_KOREAN_SEED_NAME
    assert entry["parent_checksum"] == FASTTEXT_KOREAN_SHA256
    assert entry["purpose"] == SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED
    assert external_seed_state(manifest) == "registered"
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME) == SUBSET_STATE_EXTRACTED


def test_small_subset_validation_rejects_parent_checksum_mismatch():
    manifest, entry = _manifest_and_small_entry()
    broken = dict(entry)
    broken["parent_checksum"] = "SHA256:" + "0" * 64

    result = validate_subset_entry(broken, manifest)

    assert result["valid"] is False
    assert "parent_checksum_mismatch" in result["errors"]


def test_small_subset_audit_valid_and_distinct_from_mini_fixture():
    manifest, entry = _manifest_and_small_entry()
    audit = audit_subset_artifact(manifest, subset_name=FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME)

    assert audit["valid"] is True
    assert audit["subset_name"] == FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME
    assert audit["vocab"]["line_count"] == 5000
    assert audit["vectors"]["shape"] == (5000, 300)
    assert audit["runtime_used"] is False
    assert any(candidate.get("name") == FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME for candidate in manifest["seeds"])
    assert entry["purpose"] == SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED


def test_readiness_now_prefers_medium_expanded_subset_without_auto_apply():
    engine = build_full_engine()
    readiness = assess_self_embedding_rewrite_readiness(engine)

    assert readiness["readiness"] == "ready"
    assert readiness["available_subset"]["name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert readiness["available_subset"]["vocab_size"] == 30000
    assert readiness["available_subset"]["purpose"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PURPOSE
    assert any(item["name"] == FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME for item in readiness["available_subsets"])
    assert any(item["name"] == FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME for item in readiness["available_subsets"])
    assert readiness["recommendation_data"]["automatic_application"] is False
    assert readiness["self_embedding_rewrite"] is False
    assert readiness["state_transition"] is False


def test_round31_does_not_use_subset_or_rewrite_self_embedding():
    before_fasttext = "fasttext" in sys.modules
    text = SELF_EMBEDDING.read_text(encoding="utf-8")
    manifest = load_manifest_file(SEED_MANIFEST_PATH)

    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME) == SUBSET_STATE_EXTRACTED
    assert ("fasttext" in sys.modules) == before_fasttext
    assert "vectors.npy" not in text
    assert "cc.ko.300.subset" not in text
    assert "300d" not in text.lower()


def test_round31_report_documents_small_subset_and_non_goals():
    text = REPORT.read_text(encoding="utf-8")

    assert "small 5k subset extraction" in text
    assert "production_lexical_seed" in text
    assert "cc.ko.300.subset.small.5k" in text
    assert "self_embedding rewrite" in text
    assert "not doing" in text
