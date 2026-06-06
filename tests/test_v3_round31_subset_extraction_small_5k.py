"""EVE v3 round31 — small 5k subset registration under no-vector policy."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SEED_NAME,
    FASTTEXT_KOREAN_SHA256,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTORS_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_VOCAB_CHECKSUM,
    SEED_MANIFEST_PATH,
    SEED_STATE_REGISTERED,
    SUBSET_STATE_EXTRACTED,
    audit_subset_artifact,
    assess_self_embedding_rewrite_readiness,
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


def _manifest_and_subset_entry():
    manifest = load_manifest_file(SEED_MANIFEST_PATH)
    matches = [entry for entry in manifest["seeds"] if entry.get("name") == FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME]
    assert len(matches) == 1
    return manifest, matches[0]


def _assert_vector_artifact_absent_or_ignored_untracked(path: Path) -> None:
    if not path.exists():
        return
    assert subprocess.run(["git", "check-ignore", "-q", str(path)], check=False).returncode == 0
    assert subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0


def test_small_subset_directory_and_metadata_files_exist_but_vectors_are_not_committed():
    assert SUBSET_DIR.is_dir()
    assert VOCAB.is_file()
    assert SUBSET_MANIFEST.is_file()
    _assert_vector_artifact_absent_or_ignored_untracked(VECTORS)


def test_small_subset_manifest_entry_registered_and_valid():
    manifest, entry = _manifest_and_subset_entry()
    validation = validate_manifest(manifest)

    assert validation["valid"] is True
    assert entry == fasttext_korean_subset_small_5k_entry()
    assert validate_subset_entry(entry, manifest)["valid"] is True
    assert entry["purpose"] == "production_lexical_seed"


def test_small_vocab_has_5000_lines_and_expected_checksum():
    vocab_lines = VOCAB.read_text(encoding="utf-8").splitlines()

    assert len(vocab_lines) == 5000
    assert "\ufffd" not in "".join(vocab_lines)
    assert _sha256(VOCAB) == FASTTEXT_KOREAN_SUBSET_SMALL_5K_VOCAB_CHECKSUM


def test_small_vectors_checksum_is_manifest_only_until_operator_artifact_is_present():
    manifest, entry = _manifest_and_subset_entry()
    audit = audit_subset_artifact(manifest, subset_name=FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME)

    assert entry["vectors_checksum"] == FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTORS_CHECKSUM
    _assert_vector_artifact_absent_or_ignored_untracked(VECTORS)
    assert audit["valid"] is False
    assert ({"missing_vectors_file", "vector_contents_not_read"} & set(audit["errors"]))
    assert audit["vectors"]["shape"] is None
    assert audit["vector_contents_read"] is False
    assert audit["runtime_loaded"] is False
    assert audit["vectors"]["expected_shape"] == (5000, 300)


def test_small_subset_manifest_json_matches_manifest_entry():
    manifest, entry = _manifest_and_subset_entry()
    payload = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))

    assert payload["name"] == entry["name"]
    assert payload["parent_seed"] == entry["parent_seed"]
    assert payload["parent_checksum"] == entry["parent_checksum"]
    assert payload["vocab_checksum"] == entry["vocab_checksum"]
    assert payload["vectors_checksum"] == entry["vectors_checksum"]
    assert payload["vocab_size"] == entry["vocab_size"] == 5000
    assert payload["vector_dim"] == entry["vector_dim"] == 300
    assert payload["deterministic"] is True
    assert _sha256(SUBSET_MANIFEST) == entry["subset_manifest_checksum"]


def test_small_parent_reference_and_state_ladder_unchanged():
    manifest, entry = _manifest_and_subset_entry()

    assert entry["parent_seed"] == FASTTEXT_KOREAN_SEED_NAME
    assert entry["parent_checksum"] == FASTTEXT_KOREAN_SHA256
    assert external_seed_state(manifest) == SEED_STATE_REGISTERED
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME) == SUBSET_STATE_EXTRACTED
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME) == SUBSET_STATE_EXTRACTED


def test_small_subset_audit_fails_closed_without_reading_vectors_and_is_distinct_from_mini_fixture():
    audit = audit_subset_artifact(subset_name=FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME)

    assert audit["valid"] is False
    assert ({"missing_vectors_file", "vector_contents_not_read"} & set(audit["errors"]))
    assert audit["subset_name"] == FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME
    assert audit["vocab"]["line_count"] == 5000
    assert audit["vector_contents_read"] is False
    assert audit["runtime_loaded"] is False


def test_readiness_prefers_medium_metadata_but_requires_more_audit_without_vectors():
    engine = build_full_engine()
    readiness = assess_self_embedding_rewrite_readiness(engine)

    assert readiness["readiness"] == "needs_more_audit"
    assert readiness["available_subset"]["dimension"] == 300
    assert readiness["available_subset"]["vocab_size"] == 30000
    assert readiness["available_subset"]["audit_valid"] is False
    assert readiness["available_subset"]["vector_contents_read"] is False
    assert readiness["available_subset"]["runtime_loaded"] is False
    assert readiness["recommendation_data"]["automatic_application"] is False
    assert readiness["state_transition"] is False


def test_self_embedding_adapter_unchanged_and_no_fasttext_runtime_import():
    before_fasttext = "fasttext" in sys.modules
    build_full_engine()
    text = SELF_EMBEDDING.read_text(encoding="utf-8")

    assert "vectors.npy" not in text
    assert "cc.ko.300.subset" not in text
    assert "300d" not in text.lower()
    assert ("fasttext" in sys.modules) == before_fasttext


def test_round31_report_documents_small_subset_and_no_auto_load():
    text = REPORT.read_text(encoding="utf-8")

    assert "small 5k" in text
    assert "production_lexical_seed" in text
    assert "unused until the scaffold passes" in text
