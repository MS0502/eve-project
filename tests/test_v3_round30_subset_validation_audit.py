"""EVE v3 round30 — subset validation audit before self_embedding rewrite.

Round30 is an audit/readiness round. It re-checks subset invariants and
readiness data before any self_embedding rewrite. It must not load the subset as
runtime state, import fastText, change AGP, or mark the subset as used.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SEED_NAME,
    FASTTEXT_KOREAN_SHA256,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_MANIFEST_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTORS_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_CHECKSUM,
    SEED_MANIFEST_PATH,
    SUBSET_STATE_EXTRACTED,
    assess_self_embedding_rewrite_readiness,
    audit_subset_artifact,
    external_seed_state,
    load_manifest_file,
    subset_state,
    validate_manifest,
    validate_subset_entry,
)
from main import build_full_engine

SUBSET_DIR = Path(FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH)
VOCAB = SUBSET_DIR / "vocab.txt"
VECTORS = SUBSET_DIR / "vectors.npy"
SUBSET_MANIFEST = SUBSET_DIR / "subset_manifest.json"
SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
REPORT = Path("ROUND_V3_R30_REPORT.md")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "SHA256:" + digest.hexdigest()


def _manifest_and_subset_entry():
    manifest = load_manifest_file(SEED_MANIFEST_PATH)
    subsets = [entry for entry in manifest["seeds"] if entry.get("name") == FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME]
    assert len(subsets) == 1
    return manifest, subsets[0]


def _assert_vector_artifact_absent_or_ignored_untracked(path: Path) -> None:
    if not path.exists():
        return
    assert subprocess.run(["git", "check-ignore", "-q", str(path)], check=False).returncode == 0
    assert subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0


def test_subset_checksums_still_match_manifest_metadata_without_committed_vectors():
    """Committed metadata must match; vectors checksum remains operator-local provenance."""
    manifest, entry = _manifest_and_subset_entry()

    assert _sha256(VOCAB) == entry["vocab_checksum"] == FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_CHECKSUM
    assert entry["vectors_checksum"] == FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTORS_CHECKSUM
    _assert_vector_artifact_absent_or_ignored_untracked(VECTORS)
    assert _sha256(SUBSET_MANIFEST) == entry["subset_manifest_checksum"]
    assert validate_manifest(manifest)["valid"] is True


def test_subset_shape_vocab_and_missing_vector_invariants():
    """The mini 1k vocab is intact; missing vectors fail closed through audit."""
    vocab_lines = VOCAB.read_text(encoding="utf-8").splitlines()
    audit = audit_subset_artifact(subset_name=FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME)

    assert len(vocab_lines) == 1000
    assert "\ufffd" not in "".join(vocab_lines)
    _assert_vector_artifact_absent_or_ignored_untracked(VECTORS)
    assert audit["vectors"]["shape"] is None
    assert audit["vectors"]["expected_shape"] == (1000, 300)
    assert ({"missing_vectors_file", "vector_contents_not_read"} & set(audit["errors"]))
    assert audit["vector_contents_read"] is False
    assert audit["runtime_loaded"] is False


def test_subset_manifest_required_fields_and_payload_integrity():
    """subset_manifest.json must preserve the extraction metadata."""
    manifest, entry = _manifest_and_subset_entry()
    payload = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))

    for field in (
        "name",
        "parent_seed",
        "parent_checksum",
        "selection_method",
        "filter_rule",
        "vocab_size",
        "vector_dim",
        "vocab_checksum",
        "vectors_checksum",
        "deterministic",
    ):
        assert field in payload
    assert payload["name"] == entry["name"]
    assert payload["parent_checksum"] == entry["parent_checksum"]
    assert payload["deterministic"] is True
    assert validate_manifest(manifest)["subset_entries_count"] >= 1


def test_subset_dependency_fails_without_parent_seed():
    """A subset entry cannot validate if the parent seed entry is removed."""
    manifest, entry = _manifest_and_subset_entry()
    without_parent = {"seeds": [candidate for candidate in manifest["seeds"] if candidate.get("name") != FASTTEXT_KOREAN_SEED_NAME]}

    result = validate_subset_entry(entry, without_parent)

    assert result["valid"] is False
    assert "parent_seed_not_registered" in result["errors"]


def test_parent_checksum_mismatch_invalidates_subset():
    """A parent checksum change invalidates the subset linkage."""
    manifest, entry = _manifest_and_subset_entry()
    broken = deepcopy(manifest)
    for candidate in broken["seeds"]:
        if candidate.get("name") == FASTTEXT_KOREAN_SEED_NAME:
            candidate["checksum"] = "SHA256:" + "0" * 64

    result = validate_subset_entry(entry, broken)

    assert result["valid"] is False
    assert "parent_checksum_mismatch" in result["errors"]


def test_runtime_not_using_subset_and_no_fasttext_import():
    """Round30 audit must not import fastText or wire the subset into self_embedding."""
    before_fasttext = "fasttext" in sys.modules
    self_text = SELF_EMBEDDING.read_text(encoding="utf-8")

    audit = audit_subset_artifact()

    assert audit["valid"] is False
    assert ({"missing_vectors_file", "vector_contents_not_read"} & set(audit["errors"]))
    assert audit["runtime_used"] is False
    assert audit["vector_contents_read"] is False
    assert audit["runtime_loaded"] is False
    assert audit["self_embedding_rewrite"] is False
    assert ("fasttext" in sys.modules) == before_fasttext
    assert "vectors.npy" not in self_text
    assert "cc.ko.300.subset" not in self_text
    assert "300d" not in self_text.lower()


def test_subset_missing_file_is_reported_not_fatal(tmp_path):
    """The audit helper returns structured missing-file data instead of mutating runtime."""
    manifest = load_manifest_file(SEED_MANIFEST_PATH)

    audit = audit_subset_artifact(manifest, subset_dir=tmp_path / "missing_subset")

    assert audit["valid"] is False
    assert "missing_vocab_file" in audit["errors"]
    assert "missing_vectors_file" in audit["errors"]
    assert "missing_subset_manifest_file" in audit["errors"]
    assert audit["runtime_used"] is False
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME) == SUBSET_STATE_EXTRACTED


def test_subset_audit_is_read_only_for_manifest_and_engine():
    """Audit calls must not modify manifest state, AGP mode, or engine adapters."""
    manifest_before = load_manifest_file(SEED_MANIFEST_PATH)
    engine = build_full_engine()
    agp_mode_before = engine.agp_adapter.mode
    compositor_mode_before = engine.compositor.agp_mode
    speech_mode_before = engine.speech_hub.agp_mode

    audit = audit_subset_artifact(manifest_before)
    readiness = assess_self_embedding_rewrite_readiness(engine, manifest_before)
    manifest_after = load_manifest_file(SEED_MANIFEST_PATH)

    assert audit["valid"] is False
    assert ({"missing_vectors_file", "vector_contents_not_read"} & set(audit["errors"]))
    assert audit["vector_contents_read"] is False
    assert audit["runtime_loaded"] is False
    assert readiness["self_embedding_rewrite"] is False
    assert manifest_after == manifest_before
    assert engine.agp_adapter.mode == agp_mode_before
    assert engine.compositor.agp_mode == compositor_mode_before
    assert engine.speech_hub.agp_mode == speech_mode_before
    assert external_seed_state(manifest_after) == external_seed_state(manifest_before)


def test_self_embedding_rewrite_readiness_assessment_returns_data_dict():
    """Readiness assessment is structured data, not an auto-applied recommendation."""
    engine = build_full_engine()
    readiness = assess_self_embedding_rewrite_readiness(engine)

    assert isinstance(readiness, dict)
    assert readiness["assessment_layer"] == "self_embedding_rewrite_readiness"
    assert readiness["current_embedding"]["method"] == "PMI+SVD"
    assert readiness["current_embedding"]["dimension"] == 300
    # Round31 may add a production-grade small 5k subset, so readiness may
    # prefer that subset while preserving the mini fixture in available_subsets.
    assert readiness["available_subset"]["dimension"] == 300
    assert readiness["available_subset"]["state"] == SUBSET_STATE_EXTRACTED
    assert any(item["name"] == FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME for item in readiness["available_subsets"])
    assert readiness["readiness"] == "needs_more_audit"
    assert readiness["available_subset"]["audit_valid"] is False
    assert readiness["available_subset"]["vector_contents_read"] is False
    assert readiness["available_subset"]["runtime_loaded"] is False
    assert readiness["recommendation_data"]["automatic_application"] is False
    assert readiness["state_transition"] is False


def test_round30_report_documents_audit_and_branch_options():
    """The round report must document audit results and option A/B decision point."""
    text = REPORT.read_text(encoding="utf-8")

    assert "subset validation audit" in text
    assert "invariants verified" in text
    assert "ready_with_concerns" in text
    assert "option A" in text
    assert "option B" in text
    assert "self_embedding rewrite" in text
    assert "not doing" in text
