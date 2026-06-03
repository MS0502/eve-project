"""EVE v3 Round103 — manual medium-vector validation."""

from pathlib import Path

import numpy as np

from adapters.medium_vector_manual_validation import (
    ROUND103_MANUAL_VALIDATION_VERSION,
    validate_round103_manual_medium_vectors,
    write_round103_manual_medium_vector_validation,
)


def test_round103_manual_validation_fails_closed_without_release_or_vectors(tmp_path: Path) -> None:
    report = validate_round103_manual_medium_vectors(tmp_path / "missing.npy", repo_root=tmp_path)

    assert report["validation_version"] == ROUND103_MANUAL_VALIDATION_VERSION
    assert report["round"] == 103
    assert report["validation_passed"] is False
    assert report["status"] == "blocked_manual_medium_vector_validation_incomplete"
    assert "release_restore_status_not_supplied" in report["blocking_reasons"]
    assert "candidate_vectors_file_missing" in report["blocking_reasons"]
    assert report["policy"]["commit_vectors_npy_to_pr"] is False
    assert report["policy"]["create_dummy_vectors"] is False
    assert report["policy"]["weaken_tests"] is False


def test_round103_manual_validation_rejects_fake_vectors_even_with_release_status(tmp_path: Path) -> None:
    candidate = tmp_path / "vectors.npy"
    np.save(candidate, np.zeros((1, 1), dtype=np.float32))
    release_status = {"status": "restored_medium_vectors_installed", "hard_stop_released": True}

    report = validate_round103_manual_medium_vectors(candidate, repo_root=tmp_path, release_restore_status=release_status)

    assert report["release_restore_hard_stop_released"] is True
    assert report["validation_passed"] is False
    assert "vectors_shape_mismatch" in report["blocking_reasons"]
    assert "vectors_checksum_mismatch" in report["blocking_reasons"]
    assert report["vectors_audit"]["no_copy_performed"] is True


def test_round103_export_writes_json_only(tmp_path: Path) -> None:
    report = validate_round103_manual_medium_vectors(tmp_path / "missing.npy", repo_root=tmp_path)
    out = tmp_path / "round103.json"
    export = write_round103_manual_medium_vector_validation(out, report)

    assert export["export_version"] == ROUND103_MANUAL_VALIDATION_VERSION
    assert export["json_written"] is True
    assert export["vectors_written"] is False
    assert export["binary_artifact_written_to_pr"] is False
    assert out.is_file()
