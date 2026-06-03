from pathlib import Path

from adapters.medium_vector_release_restore import (
    EXPECTED_RESTORED_ZIP_SHA256,
    RELEASE_TAG,
    WRAPPER_ASSETS,
    restore_medium_vectors_from_release,
    write_round102_release_restore_status,
)


def test_round102_release_restore_fails_closed_without_assets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    status = restore_medium_vectors_from_release(
        work_dir=tmp_path / "work",
        repo_root=repo_root,
        download=False,
        install_to_repo=True,
    )

    assert status["round"] == 102
    assert status["release"]["tag"] == RELEASE_TAG
    assert status["hard_stop_released"] is False
    assert status["status"] == "blocked_release_restore_incomplete"
    assert status["install_result"] is None
    assert status["restored_zip"]["expected_sha256"] == EXPECTED_RESTORED_ZIP_SHA256
    assert len(status["wrapper_extractions"]) == len(WRAPPER_ASSETS)
    assert {row["error"] for row in status["wrapper_extractions"]} == {"wrapper_missing"}
    assert status["vectors_audit"]["acceptable_for_manual_install"] is False
    assert not (repo_root / "seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy").exists()
    assert status["policy"]["commit_vectors_npy_to_pr"] is False
    assert status["policy"]["create_dummy_vectors"] is False


def test_round102_status_writer_records_json_only(tmp_path: Path) -> None:
    status = restore_medium_vectors_from_release(
        work_dir=tmp_path / "work",
        repo_root=tmp_path / "repo",
        download=False,
    )
    out = tmp_path / "status.json"
    write_result = write_round102_release_restore_status(out, status)

    assert write_result["json_written"] is True
    assert write_result["binary_artifact_written_to_pr"] is False
    assert out.is_file()
    assert "blocked_release_restore_incomplete" in out.read_text(encoding="utf-8")
