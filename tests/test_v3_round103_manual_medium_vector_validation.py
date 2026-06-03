from pathlib import Path

from adapters.medium_vector_manual_validation import (
    ROUND103_VALIDATION_COMMANDS,
    TARGET_VECTORS_RELATIVE_PATH,
    build_round103_manual_validation_plan,
    run_round103_manual_validation,
    write_round103_manual_validation_status,
)


def test_round103_blocks_without_manual_vectors(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    status = run_round103_manual_validation(repo_root=repo_root)

    assert status["round"] == 103
    assert status["hard_stop_released"] is False
    assert status["status"] == "blocked_manual_artifact_not_ready"
    assert status["commands_executed"] is False
    assert status["command_results"] == []
    assert status["policy"]["commit_vectors_npy_to_pr"] is False
    assert status["policy"]["create_dummy_vectors"] is False


def test_round103_plan_lists_single_operator_validation_command() -> None:
    plan = build_round103_manual_validation_plan()

    assert plan["single_validation_command"].startswith("python -m adapters.medium_vector_manual_validation")
    assert list(ROUND103_VALIDATION_COMMANDS) == plan["validation_commands"]
    assert TARGET_VECTORS_RELATIVE_PATH.endswith("vectors.npy")


def test_round103_status_writer_records_json_only(tmp_path: Path) -> None:
    status = run_round103_manual_validation(repo_root=tmp_path, execute=False)
    out = tmp_path / "round103.json"

    write_result = write_round103_manual_validation_status(out, status)

    assert write_result["json_written"] is True
    assert write_result["vectors_written"] is False
    assert write_result["binary_artifact_written_to_pr"] is False
    assert out.is_file()
    assert "blocked_manual_artifact_not_ready" in out.read_text(encoding="utf-8")
