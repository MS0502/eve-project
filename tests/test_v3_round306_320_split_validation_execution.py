from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import scripts.operator_run_round306_320_split_validation as round306_320
from scripts.operator_plan_round291_305_split_validation import ROUND291_305_VERSION


def _green_round291_305_report() -> dict:
    return {
        "version": ROUND291_305_VERSION,
        "rounds_completed": list(range(291, 306)),
        "status": "round291_305_split_validation_staged_manifest_ready",
        "success": True,
        "exit_code": 0,
        "rollback_audit_green_evidence": True,
        "artifact_staged_manifest_behavior": {
            "fails_closed_when_required_artifacts_are_missing": True,
            "does_not_read_vector_contents": True,
            "does_not_create_or_fabricate_vectors": True,
        },
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
    }


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_round306_consolidates_split_validation_green_evidence() -> None:
    evidence = round306_320.consolidate_round306_split_validation_green_evidence(_green_round291_305_report())

    assert evidence["version"] == "v3_round306_split_validation_green_evidence_consolidation"
    assert evidence["round"] == 306
    assert evidence["success"] is True
    assert evidence["checks"]["source_version_round291_305"] is True
    assert evidence["checks"]["artifact_manifest_fails_closed"] is True
    assert evidence["checks"]["artifact_manifest_does_not_read_vectors"] is True
    assert evidence["checks"]["production_persistence_no_go"] is True
    assert evidence["checks"]["runtime_mapping_default_false"] is True
    assert evidence["checks"]["enforcement_disabled"] is True
    assert evidence["blockers"] == []


def test_round307_310_artifact_free_command_runs_phase_and_reports_artifact_readiness_separately(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(command: Sequence[str], **kwargs):
        calls.append(" ".join(command))
        return _completed(0, stdout="ok\n", stderr="")

    (tmp_path / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    report = round306_320.run_artifact_free_validation_phase(
        _green_round291_305_report(), repo_root=tmp_path, runner=runner
    )

    assert report["version"] == "v3_round307_310_artifact_free_validation_command"
    assert report["success"] is True
    assert report["exit_code"] == 0
    assert calls == list(round306_320.ARTIFACT_FREE_PHASE_COMMANDS)
    assert report["artifact_dependent_readiness_reported_separately"] is True
    readiness = report["artifact_dependent_readiness"]
    assert readiness["ready"] is False
    assert readiness["fail_closed"] is True
    assert readiness["content_read"] is False
    assert readiness["vectors_created"] is False
    assert readiness["runtime_mapping_enabled_default"] is False


def test_round307_310_artifact_free_command_stops_on_first_failure(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(command: Sequence[str], **kwargs):
        calls.append(" ".join(command))
        return _completed(1 if len(calls) == 2 else 0, stdout="", stderr="failure")

    (tmp_path / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    report = round306_320.run_artifact_free_validation_phase(
        _green_round291_305_report(), repo_root=tmp_path, runner=runner
    )

    assert report["success"] is False
    assert report["exit_code"] == 1
    assert len(report["steps"]) == 2
    assert report["blockers"] == [f"artifact_free_failed:{round306_320.ARTIFACT_FREE_PHASE_COMMANDS[1]}"]


def test_round311_315_artifact_dependent_entrypoint_fails_closed_when_artifacts_missing(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    report = round306_320.run_guarded_artifact_dependent_entrypoint(
        _green_round291_305_report(),
        operator_authorized=True,
        authorization_token=round306_320.ROUND261_275_AUTHORIZATION_TOKEN,
        repo_root=tmp_path,
    )

    assert report["version"] == "v3_round311_315_guarded_artifact_dependent_entrypoint"
    assert report["success"] is False
    assert report["exit_code"] == 1
    assert report["fail_closed"] is True
    assert report["steps"] == []
    assert report["artifact_dependent_readiness"]["ready"] is False
    assert report["artifact_dependent_readiness"]["missing_artifact_ids"]
    assert report["content_read_before_safety_gate"] is False
    assert report["vectors_created"] is False
    assert report["vectors_loaded_before_safety_gate"] is False


def test_round311_315_artifact_dependent_entrypoint_requires_authorization_even_when_artifacts_exist(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for row in round306_320.DEFAULT_REQUIRED_LOCAL_ARTIFACTS:
        path = tmp_path / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("operator-local test fixture\n", encoding="utf-8")

    report = round306_320.run_guarded_artifact_dependent_entrypoint(
        _green_round291_305_report(),
        operator_authorized=False,
        authorization_token=None,
        repo_root=tmp_path,
    )

    assert report["success"] is False
    assert report["fail_closed"] is True
    assert "operator_authorization_missing_or_invalid" in report["blockers"]
    assert report["artifact_dependent_readiness"]["ready"] is True
    assert report["steps"] == []


def test_round311_315_artifact_dependent_entrypoint_ready_without_execution_when_safe_and_authorized(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for row in round306_320.DEFAULT_REQUIRED_LOCAL_ARTIFACTS:
        path = tmp_path / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("operator-local test fixture\n", encoding="utf-8")

    report = round306_320.run_guarded_artifact_dependent_entrypoint(
        _green_round291_305_report(),
        operator_authorized=True,
        authorization_token=round306_320.ROUND261_275_AUTHORIZATION_TOKEN,
        repo_root=tmp_path,
    )

    assert report["success"] is True
    assert report["exit_code"] == 0
    assert report["status"] == "round311_315_artifact_dependent_validation_ready"
    assert report["execute_requested"] is False
    assert report["artifact_dependent_readiness"]["ready"] is True
    assert report["steps"] == []
    assert report["planned_commands"] == list(round306_320.ARTIFACT_DEPENDENT_PLANNED_COMMANDS)
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_round316_320_validation_delta_records_fail_closed_artifact_dependent_result() -> None:
    artifact_free = {"success": True, "status": "round307_310_artifact_free_validation_green"}
    artifact_dependent = {"success": False, "status": "fail_closed_round311_315_artifact_dependent_validation", "fail_closed": True}

    report = round306_320.build_round316_320_validation_delta(artifact_free, artifact_dependent)

    assert report["version"] == "v3_round316_320_validation_delta_next_recommendation"
    assert report["rounds"] == [316, 317, 318, 319, 320]
    assert report["success"] is True
    assert "artifact_free_validation_green" in report["remaining_taxonomy"]
    assert "artifact_dependent_fail_closed_confirmed" in report["remaining_taxonomy"]
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_round306_320_cli_artifact_dependent_missing_artifacts_fails_closed(tmp_path: Path) -> None:
    source_path = tmp_path / "round291_305.json"
    source_path.write_text(json.dumps(_green_round291_305_report(), ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/operator_run_round306_320_split_validation.py",
            "--round291-305-json",
            str(source_path),
            "--artifact-dependent",
            "--operator-authorized",
            "--authorization-token",
            round306_320.ROUND261_275_AUTHORIZATION_TOKEN,
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["status"] == "fail_closed_round311_315_artifact_dependent_validation"
    assert payload["fail_closed"] is True
    assert payload["artifact_dependent_readiness"]["ready"] is False
    assert payload["runtime_mapping_enabled_default"] is False


def test_round322_325_artifact_dependent_diagnostics_expose_missing_paths_and_git_status(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for artifact_id in round306_320.MEDIUM30K_ARTIFACT_IDS:
        row = next(item for item in round306_320.DEFAULT_REQUIRED_LOCAL_ARTIFACTS if item["artifact_id"] == artifact_id)
        path = tmp_path / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("operator-local path placeholder for diagnostics only\n", encoding="utf-8")

    readiness = round306_320.build_artifact_dependent_readiness(repo_root=tmp_path)
    diagnostics = readiness["artifact_dependent_diagnostics"]

    assert diagnostics["version"] == "v3_round322_325_artifact_dependent_fail_closed_diagnostics"
    assert diagnostics["required_artifact_ids"] == [row["artifact_id"] for row in round306_320.DEFAULT_REQUIRED_LOCAL_ARTIFACTS]
    assert diagnostics["expected_local_paths"]["medium30k_vectors"] == "_operator_artifacts/subset_medium_30k/vectors.npy"
    assert "_operator_artifacts/subset_medium_30k/vectors.npy" in diagnostics["observed_candidate_paths"]
    assert diagnostics["complete_medium30k_candidate_present"] is True
    assert diagnostics["present_artifacts_do_not_override_missing_required_artifacts"] is True
    assert diagnostics["missing_artifact_ids"] == [
        "round236_260_acceptance_handoff_json",
        "round261_275_no_persistence_rehearsal_json",
    ]
    assert "Round261-275 rehearsal JSON" in diagnostics["why_subset_medium30k_not_accepted"]
    medium_vector_row = next(row for row in diagnostics["artifact_diagnostics"] if row["artifact_id"] == "medium30k_vectors")
    assert medium_vector_row["git_ignored"] is True
    assert medium_vector_row["git_tracked"] is False
    assert medium_vector_row["git_ignored_and_untracked"] is True
    assert medium_vector_row["missing_or_unsafe_reasons"] == []
    missing_json_row = next(row for row in diagnostics["artifact_diagnostics"] if row["artifact_id"] == "round261_275_no_persistence_rehearsal_json")
    assert missing_json_row["observed_candidate_paths"] == []
    assert "missing_expected_path" in missing_json_row["missing_or_unsafe_reasons"]
    assert diagnostics["content_read"] is False
    assert diagnostics["vector_contents_read"] is False
    assert readiness["vector_contents_read"] is False


def test_round326_330_medium30k_path_mapping_checks_paths_without_loading_vectors(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for artifact_id in round306_320.MEDIUM30K_ARTIFACT_IDS:
        row = next(item for item in round306_320.DEFAULT_REQUIRED_LOCAL_ARTIFACTS if item["artifact_id"] == artifact_id)
        path = tmp_path / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("operator-local path placeholder for diagnostics only\n", encoding="utf-8")

    readiness = round306_320.build_artifact_dependent_readiness(repo_root=tmp_path)
    mapping = readiness["medium30k_path_mapping_check"]

    assert mapping["version"] == "v3_round326_330_medium30k_manifest_path_mapping_check"
    assert mapping["success"] is True
    assert mapping["status"] == "medium30k_manifest_paths_mapped_without_vector_load"
    assert mapping["all_expected_paths_present"] is True
    assert mapping["all_expected_paths_git_ignored_and_untracked"] is True
    assert [row["artifact_id"] for row in mapping["mappings"]] == list(round306_320.MEDIUM30K_ARTIFACT_IDS)
    assert all(row["observed_candidate_path"] for row in mapping["mappings"])
    assert all(row["missing_or_unsafe_reasons"] == [] for row in mapping["mappings"])
    assert mapping["vector_contents_read"] is False
    assert mapping["vectors_loaded"] is False
    assert mapping["vectors_created"] is False
    assert mapping["artifacts_committed_or_staged"] is False
    assert mapping["runtime_mapping_default_go"] is False


def test_round331_335_validation_delta_records_diagnostic_and_path_mapping_delta(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for artifact_id in round306_320.MEDIUM30K_ARTIFACT_IDS:
        row = next(item for item in round306_320.DEFAULT_REQUIRED_LOCAL_ARTIFACTS if item["artifact_id"] == artifact_id)
        path = tmp_path / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("operator-local path placeholder for diagnostics only\n", encoding="utf-8")
    artifact_dependent = round306_320.run_guarded_artifact_dependent_entrypoint(
        _green_round291_305_report(),
        operator_authorized=True,
        authorization_token=round306_320.ROUND261_275_AUTHORIZATION_TOKEN,
        repo_root=tmp_path,
    )

    report = round306_320.build_round331_335_validation_delta(
        {"success": True, "status": "round307_310_artifact_free_validation_green"},
        artifact_dependent,
        focused_test_status="green",
        broader_pytest_status="pending_operator_run",
    )

    assert report["version"] == "v3_round331_335_diagnostic_validation_delta_next_recommendation"
    assert report["rounds"] == [331, 332, 333, 334, 335]
    assert report["success"] is True
    assert "artifact_dependent_diagnostics_actionable" in report["remaining_taxonomy"]
    assert "medium30k_path_mapping_checked_without_vector_load" in report["remaining_taxonomy"]
    assert report["improved_diagnostic_schema"]["vector_contents_read"] is False
    assert report["path_mapping_behavior"]["vectors_loaded"] is False
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False
