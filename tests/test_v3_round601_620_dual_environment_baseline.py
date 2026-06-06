from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import operator_verify_round601_620_baseline as round601_620


def _init_repo(root: Path) -> None:
    (root / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_medium30k_placeholders(root: Path) -> None:
    base = root / "_operator_artifacts" / "subset_medium_30k"
    base.mkdir(parents=True, exist_ok=True)
    (base / "vocab.txt").write_text("operator-local vocab path only\n", encoding="utf-8")
    (base / "subset_manifest.json").write_text('{"operator":"path-reference-only"}\n', encoding="utf-8")
    (base / "vectors.npy").write_bytes(b"not a numpy file and must never be opened by this guard")


def test_round601_605_artifact_free_baseline_locks_no_load_policy(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    report = round601_620.build_dual_environment_baseline_report(repo_root=tmp_path)

    assert report["version"] == "v3_round601_605_dual_environment_green_baseline_consolidation"
    assert report["success"] is True
    assert report["observed_environment"] == "artifact_free"
    assert report["confirmed_baseline"] == {
        "artifact_free_full_pytest": "1418 passed",
        "operator_codespaces_ignored_subset_medium30k_full_pytest": "1418 passed",
        "focused_portability_cluster": "66 passed",
    }
    assert report["artifact_dependent_ready"] is False
    assert report["vector_contents_read"] is False
    assert report["vectors_loaded"] is False
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_round606_610_subset_present_path_reference_cannot_green_readiness_or_load(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_medium30k_placeholders(tmp_path)

    report = round601_620.build_artifact_present_regression_guard(repo_root=tmp_path)

    assert report["version"] == "v3_round606_610_artifact_present_no_load_regression_guard"
    assert report["success"] is True
    assert report["artifact_present_path_reference_mode"] is True
    assert report["medium30k_complete_path_set_present"] is True
    assert report["required_json_artifacts_complete"] is False
    assert report["subset_path_presence_only"] is True
    assert report["artifact_dependent_ready"] is False
    assert report["checks"]["subset_path_presence_alone_does_not_make_artifact_readiness_green"] is True
    assert report["checks"]["no_vector_content_read"] is True
    assert report["checks"]["no_runtime_load"] is True
    assert report["vector_contents_read"] is False
    assert report["vectors_loaded"] is False
    assert report["vectors_created"] is False
    assert "round261_275_no_persistence_rehearsal_json" in report["missing_artifact_ids"]


def test_round611_615_operator_command_report_proves_no_artifact_staging(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_medium30k_placeholders(tmp_path)

    report = round601_620.build_operator_verification_command_report(repo_root=tmp_path)

    assert report["version"] == "v3_round611_615_operator_verification_command_report"
    assert report["success"] is True
    assert report["stable_operator_command"] == "python scripts/operator_verify_round601_620_baseline.py"
    assert report["git_artifact_safety_proof"]["command"] == "git status --short"
    assert report["git_artifact_safety_proof"]["forbidden_entries"] == []
    assert report["git_artifact_safety_proof"]["clean_of_forbidden_operator_or_vector_artifacts"] is True
    assert report["checks"]["no_vector_content_read"] is True
    assert report["checks"]["no_runtime_load"] is True
    assert report["artifacts_staged_or_committed_by_report"] is False


def test_round616_620_next_recommendation_stays_feature_safe(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    report = round601_620.build_round616_620_validation_and_next_recommendation(
        repo_root=tmp_path,
        compile_status="green",
        collect_only_status="green",
        focused_status="green",
        full_pytest_status="green_1418_passed",
        git_status_safety="green",
    )

    assert report["version"] == "v3_round616_620_validation_next_feature_safe_direction"
    assert report["success"] is True
    assert report["validation_commands"] == round601_620.VALIDATION_COMMANDS
    assert "feature-safe" in report["next_recommendation"]
    assert report["production_persistence_go"] is False
    assert report["runtime_mapping_default_go"] is False
    assert report["enforcement_go"] is False


def test_round601_620_cli_emits_compact_json_without_reading_vectors(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_medium30k_placeholders(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/operator_verify_round601_620_baseline.py",
            "--repo-root",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["version"] == "v3_round601_620_dual_environment_baseline_regression_guard"
    assert payload["success"] is True
    assert payload["rounds_completed"] == list(range(601, 621))
    assert payload["round606_610"]["artifact_present_path_reference_mode"] is True
    assert payload["round606_610"]["artifact_dependent_ready"] is False
    assert payload["git_artifact_safety_proof"]["forbidden_entries"] == []
    assert payload["production_persistence_enabled"] is False
    assert payload["runtime_mapping_enabled_default"] is False
    assert payload["enforcement_enabled"] is False
