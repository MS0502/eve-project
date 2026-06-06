"""EVE v3 Round621-640 — operator baseline-lock workflow regression tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import operator_lock_round621_640_baseline as round621_640


def _init_repo(root: Path) -> None:
    (root / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_medium30k_sentinels(root: Path) -> dict[Path, bytes]:
    base = root / "_operator_artifacts" / "subset_medium_30k"
    base.mkdir(parents=True, exist_ok=True)
    files = {
        base / "vocab.txt": b"operator-local vocab sentinel; must not be parsed\n",
        base / "subset_manifest.json": b'{"sentinel":"must not be parsed"}\n',
        base / "vectors.npy": b"not-a-real-numpy-file; contents must not be read",
    }
    for path, content in files.items():
        path.write_bytes(content)
    return files


def test_round621_625_converts_prior_command_into_pre_post_baseline_lock(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    report = round621_640.build_round621_625_baseline_lock_workflow(repo_root=tmp_path)

    assert report["version"] == "v3_round621_625_operator_baseline_lock_workflow"
    assert report["success"] is True
    assert report["operator_baseline_lock_command"] == "python scripts/operator_lock_round621_640_baseline.py"
    assert report["pre_feature_usage"] == report["post_feature_usage"]
    assert report["source_baseline_command"] == "python scripts/operator_verify_round601_620_baseline.py"
    assert report["supported_modes"] == ["artifact_free", "artifact_present_path_reference"]
    assert report["checks"]["workflow_is_path_metadata_only"] is True
    assert report["vector_contents_read"] is False
    assert report["vectors_loaded"] is False
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_round626_630_read_only_guards_do_not_read_load_enable_or_stage(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    sentinels = _write_medium30k_sentinels(tmp_path)
    before = {path: path.read_bytes() for path in sentinels}

    report = round621_640.build_round626_630_read_only_regression_guards(repo_root=tmp_path)

    after = {path: path.read_bytes() for path in sentinels}
    assert after == before
    assert report["version"] == "v3_round626_630_baseline_lock_read_only_regression_guards"
    assert report["success"] is True
    assert report["checks"] == {
        "cannot_read_vector_contents": True,
        "cannot_load_runtime_vectors": True,
        "cannot_enable_production_persistence": True,
        "cannot_enable_runtime_mapping_by_default": True,
        "cannot_enable_enforcement": True,
        "cannot_stage_forbidden_operator_artifacts": True,
        "cannot_create_operator_artifacts": True,
    }
    assert report["git_artifact_safety_proof"]["forbidden_entries"] == []
    assert report["git_artifact_safety_proof"]["artifacts_created_by_workflow"] is False
    assert report["vector_contents_read"] is False
    assert report["vectors_loaded"] is False
    assert report["vectors_created"] is False


def test_round631_635_documentation_packet_lists_dual_environment_commands() -> None:
    report = round621_640.build_round631_635_documentation_packet()

    assert report["version"] == "v3_round631_635_dual_environment_workflow_documentation_packet"
    assert report["success"] is True
    assert report["documentation_path"] == "docs/round621_640_feature_entry_readiness.md"
    assert report["validation_commands"]["compile"] == "python -m compileall -q adapters tests main.py scripts"
    assert report["validation_commands"]["round621_640_operator_lock"] == "python scripts/operator_lock_round621_640_baseline.py"
    assert report["checks"]["artifact_free_mode_documented"] is True
    assert report["checks"]["artifact_present_path_reference_mode_documented"] is True
    assert report["checks"]["explicit_operator_authorized_load_boundary_documented"] is True


def test_round636_640_readiness_recommends_exactly_one_safe_feature_track(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    report = round621_640.build_round636_640_feature_entry_readiness_report(
        repo_root=tmp_path,
        compile_status="green_compileall",
        collect_only_status="green_1423_collected",
        full_pytest_status="green_1423_passed",
        focused_status="green_round621_640_focused",
        round601_620_command_status="green_round601_620_command",
        round621_640_command_status="green_round621_640_command",
        git_status_safety="green_no_forbidden_artifacts",
    )

    assert report["version"] == "v3_round636_640_feature_entry_readiness_report"
    assert report["success"] is True
    assert report["recommended_next_feature_tracks"] == [
        "read_only_appraisal_classifier_agp_input_stabilization_audit"
    ]
    assert report["recommended_next_feature_track"] == report["recommended_next_feature_tracks"][0]
    assert report["production_persistence_go"] is False
    assert report["runtime_mapping_default_go"] is False
    assert report["enforcement_go"] is False
    assert report["vector_contents_read"] is False
    assert report["vectors_loaded"] is False


def test_round621_640_cli_emits_read_only_report_and_creates_no_artifacts(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    sentinels = _write_medium30k_sentinels(tmp_path)
    before_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    before = {path: path.read_bytes() for path in sentinels}

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/operator_lock_round621_640_baseline.py",
            "--repo-root",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    after_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    after = {path: path.read_bytes() for path in sentinels}
    assert proc.returncode == 0, proc.stderr
    assert after == before
    assert after_files == before_files
    payload = json.loads(proc.stdout)
    assert payload["version"] == "v3_round621_640_operator_baseline_lock_workflow"
    assert payload["success"] is True
    assert payload["rounds_completed"] == list(range(621, 641))
    assert payload["operator_command"] == "python scripts/operator_lock_round621_640_baseline.py"
    assert payload["operator_report_path"] == "docs/round621_640_feature_entry_readiness.md"
    assert payload["git_artifact_safety_proof"]["forbidden_entries"] == []
    assert payload["recommended_next_feature_tracks"] == [
        "read_only_appraisal_classifier_agp_input_stabilization_audit"
    ]
    assert payload["vector_contents_read"] is False
    assert payload["vectors_loaded"] is False
    assert payload["vectors_created"] is False
    assert payload["production_persistence_enabled"] is False
    assert payload["runtime_mapping_enabled_default"] is False
    assert payload["enforcement_enabled"] is False
