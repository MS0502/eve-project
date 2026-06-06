"""EVE v3 rounds291-305 — split validation and staged artifact manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.operator_plan_round276_290_rehearsal_audit as round276_290
import scripts.operator_plan_round291_305_split_validation as round291_305
import scripts.operator_plan_runtime_mapping_acceptance_handoff as handoff
import scripts.operator_rehearse_runtime_mapping_no_persistence as rehearsal


def _green_suite_report() -> dict:
    mapping_measurement = {
        "target_word": "민석",
        "negative_token": "EVE",
        "target_would_map": True,
        "target_precheck_ready": True,
        "target_mapped_in_controlled_smoke": True,
        "negative_token_blocked": True,
        "rollback_complete": True,
        "runtime_mapping_enabled_after": False,
        "enforcement_enabled_after": False,
        "runtime_mapping_persisted": False,
        "gate": {"would_map_tokens": ["민석"], "blocked_tokens": ["EVE"]},
        "precheck": {"ready_tokens": ["민석"]},
        "controlled_smoke": {"mapped_tokens": ["민석"], "rollback": {"rollback_complete": True}},
    }
    return {
        "status": "operator_local_validation_suite_green",
        "success": True,
        "exit_code": 0,
        "operator_prompt_summary": {
            "exit_code": 0,
            "status": "operator_local_validation_suite_green",
            "target_word": "민석",
            "selected_cluster_id": "concept_runtime_mapping_after_eve_self_learning_guarded_medium30k",
            "vector_lookup_after_commit": True,
            "wrapper_primary_loaded": True,
            "git_status_safety_safe": True,
            "production_persistence_enabled": False,
            "runtime_mapping_enabled_default": False,
        },
        "steps": [{}, {}, {"parsed_json": {"mapping_measurement": mapping_measurement}}],
    }


def _handoff_packet() -> dict:
    report = handoff.build_round236_260_report(_green_suite_report())
    assert report["success"] is True
    return report


def _green_rehearsal_report() -> dict:
    report = rehearsal.run_isolated_no_persistence_rehearsal(
        _handoff_packet(),
        operator_authorized=True,
        authorization_token=rehearsal.AUTHORIZATION_TOKEN,
    )
    assert report["success"] is True
    return report


def _green_round276_290_report() -> dict:
    report = round276_290.build_round276_290_report(_green_rehearsal_report())
    assert report["success"] is True
    return report


def test_round291_consolidates_rollback_audit_green_evidence() -> None:
    evidence = round291_305.consolidate_round291_rollback_audit_green_evidence(_green_round276_290_report())

    assert evidence["version"] == "v3_round291_rollback_audit_green_evidence_consolidation"
    assert evidence["round"] == 291
    assert evidence["success"] is True
    assert evidence["checks"]["runtime_mapping_not_persisted"] is True
    assert evidence["checks"]["runtime_mapping_default_false"] is True
    assert evidence["checks"]["runtime_mapping_default_unchanged"] is True
    assert evidence["checks"]["enforcement_disabled"] is True
    assert evidence["checks"]["seed_vectors_not_mutated"] is True
    assert evidence["blockers"] == []


def test_round292_295_builds_split_validation_planner_without_enabling_persistence() -> None:
    evidence = round291_305.consolidate_round291_rollback_audit_green_evidence(_green_round276_290_report())
    planner = round291_305.build_split_full_suite_validation_planner(evidence)

    assert planner["version"] == "v3_round292_295_split_full_suite_validation_planner"
    assert planner["rounds"] == [292, 293, 294, 295]
    assert planner["success"] is True
    phases = {row["phase"]: row for row in planner["phases"]}
    assert phases["artifact_free_compile_and_collection"]["requires_operator_artifacts"] is False
    assert round291_305.COMPILE_COMMAND in phases["artifact_free_compile_and_collection"]["commands"]
    assert round291_305.COLLECT_ONLY_COMMAND in phases["artifact_free_compile_and_collection"]["commands"]
    assert phases["focused_regression_segment"]["requires_operator_artifacts"] is False
    assert phases["artifact_dependent_local_segment"]["requires_operator_artifacts"] is True
    assert phases["artifact_dependent_local_segment"]["fail_closed_if_artifacts_missing"] is True
    assert phases["full_suite_and_git_safety_delta"]["git_status_must_prove_no_operator_or_vector_artifacts_staged"] is True
    assert planner["execution_policy"]["production_persistence_go"] is False
    assert planner["runtime_mapping_enabled_default"] is False
    assert planner["enforcement_enabled"] is False


def test_round296_300_manifest_records_local_artifacts_and_fails_closed_when_missing(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    manifest = round291_305.build_guarded_artifact_staged_rehearsal_manifest(repo_root=tmp_path)

    assert manifest["version"] == "v3_round296_300_guarded_artifact_staged_rehearsal_manifest"
    assert manifest["rounds"] == [296, 297, 298, 299, 300]
    assert manifest["success"] is True
    assert manifest["staging_ready"] is False
    assert manifest["fail_closed"] is True
    assert manifest["missing_artifact_ids"]
    assert manifest["unsafe_artifact_ids"] == []
    assert manifest["manifest_behavior"]["records_required_local_artifacts_without_committing_them"] is True
    assert manifest["manifest_behavior"]["does_not_create_or_fabricate_vectors"] is True
    assert all(row["local_only"] is True for row in manifest["required_artifacts"])
    assert all(row["git_ignored"] is True for row in manifest["required_artifacts"])
    assert all(row["git_tracked"] is False for row in manifest["required_artifacts"])
    assert all(row["content_read"] is False for row in manifest["required_artifacts"])
    assert all(row["vector_contents_read"] is False for row in manifest["required_artifacts"])
    assert all(row["runtime_loaded"] is False for row in manifest["required_artifacts"])
    assert all(row["artifact_safe_to_load"] is False for row in manifest["required_artifacts"])


def test_round301_305_report_contains_delta_taxonomy_and_recommendation() -> None:
    report = round291_305.build_round291_305_report(_green_round276_290_report())

    assert report["version"] == round291_305.ROUND291_305_VERSION
    assert report["rounds_completed"] == list(range(291, 306))
    assert report["success"] is True
    assert report["exit_code"] == 0
    assert report["rollback_audit_green_evidence"] is True
    assert report["artifact_staged_manifest_behavior"]["fails_closed_when_required_artifacts_are_missing"] is True
    assert "round292_295" in report["broader_validation_delta"]
    assert "production_persistence_remains_no_go" in report["remaining_taxonomy"]
    assert "runtime_mapping_enabled_default_false" in report["remaining_taxonomy"]
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_round291_305_fails_closed_when_round276_290_audit_is_not_green() -> None:
    source = _green_round276_290_report()
    source["success"] = False
    source["status"] = "blocked_round276_290_rehearsal_evidence_rollback_audit"

    report = round291_305.build_round291_305_report(source)

    assert report["success"] is False
    assert report["exit_code"] == 1
    evidence = report["round291_rollback_audit_green_evidence"]
    assert evidence["success"] is False
    assert "source_success_true" in evidence["blockers"]
    assert "source_status_green" in evidence["blockers"]


def test_round291_305_cli_writes_json_only_packet(tmp_path: Path) -> None:
    source_path = tmp_path / "round276_290.json"
    output_path = tmp_path / "round291_305.json"
    source_path.write_text(json.dumps(_green_round276_290_report(), ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/operator_plan_round291_305_split_validation.py",
            "--round276-290-json",
            str(source_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert written["success"] is True
    assert written["runtime_mapping_enabled_default"] is False
    assert written["production_persistence_enabled"] is False
    assert output_path.suffix == ".json"
