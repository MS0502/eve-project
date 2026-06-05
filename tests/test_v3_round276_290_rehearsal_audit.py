"""EVE v3 rounds276-290 — rehearsal evidence consolidation and rollback audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.operator_plan_round276_290_rehearsal_audit as audit
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


def test_round276_consolidates_no_persistence_rehearsal_green_evidence() -> None:
    evidence = audit.consolidate_no_persistence_rehearsal_green_evidence(_green_rehearsal_report())

    assert evidence["version"] == "v3_round276_no_persistence_rehearsal_green_evidence"
    assert evidence["round"] == 276
    assert evidence["success"] is True
    assert evidence["accepted_tokens"] == ["민석"]
    assert evidence["blocked_control_tokens"] == ["EVE"]
    assert evidence["checks"]["blocked_control_not_mapped"] is True
    assert evidence["checks"]["runtime_mapping_enabled_default_false"] is True
    assert evidence["checks"]["production_persistence_disabled"] is True
    assert evidence["checks"]["artifact_git_guard_clean"] is True
    assert evidence["blockers"] == []


def test_round277_280_designs_split_validation_staged_rehearsal_and_rollback_audit_without_persistence() -> None:
    evidence = audit.consolidate_no_persistence_rehearsal_green_evidence(_green_rehearsal_report())
    design = audit.design_next_isolated_rehearsal_steps(evidence)

    assert design["success"] is True
    assert design["rounds"] == [277, 278, 279, 280]
    phases = {row["phase"]: row for row in design["phases"]}
    assert "python -m pytest -q" in phases["split_full_suite_validation"]["commands"]
    assert phases["artifact_staged_rehearsal"]["allowed_output_path"].startswith("_operator_artifacts/")
    assert phases["artifact_staged_rehearsal"]["json_only"] is True
    assert phases["artifact_staged_rehearsal"]["must_not_be_committed"] is True
    assert phases["rollback_audit"]["allowed_output_path"].startswith("_operator_artifacts/")
    assert phases["go_no_go_review"]["production_persistence_go"] is False
    assert design["hard_boundaries"]["runtime_mapping_enabled_default"] is False
    assert design["hard_boundaries"]["enforcement_enabled"] is False


def test_round281_285_adds_measurable_rollback_audit_proof() -> None:
    rollback_audit = audit.build_rehearsal_rollback_audit(_green_rehearsal_report())

    assert rollback_audit["version"] == "v3_round281_285_rehearsal_rollback_audit_proof"
    assert rollback_audit["success"] is True
    assert rollback_audit["audit_score"]["passed"] == rollback_audit["audit_score"]["total"]
    assert rollback_audit["audit_score"]["rate"] == 1.0
    assert rollback_audit["audit_checks"]["runtime_mapping_not_persisted"] is True
    assert rollback_audit["audit_checks"]["embedding_lookup_not_called"] is True
    assert rollback_audit["audit_checks"]["eve_specific_vector_commit_not_called"] is True
    assert rollback_audit["audit_checks"]["blocked_control_not_mapped"] is True
    assert rollback_audit["mutation_checks"]["runtime_mapping_persisted"] is False
    assert rollback_audit["failed_checks"] == []


def test_round276_290_report_recommends_next_step_but_keeps_go_flags_false() -> None:
    report = audit.build_round276_290_report(_green_rehearsal_report())

    assert report["version"] == audit.ROUND276_290_VERSION
    assert report["rounds_completed"] == list(range(276, 291))
    assert report["success"] is True
    assert report["exit_code"] == 0
    assert report["no_persistence_rehearsal_green_evidence"] is True
    assert "deterministic audit" in report["concrete_implementation"]
    recommendation = report["round286_290_validation_delta_next_recommendation"]
    assert recommendation["production_persistence_go"] is False
    assert recommendation["runtime_mapping_default_go"] is False
    assert recommendation["enforcement_go"] is False
    assert "production_persistence_no_go" in report["remaining_taxonomy"]
    assert "EVE_blocked_control_requires_concept_SA_AGP_evidence" in report["remaining_taxonomy"]


def test_round276_290_fails_closed_when_rehearsal_evidence_is_not_green() -> None:
    source = _green_rehearsal_report()
    source["success"] = False
    source["status"] = "blocked_no_persistence_runtime_mapping_rehearsal"

    report = audit.build_round276_290_report(source)

    assert report["success"] is False
    assert report["exit_code"] == 1
    evidence = report["round276_no_persistence_rehearsal_green_evidence"]
    assert evidence["success"] is False
    assert "source_success_true" in evidence["blockers"]
    assert "source_status_green" in evidence["blockers"]


def test_round276_290_cli_writes_json_only_audit_packet(tmp_path: Path) -> None:
    rehearsal_path = tmp_path / "rehearsal.json"
    output_path = tmp_path / "round276_290_audit.json"
    rehearsal_path.write_text(json.dumps(_green_rehearsal_report(), ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/operator_plan_round276_290_rehearsal_audit.py",
            "--rehearsal-json",
            str(rehearsal_path),
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
    assert written["production_persistence_enabled"] is False
    assert written["runtime_mapping_enabled_default"] is False
    assert output_path.suffix == ".json"
