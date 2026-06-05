from __future__ import annotations

import json
from pathlib import Path

import scripts.operator_measure_runtime_mapping_acceptance_delta as delta


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


def test_round221_uses_green_operator_suite_evidence_to_select_next_repair_target() -> None:
    round221 = delta.build_round221_green_evidence_interpretation(_green_suite_report())

    assert round221["version"] == "v3_round221_operator_green_evidence_repair_target_gate"
    assert round221["round"] == 221
    assert round221["ready_for_next_narrow_repair_target"] is True
    assert round221["next_narrow_repair_target"] == "runtime_mapping_acceptance_delta_measurement_no_enablement"
    assert round221["green_evidence"]["target_word"] == "민석"
    assert round221["production_persistence_enabled"] is False
    assert round221["runtime_mapping_enabled_default"] is False
    assert round221["enforcement_enabled"] is False


def test_round222_selects_one_concrete_failure_cluster_without_enablement() -> None:
    round221 = delta.build_round221_green_evidence_interpretation(_green_suite_report())
    round222 = delta.build_round222_failure_cluster_selection(round221)

    assert round222["version"] == "v3_round222_runtime_mapping_failure_cluster_selection"
    assert round222["round"] == 222
    assert round222["selected_failure_cluster_id"] == "post_green_runtime_mapping_stage_delta_visibility_gap"
    assert round222["behavior_to_improve"] == "operator-facing measurement determinism and failure localization, not runtime enablement"
    assert "production_persistence_enablement" in round222["blocked_actions"]
    assert "runtime_mapping_enabled_default_true" in round222["blocked_actions"]
    assert round222["hard_stops"] == []


def test_round223_normalizes_minseok_acceptance_and_eve_blocked_control() -> None:
    report = _green_suite_report()
    round223 = delta.build_round223_guarded_acceptance_delta_path(report)
    rows = {row["token"]: row for row in round223["delta_rows"]}

    assert round223["version"] == "v3_round223_guarded_runtime_mapping_acceptance_delta_path"
    assert round223["round"] == 223
    assert round223["success"] is True
    assert round223["accepted_tokens"] == ["민석"]
    assert round223["blocked_control_tokens"] == ["EVE"]
    assert rows["민석"]["accepted_for_future_operator_review"] is True
    assert rows["민석"]["stages"]["runtime_mapping_enabled_after_false"] is True
    assert rows["EVE"]["blocked_control"] is True
    assert rows["EVE"]["accepted_for_future_operator_review"] is False
    assert round223["production_persistence_enabled"] is False
    assert round223["runtime_mapping_enabled_default"] is False


def test_round223_missing_mapping_evidence_fails_closed() -> None:
    report = _green_suite_report()
    report["steps"] = []

    round223 = delta.build_round223_guarded_acceptance_delta_path(report)

    assert round223["success"] is False
    assert round223["status"] == "blocked_round223_guarded_acceptance_delta"
    assert round223["hard_stops"] == ["mapping_measurement_missing_from_operator_suite_report"]
    assert round223["delta_rows"] == []


def test_round221_225_report_completes_five_rounds_from_green_evidence() -> None:
    report = delta.build_round221_225_report(_green_suite_report())

    assert report["version"] == "v3_round221_225_runtime_mapping_acceptance_delta"
    assert report["rounds_completed"] == [221, 222, 223, 224, 225]
    assert report["success"] is True
    assert report["exit_code"] == 0
    assert report["selected_failure_cluster_id"] == "post_green_runtime_mapping_stage_delta_visibility_gap"
    assert report["round224_focused_test_plan"]["focused_test_command"] == "python -m pytest -q tests/test_v3_round221_225_runtime_mapping_acceptance_delta.py"
    assert report["round225_broader_validation_delta"]["next_recommendation"].startswith("run the focused acceptance-delta command")


def test_cli_runner_reads_suite_report_or_fails_closed(tmp_path: Path) -> None:
    suite_report = tmp_path / "operator_local_validation_latest.json"
    suite_report.write_text(json.dumps(_green_suite_report(), ensure_ascii=False), encoding="utf-8")

    green = delta.run_acceptance_delta_measurement(suite_report_path=suite_report)
    blocked = delta.run_acceptance_delta_measurement(suite_report_path=tmp_path / "missing.json")

    assert green["success"] is True
    assert green["rounds_completed"] == [221, 222, 223, 224, 225]
    assert blocked["success"] is False
    assert blocked["blockers"] == ["operator_local_validation_suite_report_missing"]
    assert blocked["production_persistence_enabled"] is False
