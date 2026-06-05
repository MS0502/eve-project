import json
from pathlib import Path

import scripts.operator_select_runtime_mapping_acceptance_repair as repair


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


def test_round231_consolidates_taxonomy_green_evidence_without_enablement() -> None:
    report = repair.build_round231_235_report(_green_suite_report())
    round231 = report["round231_acceptance_taxonomy_green_evidence"]

    assert round231["version"] == "v3_round231_acceptance_taxonomy_green_evidence_consolidation"
    assert round231["round"] == 231
    assert round231["success"] is True
    assert round231["taxonomy_status"] == "runtime_mapping_acceptance_taxonomy_ready"
    assert round231["selected_reporting_gap"] == "accepted_blocked_stage_taxonomy_missing_from_acceptance_delta"
    assert round231["taxonomy_green_evidence"]["accepted_tokens"] == ["민석"]
    assert round231["taxonomy_green_evidence"]["blocked_control_tokens"] == ["EVE"]
    assert round231["policy_guard_snapshot"]["production_persistence_disabled"] is True
    assert round231["policy_guard_snapshot"]["runtime_mapping_enabled_default_false"] is True
    assert round231["policy_guard_snapshot"]["enforcement_disabled"] is True


def test_round232_selects_one_concrete_non_persistence_repair_target() -> None:
    report = repair.build_round231_235_report(_green_suite_report())
    round232 = report["round232_next_repair_target_selection"]

    assert round232["version"] == "v3_round232_next_non_persistence_repair_target_selection"
    assert round232["round"] == 232
    assert round232["success"] is True
    assert round232["selected_next_repair_target"] == "runtime_mapping_accepted_blocked_stage_matrix_reporting_no_enablement"
    assert round232["implementation_scope"] == "reporting-only matrix over existing runtime-mapping acceptance rows"
    assert "production_persistence_enablement" in round232["explicit_non_goals"]
    assert "runtime_mapping_enabled_default_true" in round232["explicit_non_goals"]
    assert "enforcement_enablement" in round232["explicit_non_goals"]


def test_round233_emits_accepted_blocked_stage_matrix_for_korean_fixture() -> None:
    report = repair.build_round231_235_report(_green_suite_report())
    round233 = report["round233_stage_matrix_reporting_improvement"]

    assert round233["version"] == "v3_round233_accepted_blocked_stage_matrix_reporting_improvement"
    assert round233["round"] == 233
    assert round233["success"] is True
    assert round233["measurement_only"] is True
    assert round233["operator_summary"]["status"] == "accepted_blocked_stage_matrix_ready"
    assert round233["operator_summary"]["accepted_tokens_for_future_review"] == ["민석"]
    assert round233["operator_summary"]["blocked_control_tokens"] == ["EVE"]
    assert round233["operator_summary"]["blocked_control_stop_stages"] == {"EVE": "blocked_at_gate"}
    assert round233["operator_summary"]["unresolved_failure_tokens"] == []
    assert round233["operator_summary"]["focused_test_command"] == repair.FOCUSED_TEST_COMMAND

    matrix = round233["accepted_blocked_stage_matrix"]
    accepted_row = next(row for row in matrix if row["token"] == "민석")
    blocked_row = next(row for row in matrix if row["token"] == "EVE")
    assert accepted_row["classification"] == "accepted_for_future_operator_review"
    assert accepted_row["operator_stop_stage"] is None
    assert blocked_row["classification"] == "blocked_control"
    assert blocked_row["operator_stop_stage"] == "blocked_at_gate"
    assert any(stage["stage"] == "would_map_at_gate" and stage["passed"] is True for stage in accepted_row["stage_outcomes"])
    assert any(stage["stage"] == "blocked_at_gate" and stage["passed"] is True for stage in blocked_row["stage_outcomes"])
    assert round233["runtime_mapping_enabled_default"] is False
    assert round233["enforcement_enabled"] is False


def test_round231_235_report_completes_five_rounds_and_keeps_policy_no_go() -> None:
    report = repair.build_round231_235_report(_green_suite_report())

    assert report["version"] == "v3_round231_235_runtime_mapping_acceptance_repair_selection"
    assert report["rounds_completed"] == [231, 232, 233, 234, 235]
    assert report["status"] == "runtime_mapping_acceptance_repair_selection_ready"
    assert report["success"] is True
    assert report["exit_code"] == 0
    assert report["taxonomy_green_evidence"]["source_success"] is True
    assert report["selected_next_repair_target"] == "runtime_mapping_accepted_blocked_stage_matrix_reporting_no_enablement"
    assert report["round234_focused_test_plan"]["focused_test_command"] == repair.FOCUSED_TEST_COMMAND
    assert report["round235_broader_validation_delta"]["next_recommendation"].startswith("use the stage matrix")
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_cli_runner_reads_suite_report_or_fails_closed(tmp_path: Path) -> None:
    suite_report = tmp_path / "operator_local_validation_latest.json"
    suite_report.write_text(json.dumps(_green_suite_report(), ensure_ascii=False), encoding="utf-8")

    green = repair.run_acceptance_repair_selection_report(suite_report_path=suite_report)
    blocked = repair.run_acceptance_repair_selection_report(suite_report_path=tmp_path / "missing.json")

    assert green["success"] is True
    assert green["rounds_completed"] == [231, 232, 233, 234, 235]
    assert green["round233_stage_matrix_reporting_improvement"]["operator_summary"]["accepted_tokens_for_future_review"] == ["민석"]
    assert blocked["success"] is False
    assert blocked["blockers"] == ["operator_local_validation_suite_report_missing"]
    assert blocked["production_persistence_enabled"] is False
    assert blocked["runtime_mapping_enabled_default"] is False
