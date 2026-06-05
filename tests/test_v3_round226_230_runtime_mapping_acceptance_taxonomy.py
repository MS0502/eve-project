import json
from pathlib import Path

import scripts.operator_report_runtime_mapping_acceptance_taxonomy as taxonomy


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


def test_round226_consolidates_acceptance_delta_green_evidence_without_enablement() -> None:
    report = taxonomy.build_round226_230_report(_green_suite_report())
    round226 = report["round226_acceptance_delta_green_consolidation"]

    assert round226["version"] == "v3_round226_acceptance_delta_green_evidence_consolidation"
    assert round226["round"] == 226
    assert round226["success"] is True
    assert round226["accepted_tokens"] == ["민석"]
    assert round226["blocked_control_tokens"] == ["EVE"]
    assert round226["selected_next_repair_target"] == "runtime_mapping_acceptance_taxonomy_reporting_no_enablement"
    assert round226["policy_guard_snapshot"]["production_persistence_disabled"] is True
    assert round226["policy_guard_snapshot"]["runtime_mapping_enabled_default_false"] is True
    assert round226["policy_guard_snapshot"]["enforcement_disabled"] is True


def test_round227_selects_reporting_gap_and_explicit_non_goals() -> None:
    report = taxonomy.build_round226_230_report(_green_suite_report())
    round227 = report["round227_reporting_gap_selection"]

    assert round227["version"] == "v3_round227_runtime_mapping_reporting_gap_selection"
    assert round227["round"] == 227
    assert round227["success"] is True
    assert round227["selected_reporting_gap"] == "accepted_blocked_stage_taxonomy_missing_from_acceptance_delta"
    assert round227["implementation_scope"] == "reporting-only taxonomy over existing acceptance-delta rows"
    assert "production_persistence_enablement" in round227["explicit_non_goals"]
    assert "runtime_mapping_enabled_default_true" in round227["explicit_non_goals"]
    assert "enforcement_enablement" in round227["explicit_non_goals"]


def test_round228_emits_operator_ready_taxonomy_for_korean_acceptance_fixture() -> None:
    report = taxonomy.build_round226_230_report(_green_suite_report())
    round228 = report["round228_acceptance_taxonomy_improvement"]

    assert round228["version"] == "v3_round228_acceptance_taxonomy_reporting_improvement"
    assert round228["round"] == 228
    assert round228["success"] is True
    assert round228["measurement_only"] is True
    assert round228["taxonomy"] == {
        "total_rows": 2,
        "accepted_count": 1,
        "blocked_control_count": 1,
        "unresolved_failure_count": 0,
        "accepted_tokens": ["민석"],
        "blocked_control_tokens": ["EVE"],
        "failed_stage_counts": {},
    }
    assert round228["operator_summary"]["status"] == "runtime_mapping_acceptance_taxonomy_ready"
    assert round228["operator_summary"]["focused_test_command"] == taxonomy.FOCUSED_TEST_COMMAND
    assert round228["runtime_mapping_enabled_default"] is False
    assert round228["enforcement_enabled"] is False


def test_round226_230_report_completes_five_rounds_and_keeps_policy_no_go() -> None:
    report = taxonomy.build_round226_230_report(_green_suite_report())

    assert report["version"] == "v3_round226_230_runtime_mapping_acceptance_taxonomy"
    assert report["rounds_completed"] == [226, 227, 228, 229, 230]
    assert report["status"] == "runtime_mapping_acceptance_taxonomy_ready"
    assert report["success"] is True
    assert report["exit_code"] == 0
    assert report["acceptance_delta_green_evidence"]["source_success"] is True
    assert report["selected_next_repair_target"] == "runtime_mapping_acceptance_taxonomy_reporting_no_enablement"
    assert report["round229_focused_test_plan"]["focused_test_command"] == taxonomy.FOCUSED_TEST_COMMAND
    assert report["round230_broader_validation_delta"]["next_recommendation"].startswith("operator may use the taxonomy")
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_cli_runner_reads_suite_report_or_fails_closed(tmp_path: Path) -> None:
    suite_report = tmp_path / "operator_local_validation_latest.json"
    suite_report.write_text(json.dumps(_green_suite_report(), ensure_ascii=False), encoding="utf-8")

    green = taxonomy.run_acceptance_taxonomy_report(suite_report_path=suite_report)
    blocked = taxonomy.run_acceptance_taxonomy_report(suite_report_path=tmp_path / "missing.json")

    assert green["success"] is True
    assert green["rounds_completed"] == [226, 227, 228, 229, 230]
    assert green["round228_acceptance_taxonomy_improvement"]["taxonomy"]["accepted_tokens"] == ["민석"]
    assert blocked["success"] is False
    assert blocked["blockers"] == ["operator_local_validation_suite_report_missing"]
    assert blocked["production_persistence_enabled"] is False
    assert blocked["runtime_mapping_enabled_default"] is False
