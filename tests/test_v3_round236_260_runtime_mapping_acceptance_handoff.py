import scripts.operator_plan_runtime_mapping_acceptance_handoff as handoff


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


def test_round236_240_builds_stage_matrix_quality_gates_without_enablement() -> None:
    report = handoff.build_round236_260_report(_green_suite_report())
    cluster = report["round236_240_stage_matrix_quality_gates"]

    assert cluster["version"] == "v3_round236_240_stage_matrix_quality_gate"
    assert cluster["rounds"] == [236, 237, 238, 239, 240]
    assert cluster["success"] is True
    assert cluster["measurement_only"] is True
    assert cluster["accepted_tokens"] == ["민석"]
    assert cluster["blocked_control_tokens"] == ["EVE"]
    assert cluster["unresolved_failure_tokens"] == []
    assert cluster["stop_stage_counts"] == {"blocked_at_gate": 1, "none": 1}
    assert cluster["quality_gates"]["blocked_controls_stop_at_gate"] is True
    assert cluster["quality_gates"]["accepted_rows_do_not_stop"] is True
    assert cluster["quality_gates"]["runtime_mapping_enabled_default_false"] is True
    assert cluster["focused_test_command"] == handoff.FOCUSED_TEST_COMMAND


def test_round241_245_handoff_packet_preserves_korean_fixture_and_blocked_control() -> None:
    report = handoff.build_round236_260_report(_green_suite_report())
    cluster = report["round241_245_operator_handoff_packet"]

    assert cluster["version"] == "v3_round241_245_operator_handoff_packet"
    assert cluster["rounds"] == [241, 242, 243, 244, 245]
    assert cluster["success"] is True
    assert cluster["operator_summary"]["status"] == "handoff_packet_ready"
    assert cluster["accepted_for_future_operator_review"] == ["민석"]
    assert cluster["blocked_controls"] == ["EVE"]
    accepted_row = next(row for row in cluster["operator_replay_rows"] if row["token"] == "민석")
    blocked_row = next(row for row in cluster["operator_replay_rows"] if row["token"] == "EVE")
    assert accepted_row["operator_action"] == "review_as_future_runtime_mapping_candidate"
    assert blocked_row["operator_action"] == "keep_as_blocked_control_until concept/SA/AGP evidence exists"
    assert cluster["runtime_mapping_enabled_default"] is False
    assert cluster["enforcement_enabled"] is False


def test_round246_250_validation_manifest_lists_focused_adjacent_and_broader_checks() -> None:
    report = handoff.build_round236_260_report(_green_suite_report())
    cluster = report["round246_250_validation_manifest"]

    assert cluster["version"] == "v3_round246_250_validation_manifest"
    assert cluster["rounds"] == [246, 247, 248, 249, 250]
    assert cluster["success"] is True
    assert cluster["validation_commands"]["focused"] == [handoff.FOCUSED_TEST_COMMAND]
    assert "tests/test_v3_round231_235_runtime_mapping_acceptance_repair_selection.py" in cluster["validation_commands"]["adjacent"][0]
    assert "python -m pytest --collect-only -q" in cluster["validation_commands"]["broader"]
    assert "python -m compileall -q ." in cluster["validation_commands"]["broader"]
    assert cluster["broader_validation_delta"].startswith("adds handoff-quality gates")


def test_round251_255_remaining_taxonomy_records_no_go_items() -> None:
    report = handoff.build_round236_260_report(_green_suite_report())
    cluster = report["round251_255_remaining_taxonomy"]

    assert cluster["version"] == "v3_round251_255_remaining_taxonomy"
    assert cluster["rounds"] == [251, 252, 253, 254, 255]
    assert cluster["success"] is True
    taxonomy = cluster["remaining_taxonomy"]
    assert taxonomy["accepted_for_future_operator_review"] == ["민석"]
    assert taxonomy["blocked_controls"] == ["EVE"]
    assert taxonomy["unresolved_failures"] == []
    assert "production_persistence_enablement" in taxonomy["explicit_no_go_items"]
    assert "runtime_mapping_enabled_default_true" in taxonomy["explicit_no_go_items"]
    assert "broad_lexical_auto_mapping" in taxonomy["explicit_no_go_items"]
    assert cluster["decision_record"]["runtime_mapping_production_enablement"] == "no_go"


def test_round256_260_final_report_completes_all_requested_rounds_and_stops_before_enablement() -> None:
    report = handoff.build_round236_260_report(_green_suite_report())
    final = report["round256_260_final_consolidation"]

    assert report["version"] == handoff.ROUND236_260_VERSION
    assert report["rounds_completed"] == list(range(236, 261))
    assert report["status"] == "runtime_mapping_acceptance_handoff_ready"
    assert report["success"] is True
    assert report["exit_code"] == 0
    assert report["blockers"] == []
    assert final["version"] == "v3_round256_260_final_consolidation"
    assert final["rounds_completed"] == list(range(236, 261))
    assert final["success"] is True
    assert final["cluster_statuses"] == {
        "round236_240_stage_matrix_quality_gates": True,
        "round241_245_operator_handoff_packet": True,
        "round246_250_validation_manifest": True,
        "round251_255_remaining_taxonomy": True,
    }
    assert final["next_recommendation"].startswith("stop short of production enablement")
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_round236_260_blocks_when_source_suite_is_not_green() -> None:
    bad_suite = {"success": False, "status": "blocked", "steps": []}
    report = handoff.build_round236_260_report(bad_suite)

    assert report["success"] is False
    assert report["status"] == "blocked_runtime_mapping_acceptance_handoff"
    assert report["exit_code"] == 1
    assert "round231_235_source_not_green" in report["blockers"]
