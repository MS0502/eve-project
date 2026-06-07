from __future__ import annotations

from adapters.affect_event_proposal_validator import ALL_AXIS_GROUPS
from adapters.affect_transition_checkpoint_rollback_dryrun_trace import (
    AUDIT_EVENT_SEQUENCE,
    CHECKPOINT_CAPTURE_SEQUENCE,
    FAILURE_HANDLING_SEQUENCE,
    GUARD_PRESERVATION_SEQUENCE,
    PREFLIGHT_VALIDATION_SEQUENCE,
    ROLLBACK_RESTORE_SEQUENCE,
    affect_checkpoint_rollback_execution_dryrun_trace_summary,
    build_affect_checkpoint_rollback_execution_dryrun_trace,
    validate_affect_checkpoint_rollback_execution_dryrun_trace,
)
from tests.fixtures.korean_conversation_fixtures import KOREAN_CONVERSATION_FIXTURES, fixture_texts

VALID_TRACE_CASES = (
    ("praise", {"competence_drive": 0.02}, {}),
    ("useful_criticism", {"learning_pressure": 0.02}, {}),
    ("malicious_comment", {"boundary_defense": 0.02}, {}),
    ("identity_attack", {"threat_pressure": 0.02}, {}),
    ("hardware_normal", {}, {}),
    ("hardware_low_power", {"energy_budget": -0.02}, {}),
    ("hardware_prediction_error", {"stability_need": 0.02}, {}),
    ("hardware_polling_tick", {}, {}),
    ("speech_output_pressure", {"expression_pressure": 0.02}, {}),
    ("listening_uncertainty", {"uncertainty_pressure": 0.02}, {}),
    (
        "imagination_negative_spiral",
        {"recovery_need": 0.02},
        {"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True},
    ),
    ("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02}, {}),
    ("self_model_update_candidate", {"self_coherence": 0.02}, {}),
)

FALSE_BOUNDARY_FLAGS = (
    "checkpoint_created",
    "rollback_created",
    "audit_written",
    "checkpoint_write_allowed",
    "rollback_write_allowed",
    "audit_write_allowed",
    "dryrun_apply_allowed",
    "live_apply_allowed",
    "apply_permission_granted",
    "state_mutation_performed",
    "runtime_mutation_performed",
    "memory_write_performed",
    "persistence_write_performed",
    "vector_read_performed",
    "vector_load_performed",
    "artifact_created_or_staged",
    "agp_bypass_allowed",
    "fallback_bypass_allowed",
    "runtime_mapping_enabled_default",
    "enforcement_enabled_default",
    "production_persistence_enabled",
    "vector_contents_read",
    "vectors_loaded",
)


def _assert_read_only_trace(trace):
    assert trace["checkpoint_capture_simulated"] is True
    assert trace["rollback_restore_simulated"] is True
    for flag in FALSE_BOUNDARY_FLAGS:
        assert trace[flag] is False, flag
    assert trace["global_synchrony_blocked"] is True
    assert trace["preflight_validation_sequence"]
    assert trace["checkpoint_capture_sequence"]
    assert trace["bounded_apply_candidate_sequence"]
    assert trace["rollback_restore_sequence"]
    assert trace["audit_event_sequence"]
    assert trace["failure_handling_sequence"]
    assert trace["guard_preservation_sequence"]
    assert trace["deterministic_sequence_id_policy"]
    assert trace["integrity_hash_policy"]
    assert trace["no_write_guarantee"]
    assert trace["no_apply_guarantee"]
    assert trace["audit_event_schema"]
    assert all(row["candidate_apply_performed"] is False for row in trace["bounded_apply_candidate_sequence"])
    assert all(row["state_mutation_performed"] is False for row in trace["bounded_apply_candidate_sequence"])


def test_known_valid_checkpoint_rollback_plans_create_read_only_execution_dryrun_traces():
    for event_category, deltas, metadata in VALID_TRACE_CASES:
        trace = build_affect_checkpoint_rollback_execution_dryrun_trace(event_category, deltas, metadata)
        assert trace["trace_passed"] is True, (event_category, trace["blocked_reasons"])
        assert trace["checkpoint_plan_passed"] is True
        assert trace["rollback_plan_passed"] is True
        assert trace["execution_trace_built"] is True
        _assert_read_only_trace(trace)
        assert validate_affect_checkpoint_rollback_execution_dryrun_trace(trace)["validation_passed"] is True


def test_unknown_event_category_fails_closed_and_plan_failure_blocks_trace_readiness():
    unknown = build_affect_checkpoint_rollback_execution_dryrun_trace("unknown_event_category", {})
    assert unknown["trace_passed"] is False
    assert unknown["execution_trace_built"] is False
    assert "unknown_event_category_fail_closed" in unknown["blocked_reasons"]
    assert "checkpoint_rollback_plan_failure_blocks_trace_readiness" in unknown["blocked_reasons"]
    assert "rollback_plan_failure_blocks_trace_readiness" in unknown["blocked_reasons"]
    assert validate_affect_checkpoint_rollback_execution_dryrun_trace(unknown)["validation_passed"] is False
    _assert_read_only_trace(unknown)

    bridge_failure = build_affect_checkpoint_rollback_execution_dryrun_trace(
        "praise", {"competence_drive": 0.02}, {"dryrun_bridge_failure_simulated": True}
    )
    assert bridge_failure["trace_passed"] is False
    assert bridge_failure["checkpoint_plan_passed"] is False
    assert "checkpoint_rollback_plan_failure_blocks_trace_readiness" in bridge_failure["blocked_reasons"]
    _assert_read_only_trace(bridge_failure)


def test_trace_success_does_not_imply_checkpoint_rollback_audit_or_apply_permission():
    trace = build_affect_checkpoint_rollback_execution_dryrun_trace("praise", {"competence_drive": 0.02})
    assert trace["trace_passed"] is True
    for flag in (
        "checkpoint_created",
        "rollback_created",
        "audit_written",
        "dryrun_apply_allowed",
        "live_apply_allowed",
        "apply_permission_granted",
    ):
        assert trace[flag] is False
    assert any("does_not_create_checkpoint" in warning for warning in trace["warnings"])
    assert any("does_not_create_rollback" in warning for warning in trace["warnings"])
    assert any("does_not_write_audit" in warning for warning in trace["warnings"])


def test_traces_never_request_runtime_persistence_memory_vector_or_agp_fallback_bypass():
    trace = build_affect_checkpoint_rollback_execution_dryrun_trace(
        "praise",
        {"competence_drive": 0.02},
        {
            "runtime_mutation_requested": True,
            "persistence_write_requested": True,
            "memory_write_requested": True,
            "vector_read_requested": True,
            "vector_load_requested": True,
            "agp_bypass_requested": True,
            "fallback_bypass_requested": True,
        },
    )
    assert trace["trace_passed"] is False
    for reason in (
        "runtime_mutation_request_blocks_checkpoint_plan",
        "persistence_request_blocks_checkpoint_plan",
        "memory_write_request_blocks_checkpoint_plan",
        "vector_read_or_load_request_blocks_checkpoint_plan",
        "agp_or_fallback_bypass_request_blocks_checkpoint_plan",
    ):
        assert reason in trace["blocked_reasons"]
    _assert_read_only_trace(trace)


def test_hostile_social_traces_preserve_quarantine_appraisal_gate_and_block_identity_memory_updates():
    for event_category in ("malicious_comment", "identity_attack"):
        deltas = {"boundary_defense" if event_category == "malicious_comment" else "threat_pressure": 0.02}
        trace = build_affect_checkpoint_rollback_execution_dryrun_trace(event_category, deltas)
        guards = trace["guard_preservation_result"]
        assert trace["trace_passed"] is True
        assert guards["hostile_social_trace"] is True
        assert guards["hostile_social_preserves_quarantine_appraisal_gate"] is True
        assert guards["hostile_social_blocks_identity_self_memory_update"] is True
        assert trace["memory_write_performed"] is False

    blocked = build_affect_checkpoint_rollback_execution_dryrun_trace(
        "identity_attack", {"threat_pressure": 0.02}, {"core_identity_update_requested": True}
    )
    assert blocked["trace_passed"] is False
    assert "self_or_memory_update_request_blocks_hostile_social_event" in blocked["blocked_reasons"]


def test_useful_criticism_requires_appraisal_before_memory_or_self_model_update():
    trace = build_affect_checkpoint_rollback_execution_dryrun_trace("useful_criticism", {"learning_pressure": 0.02})
    assert trace["trace_passed"] is True
    assert trace["guard_preservation_result"]["useful_criticism_preserves_appraisal_before_memory_self_model"] is True

    blocked = build_affect_checkpoint_rollback_execution_dryrun_trace(
        "useful_criticism", {"learning_pressure": 0.02}, {"self_model_update_requested": True}
    )
    assert blocked["trace_passed"] is False
    assert "useful_criticism_requires_appraisal_before_memory_or_self_model_update" in blocked["blocked_reasons"]


def test_hardware_trace_boundaries_zero_delta_operational_non_panic_diagnostic_and_polling_safe():
    normal = build_affect_checkpoint_rollback_execution_dryrun_trace("hardware_normal", {})
    assert normal["trace_passed"] is True
    assert normal["proposed_axis_deltas"] == {}
    assert normal["guard_preservation_result"]["hardware_normal_zero_delta_required"] is True

    normal_bad = build_affect_checkpoint_rollback_execution_dryrun_trace("hardware_normal", {"stability_need": 0.01})
    assert normal_bad["trace_passed"] is False
    assert "hardware_normal_must_remain_zero_delta" in normal_bad["blocked_reasons"]

    social_axis = build_affect_checkpoint_rollback_execution_dryrun_trace("hardware_low_power", {"social_trust": 0.01})
    assert social_axis["trace_passed"] is False
    assert "hardware_events_cannot_target_social_self_identity_axes" in social_axis["blocked_reasons"]

    low_power = build_affect_checkpoint_rollback_execution_dryrun_trace("hardware_low_power", {"energy_budget": -0.02})
    assert low_power["trace_passed"] is True
    assert low_power["guard_preservation_result"]["hardware_non_panic_operational_only"] is True

    prediction_error = build_affect_checkpoint_rollback_execution_dryrun_trace("hardware_prediction_error", {"stability_need": 0.02})
    assert prediction_error["trace_passed"] is True
    assert prediction_error["guard_preservation_result"]["hardware_prediction_error_diagnostic_operational_only"] is True

    polling = build_affect_checkpoint_rollback_execution_dryrun_trace(
        "hardware_polling_tick", {}, {"recursive_concern_loop_requested": True}
    )
    assert polling["trace_passed"] is False
    assert "hardware_polling_tick_cannot_create_recursive_concern_loop" in polling["blocked_reasons"]


def test_speech_listening_imagination_and_memory_self_guards_are_preserved():
    speech = build_affect_checkpoint_rollback_execution_dryrun_trace("speech_output_pressure", {"expression_pressure": 0.02})
    assert speech["trace_passed"] is True
    assert speech["guard_preservation_result"]["speech_pressure_preserves_agp_fallback_gates"] is True
    assert speech["agp_bypass_allowed"] is False
    assert speech["fallback_bypass_allowed"] is False

    listening = build_affect_checkpoint_rollback_execution_dryrun_trace(
        "listening_uncertainty", {"uncertainty_pressure": 0.02}, {"relabel_neutral_as_hostile_requested": True}
    )
    assert listening["trace_passed"] is False
    assert "listening_uncertainty_cannot_relabel_neutral_as_hostile" in listening["blocked_reasons"]

    imagination = build_affect_checkpoint_rollback_execution_dryrun_trace("imagination_negative_spiral", {"recovery_need": 0.02})
    assert imagination["trace_passed"] is False
    assert "imagination_negative_spiral_requires_budget_cooldown_reality_check" in imagination["blocked_reasons"]

    memory = build_affect_checkpoint_rollback_execution_dryrun_trace("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02})
    assert memory["trace_passed"] is True
    assert memory["guard_preservation_result"]["memory_self_update_preserves_appraisal_quarantine"] is True
    assert memory["memory_write_performed"] is False


def test_one_event_cannot_build_all_axis_execution_traces_and_global_synchrony_remains_blocked():
    all_axis_like = {axis: 0.01 for axis in tuple(sorted(ALL_AXIS_GROUPS))[: len(ALL_AXIS_GROUPS)]}
    trace = build_affect_checkpoint_rollback_execution_dryrun_trace("praise", all_axis_like, {"global_synchrony_requested": True})
    assert trace["trace_passed"] is False
    assert "one_event_cannot_build_all_axis_checkpoint_plans" in trace["blocked_reasons"]
    assert "global_synchrony_request_blocks_checkpoint_plan" in trace["blocked_reasons"]
    assert len(trace["bounded_apply_candidate_sequence"]) == len(ALL_AXIS_GROUPS)
    assert validate_affect_checkpoint_rollback_execution_dryrun_trace(trace)["validation_passed"] is False
    assert trace["global_synchrony_blocked"] is True


def test_trace_schema_includes_all_required_sequences_policies_and_guarantees():
    summary = affect_checkpoint_rollback_execution_dryrun_trace_summary()
    schema = summary["trace_schema_summary"]
    assert schema["preflight_validation_sequence"] == PREFLIGHT_VALIDATION_SEQUENCE
    assert schema["checkpoint_capture_sequence"] == CHECKPOINT_CAPTURE_SEQUENCE
    assert schema["rollback_restore_sequence"] == ROLLBACK_RESTORE_SEQUENCE
    assert schema["audit_event_sequence"] == AUDIT_EVENT_SEQUENCE
    assert schema["failure_handling_sequence"] == FAILURE_HANDLING_SEQUENCE
    assert schema["guard_preservation_sequence"] == GUARD_PRESERVATION_SEQUENCE
    assert schema["deterministic_sequence_id_policy"]
    assert schema["integrity_hash_policy"]
    assert schema["no_write_guarantee"]
    assert schema["no_apply_guarantee"]
    assert any("memory_write_guard" in step for step in GUARD_PRESERVATION_SEQUENCE)
    assert any("persistence_guard" in step for step in GUARD_PRESERVATION_SEQUENCE)
    assert any("vector_content" in step for step in GUARD_PRESERVATION_SEQUENCE)


def test_no_live_emotion_hormone_memory_mutation_persistence_runtime_or_vector_enablement_occurs():
    before = build_affect_checkpoint_rollback_execution_dryrun_trace("praise", {"competence_drive": 0.02})
    after = build_affect_checkpoint_rollback_execution_dryrun_trace("praise", {"competence_drive": 0.02})
    assert before == after
    _assert_read_only_trace(before)


def test_korean_fixtures_and_minseok_token_remain_preserved():
    assert len(KOREAN_CONVERSATION_FIXTURES) == 20
    assert "EVE 프로젝트" in fixture_texts()
    token = "민석"
    trace = build_affect_checkpoint_rollback_execution_dryrun_trace("praise", {"competence_drive": 0.02}, {"notes": token})
    assert token == "민석"
    assert trace["trace_passed"] is True
    assert trace["artifact_created_or_staged"] is False
