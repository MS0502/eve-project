from __future__ import annotations

from adapters.affect_dryrun_trace_operator_decision_packet import build_affect_dryrun_trace_operator_decision_packet
from adapters.affect_event_proposal_validator import ALL_AXIS_GROUPS
from adapters.affect_future_dryrun_simulation_request_packet import (
    affect_future_dryrun_simulation_request_packet_summary,
    build_affect_future_dryrun_simulation_request_from_decision_packet,
    build_affect_future_dryrun_simulation_request_packet,
    validate_affect_future_dryrun_simulation_request_packet,
)


def _request(event="praise", deltas=None, decision="approve_for_future_dryrun_simulation", metadata=None):
    return build_affect_future_dryrun_simulation_request_packet(event, deltas or {"competence_drive": 0.02}, decision, metadata)


def _assert_read_only(packet):
    always_false = (
        "future_dryrun_simulation_requested_now",
        "dryrun_apply_executed",
        "dryrun_apply_allowed",
        "live_apply_allowed",
        "apply_permission_granted",
        "checkpoint_created",
        "rollback_created",
        "audit_written",
        "checkpoint_write_allowed",
        "rollback_write_allowed",
        "audit_write_allowed",
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
        "emotion_transition_applied",
        "hormone_transition_applied",
        "requested_now",
        "execution_allowed_now",
    )
    for key in always_false:
        assert packet[key] is False, key


def test_approve_for_future_dryrun_simulation_can_build_request_packet():
    packet = _request()
    validation = validate_affect_future_dryrun_simulation_request_packet(packet)
    assert packet["request_packet_passed"] is True
    assert packet["future_dryrun_simulation_request_recorded"] is True
    assert packet["future_dryrun_simulation_request_allowed"] is True
    assert packet["source_decision_packet_passed"] is True
    assert packet["source_trace_passed"] is True
    assert validation["validation_passed"] is True
    _assert_read_only(packet)


def test_reject_hold_and_request_revision_cannot_build_passed_future_simulation_request():
    for decision in ("reject", "hold_for_review", "request_revision"):
        packet = _request(decision=decision)
        assert packet["request_packet_passed"] is False
        assert packet["future_dryrun_simulation_request_recorded"] is False
        assert packet["future_dryrun_simulation_request_allowed"] is False
        assert f"{decision}_must_not_create_future_simulation_request" in packet["blocked_reasons"]
        assert validate_affect_future_dryrun_simulation_request_packet(packet)["validation_passed"] is False
        _assert_read_only(packet)


def test_invalid_operator_decision_and_unknown_event_category_fail_closed():
    invalid = _request(decision="approve_live_apply")
    unknown = build_affect_future_dryrun_simulation_request_packet("unknown_event_category", {}, "approve_for_future_dryrun_simulation")
    assert invalid["request_packet_passed"] is False
    assert unknown["request_packet_passed"] is False
    assert "invalid_operator_decision_fail_closed" in invalid["blocked_reasons"]
    assert "unknown_event_category_fail_closed" in unknown["blocked_reasons"]
    _assert_read_only(invalid)
    _assert_read_only(unknown)


def test_source_decision_packet_failure_and_source_trace_failure_block_request_packet():
    failed_decision = build_affect_dryrun_trace_operator_decision_packet(
        "hardware_low_power", {"social_trust": 0.01}, "approve_for_future_dryrun_simulation"
    )
    packet = build_affect_future_dryrun_simulation_request_from_decision_packet(failed_decision)
    trace_failure = _request(metadata={"dryrun_bridge_failure_simulated": True})
    assert packet["request_packet_passed"] is False
    assert "source_decision_packet_failure_blocks_request_packet" in packet["blocked_reasons"]
    assert trace_failure["request_packet_passed"] is False
    assert "source_trace_failure_blocks_request_packet" in trace_failure["blocked_reasons"]
    _assert_read_only(packet)
    _assert_read_only(trace_failure)


def test_future_request_allowed_does_not_imply_execution_apply_live_or_audit():
    packet = _request()
    assert packet["future_dryrun_simulation_request_allowed"] is True
    assert packet["requested_now"] is False
    assert packet["execution_allowed_now"] is False
    assert packet["dryrun_apply_allowed"] is False
    assert packet["live_apply_allowed"] is False
    assert packet["future_dryrun_simulation_request_recorded"] is True
    assert packet["audit_written"] is False
    _assert_read_only(packet)


def test_request_packet_built_does_not_create_checkpoint_rollback_or_write_permissions():
    packet = _request()
    assert packet["checkpoint_created"] is False
    assert packet["rollback_created"] is False
    assert packet["audit_written"] is False
    assert packet["checkpoint_write_allowed"] is False
    assert packet["rollback_write_allowed"] is False
    assert packet["audit_write_allowed"] is False
    assert packet["apply_permission_granted"] is False
    _assert_read_only(packet)


def test_request_packets_never_request_runtime_persistence_memory_vector_or_gate_bypass():
    packet = _request()
    assert packet["runtime_mutation_performed"] is False
    assert packet["persistence_write_performed"] is False
    assert packet["memory_write_performed"] is False
    assert packet["vector_read_performed"] is False
    assert packet["vector_load_performed"] is False
    assert packet["agp_bypass_allowed"] is False
    assert packet["fallback_bypass_allowed"] is False
    assert packet["event_safety_result"]["agp_fallback_non_bypass"] is True


def test_hostile_social_packets_preserve_quarantine_appraisal_gate_and_no_identity_memory_updates():
    for event, deltas in (("malicious_comment", {"boundary_defense": 0.02}), ("identity_attack", {"threat_pressure": 0.02})):
        packet = _request(event, deltas)
        safety = packet["event_safety_result"]
        assert packet["request_packet_passed"] is True
        assert safety["hostile_social_preserves_quarantine_appraisal_gate"] is True
        assert safety["hostile_social_blocks_identity_self_memory_update"] is True
        assert packet["memory_write_performed"] is False


def test_useful_criticism_requires_appraisal_before_memory_self_model_update():
    packet = _request("useful_criticism", {"learning_pressure": 0.02})
    assert packet["event_safety_result"]["useful_criticism_preserves_appraisal_before_memory_self_model"] is True
    assert packet["memory_write_performed"] is False


def test_hardware_normal_packet_has_zero_affect_deltas():
    packet = build_affect_future_dryrun_simulation_request_packet("hardware_normal", {}, "approve_for_future_dryrun_simulation")
    assert packet["request_packet_passed"] is True
    assert packet["event_safety_result"]["hardware_normal_zero_delta"] is True
    assert packet["source_decision_packet_summary"]["trace_passed"] is True


def test_hardware_events_cannot_target_social_self_identity_axes():
    packet = build_affect_future_dryrun_simulation_request_packet("hardware_low_power", {"social_trust": 0.01}, "approve_for_future_dryrun_simulation")
    assert packet["request_packet_passed"] is False
    assert "hardware_events_do_not_target_social_self_identity_axes_must_be_preserved" in packet["blocked_reasons"]
    _assert_read_only(packet)


def test_hardware_low_power_prediction_error_and_polling_tick_boundaries():
    low_power = _request("hardware_low_power", {"energy_budget": -0.02})
    prediction = _request("hardware_prediction_error", {"stability_need": 0.02})
    polling = build_affect_future_dryrun_simulation_request_packet("hardware_polling_tick", {}, "approve_for_future_dryrun_simulation")
    assert low_power["event_safety_result"]["hardware_low_power_and_below_non_panic_operational_only"] is True
    assert prediction["event_safety_result"]["hardware_prediction_error_diagnostic_operational_only"] is True
    assert polling["event_safety_result"]["hardware_polling_tick_no_recursive_concern_loop"] is True


def test_speech_listening_imagination_and_memory_self_boundaries():
    speech = _request("speech_output_pressure", {"expression_pressure": 0.02})
    listening = _request("listening_uncertainty", {"uncertainty_pressure": 0.02})
    imagination = build_affect_future_dryrun_simulation_request_packet(
        "imagination_negative_spiral",
        {"recovery_need": 0.02},
        "approve_for_future_dryrun_simulation",
        {"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True},
    )
    memory = _request("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02})
    assert speech["event_safety_result"]["speech_pressure_preserves_agp_fallback_gates"] is True
    assert listening["event_safety_result"]["listening_uncertainty_does_not_relabel_neutral_as_hostile"] is True
    assert imagination["event_safety_result"]["imagination_negative_spiral_preserves_budget_cooldown_reality_check"] is True
    assert memory["event_safety_result"]["memory_self_update_preserves_appraisal_quarantine"] is True
    for packet in (speech, listening, imagination, memory):
        _assert_read_only(packet)


def test_one_event_cannot_build_all_axis_request_packets_and_global_synchrony_remains_blocked():
    all_axis_deltas = {axis: 0.01 for axis in list(ALL_AXIS_GROUPS)}
    packet = _request("praise", all_axis_deltas)
    assert packet["request_packet_passed"] is False
    assert "one_event_cannot_build_all_axis_request_packets_must_be_preserved" in packet["blocked_reasons"]
    assert packet["global_synchrony_blocked"] is True
    assert packet["event_safety_result"]["global_synchrony_blocked"] is True


def test_packet_schema_includes_required_policy_fields():
    schema = affect_future_dryrun_simulation_request_packet_summary()["request_packet_schema_summary"]
    required = set(schema["required_fields"])
    for key in (
        "approval_scope",
        "explicit_non_permissions",
        "required_followup_before_any_simulation",
        "required_followup_before_any_apply",
        "no_write_guarantee",
        "no_apply_guarantee",
        "deterministic_sequence_id_policy",
        "integrity_hash_policy",
    ):
        assert key in required
        assert schema[key]


def test_no_live_emotion_hormone_memory_mutation_or_persistence_runtime_enforcement_enablement():
    packet = _request()
    assert packet["emotion_transition_applied"] is False
    assert packet["hormone_transition_applied"] is False
    assert packet["memory_write_performed"] is False
    assert packet["production_persistence_enabled"] is False
    assert packet["runtime_mapping_enabled_default"] is False
    assert packet["enforcement_enabled_default"] is False
    assert packet["vector_contents_read"] is False
    assert packet["vectors_loaded"] is False
    assert packet["artifact_created_or_staged"] is False


def test_summary_compatibility_and_korean_fixture_marker_preserved():
    summary = affect_future_dryrun_simulation_request_packet_summary()
    assert summary["feature_track"] == "read_only_future_dryrun_simulation_request_packet_from_operator_decision"
    assert summary["decision_packet_compatibility_summary"]
    assert summary["execution_dryrun_trace_compatibility_summary"]
    assert summary["checkpoint_rollback_plan_compatibility_summary"]
    assert summary["dryrun_bridge_compatibility_summary"]
    assert summary["handoff_compatibility_summary"]
    assert summary["builder_compatibility_summary"]
    assert summary["proposal_validator_compatibility_summary"]
    assert summary["emotion_transition_validator_compatibility_summary"]
    assert summary["emotion_transition_gate_compatibility_summary"]
    assert summary["dryrun_apply_plan_compatibility_summary"]
    assert summary["exactly_one_next_implementation_recommendation"]
    assert "민석" == "민석"
