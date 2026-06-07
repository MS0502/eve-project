from __future__ import annotations

from adapters.affect_dryrun_trace_operator_decision_packet import (
    EXPLICIT_NON_PERMISSIONS,
    REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
    SUPPORTED_OPERATOR_DECISIONS,
    affect_dryrun_trace_operator_decision_packet_summary,
    build_affect_dryrun_trace_operator_decision_packet,
    validate_affect_dryrun_trace_operator_decision_packet,
)
from adapters.affect_event_proposal_validator import ALL_AXIS_GROUPS
from pathlib import Path

from tests.fixtures.korean_conversation_fixtures import KOREAN_CONVERSATION_FIXTURES, fixture_texts

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


def _packet(event="praise", deltas=None, decision="approve_for_future_dryrun_simulation", metadata=None):
    return build_affect_dryrun_trace_operator_decision_packet(event, deltas or {"competence_drive": 0.02}, decision, metadata)


def _assert_read_only_packet(packet):
    for flag in FALSE_BOUNDARY_FLAGS:
        assert packet[flag] is False, flag
    assert packet["global_synchrony_blocked"] is True
    assert packet["approval_scope"]
    assert packet["explicit_non_permissions"] == EXPLICIT_NON_PERMISSIONS
    assert packet["required_followup_before_any_apply"] == REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY
    assert packet["deterministic_sequence_id_policy"]
    assert packet["integrity_hash_policy"]
    assert packet["no_write_guarantee"]
    assert packet["no_apply_guarantee"]
    assert packet["bounded_apply_candidate_sequence_summary"]["candidate_apply_performed_any"] is False
    assert packet["bounded_apply_candidate_sequence_summary"]["state_mutation_performed_any"] is False


def test_valid_traces_can_build_supported_decision_packets():
    for decision in SUPPORTED_OPERATOR_DECISIONS:
        packet = _packet(decision=decision)
        assert packet["packet_passed"] is True, (decision, packet["blocked_reasons"])
        assert packet["decision_recorded"] is True
        assert packet["decision_packet_built"] is True
        assert packet["trace_passed"] is True
        assert packet["trace_accepted"] is True
        assert validate_affect_dryrun_trace_operator_decision_packet(packet)["validation_passed"] is True
        _assert_read_only_packet(packet)


def test_decision_flags_are_exclusive_for_supported_decisions():
    expectations = {
        "approve_for_future_dryrun_simulation": "approve_for_future_dryrun_simulation",
        "reject": "reject_requested",
        "hold_for_review": "hold_for_review_requested",
        "request_revision": "revision_requested",
    }
    for decision, flag in expectations.items():
        packet = _packet(decision=decision)
        assert packet[flag] is True
        assert sum(int(packet[name]) for name in expectations.values()) == 1


def test_invalid_operator_decision_and_unknown_event_category_fail_closed():
    invalid = _packet(decision="approve_live_apply")
    assert invalid["packet_passed"] is False
    assert invalid["decision_recorded"] is False
    assert "invalid_operator_decision_fail_closed" in invalid["blocked_reasons"]
    assert validate_affect_dryrun_trace_operator_decision_packet(invalid)["validation_passed"] is False
    _assert_read_only_packet(invalid)

    unknown = build_affect_dryrun_trace_operator_decision_packet("unknown_event_category", {}, "reject")
    assert unknown["packet_passed"] is False
    assert unknown["trace_passed"] is False
    assert "unknown_event_category_fail_closed" in unknown["blocked_reasons"]
    assert validate_affect_dryrun_trace_operator_decision_packet(unknown)["validation_passed"] is False
    _assert_read_only_packet(unknown)


def test_trace_failure_blocks_approval_but_not_read_only_boundaries():
    packet = _packet(metadata={"dryrun_bridge_failure_simulated": True})
    assert packet["packet_passed"] is False
    assert packet["trace_passed"] is False
    assert "trace_failure_blocks_approve_for_future_dryrun_simulation" in packet["blocked_reasons"]
    _assert_read_only_packet(packet)


def test_approve_for_future_dryrun_simulation_does_not_imply_apply_permission():
    packet = _packet()
    assert packet["approve_for_future_dryrun_simulation"] is True
    assert packet["dryrun_apply_allowed"] is False
    assert packet["live_apply_allowed"] is False
    assert packet["apply_permission_granted"] is False


def test_reject_hold_and_request_revision_grant_no_apply_permission():
    for decision in ("reject", "hold_for_review", "request_revision"):
        packet = _packet(decision=decision)
        assert packet["dryrun_apply_allowed"] is False
        assert packet["live_apply_allowed"] is False
        assert packet["apply_permission_granted"] is False
        _assert_read_only_packet(packet)


def test_decision_recorded_and_packet_built_do_not_imply_audit_checkpoint_or_rollback():
    packet = _packet()
    assert packet["decision_recorded"] is True
    assert packet["decision_packet_built"] is True
    assert packet["audit_written"] is False
    assert packet["checkpoint_created"] is False
    assert packet["rollback_created"] is False
    assert packet["checkpoint_write_allowed"] is False
    assert packet["rollback_write_allowed"] is False
    assert packet["audit_write_allowed"] is False


def test_packets_never_request_runtime_persistence_memory_vector_or_gate_bypass():
    packet = _packet()
    assert packet["runtime_mutation_performed"] is False
    assert packet["persistence_write_performed"] is False
    assert packet["memory_write_performed"] is False
    assert packet["vector_read_performed"] is False
    assert packet["vector_load_performed"] is False
    assert packet["vector_contents_read"] is False
    assert packet["vectors_loaded"] is False
    assert packet["agp_bypass_allowed"] is False
    assert packet["fallback_bypass_allowed"] is False
    assert packet["event_safety_result"]["agp_fallback_non_bypass"] is True


def test_hostile_social_packets_preserve_quarantine_appraisal_gate_and_no_identity_memory_updates():
    for event, deltas in (("malicious_comment", {"boundary_defense": 0.02}), ("identity_attack", {"threat_pressure": 0.02})):
        packet = _packet(event, deltas, "hold_for_review")
        assert packet["packet_passed"] is True
        safety = packet["event_safety_result"]
        assert safety["hostile_social_preserves_quarantine_appraisal_gate"] is True
        assert safety["hostile_social_blocks_identity_self_memory_update"] is True
        assert packet["memory_write_performed"] is False
        assert packet["event_safety_result"]["global_synchrony_blocked"] is True


def test_useful_criticism_requires_appraisal_before_memory_self_model_update():
    packet = _packet("useful_criticism", {"learning_pressure": 0.02}, "hold_for_review")
    assert packet["packet_passed"] is True
    assert packet["event_safety_result"]["useful_criticism_preserves_appraisal_before_memory_self_model"] is True
    assert packet["memory_write_performed"] is False


def test_hardware_normal_packet_has_zero_affect_deltas():
    packet = build_affect_dryrun_trace_operator_decision_packet("hardware_normal", {}, "approve_for_future_dryrun_simulation")
    assert packet["packet_passed"] is True
    assert packet["event_safety_result"]["hardware_normal_zero_delta"] is True
    assert packet["bounded_apply_candidate_sequence_summary"]["axes"] == ("no_axis_delta_candidate",)


def test_hardware_events_cannot_target_social_self_identity_axes():
    packet = build_affect_dryrun_trace_operator_decision_packet("hardware_low_power", {"social_trust": 0.01}, "hold_for_review")
    assert packet["packet_passed"] is False
    assert "hardware_events_do_not_target_social_self_identity_axes_must_be_preserved" in packet["blocked_reasons"]
    _assert_read_only_packet(packet)


def test_hardware_low_power_prediction_error_and_polling_tick_boundaries():
    low_power = build_affect_dryrun_trace_operator_decision_packet("hardware_low_power", {"energy_budget": -0.02}, "hold_for_review")
    prediction = build_affect_dryrun_trace_operator_decision_packet("hardware_prediction_error", {"stability_need": 0.02}, "hold_for_review")
    polling = build_affect_dryrun_trace_operator_decision_packet("hardware_polling_tick", {}, "hold_for_review")
    assert low_power["event_safety_result"]["hardware_low_power_and_below_non_panic_operational_only"] is True
    assert prediction["event_safety_result"]["hardware_prediction_error_diagnostic_operational_only"] is True
    assert polling["event_safety_result"]["hardware_polling_tick_no_recursive_concern_loop"] is True
    assert low_power["packet_passed"] is True
    assert prediction["packet_passed"] is True
    assert polling["packet_passed"] is True


def test_speech_listening_imagination_and_memory_self_boundaries():
    speech = _packet("speech_output_pressure", {"expression_pressure": 0.02}, "hold_for_review")
    listening = _packet("listening_uncertainty", {"uncertainty_pressure": 0.02}, "hold_for_review")
    imagination = build_affect_dryrun_trace_operator_decision_packet(
        "imagination_negative_spiral",
        {"recovery_need": 0.02},
        "hold_for_review",
        {"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True},
    )
    memory = _packet("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02}, "hold_for_review")
    assert speech["event_safety_result"]["speech_pressure_preserves_agp_fallback_gates"] is True
    assert listening["event_safety_result"]["listening_uncertainty_does_not_relabel_neutral_as_hostile"] is True
    assert imagination["event_safety_result"]["imagination_negative_spiral_preserves_budget_cooldown_reality_check"] is True
    assert memory["event_safety_result"]["memory_self_update_preserves_appraisal_quarantine"] is True
    for packet in (speech, listening, imagination, memory):
        assert packet["packet_passed"] is True
        assert packet["memory_write_performed"] is False


def test_one_event_cannot_build_all_axis_decision_packets_and_global_synchrony_remains_blocked():
    all_axis_deltas = {axis: 0.01 for axis in list(ALL_AXIS_GROUPS)[: len(ALL_AXIS_GROUPS)]}
    packet = _packet("praise", all_axis_deltas, "hold_for_review")
    assert packet["packet_passed"] is False
    assert "one_event_cannot_build_all_axis_decision_packets_must_be_preserved" in packet["blocked_reasons"]
    assert packet["global_synchrony_blocked"] is True
    assert packet["event_safety_result"]["global_synchrony_blocked"] is True


def test_packet_schema_includes_required_review_nonpermission_and_integrity_fields():
    packet = _packet()
    for key in (
        "approval_scope",
        "explicit_non_permissions",
        "required_followup_before_any_apply",
        "no_write_guarantee",
        "no_apply_guarantee",
        "deterministic_sequence_id_policy",
        "integrity_hash_policy",
    ):
        assert packet[key]
    schema = affect_dryrun_trace_operator_decision_packet_summary()["decision_packet_schema_summary"]
    for key in (
        "approval_scope",
        "explicit_non_permissions",
        "required_followup_before_any_apply",
        "deterministic_sequence_id_policy",
        "integrity_hash_policy",
        "no_write_guarantee",
        "no_apply_guarantee",
    ):
        assert key in schema


def test_no_live_emotion_hormone_memory_mutation_or_persistence_enablement_defaults():
    packet = _packet()
    assert packet["state_mutation_performed"] is False
    assert packet["runtime_mutation_performed"] is False
    assert packet["memory_write_performed"] is False
    assert packet["persistence_write_performed"] is False
    assert packet["production_persistence_enabled"] is False
    assert packet["runtime_mapping_enabled_default"] is False
    assert packet["enforcement_enabled_default"] is False
    assert packet["vector_contents_read"] is False
    assert packet["vectors_loaded"] is False
    assert packet["artifact_created_or_staged"] is False


def test_summary_exposes_compatibility_and_safety_rules():
    summary = affect_dryrun_trace_operator_decision_packet_summary()
    assert summary["feature_track"] == "read_only_operator_decision_packet_for_affect_dryrun_trace"
    assert summary["supported_decisions"] == SUPPORTED_OPERATOR_DECISIONS
    assert summary["decision_safety_rules"]["global_synchrony_blocked"] is True
    assert summary["decision_safety_rules"]["approval_does_not_grant_live_apply_permission"] is True
    assert summary["execution_trace_compatibility_summary"]["trace_version"]
    for flag in FALSE_BOUNDARY_FLAGS:
        assert summary[flag] is False


def test_korean_fixtures_and_minseok_remain_preserved():
    texts = fixture_texts()
    assert texts == [row["text"] for row in KOREAN_CONVERSATION_FIXTURES]
    assert any(row["category"] == "minsok" for row in KOREAN_CONVERSATION_FIXTURES)
    assert "민석" in Path("adapters/user_presence_adapter.py").read_text(encoding="utf-8")
