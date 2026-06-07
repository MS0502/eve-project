from copy import deepcopy

from adapters.affect_event_proposal_validator import ALL_AXIS_GROUPS
from adapters.affect_event_to_axis_proposal_map import event_to_axis_proposal_map
from adapters.affect_future_dryrun_simulation_request_packet import build_affect_future_dryrun_simulation_request_packet
from adapters.affect_future_dryrun_simulation_runner_preflight import (
    FEATURE_TRACK,
    build_affect_future_dryrun_simulation_runner_preflight,
    build_affect_future_dryrun_simulation_runner_preflight_from_request_packet,
    affect_future_dryrun_simulation_runner_preflight_summary,
    validate_affect_future_dryrun_simulation_runner_preflight,
)


def _preflight(event="praise", deltas=None, decision="approve_for_future_dryrun_simulation", metadata=None):
    return build_affect_future_dryrun_simulation_runner_preflight(event, deltas or {"competence_drive": 0.02}, decision, metadata)


def _request(**overrides):
    packet = build_affect_future_dryrun_simulation_request_packet("praise", {"competence_drive": 0.02}, "approve_for_future_dryrun_simulation")
    packet.update(overrides)
    return packet


def _assert_no_write_no_apply(preflight):
    for key in (
        "future_simulation_execution_requested_now",
        "future_simulation_executed",
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
        "execution_requested_now",
        "execution_allowed_now",
    ):
        assert preflight[key] is False, key


def test_valid_request_packet_can_build_runner_preflight():
    request = build_affect_future_dryrun_simulation_request_packet("praise", {"competence_drive": 0.02}, "approve_for_future_dryrun_simulation")
    preflight = build_affect_future_dryrun_simulation_runner_preflight_from_request_packet(request)
    validation = validate_affect_future_dryrun_simulation_runner_preflight(preflight)
    assert preflight["runner_preflight_passed"] is True
    assert preflight["runner_preflight_built"] is True
    assert preflight["request_packet_accepted"] is True
    assert preflight["future_simulation_runner_eligible"] is True
    assert validation["validation_passed"] is True
    _assert_no_write_no_apply(preflight)


def test_non_approval_request_packets_cannot_build_passed_runner_preflight():
    for decision in ("reject", "hold_for_review", "request_revision"):
        preflight = _preflight("praise", {"competence_drive": 0.02}, decision)
        assert preflight["runner_preflight_passed"] is False
        assert preflight["request_packet_accepted"] is False
        assert preflight["future_simulation_runner_eligible"] is False
        _assert_no_write_no_apply(preflight)


def test_unknown_event_category_fails_closed():
    preflight = _preflight("unknown_event", {})
    assert preflight["runner_preflight_passed"] is False
    assert "unknown_event_category_fail_closed" in preflight["blocked_reasons"]
    assert validate_affect_future_dryrun_simulation_runner_preflight(preflight)["validation_passed"] is False


def test_source_request_packet_failure_and_source_allowed_false_block_runner_preflight():
    failed = build_affect_future_dryrun_simulation_runner_preflight_from_request_packet(_request(request_packet_passed=False))
    not_allowed = build_affect_future_dryrun_simulation_runner_preflight_from_request_packet(_request(future_dryrun_simulation_request_allowed=False))
    assert failed["runner_preflight_passed"] is False
    assert "source_request_packet_failure_blocks_runner_preflight" in failed["blocked_reasons"]
    assert not_allowed["runner_preflight_passed"] is False
    assert "source_future_request_allowed_false_blocks_runner_preflight" in not_allowed["blocked_reasons"]


def test_source_execution_or_apply_permissions_block_runner_preflight():
    cases = (
        ("requested_now", True, "source_requested_now_true_blocks_runner_preflight"),
        ("future_dryrun_simulation_requested_now", True, "source_requested_now_true_blocks_runner_preflight"),
        ("dryrun_apply_allowed", True, "source_dryrun_apply_allowed_true_blocks_runner_preflight"),
        ("live_apply_allowed", True, "source_live_apply_allowed_true_blocks_runner_preflight"),
    )
    for key, value, reason in cases:
        preflight = build_affect_future_dryrun_simulation_runner_preflight_from_request_packet(_request(**{key: value}))
        assert preflight["runner_preflight_passed"] is False
        assert reason in preflight["blocked_reasons"]
        _assert_no_write_no_apply(preflight)


def test_runner_preflight_success_does_not_imply_execution_simulation_apply_or_permissions():
    preflight = _preflight()
    assert preflight["runner_preflight_passed"] is True
    assert preflight["future_simulation_execution_requested_now"] is False
    assert preflight["future_simulation_executed"] is False
    assert preflight["dryrun_apply_executed"] is False
    assert preflight["dryrun_apply_allowed"] is False
    assert preflight["live_apply_allowed"] is False
    assert preflight["apply_permission_granted"] is False


def test_artifact_checkpoint_rollback_audit_flags_are_always_false():
    preflight = _preflight()
    for key in (
        "checkpoint_created",
        "rollback_created",
        "audit_written",
        "checkpoint_write_allowed",
        "rollback_write_allowed",
        "audit_write_allowed",
        "artifact_created_or_staged",
    ):
        assert preflight[key] is False


def test_runner_preflights_never_request_runtime_persistence_memory_vector_or_bypass():
    preflight = _preflight()
    for key in (
        "runtime_mutation_performed",
        "persistence_write_performed",
        "memory_write_performed",
        "vector_read_performed",
        "vector_load_performed",
        "vector_contents_read",
        "vectors_loaded",
        "agp_bypass_allowed",
        "fallback_bypass_allowed",
    ):
        assert preflight[key] is False


def test_hostile_social_runner_preflights_preserve_quarantine_appraisal_gate_and_no_identity_memory_updates():
    for event in ("malicious_comment", "identity_attack"):
        preflight = _preflight(event, {"boundary_defense": 0.02} if event == "malicious_comment" else {"threat_pressure": 0.02})
        safety = preflight["event_safety_result"]
        assert safety["hostile_social_preserves_quarantine_appraisal_gate"] is True
        assert safety["hostile_social_blocks_identity_self_memory_update"] is True
        assert preflight["memory_write_performed"] is False
        assert preflight["state_mutation_performed"] is False


def test_useful_criticism_requires_appraisal_before_memory_self_model_update():
    preflight = _preflight("useful_criticism", {"learning_pressure": 0.02})
    assert preflight["event_safety_result"]["useful_criticism_preserves_appraisal_before_memory_self_model"] is True
    assert preflight["memory_write_performed"] is False


def test_hardware_normal_zero_delta_and_hardware_events_cannot_target_social_self_identity_axes():
    normal = build_affect_future_dryrun_simulation_runner_preflight("hardware_normal", {})
    bad = build_affect_future_dryrun_simulation_runner_preflight("hardware_low_power", {"social_trust": 0.01})
    assert normal["runner_preflight_passed"] is True
    assert normal["event_safety_result"]["hardware_normal_zero_delta"] is True
    assert bad["runner_preflight_passed"] is False
    assert "hardware_events_do_not_target_social_self_identity_axes_must_be_preserved" in bad["blocked_reasons"]


def test_hardware_low_power_prediction_error_and_polling_tick_boundaries():
    low_power = build_affect_future_dryrun_simulation_runner_preflight("hardware_low_power", {"energy_budget": -0.02})
    prediction = build_affect_future_dryrun_simulation_runner_preflight("hardware_prediction_error", {"stability_need": 0.02})
    polling = build_affect_future_dryrun_simulation_runner_preflight("hardware_polling_tick", {})
    assert low_power["event_safety_result"]["hardware_low_power_and_below_non_panic_operational_only"] is True
    assert prediction["event_safety_result"]["hardware_prediction_error_diagnostic_operational_only"] is True
    assert polling["event_safety_result"]["hardware_polling_tick_no_recursive_concern_loop"] is True


def test_speech_listening_imagination_and_memory_self_boundaries():
    speech = build_affect_future_dryrun_simulation_runner_preflight("speech_output_pressure", {"expression_pressure": 0.02})
    listening = build_affect_future_dryrun_simulation_runner_preflight("listening_uncertainty", {"uncertainty_pressure": 0.02})
    imagination = build_affect_future_dryrun_simulation_runner_preflight(
        "imagination_negative_spiral",
        {"recovery_need": 0.02},
        metadata={"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True},
    )
    memory = build_affect_future_dryrun_simulation_runner_preflight("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02})
    assert speech["event_safety_result"]["speech_pressure_preserves_agp_fallback_gates"] is True
    assert listening["event_safety_result"]["listening_uncertainty_does_not_relabel_neutral_as_hostile"] is True
    assert imagination["event_safety_result"]["imagination_negative_spiral_preserves_budget_cooldown_reality_check"] is True
    assert memory["event_safety_result"]["memory_self_update_preserves_appraisal_quarantine"] is True


def test_one_event_cannot_build_all_axis_runner_preflights_and_global_synchrony_remains_blocked():
    proposal_map = event_to_axis_proposal_map()
    all_axis_deltas = {axis: 0.01 for row in proposal_map.values() for axis in row.get("allowed_axis_deltas", {})}
    all_axis_deltas.update({f"synthetic_axis_{idx}": 0.01 for idx in range(len(ALL_AXIS_GROUPS))})
    preflight = build_affect_future_dryrun_simulation_runner_preflight("praise", all_axis_deltas)
    assert preflight["runner_preflight_passed"] is False
    assert preflight["global_synchrony_blocked"] is True
    assert preflight["event_safety_result"]["global_synchrony_blocked"] is True


def test_preflight_schema_includes_required_policy_fields():
    schema = affect_future_dryrun_simulation_runner_preflight_summary()["runner_preflight_schema_summary"]
    required = set(schema["required_fields"])
    for key in (
        "runner_eligibility_scope",
        "explicit_non_permissions",
        "required_followup_before_runner_execution",
        "required_followup_before_any_apply",
        "no_write_guarantee",
        "no_apply_guarantee",
        "deterministic_sequence_id_policy",
        "integrity_hash_policy",
    ):
        assert key in required
        assert schema[key]


def test_no_live_emotion_hormone_memory_mutation_persistence_runtime_enforcement_or_vector_load():
    preflight = _preflight()
    assert preflight["emotion_transition_applied"] is False
    assert preflight["hormone_transition_applied"] is False
    assert preflight["memory_write_performed"] is False
    assert preflight["production_persistence_enabled"] is False
    assert preflight["runtime_mapping_enabled_default"] is False
    assert preflight["enforcement_enabled_default"] is False
    assert preflight["vector_contents_read"] is False
    assert preflight["vectors_loaded"] is False
    assert preflight["artifact_created_or_staged"] is False


def test_summary_compatibility_and_korean_fixture_marker_preserved():
    summary = affect_future_dryrun_simulation_runner_preflight_summary()
    assert summary["feature_track"] == FEATURE_TRACK
    assert summary["future_request_packet_compatibility_summary"]
    assert summary["operator_decision_packet_compatibility_summary"]
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


def test_preflight_validation_blocks_tampered_write_or_apply_flags():
    preflight = _preflight()
    for key in ("future_simulation_executed", "dryrun_apply_allowed", "live_apply_allowed", "audit_written", "checkpoint_created", "rollback_created"):
        tampered = deepcopy(preflight)
        tampered[key] = True
        validation = validate_affect_future_dryrun_simulation_runner_preflight(tampered)
        assert validation["validation_passed"] is False
        assert any(key in reason for reason in validation["blocked_reasons"])
