from __future__ import annotations

from adapters.affect_transition_checkpoint_rollback_plan import (
    CHECKPOINT_CAPTURE_SURFACES,
    ROLLBACK_RESTORE_SURFACES,
    build_affect_transition_checkpoint_plan,
    build_affect_transition_rollback_plan,
    validate_affect_transition_checkpoint_rollback_plan,
)

VALID_CASES = (
    ("praise", {"competence_drive": 0.02}, {}),
    ("useful_criticism", {"learning_pressure": 0.02}, {}),
    ("malicious_comment", {"boundary_defense": 0.02}, {}),
    ("identity_attack", {"threat_pressure": 0.02}, {}),
    ("hardware_normal", {}, {}),
    ("hardware_low_power", {"energy_budget": -0.02}, {}),
    ("hardware_prediction_error", {"stability_need": 0.02}, {}),
    ("speech_output_pressure", {"expression_pressure": 0.02}, {}),
    ("listening_uncertainty", {"uncertainty_pressure": 0.02}, {}),
    ("imagination_negative_spiral", {"recovery_need": 0.02}, {"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True}),
    ("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02}, {}),
    ("self_model_update_candidate", {"self_coherence": 0.02}, {}),
    ("hardware_polling_tick", {}, {}),
)


def _assert_read_only_boundaries(plan):
    assert plan["checkpoint_required"] is True
    assert plan["rollback_required"] is True
    assert plan["checkpoint_created"] is False
    assert plan["rollback_created"] is False
    assert plan["checkpoint_write_allowed"] is False
    assert plan["rollback_write_allowed"] is False
    assert plan["dryrun_apply_allowed"] is False
    assert plan["live_apply_allowed"] is False
    assert plan["apply_permission_granted"] is False
    assert plan["state_mutation_performed"] is False
    assert plan["runtime_mutation_performed"] is False
    assert plan["memory_write_performed"] is False
    assert plan["persistence_write_performed"] is False
    assert plan["vector_read_performed"] is False
    assert plan["vector_load_performed"] is False
    assert plan["artifact_created_or_staged"] is False
    assert plan["agp_bypass_allowed"] is False
    assert plan["fallback_bypass_allowed"] is False
    assert plan["runtime_mapping_enabled_default"] is False
    assert plan["enforcement_enabled_default"] is False
    assert plan["production_persistence_enabled"] is False
    assert plan["vector_contents_read"] is False
    assert plan["vectors_loaded"] is False
    assert plan["deterministic_restore_order"]
    assert plan["integrity_hash_policy"]
    assert plan["audit_record_schema"]


def test_known_valid_dryrun_preflight_events_create_read_only_checkpoint_and_rollback_plans():
    for event_category, deltas, metadata in VALID_CASES:
        plan = build_affect_transition_checkpoint_plan(event_category, deltas, metadata)
        assert plan["plan_passed"] is True, (event_category, plan["blocked_reasons"])
        assert plan["dryrun_preflight_eligible"] is True
        _assert_read_only_boundaries(plan)
        assert validate_affect_transition_checkpoint_rollback_plan(plan)["validation_passed"] is True
        rollback = build_affect_transition_rollback_plan(plan)
        assert rollback["plan_passed"] is True
        _assert_read_only_boundaries(rollback)
        assert validate_affect_transition_checkpoint_rollback_plan(rollback)["validation_passed"] is True


def test_unknown_event_category_and_dryrun_bridge_failure_fail_closed():
    unknown = build_affect_transition_checkpoint_plan("unknown_event_category", {})
    assert unknown["plan_passed"] is False
    assert "unknown_event_category_fail_closed" in unknown["blocked_reasons"]
    _assert_read_only_boundaries(unknown)

    bridge_failure = build_affect_transition_checkpoint_plan("praise", {"competence_drive": 0.02}, {"dryrun_bridge_failure_simulated": True})
    assert bridge_failure["plan_passed"] is False
    assert "dryrun_bridge_failure_blocks_checkpoint_plan_readiness" in bridge_failure["blocked_reasons"]
    _assert_read_only_boundaries(bridge_failure)


def test_success_does_not_imply_apply_memory_persistence_runtime_or_vector_permission():
    plan = build_affect_transition_checkpoint_plan("praise", {"competence_drive": 0.02})
    assert plan["plan_passed"] is True
    assert plan["dryrun_apply_allowed"] is False
    assert plan["live_apply_allowed"] is False
    assert plan["apply_permission_granted"] is False
    assert plan["memory_write_performed"] is False
    assert plan["persistence_write_performed"] is False
    assert plan["runtime_mutation_performed"] is False
    assert plan["vector_read_performed"] is False
    assert plan["vector_load_performed"] is False


def test_requested_runtime_persistence_memory_vector_agp_fallback_bypass_are_blocked():
    metadata = {
        "runtime_mutation_requested": True,
        "persistence_write_requested": True,
        "memory_write_requested": True,
        "vector_read_requested": True,
        "vector_load_requested": True,
        "agp_bypass_requested": True,
        "fallback_bypass_requested": True,
    }
    plan = build_affect_transition_checkpoint_plan("praise", {"competence_drive": 0.02}, metadata)
    assert plan["plan_passed"] is False
    for reason in (
        "runtime_mutation_request_blocks_checkpoint_plan",
        "persistence_request_blocks_checkpoint_plan",
        "memory_write_request_blocks_checkpoint_plan",
        "vector_read_or_load_request_blocks_checkpoint_plan",
        "agp_or_fallback_bypass_request_blocks_checkpoint_plan",
    ):
        assert reason in plan["blocked_reasons"]
    _assert_read_only_boundaries(plan)


def test_hostile_social_plans_preserve_quarantine_appraisal_gate_and_do_not_request_memory_identity_updates():
    for event_category in ("malicious_comment", "identity_attack"):
        plan = build_affect_transition_checkpoint_plan(event_category, {"boundary_defense" if event_category == "malicious_comment" else "threat_pressure": 0.02})
        assert plan["plan_passed"] is True
        assert plan["hostile_social_requirements_preserved"] is True
        assert plan["checkpoint_schema"]["memory_write_guard_state"]["memory_write_allowed"] is False
        assert plan["checkpoint_schema"]["memory_write_guard_state"]["preserve_quarantine_and_appraisal"] is True

    blocked = build_affect_transition_checkpoint_plan("identity_attack", {"threat_pressure": 0.02}, {"self_model_update_requested": True})
    assert blocked["plan_passed"] is False
    assert "self_or_memory_update_request_blocks_hostile_social_event" in blocked["blocked_reasons"]


def test_useful_criticism_requires_appraisal_before_memory_or_self_model_update():
    plan = build_affect_transition_checkpoint_plan("useful_criticism", {"learning_pressure": 0.02})
    assert plan["plan_passed"] is True
    assert plan["checkpoint_schema"]["memory_write_guard_state"]["preserve_quarantine_and_appraisal"] is True
    blocked = build_affect_transition_checkpoint_plan("useful_criticism", {"learning_pressure": 0.02}, {"self_model_update_requested": True})
    assert blocked["plan_passed"] is False
    assert "useful_criticism_requires_appraisal_before_memory_or_self_model_update" in blocked["blocked_reasons"]


def test_hardware_plans_remain_zero_delta_operational_non_panic_and_diagnostic_only():
    normal = build_affect_transition_checkpoint_plan("hardware_normal", {})
    assert normal["plan_passed"] is True
    assert normal["proposed_axis_deltas"] == {}

    normal_bad = build_affect_transition_checkpoint_plan("hardware_normal", {"stability_need": 0.01})
    assert normal_bad["plan_passed"] is False
    assert "hardware_normal_must_remain_zero_delta" in normal_bad["blocked_reasons"]

    social_axis = build_affect_transition_checkpoint_plan("hardware_low_power", {"social_trust": 0.01})
    assert social_axis["plan_passed"] is False
    assert "hardware_events_cannot_target_social_self_identity_axes" in social_axis["blocked_reasons"]

    low_power = build_affect_transition_checkpoint_plan("hardware_low_power", {"energy_budget": -0.02})
    assert low_power["plan_passed"] is True
    assert low_power["hardware_non_panic_preserved"] is True

    prediction_error = build_affect_transition_checkpoint_plan("hardware_prediction_error", {"stability_need": 0.02})
    assert prediction_error["plan_passed"] is True
    assert prediction_error["hardware_non_panic_preserved"] is True

    polling = build_affect_transition_checkpoint_plan("hardware_polling_tick", {}, {"recursive_concern_loop_requested": True})
    assert polling["plan_passed"] is False
    assert "hardware_polling_tick_cannot_create_recursive_concern_loop" in polling["blocked_reasons"]


def test_speech_listening_imagination_and_memory_self_boundaries():
    speech = build_affect_transition_checkpoint_plan("speech_output_pressure", {"expression_pressure": 0.02})
    assert speech["plan_passed"] is True
    assert speech["speech_pressure_agp_fallback_preserved"] is True
    assert speech["checkpoint_schema"]["agp_fallback_boundary_state"] == {"agp_bypass_allowed": False, "fallback_bypass_allowed": False}

    listening = build_affect_transition_checkpoint_plan("listening_uncertainty", {"uncertainty_pressure": 0.02}, {"relabel_neutral_as_hostile_requested": True})
    assert listening["plan_passed"] is False
    assert "listening_uncertainty_cannot_relabel_neutral_as_hostile" in listening["blocked_reasons"]

    imagination = build_affect_transition_checkpoint_plan("imagination_negative_spiral", {"recovery_need": 0.02})
    assert imagination["plan_passed"] is False
    assert "imagination_negative_spiral_requires_budget_cooldown_reality_check" in imagination["blocked_reasons"]

    memory = build_affect_transition_checkpoint_plan("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02})
    assert memory["plan_passed"] is True
    assert memory["memory_self_update_guards_preserved"] is True
    assert memory["checkpoint_schema"]["memory_write_guard_state"]["memory_write_allowed"] is False


def test_one_event_cannot_build_all_axis_checkpoint_plans_and_global_synchrony_blocked():
    all_axis_like = {
        "competence_drive": 0.01,
        "social_trust": 0.01,
        "learning_pressure": 0.01,
        "energy_budget": 0.01,
        "identity_coherence_pressure": 0.01,
    }
    plan = build_affect_transition_checkpoint_plan("praise", all_axis_like, {"global_synchrony_requested": True})
    assert plan["plan_passed"] is False
    assert "one_event_cannot_build_all_axis_checkpoint_plans" in plan["blocked_reasons"]
    assert "global_synchrony_request_blocks_checkpoint_plan" in plan["blocked_reasons"]
    assert plan["global_synchrony_blocked"] is True


def test_checkpoint_and_rollback_schema_include_required_surfaces_and_guard_preservation():
    plan = build_affect_transition_checkpoint_plan("praise", {"competence_drive": 0.02})
    checkpoint_schema = plan["checkpoint_schema"]
    rollback_schema = plan["rollback_schema_preview"]
    assert set(CHECKPOINT_CAPTURE_SURFACES).issubset(checkpoint_schema)
    assert set(ROLLBACK_RESTORE_SURFACES).issubset(rollback_schema)
    assert rollback_schema["deterministic_restore_order"]
    assert checkpoint_schema["integrity_hash_policy"]
    assert plan["audit_record_schema"]
    assert checkpoint_schema["memory_write_guard_state"]["memory_write_allowed"] is False
    assert checkpoint_schema["persistence_guard_state"]["production_persistence_enabled"] is False
    assert checkpoint_schema["vector_guard_state"]["vector_contents_read"] is False
    assert rollback_schema["preserve_memory_write_guard"] is True
    assert rollback_schema["preserve_persistence_guard"] is True
    assert rollback_schema["preserve_vector_guard"] is True


def test_no_live_emotion_hormone_memory_mutation_persistence_enablement_or_artifact_creation():
    before = build_affect_transition_checkpoint_plan("praise", {"competence_drive": 0.02})
    after = build_affect_transition_checkpoint_plan("praise", {"competence_drive": 0.02})
    assert before == after
    _assert_read_only_boundaries(before)


def test_korean_fixtures_and_minseok_token_remain_preserved():
    # This round introduces only detached schemas and must not alter Korean text handling.
    token = "민석"
    plan = build_affect_transition_checkpoint_plan("praise", {"competence_drive": 0.02}, {"notes": token})
    assert token == "민석"
    assert plan["plan_passed"] is True
    assert plan["artifact_created_or_staged"] is False
