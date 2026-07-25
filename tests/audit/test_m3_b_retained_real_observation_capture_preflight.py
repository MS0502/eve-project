from __future__ import annotations

from scripts.audit.m3_b_retained_real_observation_capture_preflight import audit_repository


def test_retained_real_observation_capture_preflight_audit_is_deterministic_and_exact():
    first = audit_repository()
    second = audit_repository()
    assert first == second
    assert first["errors"] == []
    assert first["source_binding_count"] == 37
    assert first["source_binding_complete"] is True
    assert first["source_binding_cumulative_counts"] == [4, 6, 12, 19, 25, 31, 37]
    assert len(first["source_binding_group_digests"]) == 7
    assert first["deterministic_preflight_equal"] is True
    assert first["production_capture_future_path_present"] is False
    assert first["retention_sink_future_path_present"] is False
    assert first["production_capture_adapter_present"] is False
    assert first["retention_sink_present"] is False
    assert first["retained_real_observation_count"] == 0
    assert first["positive_confidence_real_observation_count"] == 0
    assert first["observation_window_eligible"] is False
    assert first["observation_window_started"] is False
    assert first["observation_window_satisfied"] is False
    assert first["blockers"] == [
        "REGISTRY_RETAINED_REAL_OBSERVATION_CAPTURE_ABSENT",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
        "REGISTRY_OBSERVATION_WINDOW_NOT_STARTED",
    ]
    assert first["runtime_hook_installed"] is False
    assert first["scheduler_installed"] is False
    assert first["persistence_accessed"] is False
    assert first["event_append_performed"] is False
    assert first["registry_owner_mutated"] is False
    assert first["live_affect_mutated"] is False
    assert first["live_drive_mutated"] is False
    assert first["named_state_mutated"] is False
    assert first["goal_memory_self_expression_mutated"] is False
    assert first["m3_b_complete"] is False
    assert first["m3_c_open"] is False
    assert first["m3_e_authority_open"] is False
    assert first["cutover_authorized"] is False
    assert first["legacy_runtime_authoritative"] is True
    assert first["legacy_persistence_authoritative"] is True
