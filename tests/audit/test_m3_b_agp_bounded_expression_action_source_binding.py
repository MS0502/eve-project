from __future__ import annotations

from scripts.audit.m3_b_agp_bounded_expression_action_source_binding import audit_repository


def test_agp_bounded_expression_action_source_binding_audit_is_deterministic_and_exact():
    first = audit_repository()
    second = audit_repository()
    assert first == second
    assert first["errors"] == []
    assert first["audit_fixture_only"] is True
    assert first["audit_fixture_is_production_observation"] is False
    assert first["appraised_binding_count"] == 6
    assert first["total_bound_axis_count"] == 37
    assert first["remaining_axis_count"] == 0
    assert first["retained_real_observation_count"] == 0
    assert first["positive_confidence_real_observation_count"] == 0
    assert set(first["bound_axes"]) == {
        "expression_pressure",
        "expression_inhibition",
        "action_readiness",
        "risk_tolerance",
        "patience_level",
        "conflict_avoidance",
    }
    assert first["deterministic_evidence_equal"] is True
    assert first["raw_digest_recalculation_verified"] is True
    assert first["agp_and_appraisal_gate_rejection_verified"] is True
    assert all(first["rejection_checks"].values())
    assert first["production_capture_present"] is False
    assert first["expression_or_action_executed"] is False
    assert first["memory_write_performed"] is False
    assert first["persistence_accessed"] is False
    assert first["event_append_performed"] is False
    assert first["registry_owner_materialized"] is False
    assert first["observation_window_started"] is False
    assert first["m3_b_complete"] is False
    assert first["m3_c_open"] is False
    assert first["m3_e_authority_open"] is False
    assert first["cutover_authorized"] is False
