from __future__ import annotations

from scripts.audit.m3_b_quarantined_risk_source_binding import audit_repository


def test_quarantined_risk_source_binding_audit_is_deterministic_and_exact():
    first = audit_repository()
    second = audit_repository()

    assert first == second
    assert first["schema_version"] == "eve.m3-b.quarantined-risk-source-binding-audit.v1"
    assert first["baseline_sha"] == "c9b46e2f0d509d78b6b2802e180e7a3a4be741b3"
    assert first["authority"] == "shadow_only_quarantined_risk_source_binding"
    assert first["audit_fixture_only"] is True
    assert first["audit_fixture_is_production_observation"] is False
    assert first["appraised_binding_count"] == 6
    assert first["total_bound_axis_count"] == 12
    assert first["remaining_axis_count"] == 25
    assert first["bound_axes"] == [
        "threat_pressure",
        "uncertainty_pressure",
        "self_protection",
        "boundary_defense",
        "trust_risk",
        "exposure_risk",
    ]
    assert set(first["derived_evidence"]) == set(first["bound_axes"])
    assert first["deterministic_evidence_equal"] is True
    assert first["raw_digest_recalculation_verified"] is True
    assert first["quarantine_and_appraisal_gate_rejection_verified"] is True
    assert all(first["rejection_checks"].values())
    assert first["static_surface"]["no_io_polling_scheduler_event_or_runtime_surface"] is True
    assert first["production_capture_present"] is False
    assert first["runtime_capture_installed"] is False
    assert first["raw_social_feedback_ingested"] is False
    assert first["persistence_accessed"] is False
    assert first["event_append_performed"] is False
    assert first["observation_window_started"] is False
    assert first["m3_b_complete"] is False
    assert first["m3_c_open"] is False
    assert first["m3_e_authority_open"] is False
    assert first["cutover_authorized"] is False
    assert first["errors"] == []
