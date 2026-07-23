from __future__ import annotations

from pathlib import Path

from scripts.audit.m3_b_appraised_survival_source_binding import audit_repository


def test_appraised_survival_source_binding_audit_is_deterministic_and_exact():
    first = audit_repository(Path("."))
    second = audit_repository(Path("."))
    assert first == second
    assert first["errors"] == []
    assert first["appraised_binding_count"] == 2
    assert first["total_bound_axis_count"] == 6
    assert first["remaining_axis_count"] == 31
    assert first["bound_axes"] == ["stress_load", "stability_need"]
    assert first["deterministic_evidence_equal"] is True
    assert first["raw_digest_recalculation_verified"] is True
    assert first["appraisal_gate_rejection_verified"] is True
    assert first["static_surface"][
        "no_io_polling_scheduler_event_or_runtime_surface"
    ] is True
    assert set(first["derived_evidence"]) == {"stress_load", "stability_need"}
    assert first["production_capture_present"] is False
    assert first["raw_social_feedback_ingested"] is False
    assert first["registry_owner_materialized"] is False
    assert first["observation_window_started"] is False
    assert first["m3_b_complete"] is False
    assert first["m3_c_open"] is False
    assert first["m3_e_authority_open"] is False
    assert first["cutover_authorized"] is False
