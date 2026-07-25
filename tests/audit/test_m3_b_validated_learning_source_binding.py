from __future__ import annotations

from scripts.audit.m3_b_validated_learning_source_binding import audit_repository


def test_validated_learning_source_binding_audit_is_deterministic_and_exact():
    first = audit_repository()
    second = audit_repository()

    assert first == second
    assert first["schema_version"] == "eve.m3-b.validated-learning-source-binding-audit.v1"
    assert first["baseline_sha"] == "6c57f41114fbe0a203e559a27b187f6801ad7640"
    assert first["authority"] == "shadow_only_validated_learning_source_binding"
    assert first["audit_fixture_only"] is True
    assert first["audit_fixture_is_production_observation"] is False
    assert first["appraised_binding_count"] == 6
    assert first["total_bound_axis_count"] == 25
    assert first["remaining_axis_count"] == 12
    assert first["bound_axes"] == [
        "curiosity_drive",
        "novelty_seeking",
        "learning_pressure",
        "memory_consolidation_pressure",
        "prediction_error_pressure",
        "competence_drive",
    ]
    assert set(first["derived_evidence"]) == set(first["bound_axes"])
    assert first["deterministic_evidence_equal"] is True
    assert first["raw_digest_recalculation_verified"] is True
    assert first["validation_and_appraisal_gate_rejection_verified"] is True
    assert all(first["rejection_checks"].values())
    assert first["production_capture_present"] is False
    assert first["runtime_capture_installed"] is False
    assert first["hardware_polling_installed"] is False
    assert first["raw_social_feedback_ingested"] is False
    assert first["learning_mutation_performed"] is False
    assert first["memory_write_performed"] is False
    assert first["scheduler_installed"] is False
    assert first["persistence_accessed"] is False
    assert first["event_append_performed"] is False
    assert first["registry_owner_materialized"] is False
    assert first["observation_window_started"] is False
    assert first["observation_window_satisfied"] is False
    assert first["m3_b_complete"] is False
    assert first["m3_c_open"] is False
    assert first["m3_e_authority_open"] is False
    assert first["cutover_authorized"] is False
    assert first["legacy_runtime_authoritative"] is True
    assert first["legacy_persistence_authoritative"] is True
    assert first["errors"] == []
