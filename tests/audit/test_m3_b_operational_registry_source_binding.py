from __future__ import annotations

from scripts.audit.m3_b_operational_registry_source_binding import (
    ROOT,
    audit_repository,
)


def test_operational_source_binding_audit_is_deterministic_and_exact():
    first = audit_repository(ROOT)
    second = audit_repository(ROOT)
    assert first == second
    assert first["errors"] == []
    assert first["audit_fixture_only"] is True
    assert first["audit_fixture_is_production_observation"] is False
    assert first["binding_count"] == 4
    assert first["remaining_axis_count"] == 33
    assert first["bound_axes"] == [
        "energy_budget",
        "fatigue_pressure",
        "recovery_need",
        "overload_risk",
    ]
    assert first["deterministic_evidence_equal"] is True
    assert first["raw_digest_recalculation_verified"] is True
    assert len(first["binding_set_digest"]) == 64
    assert len(first["report_digest"]) == 64
    assert set(first["derived_evidence"]) == set(first["bound_axes"])
    assert all(
        0.0 < item["confidence"] <= 1.0
        and 0.0 <= item["value"] <= 1.0
        and len(item["evidence_digest"]) == 64
        and len(item["raw_observation_digest"]) == 64
        and len(item["source_integrity_digest"]) == 64
        for item in first["derived_evidence"].values()
    )


def test_operational_source_binding_audit_preserves_authority_boundary():
    report = audit_repository(ROOT)
    assert report["blockers"] == [
        "REGISTRY_APPRAISED_33_AXIS_SOURCE_BINDINGS_INCOMPLETE",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
    ]
    assert all(report["rejection_checks"].values())
    assert report["static_surface"] == {
        "forbidden_calls": [],
        "forbidden_imports": [],
        "no_io_polling_scheduler_event_or_runtime_surface": True,
    }
    for field in (
        "production_capture_present",
        "runtime_capture_installed",
        "hardware_polling_installed",
        "scheduler_installed",
        "persistence_accessed",
        "event_append_performed",
        "registry_owner_materialized",
        "observation_window_started",
        "observation_window_satisfied",
        "m3_b_complete",
        "m3_c_open",
        "m3_e_authority_open",
        "cutover_authorized",
    ):
        assert report[field] is False
    assert report["legacy_runtime_authoritative"] is True
    assert report["legacy_persistence_authoritative"] is True
