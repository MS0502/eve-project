from __future__ import annotations

from scripts.audit.m3_b_registry_observation_source_manifest import (
    ROOT,
    audit_repository,
)


def test_source_manifest_audit_is_deterministic_and_exact():
    first = audit_repository(ROOT)
    second = audit_repository(ROOT)
    assert first == second
    assert first["errors"] == []
    assert first["axis_count"] == 37
    assert first["canonical_axis_order"] is True
    assert first["structurally_complete"] is True
    assert first["deterministic_manifest_equal"] is True
    assert first["entry_digest_count"] == 37
    assert len(first["manifest_digest"]) == 64
    assert len(first["report_digest"]) == 64


def test_source_manifest_audit_preserves_preflight_authority_boundary():
    report = audit_repository(ROOT)
    assert report["hardware_direct_axes"] == [
        "energy_budget",
        "fatigue_pressure",
        "recovery_need",
        "overload_risk",
    ]
    assert report["real_source_binding_count"] == 0
    assert report["blockers"] == [
        "REGISTRY_REAL_OBSERVATION_SOURCE_BINDINGS_INCOMPLETE",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
    ]
    assert all(report["rejection_checks"].values())
    assert report["static_surface"] == {
        "forbidden_calls": [],
        "forbidden_imports": [],
        "no_io_hardware_polling_scheduler_event_or_runtime_surface": True,
    }
    for field in (
        "real_observation_values_present",
        "real_source_bindings_present",
        "capture_ready",
        "runtime_capture_installed",
        "hardware_polling_installed",
        "scheduler_installed",
        "persistence_accessed",
        "event_append_performed",
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
