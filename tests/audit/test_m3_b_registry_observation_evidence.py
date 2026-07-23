from __future__ import annotations

from scripts.audit.m3_b_registry_observation_evidence import ROOT, audit_repository


def test_registry_observation_evidence_audit_is_deterministic_and_exact():
    first = audit_repository(ROOT)
    second = audit_repository(ROOT)
    assert first == second
    assert first["errors"] == []
    assert first["audit_fixture_only"] is True
    assert first["audit_fixture_is_production_observation_evidence"] is False
    assert first["axis_count"] == 37
    assert first["positive_confidence_count"] == 37
    assert first["exact_positive_confidence_coverage"] is True
    assert first["deterministic_bundle_equal"] is True
    assert first["deterministic_owner_equal"] is True
    assert first["predecessor_owner_unchanged"] is True
    assert len(first["bundle_digest"]) == 64
    assert len(first["materialized_owner_digest"]) == 64
    assert len(first["report_digest"]) == 64


def test_registry_observation_evidence_audit_proves_fail_closed_boundary():
    report = audit_repository(ROOT)
    assert report["rejection_checks"] == {
        "baseline_derived": True,
        "default_derived": True,
        "genesis_derived": True,
        "missing_raw_reference": True,
        "proposal_only": True,
        "synthetic": True,
        "zero_confidence": True,
    }
    assert report["fixture_packet_positive_confidence_count"] == 63
    assert report["fixture_packet_zero_confidence_count"] == 0
    assert report["fixture_packet_window_blockers"] == []
    assert report["fixture_packet_calculated_start_eligible"] is True
    assert report["production_observation_window_started"] is False
    assert report["production_observation_window_satisfied"] is False
    assert report["static_surface"] == {
        "forbidden_calls": [],
        "forbidden_imports": [],
        "no_io_persistence_scheduler_event_or_runtime_surface": True,
    }
    assert report["legacy_runtime_authoritative"] is True
    assert report["legacy_persistence_authoritative"] is True
    assert report["runtime_hook_installed"] is False
    assert report["scheduler_installed"] is False
    assert report["persistence_accessed"] is False
    assert report["event_append_performed"] is False
    assert report["live_affect_mutated"] is False
    assert report["live_drive_mutated"] is False
    assert report["named_state_mutated"] is False
    assert report["goal_memory_self_expression_mutated"] is False
    assert report["observation_window_started"] is False
    assert report["observation_window_satisfied"] is False
    assert report["m3_b_complete"] is False
    assert report["m3_c_open"] is False
    assert report["m3_e_authority_open"] is False
    assert report["cutover_authorized"] is False
