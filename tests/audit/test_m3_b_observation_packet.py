from __future__ import annotations

from scripts.audit.m3_b_observation_packet import ROOT, audit_repository


def test_combined_packet_audit_is_deterministic_and_exposes_exact_window_blocker():
    first = audit_repository(ROOT)
    second = audit_repository(ROOT)
    assert first == second
    assert first["errors"] == []
    assert first["axis_count"] == 63
    assert first["legacy_axis_count"] == 26
    assert first["registry_axis_count"] == 37
    assert first["structurally_complete"] is True
    assert first["strict_projection_input_ready"] is True
    assert first["genesis_positive_confidence_count"] == 26
    assert first["genesis_zero_confidence_count"] == 37
    assert first["partial_positive_confidence_count"] == 28
    assert first["partial_zero_confidence_count"] == 35
    assert first["window_blockers"] == ["REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"]
    assert first["observation_window_start_eligible"] is False
    assert first["observation_window_started"] is False
    assert first["m3_b_complete"] is False
    assert len(first["source_set_digest"]) == 64
    assert len(first["packet_digest"]) == 64
    assert len(first["report_digest"]) == 64


def test_combined_packet_audit_proves_sources_unchanged_and_no_live_surface():
    report = audit_repository(ROOT)
    assert report["deterministic_replay_equal"] is True
    assert report["legacy_source_unchanged"] is True
    assert report["registry_owner_unchanged"] is True
    assert report["static_surface"] == {
        "forbidden_calls": [],
        "forbidden_imports": [],
        "no_io_persistence_scheduler_event_projection_or_runtime_surface": True,
    }
    assert report["legacy_runtime_authoritative"] is True
    assert report["legacy_persistence_authoritative"] is True
    assert report["projection_performed"] is False
    assert report["observation_window_satisfied"] is False
    assert report["persistence_accessed"] is False
    assert report["event_append_performed"] is False
    assert report["live_affect_mutated"] is False
    assert report["live_drive_mutated"] is False
    assert report["named_state_mutated"] is False
    assert report["goal_memory_self_expression_mutated"] is False
    assert report["m3_c_open"] is False
    assert report["m3_e_authority_open"] is False
    assert report["cutover_authorized"] is False
