from __future__ import annotations

from scripts.audit.m3_b_registry_affect_owner import ROOT, audit_repository


def test_registry_owner_audit_is_deterministic_and_resolves_the_owner_contract_blocker():
    first = audit_repository(ROOT)
    second = audit_repository(ROOT)
    assert first == second
    assert first["errors"] == []
    assert first["axis_count"] == 37
    assert first["current_value_owner_contract_found"] is True
    assert first["deterministic_genesis_equal"] is True
    assert first["genesis_is_observation_evidence"] is False
    assert first["proposal_metadata_is_current_state"] is False
    assert first["read_only_observation_count"] == 37
    assert first["snapshot_identity_schema_provenance_integrity_complete"] is True
    assert first["remaining_source_ownership_blockers"] == []
    assert first["next_required_artifact"] == "real combined 63-axis read-only observation packet and observation-window evidence"
    assert first["observation_window_started"] is False
    assert first["m3_b_complete"] is False
    assert first["m3_c_open"] is False
    assert len(first["report_digest"]) == 64


def test_registry_owner_audit_proves_no_live_or_io_surface():
    report = audit_repository(ROOT)
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
    assert report["goal_memory_self_expression_mutated"] is False
    assert report["m3_e_authority_open"] is False
    assert report["cutover_authorized"] is False
