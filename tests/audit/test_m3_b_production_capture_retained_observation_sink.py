from __future__ import annotations

from scripts.audit.m3_b_production_capture_retained_observation_sink import audit_repository


def test_production_capture_retention_sink_audit_is_deterministic_and_fail_closed():
    first = audit_repository()
    second = audit_repository()
    assert first == second
    assert first["errors"] == []
    assert first["audit_fixture_only"] is True
    assert first["audit_fixture_is_production_observation"] is False
    assert first["production_capture_adapter_present"] is True
    assert first["immutable_retention_sink_present"] is True
    assert first["durable_store_type"] == "SQLiteShadowStore"
    assert first["append_only_chain_required"] is True
    assert first["readback_verification_required"] is True
    assert first["auto_initialize"] is False
    assert first["auto_append"] is False
    assert first["registered_production_source_verifier_count"] == 0
    assert first["registered_production_source_verifier_contracts"] == []
    assert first["unregistered_verifier_execution_rejected"] is True
    assert first["caller_authored_verification_rejected"] is True
    assert first["retained_real_observation_count"] == 0
    assert first["positive_confidence_real_observation_count"] == 0
    assert first["observation_window_eligible"] is False
    assert first["observation_window_started"] is False
    assert first["m3_b_complete"] is False
    assert first["m3_c_open"] is False
    assert first["m3_e_authority_open"] is False
    assert first["cutover_authorized"] is False
    assert first["blockers"] == [
        "REGISTRY_PRODUCTION_SOURCE_VERIFIER_COVERAGE_INCOMPLETE",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
        "REGISTRY_OBSERVATION_WINDOW_NOT_STARTED",
    ]
    assert first["legacy_runtime_authoritative"] is True
    assert first["legacy_persistence_authoritative"] is True
