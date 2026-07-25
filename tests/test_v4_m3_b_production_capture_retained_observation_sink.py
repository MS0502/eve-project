from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

import core.m3_b_registry_production_capture_adapter as capture_module
from core.m3_b_registry_observation_evidence import RegistryAxisPositiveConfidenceEvidence
from core.m3_b_registry_production_capture_adapter import (
    PRODUCTION_SOURCE_VERIFIER_BLOCKER,
    REGISTERED_PRODUCTION_SOURCE_VERIFIERS,
    ProductionCaptureRecord,
    ProductionSourceVerification,
    ProductionSourceVerifierRegistration,
    ProductionSourceVerifierResult,
    RegistryProductionCaptureAdapter,
    RegistryProductionCaptureError,
    execute_registered_production_verifier,
    production_capture_capability_status,
)
from core.m3_b_registry_retained_real_observation_sink import (
    RETENTION_EVENT_TYPE,
    RETENTION_STREAM_ID,
    RegistryRetainedObservationSinkError,
    RetainedRealObservationSink,
    build_retained_real_observation_event,
    retention_sink_capability_status,
)
from core.sqlite_shadow_store import SQLiteShadowStore, StoreNotInitialized


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence() -> RegistryAxisPositiveConfidenceEvidence:
    return RegistryAxisPositiveConfidenceEvidence(
        axis="energy_budget",
        value=0.6,
        confidence=0.8,
        observed_tick=20,
        observation_id="test:energy-budget:observation:20",
        source_family="operational_metrics_or_appraised_load_trace",
        source_instance_id="test:device-runtime:v1",
        source_snapshot_id="test:device-runtime:snapshot:20",
        source_schema_version="test.device-runtime.v1",
        source_integrity_digest=_sha("source:20"),
        raw_observation_digest=_sha("raw:20"),
        acquisition_method="explicit_test_fixture_matching_production_contract",
        verification_method="test_fixture_exact_digest_verification",
        model_or_rule_version="test.energy-budget.derivation.v1",
    )


def _verifier_result(
    evidence: RegistryAxisPositiveConfidenceEvidence,
    *,
    fixture_only: bool = False,
    source_snapshot_id: str | None = None,
    observation_evidence_digest: str | None = None,
) -> ProductionSourceVerifierResult:
    return ProductionSourceVerifierResult(
        source_instance_id=evidence.source_instance_id,
        source_snapshot_id=source_snapshot_id or evidence.source_snapshot_id,
        source_schema_version=evidence.source_schema_version,
        source_integrity_digest=evidence.source_integrity_digest,
        raw_observation_digest=evidence.raw_observation_digest,
        observation_evidence_digest=observation_evidence_digest or evidence.evidence_digest,
        verifier_trace_digest=_sha("verifier-trace:20"),
        verified_logical_tick=20,
        verification_environment="test_fixture" if fixture_only else "production",
        fixture_only=fixture_only,
    )


def _register_test_verifier(
    monkeypatch: pytest.MonkeyPatch,
    verifier=None,
) -> None:
    if verifier is None:
        verifier = lambda evidence, source_material: _verifier_result(evidence)
    registration = ProductionSourceVerifierRegistration(
        source_contract_id="eve:m3-b:registry-source:energy_budget:v1",
        verifier_id="test.production.energy-budget-verifier",
        verifier_version="v1",
        verifier=verifier,
    )
    monkeypatch.setattr(
        capture_module,
        "REGISTERED_PRODUCTION_SOURCE_VERIFIERS",
        {registration.source_contract_id: registration},
    )


def _issued_verification(
    monkeypatch: pytest.MonkeyPatch,
    evidence: RegistryAxisPositiveConfidenceEvidence,
    *,
    verifier=None,
) -> ProductionSourceVerification:
    _register_test_verifier(monkeypatch, verifier=verifier)
    return execute_registered_production_verifier(
        evidence,
        {"source": "test-only-disposable-runtime"},
    )


def _capture(monkeypatch: pytest.MonkeyPatch) -> ProductionCaptureRecord:
    evidence = _evidence()
    verification = _issued_verification(monkeypatch, evidence)
    return RegistryProductionCaptureAdapter().capture(
        evidence,
        verification,
        capture_id="test:capture:energy-budget:20",
        capture_tick=20,
    )


def test_capability_presence_does_not_claim_any_real_observation_or_window():
    capture = production_capture_capability_status()
    sink = retention_sink_capability_status()
    assert REGISTERED_PRODUCTION_SOURCE_VERIFIERS == {}
    assert capture.production_capture_adapter_present is True
    assert capture.immutable_retention_sink_required is True
    assert capture.registered_production_source_verifier_count == 0
    assert capture.retained_real_observation_count == 0
    assert capture.positive_confidence_real_observation_count == 0
    assert capture.observation_window_eligible is False
    assert capture.observation_window_started is False
    assert capture.m3_b_complete is False
    assert capture.m3_c_open is False
    assert capture.m3_e_authority_open is False
    assert capture.cutover_authorized is False
    assert capture.blockers == (
        PRODUCTION_SOURCE_VERIFIER_BLOCKER,
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
        "REGISTRY_OBSERVATION_WINDOW_NOT_STARTED",
    )
    assert sink.immutable_retention_sink_present is True
    assert sink.durable_store_type == "SQLiteShadowStore"
    assert sink.auto_initialize is False
    assert sink.auto_append is False
    assert sink.retained_real_observation_count == 0
    assert sink.positive_confidence_real_observation_count == 0
    assert sink.observation_window_started is False
    assert sink.m3_e_authority_open is False


def test_unregistered_verifier_cannot_issue_a_production_verification():
    evidence = _evidence()
    with pytest.raises(RegistryProductionCaptureError, match="not registered"):
        execute_registered_production_verifier(
            evidence,
            {"source": "unregistered-test-source"},
        )


def test_direct_verification_metadata_cannot_bypass_registered_verifier_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    evidence = _evidence()
    _register_test_verifier(monkeypatch)
    with pytest.raises(RegistryProductionCaptureError, match="issued by registered verifier execution"):
        ProductionSourceVerification(
            axis=evidence.axis,
            source_contract_id="eve:m3-b:registry-source:energy_budget:v1",
            source_family=evidence.source_family,
            source_instance_id=evidence.source_instance_id,
            source_snapshot_id=evidence.source_snapshot_id,
            source_schema_version=evidence.source_schema_version,
            source_integrity_digest=evidence.source_integrity_digest,
            raw_observation_digest=evidence.raw_observation_digest,
            observation_evidence_digest=evidence.evidence_digest,
            verifier_id="test.production.energy-budget-verifier",
            verifier_version="v1",
            verifier_trace_digest=_sha("caller-authored-trace"),
            verified_logical_tick=20,
        )


def test_registered_verifier_callable_is_executed_before_verification_is_issued(
    monkeypatch: pytest.MonkeyPatch,
):
    evidence = _evidence()
    calls: list[dict] = []

    def verifier(observation, source_material):
        calls.append(dict(source_material))
        return _verifier_result(observation)

    verification = _issued_verification(monkeypatch, evidence, verifier=verifier)
    assert calls == [{"source": "test-only-disposable-runtime"}]
    assert verification.counts_as_real is True
    assert verification.observation_evidence_digest == evidence.evidence_digest
    assert verification.verifier_id == "test.production.energy-budget-verifier"


def test_fixture_verification_can_never_become_a_retained_real_capture_even_if_registered(
    monkeypatch: pytest.MonkeyPatch,
):
    evidence = _evidence()

    def fixture_verifier(observation, source_material):
        return _verifier_result(observation, fixture_only=True)

    fixture = _issued_verification(monkeypatch, evidence, verifier=fixture_verifier)
    assert fixture.counts_as_real is False
    with pytest.raises(RegistryProductionCaptureError, match="test fixtures"):
        RegistryProductionCaptureAdapter().capture(
            evidence,
            fixture,
            capture_id="test:capture:fixture",
            capture_tick=20,
        )


def test_verifier_output_must_bind_exact_evidence_identity_and_digests(
    monkeypatch: pytest.MonkeyPatch,
):
    evidence = _evidence()

    def mismatched_snapshot(observation, source_material):
        return _verifier_result(observation, source_snapshot_id="test:other:snapshot")

    with pytest.raises(RegistryProductionCaptureError, match="does not bind the exact observation evidence"):
        _issued_verification(monkeypatch, evidence, verifier=mismatched_snapshot)

    def mismatched_digest(observation, source_material):
        return _verifier_result(
            observation,
            observation_evidence_digest=_sha("other-evidence"),
        )

    with pytest.raises(RegistryProductionCaptureError, match="does not bind the exact observation evidence"):
        _issued_verification(monkeypatch, evidence, verifier=mismatched_digest)


def test_capture_cannot_claim_persistence_window_mutation_or_authority(
    monkeypatch: pytest.MonkeyPatch,
):
    capture = _capture(monkeypatch)
    with pytest.raises(FrozenInstanceError):
        capture.m3_b_complete = True  # type: ignore[misc]
    with pytest.raises(RegistryProductionCaptureError, match="cannot claim"):
        replace(capture, persistence_accessed=True)
    with pytest.raises(RegistryProductionCaptureError, match="cannot claim"):
        replace(capture, observation_window_started=True)
    with pytest.raises(RegistryProductionCaptureError, match="cannot claim"):
        replace(capture, m3_e_authority_open=True)


def test_sink_construction_performs_no_io_and_requires_explicit_store_initialize(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "retained.sqlite3"
    store = SQLiteShadowStore(database)
    sink = RetainedRealObservationSink(store)
    assert sink.database_path == str(database)
    assert database.exists() is False
    capture = _capture(monkeypatch)
    with pytest.raises(StoreNotInitialized):
        sink.append(
            capture,
            event_id="test:retained:energy-budget:1",
            sequence=1,
            correlation_id="test:retained:correlation:1",
        )
    assert database.exists() is False


def test_registered_verifier_simulation_proves_durable_append_without_changing_repository_status(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    # This registration is process-local test evidence only. The repository table
    # remains empty and this disposable simulation is not a production observation.
    capture = _capture(monkeypatch)
    database = tmp_path / "retained.sqlite3"
    store = SQLiteShadowStore(database)
    initialization = store.initialize()
    assert initialization.wal_enabled is True
    sink = RetainedRealObservationSink(store)
    receipt = sink.append(
        capture,
        event_id="test:retained:energy-budget:1",
        sequence=1,
        correlation_id="test:retained:correlation:1",
    )
    assert receipt.axis == "energy_budget"
    assert receipt.capture_id == capture.capture_id
    assert receipt.capture_digest == capture.capture_digest
    assert receipt.verification_digest == capture.verification.verification_digest
    assert receipt.store_before_count == 0
    assert receipt.store_after_count == 1
    assert receipt.store_before_chain_digest == "0" * 64
    assert receipt.store_before_chain_digest != receipt.store_after_chain_digest
    assert receipt.readback_verified is True
    assert receipt.retained_real_observation_delta == 1
    assert receipt.observation_window_started is False
    assert receipt.registry_owner_mutated is False
    assert receipt.m3_b_complete is False
    assert receipt.m3_e_authority_open is False
    assert receipt.cutover_authorized is False

    events = store.events(stream_id=RETENTION_STREAM_ID)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == RETENTION_EVENT_TYPE
    assert event.authority == "shadow_only"
    payload = json.loads(event.payload_json)
    assert payload["classification"] == "retained_real_observation"
    assert payload["immutable"] is True
    assert payload["axis"] == "energy_budget"
    assert payload["capture_digest"] == capture.capture_digest
    assert payload["source_evidence_digest"] == capture.evidence.evidence_digest
    assert payload["source_verification_digest"] == capture.verification.verification_digest
    # events() recalculates and validates the complete persisted event chain before
    # returning records, so this is the public integrity check for event-only state.
    assert store.events() == events


def test_retention_event_material_is_deterministic_under_test_registration(
    monkeypatch: pytest.MonkeyPatch,
):
    capture = _capture(monkeypatch)
    first = build_retained_real_observation_event(
        capture,
        event_id="test:retained:deterministic:1",
        sequence=1,
        correlation_id="test:retained:correlation:deterministic",
    )
    second = build_retained_real_observation_event(
        capture,
        event_id="test:retained:deterministic:1",
        sequence=1,
        correlation_id="test:retained:correlation:deterministic",
    )
    assert first == second
    assert first.digest == second.digest
    assert first.authority == "shadow_only"


def test_sink_rejects_non_capture_objects_without_touching_store(tmp_path):
    store = SQLiteShadowStore(tmp_path / "retained.sqlite3")
    store.initialize()
    before = store.events()
    with pytest.raises(RegistryRetainedObservationSinkError, match="exact immutable"):
        build_retained_real_observation_event(
            object(),  # type: ignore[arg-type]
            event_id="test:invalid:event",
            sequence=1,
            correlation_id="test:invalid:correlation",
        )
    after = store.events()
    assert before == after == ()
