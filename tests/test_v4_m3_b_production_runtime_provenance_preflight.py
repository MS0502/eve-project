from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import MappingProxyType

import pytest

import core.m3_b_production_runtime_provenance_preflight as provenance_module
from core.m3_b_production_runtime_provenance_preflight import (
    OBSERVATION_WINDOW_BLOCKER,
    POSITIVE_CONFIDENCE_BLOCKER,
    PRODUCTION_SOURCE_VERIFIER_BLOCKER,
    REGISTERED_RUNTIME_PROVENANCE_VERIFIERS,
    RUNTIME_PROVENANCE_ANCHOR_ABSENT,
    RUNTIME_PROVENANCE_VERIFIER_ABSENT,
    ProductionRuntimeProvenanceError,
    ProductionRuntimeProvenanceVerification,
    RuntimeProvenanceCandidate,
    RuntimeProvenanceVerifierRegistration,
    RuntimeProvenanceVerifierResult,
    execute_registered_runtime_provenance_verifier,
    production_runtime_provenance_capability_status,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _material() -> dict:
    return {
        "attestation_type": "test-only-independent-launch-attestation",
        "runtime_instance_id": "runtime:test:1",
        "source_instance_id": "runtime:ai-adapter:primary",
        "repository_head_sha": "2e2ad2ffa4d320ad9f8439a6b13097fa72fa40bb",
        "entrypoint_id": "main.py:repl",
    }


def _candidate(*, trust_domain: str = "test.fixture.trust-domain", fixture_only: bool = True):
    material = _material()
    return RuntimeProvenanceCandidate(
        trust_domain=trust_domain,
        runtime_instance_id="runtime:test:1",
        source_instance_id="runtime:ai-adapter:primary",
        repository_head_sha="2e2ad2ffa4d320ad9f8439a6b13097fa72fa40bb",
        entrypoint_id="main.py:repl",
        launch_attestation_id="attestation:test:1",
        launch_attestation_digest=_digest(material),
        logical_tick=2,
        fixture_only=fixture_only,
    )


def test_repository_runtime_provenance_registry_is_empty_and_runtime_immutable():
    assert type(REGISTERED_RUNTIME_PROVENANCE_VERIFIERS) is MappingProxyType
    assert len(REGISTERED_RUNTIME_PROVENANCE_VERIFIERS) == 0
    assert not hasattr(REGISTERED_RUNTIME_PROVENANCE_VERIFIERS, "__setitem__")


def test_production_looking_self_authored_candidate_never_counts_as_verified_anchor():
    candidate = _candidate(trust_domain="production-looking.self-claim", fixture_only=False)
    status = production_runtime_provenance_capability_status()
    assert candidate.fixture_only is False
    assert candidate.entrypoint_id == "main.py:repl"
    assert status.verified_production_runtime_anchor_count == 0
    assert status.registered_runtime_provenance_verifier_count == 0
    assert "main_py_repl_path" in status.self_authored_claims_not_trusted
    assert "fixture_only_false" in status.self_authored_claims_not_trusted


def test_unregistered_runtime_provenance_verifier_fails_closed():
    candidate = _candidate()
    with pytest.raises(ProductionRuntimeProvenanceError, match="not registered"):
        execute_registered_runtime_provenance_verifier(candidate, _material())


def test_attestation_material_must_match_candidate_before_verifier_lookup():
    candidate = _candidate()
    mismatched = {**_material(), "runtime_instance_id": "runtime:other"}
    with pytest.raises(ProductionRuntimeProvenanceError, match="digest does not match"):
        execute_registered_runtime_provenance_verifier(candidate, mismatched)


def test_direct_verified_provenance_construction_is_impossible():
    candidate = _candidate()
    with pytest.raises(ProductionRuntimeProvenanceError, match="issued by registered verifier execution"):
        ProductionRuntimeProvenanceVerification(
            trust_domain=candidate.trust_domain,
            runtime_instance_id=candidate.runtime_instance_id,
            source_instance_id=candidate.source_instance_id,
            repository_head_sha=candidate.repository_head_sha,
            entrypoint_id=candidate.entrypoint_id,
            launch_attestation_id=candidate.launch_attestation_id,
            launch_attestation_digest=candidate.launch_attestation_digest,
            candidate_digest=candidate.candidate_digest,
            verifier_id="caller-authored-verifier",
            verifier_version="v1",
            verifier_trace_digest=_digest({"trace": "caller-authored"}),
            verified_logical_tick=2,
        )


def test_fixture_only_registered_verifier_simulation_cannot_become_production(
    monkeypatch: pytest.MonkeyPatch,
):
    candidate = _candidate()

    def verifier(observed_candidate, attestation_material):
        assert observed_candidate == candidate
        assert attestation_material == _material()
        return RuntimeProvenanceVerifierResult(
            runtime_instance_id=candidate.runtime_instance_id,
            source_instance_id=candidate.source_instance_id,
            repository_head_sha=candidate.repository_head_sha,
            entrypoint_id=candidate.entrypoint_id,
            launch_attestation_id=candidate.launch_attestation_id,
            launch_attestation_digest=candidate.launch_attestation_digest,
            candidate_digest=candidate.candidate_digest,
            verifier_trace_digest=_digest({"trace": "test-fixture-verified"}),
            verified_logical_tick=2,
            verification_environment="test_fixture",
            fixture_only=True,
        )

    registration = RuntimeProvenanceVerifierRegistration(
        trust_domain=candidate.trust_domain,
        verifier_id="test.runtime-provenance-verifier",
        verifier_version="v1",
        verifier=verifier,
    )
    monkeypatch.setattr(
        provenance_module,
        "REGISTERED_RUNTIME_PROVENANCE_VERIFIERS",
        MappingProxyType({candidate.trust_domain: registration}),
    )
    verification = execute_registered_runtime_provenance_verifier(candidate, _material())
    assert verification.counts_as_production is False
    assert verification.fixture_only is True
    with pytest.raises(ProductionRuntimeProvenanceError, match="issued by registered verifier execution"):
        replace(
            verification,
            verifier_trace_digest=_digest({"trace": "caller-replaced"}),
        )


def test_preflight_capability_keeps_every_production_and_m3_boundary_closed():
    status = production_runtime_provenance_capability_status()
    assert status.verifier_registry_immutable is True
    assert status.registered_runtime_provenance_verifier_count == 0
    assert status.verified_production_runtime_anchor_count == 0
    assert status.registered_production_source_verifier_count == 0
    assert status.retained_real_observation_count == 0
    assert status.positive_confidence_real_observation_count == 0
    assert status.observation_window_eligible is False
    assert status.observation_window_started is False
    assert status.m3_b_complete is False
    assert status.m3_c_open is False
    assert status.m3_e_authority_open is False
    assert status.cutover_authorized is False
    assert status.blockers == (
        RUNTIME_PROVENANCE_VERIFIER_ABSENT,
        RUNTIME_PROVENANCE_ANCHOR_ABSENT,
        PRODUCTION_SOURCE_VERIFIER_BLOCKER,
        POSITIVE_CONFIDENCE_BLOCKER,
        OBSERVATION_WINDOW_BLOCKER,
    )
