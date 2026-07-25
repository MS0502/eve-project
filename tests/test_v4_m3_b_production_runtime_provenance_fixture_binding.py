from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import pytest

import core.m3_b_production_runtime_provenance_preflight as provenance_module
from core.m3_b_production_runtime_provenance_preflight import (
    ProductionRuntimeProvenanceError,
    RuntimeProvenanceCandidate,
    RuntimeProvenanceVerifierRegistration,
    RuntimeProvenanceVerifierResult,
    execute_registered_runtime_provenance_verifier,
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


def test_fixture_candidate_cannot_be_reclassified_as_production_by_verifier_result(
    monkeypatch: pytest.MonkeyPatch,
):
    material = {
        "attestation_type": "test-only-independent-launch-attestation",
        "runtime_instance_id": "runtime:test:fixture-binding",
    }
    candidate = RuntimeProvenanceCandidate(
        trust_domain="test.fixture-binding",
        runtime_instance_id="runtime:test:fixture-binding",
        source_instance_id="runtime:ai-adapter:primary",
        repository_head_sha="2e2ad2ffa4d320ad9f8439a6b13097fa72fa40bb",
        entrypoint_id="main.py:repl",
        launch_attestation_id="attestation:test:fixture-binding",
        launch_attestation_digest=_digest(material),
        logical_tick=2,
        fixture_only=True,
    )

    def verifier(observed_candidate, attestation_material):
        return RuntimeProvenanceVerifierResult(
            runtime_instance_id=observed_candidate.runtime_instance_id,
            source_instance_id=observed_candidate.source_instance_id,
            repository_head_sha=observed_candidate.repository_head_sha,
            entrypoint_id=observed_candidate.entrypoint_id,
            launch_attestation_id=observed_candidate.launch_attestation_id,
            launch_attestation_digest=observed_candidate.launch_attestation_digest,
            candidate_digest=observed_candidate.candidate_digest,
            verifier_trace_digest=_digest({"trace": "invalid-fixture-flip"}),
            verified_logical_tick=observed_candidate.logical_tick,
            verification_environment="production",
            fixture_only=False,
        )

    registration = RuntimeProvenanceVerifierRegistration(
        trust_domain=candidate.trust_domain,
        verifier_id="test.fixture-flip-verifier",
        verifier_version="v1",
        verifier=verifier,
    )
    monkeypatch.setattr(
        provenance_module,
        "REGISTERED_RUNTIME_PROVENANCE_VERIFIERS",
        MappingProxyType({candidate.trust_domain: registration}),
    )
    with pytest.raises(ProductionRuntimeProvenanceError, match="fixture classification"):
        execute_registered_runtime_provenance_verifier(candidate, material)
