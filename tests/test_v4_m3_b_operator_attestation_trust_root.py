from __future__ import annotations

import hashlib
from dataclasses import replace
from types import MappingProxyType

import pytest

import core.m3_b_operator_attestation_trust_root as trust_module
from core.m3_b_operator_attestation_trust_root import (
    PRIMARY_OPERATOR_ID,
    PRIMARY_OPERATOR_TRUST_DOMAIN,
    REVIEWED_OPERATOR_ATTESTATIONS,
    OperatorAttestationError,
    OperatorLaunchBinding,
    OperatorPublicLaunchAttestation,
    ReviewedOperatorAttestationRegistration,
    ReviewedOperatorAttestationVerification,
    build_operator_public_launch_attestation,
    verify_operator_private_binding,
    verify_reviewed_operator_attestation,
)


HEAD = "1b5e8ffb268a47d215bcc6f6ef1235b147559bb7"
NONCE = b"operator-private-fixture-nonce-32bytes!!"


def _binding(*, fixture_only: bool = True) -> OperatorLaunchBinding:
    return OperatorLaunchBinding(
        runtime_instance_id="runtime:phone:fixture-1",
        source_instance_id="runtime:ai-adapter:primary",
        repository_head_sha=HEAD,
        entrypoint_id="main.py:repl",
        launch_attestation_id="operator-attestation:fixture-1",
        logical_tick=7,
        fixture_only=fixture_only,
    )


def _candidate(attestation: OperatorPublicLaunchAttestation) -> dict:
    return {
        "trust_domain": attestation.trust_domain,
        "runtime_instance_id": attestation.runtime_instance_id,
        "source_instance_id": attestation.source_instance_id,
        "repository_head_sha": attestation.repository_head_sha,
        "entrypoint_id": attestation.entrypoint_id,
        "launch_attestation_id": attestation.launch_attestation_id,
        "launch_attestation_digest": attestation.attestation_digest,
        "logical_tick": attestation.logical_tick,
        "fixture_only": attestation.fixture_only,
    }


def _registration(attestation: OperatorPublicLaunchAttestation) -> ReviewedOperatorAttestationRegistration:
    trace = verify_operator_private_binding(attestation, NONCE)
    return ReviewedOperatorAttestationRegistration(
        attestation_digest=attestation.attestation_digest,
        review_record_digest=hashlib.sha256(b"fixture-review-record").hexdigest(),
        local_verification_trace_digest=trace,
        runtime_instance_id=attestation.runtime_instance_id,
        source_instance_id=attestation.source_instance_id,
        repository_head_sha=attestation.repository_head_sha,
        entrypoint_id=attestation.entrypoint_id,
        launch_attestation_id=attestation.launch_attestation_id,
        logical_tick=attestation.logical_tick,
        private_nonce_commitment_digest=attestation.private_nonce_commitment_digest,
        nonce_binding_digest=attestation.nonce_binding_digest,
        fixture_only=attestation.fixture_only,
    )


def test_c1_reviewed_registry_is_empty_immutable_and_has_one_operator_identity():
    assert type(REVIEWED_OPERATOR_ATTESTATIONS) is MappingProxyType
    assert len(REVIEWED_OPERATOR_ATTESTATIONS) == 0
    assert not hasattr(REVIEWED_OPERATOR_ATTESTATIONS, "__setitem__")
    assert PRIMARY_OPERATOR_TRUST_DOMAIN == "eve.operator-attestation.primary.v1"
    assert PRIMARY_OPERATOR_ID == "primary-operator"


def test_private_nonce_never_enters_public_attestation_mapping():
    attestation = build_operator_public_launch_attestation(_binding(), NONCE)
    public = attestation.to_mapping()
    assert "private_nonce" not in public
    assert "nonce" not in public
    assert public["private_nonce_commitment_digest"] == hashlib.sha256(NONCE).hexdigest()
    assert len(public["nonce_binding_digest"]) == 64
    assert NONCE.decode("ascii") not in str(public)


def test_operator_private_binding_recomputes_locally_and_wrong_nonce_fails():
    attestation = build_operator_public_launch_attestation(_binding(), NONCE)
    trace = verify_operator_private_binding(attestation, NONCE)
    assert len(trace) == 64
    with pytest.raises(OperatorAttestationError, match="commitment does not match"):
        verify_operator_private_binding(attestation, b"different-private-fixture-nonce-32!!")


def test_public_attestation_digest_alone_cannot_cross_review_boundary():
    attestation = build_operator_public_launch_attestation(_binding(fixture_only=False), NONCE)
    with pytest.raises(OperatorAttestationError, match="not repository-reviewed"):
        verify_reviewed_operator_attestation(_candidate(attestation), attestation.to_mapping())


def test_reviewed_fixture_attestation_can_be_verified_but_never_counts_as_production(
    monkeypatch: pytest.MonkeyPatch,
):
    attestation = build_operator_public_launch_attestation(_binding(fixture_only=True), NONCE)
    registration = _registration(attestation)
    monkeypatch.setattr(
        trust_module,
        "REVIEWED_OPERATOR_ATTESTATIONS",
        MappingProxyType({attestation.attestation_digest: registration}),
    )
    verification = verify_reviewed_operator_attestation(
        _candidate(attestation),
        attestation.to_mapping(),
    )
    assert verification.counts_as_production is False
    assert verification.fixture_only is True
    assert verification.attestation_digest == attestation.attestation_digest
    with pytest.raises(OperatorAttestationError, match="issued by exact registry lookup"):
        replace(
            verification,
            verifier_trace_digest=hashlib.sha256(b"caller-replaced").hexdigest(),
        )


def test_reviewed_registration_cannot_be_replayed_for_different_runtime_candidate(
    monkeypatch: pytest.MonkeyPatch,
):
    attestation = build_operator_public_launch_attestation(_binding(), NONCE)
    registration = _registration(attestation)
    monkeypatch.setattr(
        trust_module,
        "REVIEWED_OPERATOR_ATTESTATIONS",
        MappingProxyType({attestation.attestation_digest: registration}),
    )
    candidate = _candidate(attestation)
    candidate["runtime_instance_id"] = "runtime:phone:other"
    with pytest.raises(OperatorAttestationError, match="does not bind exact runtime candidate"):
        verify_reviewed_operator_attestation(candidate, attestation.to_mapping())


def test_reviewed_registration_must_match_public_nonce_binding_exactly(
    monkeypatch: pytest.MonkeyPatch,
):
    attestation = build_operator_public_launch_attestation(_binding(), NONCE)
    registration = _registration(attestation)
    mismatched = ReviewedOperatorAttestationRegistration(
        attestation_digest=registration.attestation_digest,
        review_record_digest=registration.review_record_digest,
        local_verification_trace_digest=registration.local_verification_trace_digest,
        runtime_instance_id=registration.runtime_instance_id,
        source_instance_id=registration.source_instance_id,
        repository_head_sha=registration.repository_head_sha,
        entrypoint_id=registration.entrypoint_id,
        launch_attestation_id=registration.launch_attestation_id,
        logical_tick=registration.logical_tick,
        private_nonce_commitment_digest=registration.private_nonce_commitment_digest,
        nonce_binding_digest=hashlib.sha256(b"wrong-binding").hexdigest(),
        fixture_only=True,
    )
    monkeypatch.setattr(
        trust_module,
        "REVIEWED_OPERATOR_ATTESTATIONS",
        MappingProxyType({attestation.attestation_digest: mismatched}),
    )
    with pytest.raises(OperatorAttestationError, match="does not match public record"):
        verify_reviewed_operator_attestation(_candidate(attestation), attestation.to_mapping())


def test_direct_reviewed_verification_construction_is_impossible():
    digest = hashlib.sha256(b"x").hexdigest()
    with pytest.raises(OperatorAttestationError, match="issued by exact registry lookup"):
        ReviewedOperatorAttestationVerification(
            attestation_digest=digest,
            review_record_digest=hashlib.sha256(b"review").hexdigest(),
            verifier_trace_digest=hashlib.sha256(b"trace").hexdigest(),
            verified_logical_tick=1,
            fixture_only=True,
        )
