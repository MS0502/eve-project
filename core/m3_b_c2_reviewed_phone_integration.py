"""Reviewed C2 integration for the exact non-CI phone prediction-error witness.

This module is the first reviewed activation layer above the historical C1/C2
preflights. It does not mutate those historical registries. Instead it pins one
exact operator-reviewed phone witness and exposes immutable C2 registries whose
verifications can only be issued by the functions in this module.

Importing this module performs no runtime I/O, persistence, retention append,
registry-owner mutation, observation-window transition, or authority promotion.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import InitVar, dataclass
from types import MappingProxyType
from typing import Any, Mapping

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_operator_attestation_trust_root import OperatorPublicLaunchAttestation
from core.m3_b_production_runtime_provenance_preflight import RuntimeProvenanceCandidate
from core.m3_b_registry_observation_evidence import RegistryAxisPositiveConfidenceEvidence
from core.m3_b_registry_observation_source_manifest import registry_observation_source_manifest

C2_SCHEMA_VERSION = "eve.m3-b.c2-reviewed-phone-integration.v1"
PUBLIC_REVIEW_SCHEMA_VERSION = "eve.m3-b.phone-prediction-error-public-review.v2"
ATTESTATION_DIGEST = "85b55eee61618ad98476f71c4dadcb9b2e4383d79aefd93a41a2c34634efecda"
PUBLIC_REVIEW_DIGEST = "6a3d34120d9773f28544aa82d963cf2e65220f6f899aeab42c132660f87ad81e"
LOCAL_VERIFICATION_TRACE_DIGEST = "9a1662e233d7b22ca00efae3a6db67a8278a79e648ea2326d5efc081dfc5b77f"
EVIDENCE_DIGEST = "14549d2b9f37f2a8b00a5bc9de61dbdad8e12dbb8a4d4e08e254ef0e9848b3dc"
PRIVATE_MATERIAL_DIGEST = "7db1fd5fd8a06848f4eb8f0d4d3905e99293ea605bc280ec4fe96bf6d157766c"
PINNED_WITNESS_HEAD = "b4968be9aeb6eefc7274f9985ab333f08e470daf"
TRUST_DOMAIN = "eve.operator-attestation.primary.v1"
OPERATOR_ID = "primary-operator"
RUNTIME_INSTANCE_ID = "runtime:phone:primary:b4968be9aeb6"
SOURCE_INSTANCE_ID = "runtime:ai-adapter:primary"
ENTRYPOINT_ID = "scripts/operator/m3_b_phone_prediction_error_witness.py:main"
LAUNCH_ATTESTATION_ID = "operator-attestation:phone:b4968be9aeb6"
SOURCE_CONTRACT_ID = "eve:m3-b:registry-source:prediction_error_pressure:v1"
AXIS = "prediction_error_pressure"
RUNTIME_VERIFIER_ID = "eve.m3-b.c2.primary-operator-runtime-verifier"
RUNTIME_VERIFIER_VERSION = "v1"
SOURCE_VERIFIER_ID = "eve.m3-b.c2.prediction-error-production-verifier"
SOURCE_VERIFIER_VERSION = "v1"
ZERO_DIGEST = "0" * 64
_RUNTIME_TOKEN = object()
_SOURCE_TOKEN = object()
_CAPTURE_TOKEN = object()


class C2ReviewedPhoneIntegrationError(ValueError):
    """Raised when reviewed C2 material deviates from the pinned phone witness."""


def _canonical(value: Any, field: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise C2ReviewedPhoneIntegrationError(f"{field} is not canonical JSON material") from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == ZERO_DIGEST
    ):
        raise C2ReviewedPhoneIntegrationError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _source_entry():
    for entry in registry_observation_source_manifest().entries:
        if entry.axis == AXIS:
            if entry.source_contract_id != SOURCE_CONTRACT_ID:
                raise C2ReviewedPhoneIntegrationError("prediction-error source contract drift")
            return entry
    raise C2ReviewedPhoneIntegrationError("prediction-error axis missing from source manifest")


@dataclass(frozen=True, slots=True)
class C2ReviewedAttestationRegistration:
    attestation_digest: str
    public_review_digest: str
    local_verification_trace_digest: str
    runtime_instance_id: str
    source_instance_id: str
    repository_head_sha: str
    entrypoint_id: str
    launch_attestation_id: str
    private_nonce_commitment_digest: str
    nonce_binding_digest: str
    fixture_only: bool = False
    trust_domain: str = TRUST_DOMAIN
    operator_id: str = OPERATOR_ID
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        for field in (
            "attestation_digest",
            "public_review_digest",
            "local_verification_trace_digest",
            "private_nonce_commitment_digest",
            "nonce_binding_digest",
        ):
            _sha256(getattr(self, field), field)
        if (
            self.runtime_instance_id != RUNTIME_INSTANCE_ID
            or self.source_instance_id != SOURCE_INSTANCE_ID
            or self.repository_head_sha != PINNED_WITNESS_HEAD
            or self.entrypoint_id != ENTRYPOINT_ID
            or self.launch_attestation_id != LAUNCH_ATTESTATION_ID
            or self.trust_domain != TRUST_DOMAIN
            or self.operator_id != OPERATOR_ID
            or self.fixture_only
            or self.authority != SHADOW_AUTHORITY
        ):
            raise C2ReviewedPhoneIntegrationError("reviewed attestation registration drift")


@dataclass(frozen=True, slots=True)
class C2VerifierRegistration:
    binding_id: str
    verifier_id: str
    verifier_version: str
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        for value in (self.binding_id, self.verifier_id, self.verifier_version):
            if not isinstance(value, str) or not value.strip():
                raise C2ReviewedPhoneIntegrationError("verifier registration identifiers must be non-empty")
        if self.authority != SHADOW_AUTHORITY:
            raise C2ReviewedPhoneIntegrationError("C2 verifier registration must remain shadow_only")


C2_REVIEWED_OPERATOR_ATTESTATIONS: Mapping[str, C2ReviewedAttestationRegistration] = MappingProxyType({
    ATTESTATION_DIGEST: C2ReviewedAttestationRegistration(
        attestation_digest=ATTESTATION_DIGEST,
        public_review_digest=PUBLIC_REVIEW_DIGEST,
        local_verification_trace_digest=LOCAL_VERIFICATION_TRACE_DIGEST,
        runtime_instance_id=RUNTIME_INSTANCE_ID,
        source_instance_id=SOURCE_INSTANCE_ID,
        repository_head_sha=PINNED_WITNESS_HEAD,
        entrypoint_id=ENTRYPOINT_ID,
        launch_attestation_id=LAUNCH_ATTESTATION_ID,
        private_nonce_commitment_digest="66b676d5d951629a0dbb9348b63ae4e5710bae182abdf1a0587797d1b31d786d",
        nonce_binding_digest="9342e0523b30389ad4fe6f7a6582d047fc8e9eb15d8c668a5ee2577b7c5fca60",
    )
})
C2_RUNTIME_PROVENANCE_VERIFIERS: Mapping[str, C2VerifierRegistration] = MappingProxyType({
    TRUST_DOMAIN: C2VerifierRegistration(
        binding_id=TRUST_DOMAIN,
        verifier_id=RUNTIME_VERIFIER_ID,
        verifier_version=RUNTIME_VERIFIER_VERSION,
    )
})
C2_PRODUCTION_SOURCE_VERIFIERS: Mapping[str, C2VerifierRegistration] = MappingProxyType({
    SOURCE_CONTRACT_ID: C2VerifierRegistration(
        binding_id=SOURCE_CONTRACT_ID,
        verifier_id=SOURCE_VERIFIER_ID,
        verifier_version=SOURCE_VERIFIER_VERSION,
    )
})


def verify_public_review(value: Mapping[str, Any]) -> tuple[OperatorPublicLaunchAttestation, RegistryAxisPositiveConfidenceEvidence]:
    """Validate the complete public-safe v2 witness against the reviewed pin."""

    if not isinstance(value, Mapping):
        raise C2ReviewedPhoneIntegrationError("public review must be a mapping")
    review = dict(value)
    claimed_digest = review.get("public_review_digest")
    if claimed_digest != PUBLIC_REVIEW_DIGEST:
        raise C2ReviewedPhoneIntegrationError("public review digest is not the reviewed digest")
    material = dict(review)
    material.pop("public_review_digest", None)
    if _digest(material, "phone_prediction_error_public_review") != PUBLIC_REVIEW_DIGEST:
        raise C2ReviewedPhoneIntegrationError("public review canonical digest mismatch")
    if (
        review.get("schema_version") != PUBLIC_REVIEW_SCHEMA_VERSION
        or review.get("authority") != SHADOW_AUTHORITY
        or review.get("fixture_only") is not False
        or review.get("raw_record_count") != 2
        or review.get("private_material_digest") != PRIVATE_MATERIAL_DIGEST
        or review.get("private_raw_location") != "operator_private_companion_only"
    ):
        raise C2ReviewedPhoneIntegrationError("public review envelope does not match reviewed C2 contract")
    for field in (
        "reviewed_attestation_registered",
        "runtime_provenance_verifier_registered",
        "production_source_verifier_registered",
        "retained_real_observation",
        "observation_window_started",
        "m3_b_complete",
        "m3_c_open",
        "m3_e_authority_open",
        "cutover_authorized",
    ):
        if review.get(field) is not False:
            raise C2ReviewedPhoneIntegrationError(f"phone witness cannot pre-claim {field}")

    attestation = OperatorPublicLaunchAttestation.from_mapping(review.get("attestation", {}))
    if attestation.attestation_digest != ATTESTATION_DIGEST:
        raise C2ReviewedPhoneIntegrationError("public attestation digest mismatch")
    registration = C2_REVIEWED_OPERATOR_ATTESTATIONS.get(attestation.attestation_digest)
    if type(registration) is not C2ReviewedAttestationRegistration:
        raise C2ReviewedPhoneIntegrationError("attestation is not C2 repository-reviewed")
    expected_attestation = (
        registration.runtime_instance_id,
        registration.source_instance_id,
        registration.repository_head_sha,
        registration.entrypoint_id,
        registration.launch_attestation_id,
        registration.private_nonce_commitment_digest,
        registration.nonce_binding_digest,
        registration.fixture_only,
        registration.trust_domain,
        registration.operator_id,
    )
    observed_attestation = (
        attestation.runtime_instance_id,
        attestation.source_instance_id,
        attestation.repository_head_sha,
        attestation.entrypoint_id,
        attestation.launch_attestation_id,
        attestation.private_nonce_commitment_digest,
        attestation.nonce_binding_digest,
        attestation.fixture_only,
        attestation.trust_domain,
        attestation.operator_id,
    )
    if observed_attestation != expected_attestation:
        raise C2ReviewedPhoneIntegrationError("public attestation does not match reviewed registration")

    local_review = review.get("attestation_local_review")
    if not isinstance(local_review, Mapping):
        raise C2ReviewedPhoneIntegrationError("attestation local review is missing")
    if (
        local_review.get("attestation_digest") != ATTESTATION_DIGEST
        or local_review.get("local_verification_trace_digest") != LOCAL_VERIFICATION_TRACE_DIGEST
        or local_review.get("fixture_only") is not False
        or local_review.get("repository_head_sha") != PINNED_WITNESS_HEAD
        or local_review.get("runtime_instance_id") != RUNTIME_INSTANCE_ID
        or local_review.get("source_instance_id") != SOURCE_INSTANCE_ID
        or local_review.get("trust_domain") != TRUST_DOMAIN
    ):
        raise C2ReviewedPhoneIntegrationError("local verification summary does not match reviewed registration")

    evidence_mapping = review.get("evidence")
    if not isinstance(evidence_mapping, Mapping):
        raise C2ReviewedPhoneIntegrationError("public review lacks positive-confidence evidence")
    try:
        evidence = RegistryAxisPositiveConfidenceEvidence(**dict(evidence_mapping))
    except (TypeError, ValueError) as exc:
        raise C2ReviewedPhoneIntegrationError("public review evidence fails exact evidence schema") from exc
    if (
        evidence.axis != AXIS
        or evidence.source_instance_id != SOURCE_INSTANCE_ID
        or evidence.evidence_digest != EVIDENCE_DIGEST
        or review.get("evidence_digest") != EVIDENCE_DIGEST
        or review.get("evidence_observed_tick") != evidence.observed_tick
    ):
        raise C2ReviewedPhoneIntegrationError("public evidence does not match reviewed prediction-error pin")
    _source_entry()
    return attestation, evidence


@dataclass(frozen=True, slots=True)
class C2RuntimeProvenanceVerification:
    candidate_digest: str
    attestation_digest: str
    public_review_digest: str
    runtime_instance_id: str
    source_instance_id: str
    repository_head_sha: str
    verifier_id: str
    verifier_version: str
    verifier_trace_digest: str
    verified_logical_tick: int
    fixture_only: bool = False
    non_ci_runtime_verified: bool = True
    production_launch_verified: bool = True
    independent_trust_root_verified: bool = True
    authority: str = SHADOW_AUTHORITY
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RUNTIME_TOKEN:
            raise C2ReviewedPhoneIntegrationError("runtime provenance must be issued by C2 reviewed verifier")
        for field in (
            "candidate_digest",
            "attestation_digest",
            "public_review_digest",
            "verifier_trace_digest",
        ):
            _sha256(getattr(self, field), field)
        if (
            self.fixture_only
            or not self.non_ci_runtime_verified
            or not self.production_launch_verified
            or not self.independent_trust_root_verified
            or self.authority != SHADOW_AUTHORITY
        ):
            raise C2ReviewedPhoneIntegrationError("C2 runtime provenance cannot weaken reviewed production proof")

    @property
    def counts_as_production(self) -> bool:
        return True

    def to_mapping(self) -> dict[str, Any]:
        return {
            "attestation_digest": self.attestation_digest,
            "authority": self.authority,
            "candidate_digest": self.candidate_digest,
            "fixture_only": self.fixture_only,
            "independent_trust_root_verified": self.independent_trust_root_verified,
            "non_ci_runtime_verified": self.non_ci_runtime_verified,
            "production_launch_verified": self.production_launch_verified,
            "public_review_digest": self.public_review_digest,
            "repository_head_sha": self.repository_head_sha,
            "runtime_instance_id": self.runtime_instance_id,
            "source_instance_id": self.source_instance_id,
            "verified_logical_tick": self.verified_logical_tick,
            "verifier_id": self.verifier_id,
            "verifier_trace_digest": self.verifier_trace_digest,
            "verifier_version": self.verifier_version,
        }

    @property
    def verification_digest(self) -> str:
        return _digest(self.to_mapping(), "c2_runtime_provenance_verification")


def verify_runtime_provenance(public_review: Mapping[str, Any]) -> C2RuntimeProvenanceVerification:
    attestation, _evidence = verify_public_review(public_review)
    registration = C2_RUNTIME_PROVENANCE_VERIFIERS.get(attestation.trust_domain)
    if type(registration) is not C2VerifierRegistration:
        raise C2ReviewedPhoneIntegrationError("C2 runtime verifier is not registered")
    candidate = RuntimeProvenanceCandidate(
        trust_domain=attestation.trust_domain,
        runtime_instance_id=attestation.runtime_instance_id,
        source_instance_id=attestation.source_instance_id,
        repository_head_sha=attestation.repository_head_sha,
        entrypoint_id=attestation.entrypoint_id,
        launch_attestation_id=attestation.launch_attestation_id,
        launch_attestation_digest=attestation.attestation_digest,
        logical_tick=attestation.logical_tick,
        fixture_only=False,
    )
    trace = _digest(
        {
            "attestation_digest": attestation.attestation_digest,
            "candidate_digest": candidate.candidate_digest,
            "local_verification_trace_digest": LOCAL_VERIFICATION_TRACE_DIGEST,
            "public_review_digest": PUBLIC_REVIEW_DIGEST,
            "review": "repository_reviewed_non_ci_phone_launch",
            "verifier_id": registration.verifier_id,
            "verifier_version": registration.verifier_version,
        },
        "c2_runtime_provenance_verifier_trace",
    )
    return C2RuntimeProvenanceVerification(
        candidate_digest=candidate.candidate_digest,
        attestation_digest=attestation.attestation_digest,
        public_review_digest=PUBLIC_REVIEW_DIGEST,
        runtime_instance_id=attestation.runtime_instance_id,
        source_instance_id=attestation.source_instance_id,
        repository_head_sha=attestation.repository_head_sha,
        verifier_id=registration.verifier_id,
        verifier_version=registration.verifier_version,
        verifier_trace_digest=trace,
        verified_logical_tick=attestation.logical_tick,
        _issuance_token=_RUNTIME_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class C2ProductionSourceVerification:
    axis: str
    source_contract_id: str
    source_family: str
    source_instance_id: str
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    raw_observation_digest: str
    observation_evidence_digest: str
    runtime_provenance_verification_digest: str
    verifier_id: str
    verifier_version: str
    verifier_trace_digest: str
    verified_logical_tick: int
    fixture_only: bool = False
    synthetic: bool = False
    production_origin_verified: bool = True
    runtime_capture_verified: bool = True
    authority: str = SHADOW_AUTHORITY
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SOURCE_TOKEN:
            raise C2ReviewedPhoneIntegrationError("source verification must be issued by C2 registered verifier")
        for field in (
            "source_integrity_digest",
            "raw_observation_digest",
            "observation_evidence_digest",
            "runtime_provenance_verification_digest",
            "verifier_trace_digest",
        ):
            _sha256(getattr(self, field), field)
        if (
            self.axis != AXIS
            or self.source_contract_id != SOURCE_CONTRACT_ID
            or self.fixture_only
            or self.synthetic
            or not self.production_origin_verified
            or not self.runtime_capture_verified
            or self.authority != SHADOW_AUTHORITY
        ):
            raise C2ReviewedPhoneIntegrationError("C2 source verification cannot weaken reviewed real-source proof")

    @property
    def counts_as_real(self) -> bool:
        return True

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "axis": self.axis,
            "fixture_only": self.fixture_only,
            "observation_evidence_digest": self.observation_evidence_digest,
            "production_origin_verified": self.production_origin_verified,
            "raw_observation_digest": self.raw_observation_digest,
            "runtime_capture_verified": self.runtime_capture_verified,
            "runtime_provenance_verification_digest": self.runtime_provenance_verification_digest,
            "source_contract_id": self.source_contract_id,
            "source_family": self.source_family,
            "source_instance_id": self.source_instance_id,
            "source_integrity_digest": self.source_integrity_digest,
            "source_schema_version": self.source_schema_version,
            "source_snapshot_id": self.source_snapshot_id,
            "synthetic": self.synthetic,
            "verified_logical_tick": self.verified_logical_tick,
            "verifier_id": self.verifier_id,
            "verifier_trace_digest": self.verifier_trace_digest,
            "verifier_version": self.verifier_version,
        }

    @property
    def verification_digest(self) -> str:
        return _digest(self.to_mapping(), "c2_production_source_verification")


def verify_prediction_error_source(
    public_review: Mapping[str, Any],
    runtime_verification: C2RuntimeProvenanceVerification | None = None,
) -> C2ProductionSourceVerification:
    _attestation, evidence = verify_public_review(public_review)
    runtime = runtime_verification or verify_runtime_provenance(public_review)
    if type(runtime) is not C2RuntimeProvenanceVerification or not runtime.counts_as_production:
        raise C2ReviewedPhoneIntegrationError("C2 source verifier requires issued runtime provenance")
    if (
        runtime.attestation_digest != ATTESTATION_DIGEST
        or runtime.public_review_digest != PUBLIC_REVIEW_DIGEST
        or runtime.source_instance_id != evidence.source_instance_id
        or runtime.repository_head_sha != PINNED_WITNESS_HEAD
    ):
        raise C2ReviewedPhoneIntegrationError("runtime provenance does not bind reviewed source evidence")
    entry = _source_entry()
    registration = C2_PRODUCTION_SOURCE_VERIFIERS.get(entry.source_contract_id)
    if type(registration) is not C2VerifierRegistration:
        raise C2ReviewedPhoneIntegrationError("prediction-error production verifier is not registered")
    trace = _digest(
        {
            "evidence_digest": evidence.evidence_digest,
            "public_review_digest": PUBLIC_REVIEW_DIGEST,
            "runtime_provenance_verification_digest": runtime.verification_digest,
            "source_contract_id": entry.source_contract_id,
            "source_integrity_digest": evidence.source_integrity_digest,
            "verifier_id": registration.verifier_id,
            "verifier_version": registration.verifier_version,
        },
        "c2_prediction_error_source_verifier_trace",
    )
    return C2ProductionSourceVerification(
        axis=evidence.axis,
        source_contract_id=entry.source_contract_id,
        source_family=evidence.source_family,
        source_instance_id=evidence.source_instance_id,
        source_snapshot_id=evidence.source_snapshot_id,
        source_schema_version=evidence.source_schema_version,
        source_integrity_digest=evidence.source_integrity_digest,
        raw_observation_digest=evidence.raw_observation_digest,
        observation_evidence_digest=evidence.evidence_digest,
        runtime_provenance_verification_digest=runtime.verification_digest,
        verifier_id=registration.verifier_id,
        verifier_version=registration.verifier_version,
        verifier_trace_digest=trace,
        verified_logical_tick=evidence.observed_tick,
        _issuance_token=_SOURCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class C2ReviewedProductionCapture:
    capture_id: str
    capture_tick: int
    evidence: RegistryAxisPositiveConfidenceEvidence
    runtime_verification: C2RuntimeProvenanceVerification
    source_verification: C2ProductionSourceVerification
    authority: str = SHADOW_AUTHORITY
    retained_real_observation_eligible: bool = True
    retained_real_observation: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _CAPTURE_TOKEN:
            raise C2ReviewedPhoneIntegrationError("C2 capture must be issued by reviewed integration")
        if (
            self.evidence.evidence_digest != EVIDENCE_DIGEST
            or self.runtime_verification.verification_digest
            != self.source_verification.runtime_provenance_verification_digest
            or self.source_verification.observation_evidence_digest != self.evidence.evidence_digest
            or not self.runtime_verification.counts_as_production
            or not self.source_verification.counts_as_real
            or not self.retained_real_observation_eligible
            or self.authority != SHADOW_AUTHORITY
        ):
            raise C2ReviewedPhoneIntegrationError("C2 capture binding is incomplete")
        if any((
            self.retained_real_observation,
            self.observation_window_started,
            self.m3_b_complete,
            self.m3_c_open,
            self.m3_e_authority_open,
            self.cutover_authorized,
        )):
            raise C2ReviewedPhoneIntegrationError("pre-retention C2 capture cannot claim later authority")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "capture_id": self.capture_id,
            "capture_tick": self.capture_tick,
            "cutover_authorized": self.cutover_authorized,
            "evidence": self.evidence.to_mapping(),
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "observation_window_started": self.observation_window_started,
            "retained_real_observation": self.retained_real_observation,
            "retained_real_observation_eligible": self.retained_real_observation_eligible,
            "runtime_verification": self.runtime_verification.to_mapping(),
            "source_verification": self.source_verification.to_mapping(),
        }

    @property
    def capture_digest(self) -> str:
        return _digest(self.to_mapping(), "c2_reviewed_production_capture")


def build_reviewed_capture(public_review: Mapping[str, Any]) -> C2ReviewedProductionCapture:
    _attestation, evidence = verify_public_review(public_review)
    runtime = verify_runtime_provenance(public_review)
    source = verify_prediction_error_source(public_review, runtime)
    return C2ReviewedProductionCapture(
        capture_id=f"c2:prediction-error:{evidence.observation_id}",
        capture_tick=evidence.observed_tick,
        evidence=evidence,
        runtime_verification=runtime,
        source_verification=source,
        _issuance_token=_CAPTURE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class C2IntegrationStatus:
    reviewed_real_operator_attestation_count: int = 1
    registered_runtime_provenance_verifier_count: int = 1
    verified_production_runtime_anchor_count: int = 1
    registered_production_source_verifier_count: int = 1
    verified_positive_confidence_candidate_count: int = 1
    retained_real_observation_count: int = 0
    observation_window_eligible: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False
    authority: str = SHADOW_AUTHORITY
    schema_version: str = C2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.reviewed_real_operator_attestation_count != len(C2_REVIEWED_OPERATOR_ATTESTATIONS)
            or self.registered_runtime_provenance_verifier_count != len(C2_RUNTIME_PROVENANCE_VERIFIERS)
            or self.registered_production_source_verifier_count != len(C2_PRODUCTION_SOURCE_VERIFIERS)
            or self.verified_production_runtime_anchor_count != 1
            or self.verified_positive_confidence_candidate_count != 1
            or self.retained_real_observation_count != 0
            or self.observation_window_eligible
            or self.observation_window_started
            or self.m3_b_complete
            or self.m3_c_open
            or self.m3_e_authority_open
            or self.cutover_authorized
            or self.authority != SHADOW_AUTHORITY
            or self.schema_version != C2_SCHEMA_VERSION
        ):
            raise C2ReviewedPhoneIntegrationError("C2 integration status overstates reviewed activation")


def integration_status(public_review: Mapping[str, Any]) -> C2IntegrationStatus:
    capture = build_reviewed_capture(public_review)
    if not (
        capture.runtime_verification.counts_as_production
        and capture.source_verification.counts_as_real
        and capture.evidence.confidence > 0.0
    ):
        raise C2ReviewedPhoneIntegrationError("reviewed C2 integration cannot establish verified candidate")
    return C2IntegrationStatus()
