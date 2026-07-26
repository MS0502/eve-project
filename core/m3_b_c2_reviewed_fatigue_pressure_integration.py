"""Reviewed C2 integration for the exact real phone ``fatigue_pressure`` witness.

This pins the operator-reviewed public witness produced from merged PR #205 and
creates immutable runtime/source verification plus capture objects. Importing the
module performs no runtime I/O, persistence append, observation-window start, or
authority promotion.
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

C2_SCHEMA_VERSION = "eve.m3-b.c2-reviewed-fatigue-pressure-integration.v1"
PUBLIC_REVIEW_SCHEMA_VERSION = "eve.m3-b.phone-fatigue-pressure-public-review.v1"
MEASUREMENT_POLICY_VERSION = "eve.m3-b.phone-fatigue-pressure-measurement-policy.v1"
ATTESTATION_DIGEST = "421da78df1035dd994df3098c1345a448fca59b7a36f9d8cc2fb8c3dce0d4db8"
PUBLIC_REVIEW_DIGEST = "4b88c7734234ac2982836b95bf392fe143bc928119d4af515e576b39e480af61"
LOCAL_VERIFICATION_TRACE_DIGEST = "1c7649c3139c87fd30a3059aa3a678b8c3713112b932c6452fb39cf689efda37"
EVIDENCE_DIGEST = "017e189e1a35a26ce47a0372fe558e069679bf03e438ff9767bf3e0f4196a707"
PRIVATE_MATERIAL_DIGEST = "2593ea4a0ab435796911a2ddef7eb444b88148e695752710bfb6faa087302c00"
PINNED_WITNESS_HEAD = "1ac94c402d6fb8935614d0a72cda3e622b69ec82"
TRUST_DOMAIN = "eve.operator-attestation.primary.v1"
OPERATOR_ID = "primary-operator"
RUNTIME_INSTANCE_ID = "runtime:phone:primary:1ac94c402d6f"
SOURCE_INSTANCE_ID = "runtime:phone-operational-fatigue:primary"
ENTRYPOINT_ID = "scripts/operator/m3_b_phone_fatigue_pressure_witness.py:main"
LAUNCH_ATTESTATION_ID = "operator-attestation:phone:1ac94c402d6f"
SOURCE_CONTRACT_ID = "eve:m3-b:registry-source:fatigue_pressure:v1"
SOURCE_FAMILY = "operational_metrics_or_appraised_load_trace"
AXIS = "fatigue_pressure"
RUNTIME_VERIFIER_ID = "eve.m3-b.c2.fatigue-pressure-runtime-verifier"
SOURCE_VERIFIER_ID = "eve.m3-b.c2.fatigue-pressure-production-verifier"
VERIFIER_VERSION = "v1"
PROCESS_CPU_METHODS = ("os_times_process_cpu_v1",)
QUEUE_METHODS = ("kernel_loadavg_1m_normalized_v1",)
TASK_SWITCH_METHODS = ("getrusage_context_switch_delta_v1",)
SNAPSHOT_INTEGRITY_DIGESTS = (
    "bfdf87c7806daa2b860b339a07d71aa5498ad466b5761c738f7f3512312038dc",
    "b4f947aad0c8f7f4a49747166afd9e06093ab77232830a845aef60874608db12",
    "abc8482403819c0edeefd097b510ce701ddc3f092bb9040dba80cf9a53ed3a2e",
)
PRIVATE_NONCE_COMMITMENT_DIGEST = "66b676d5d951629a0dbb9348b63ae4e5710bae182abdf1a0587797d1b31d786d"
NONCE_BINDING_DIGEST = "3ed8f5b017c2f0b2982189d84beb822539ba14b4c2061a9c64777a500e9dafe1"
ZERO_DIGEST = "0" * 64
_RUNTIME_TOKEN = object()
_SOURCE_TOKEN = object()
_CAPTURE_TOKEN = object()


class C2ReviewedFatiguePressureIntegrationError(ValueError):
    pass


def _canonical(value: Any, field: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise C2ReviewedFatiguePressureIntegrationError(f"{field} is not canonical JSON material") from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) or value == ZERO_DIGEST:
        raise C2ReviewedFatiguePressureIntegrationError(f"{field} must be a non-placeholder lowercase SHA-256 digest")
    return value


def _source_entry():
    for entry in registry_observation_source_manifest().entries:
        if entry.axis == AXIS:
            if (
                entry.source_contract_id != SOURCE_CONTRACT_ID
                or entry.source_family != SOURCE_FAMILY
                or entry.minimum_raw_record_count != 3
                or entry.minimum_logical_span_ticks != 2
                or entry.appraisal_required is not False
                or entry.hardware_direct_input_allowed is not True
            ):
                raise C2ReviewedFatiguePressureIntegrationError("fatigue-pressure source contract drift")
            return entry
    raise C2ReviewedFatiguePressureIntegrationError("fatigue-pressure axis missing from source manifest")


@dataclass(frozen=True, slots=True)
class ReviewedRegistration:
    attestation_digest: str
    public_review_digest: str
    local_verification_trace_digest: str
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        _sha256(self.attestation_digest, "attestation_digest")
        _sha256(self.public_review_digest, "public_review_digest")
        _sha256(self.local_verification_trace_digest, "local_verification_trace_digest")
        if self.authority != SHADOW_AUTHORITY:
            raise C2ReviewedFatiguePressureIntegrationError("review registration must remain shadow_only")


C2_FATIGUE_REVIEWED_OPERATOR_ATTESTATIONS = MappingProxyType({
    ATTESTATION_DIGEST: ReviewedRegistration(
        ATTESTATION_DIGEST, PUBLIC_REVIEW_DIGEST, LOCAL_VERIFICATION_TRACE_DIGEST
    )
})
C2_FATIGUE_RUNTIME_PROVENANCE_VERIFIERS = MappingProxyType({TRUST_DOMAIN: (RUNTIME_VERIFIER_ID, VERIFIER_VERSION)})
C2_FATIGUE_PRODUCTION_SOURCE_VERIFIERS = MappingProxyType({SOURCE_CONTRACT_ID: (SOURCE_VERIFIER_ID, VERIFIER_VERSION)})


def verify_public_review(value: Mapping[str, Any]) -> tuple[OperatorPublicLaunchAttestation, RegistryAxisPositiveConfidenceEvidence]:
    if not isinstance(value, Mapping):
        raise C2ReviewedFatiguePressureIntegrationError("public review must be a mapping")
    review = dict(value)
    if review.get("public_review_digest") != PUBLIC_REVIEW_DIGEST:
        raise C2ReviewedFatiguePressureIntegrationError("public review digest is not the reviewed digest")
    material = dict(review)
    material.pop("public_review_digest", None)
    if _digest(material, "phone_fatigue_pressure_public_review") != PUBLIC_REVIEW_DIGEST:
        raise C2ReviewedFatiguePressureIntegrationError("public review canonical digest mismatch")
    if (
        review.get("schema_version") != PUBLIC_REVIEW_SCHEMA_VERSION
        or review.get("measurement_policy_version") != MEASUREMENT_POLICY_VERSION
        or review.get("authority") != SHADOW_AUTHORITY
        or review.get("axis") != AXIS
        or review.get("source_instance_id") != SOURCE_INSTANCE_ID
        or review.get("fixture_only") is not False
        or review.get("raw_record_count") != 3
        or review.get("private_material_digest") != PRIVATE_MATERIAL_DIGEST
        or review.get("private_raw_location") != "operator_private_companion_only"
        or tuple(review.get("process_cpu_measurement_methods", ())) != PROCESS_CPU_METHODS
        or tuple(review.get("queue_measurement_methods", ())) != QUEUE_METHODS
        or tuple(review.get("task_switch_measurement_methods", ())) != TASK_SWITCH_METHODS
        or tuple(review.get("snapshot_integrity_digests", ())) != SNAPSHOT_INTEGRITY_DIGESTS
        or review.get("tick_hz") != 1_000_000
    ):
        raise C2ReviewedFatiguePressureIntegrationError("public review envelope does not match reviewed fatigue-pressure contract")
    for field in (
        "reviewed_attestation_registered", "runtime_provenance_verifier_registered",
        "production_source_verifier_registered", "retained_real_observation",
        "observation_window_started", "m3_b_complete", "m3_c_open",
        "m3_e_authority_open", "cutover_authorized",
    ):
        if review.get(field) is not False:
            raise C2ReviewedFatiguePressureIntegrationError(f"phone witness cannot pre-claim {field}")

    attestation = OperatorPublicLaunchAttestation.from_mapping(review.get("attestation", {}))
    if (
        attestation.attestation_digest != ATTESTATION_DIGEST
        or attestation.repository_head_sha != PINNED_WITNESS_HEAD
        or attestation.runtime_instance_id != RUNTIME_INSTANCE_ID
        or attestation.source_instance_id != SOURCE_INSTANCE_ID
        or attestation.entrypoint_id != ENTRYPOINT_ID
        or attestation.launch_attestation_id != LAUNCH_ATTESTATION_ID
        or attestation.private_nonce_commitment_digest != PRIVATE_NONCE_COMMITMENT_DIGEST
        or attestation.nonce_binding_digest != NONCE_BINDING_DIGEST
        or attestation.fixture_only
        or attestation.trust_domain != TRUST_DOMAIN
        or attestation.operator_id != OPERATOR_ID
    ):
        raise C2ReviewedFatiguePressureIntegrationError("public attestation does not match reviewed fatigue-pressure registration")
    if attestation.attestation_digest not in C2_FATIGUE_REVIEWED_OPERATOR_ATTESTATIONS:
        raise C2ReviewedFatiguePressureIntegrationError("attestation is not repository-reviewed for fatigue_pressure")

    local = review.get("attestation_local_review")
    if not isinstance(local, Mapping) or (
        local.get("attestation_digest") != ATTESTATION_DIGEST
        or local.get("local_verification_trace_digest") != LOCAL_VERIFICATION_TRACE_DIGEST
        or local.get("fixture_only") is not False
        or local.get("repository_head_sha") != PINNED_WITNESS_HEAD
        or local.get("runtime_instance_id") != RUNTIME_INSTANCE_ID
        or local.get("source_instance_id") != SOURCE_INSTANCE_ID
        or local.get("trust_domain") != TRUST_DOMAIN
        or local.get("private_nonce_commitment_digest") != PRIVATE_NONCE_COMMITMENT_DIGEST
    ):
        raise C2ReviewedFatiguePressureIntegrationError("local verification summary does not match reviewed fatigue-pressure registration")

    mapping = review.get("evidence")
    if not isinstance(mapping, Mapping):
        raise C2ReviewedFatiguePressureIntegrationError("public review lacks positive-confidence evidence")
    try:
        evidence = RegistryAxisPositiveConfidenceEvidence(**dict(mapping))
    except (TypeError, ValueError) as exc:
        raise C2ReviewedFatiguePressureIntegrationError("public review evidence fails exact evidence schema") from exc
    if (
        evidence.axis != AXIS or evidence.source_family != SOURCE_FAMILY
        or evidence.source_instance_id != SOURCE_INSTANCE_ID
        or evidence.evidence_digest != EVIDENCE_DIGEST
        or review.get("evidence_digest") != EVIDENCE_DIGEST
        or review.get("evidence_observed_tick") != evidence.observed_tick
        or evidence.synthetic or evidence.proposal_only
        or evidence.verification_status != "verified"
    ):
        raise C2ReviewedFatiguePressureIntegrationError("public evidence does not match reviewed fatigue-pressure pin")
    if evidence.source_family != _source_entry().source_family:
        raise C2ReviewedFatiguePressureIntegrationError("evidence source family does not match fatigue-pressure manifest")
    return attestation, evidence


@dataclass(frozen=True, slots=True)
class FatigueRuntimeVerification:
    candidate_digest: str
    verifier_trace_digest: str
    verified_logical_tick: int
    authority: str = SHADOW_AUTHORITY
    fixture_only: bool = False
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RUNTIME_TOKEN:
            raise C2ReviewedFatiguePressureIntegrationError("runtime provenance must be issued by reviewed fatigue-pressure verifier")
        _sha256(self.candidate_digest, "candidate_digest")
        _sha256(self.verifier_trace_digest, "verifier_trace_digest")
        if self.authority != SHADOW_AUTHORITY or self.fixture_only:
            raise C2ReviewedFatiguePressureIntegrationError("fatigue-pressure runtime verification cannot weaken production proof")

    @property
    def counts_as_production(self) -> bool:
        return True

    def to_mapping(self) -> dict[str, Any]:
        return {
            "attestation_digest": ATTESTATION_DIGEST,
            "authority": self.authority,
            "candidate_digest": self.candidate_digest,
            "fixture_only": self.fixture_only,
            "public_review_digest": PUBLIC_REVIEW_DIGEST,
            "repository_head_sha": PINNED_WITNESS_HEAD,
            "runtime_instance_id": RUNTIME_INSTANCE_ID,
            "source_instance_id": SOURCE_INSTANCE_ID,
            "verified_logical_tick": self.verified_logical_tick,
            "verifier_id": RUNTIME_VERIFIER_ID,
            "verifier_trace_digest": self.verifier_trace_digest,
            "verifier_version": VERIFIER_VERSION,
        }

    @property
    def verification_digest(self) -> str:
        return _digest(self.to_mapping(), "c2_fatigue_pressure_runtime_provenance_verification")


def verify_runtime_provenance(public_review: Mapping[str, Any]) -> FatigueRuntimeVerification:
    attestation, _ = verify_public_review(public_review)
    if attestation.trust_domain not in C2_FATIGUE_RUNTIME_PROVENANCE_VERIFIERS:
        raise C2ReviewedFatiguePressureIntegrationError("fatigue-pressure runtime verifier is not registered")
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
    trace = _digest({
        "attestation_digest": ATTESTATION_DIGEST,
        "candidate_digest": candidate.candidate_digest,
        "local_verification_trace_digest": LOCAL_VERIFICATION_TRACE_DIGEST,
        "public_review_digest": PUBLIC_REVIEW_DIGEST,
        "review": "repository_reviewed_non_ci_phone_fatigue_pressure_launch",
        "verifier_id": RUNTIME_VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
    }, "c2_fatigue_pressure_runtime_provenance_verifier_trace")
    return FatigueRuntimeVerification(candidate.candidate_digest, trace, attestation.logical_tick, _issuance_token=_RUNTIME_TOKEN)


@dataclass(frozen=True, slots=True)
class FatigueSourceVerification:
    source_integrity_digest: str
    raw_observation_digest: str
    runtime_provenance_verification_digest: str
    verifier_trace_digest: str
    verified_logical_tick: int
    authority: str = SHADOW_AUTHORITY
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SOURCE_TOKEN:
            raise C2ReviewedFatiguePressureIntegrationError("source verification must be issued by registered fatigue-pressure verifier")
        for field in ("source_integrity_digest", "raw_observation_digest", "runtime_provenance_verification_digest", "verifier_trace_digest"):
            _sha256(getattr(self, field), field)
        if self.authority != SHADOW_AUTHORITY:
            raise C2ReviewedFatiguePressureIntegrationError("fatigue-pressure source verification must remain shadow_only")

    @property
    def counts_as_real(self) -> bool:
        return True

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority, "axis": AXIS, "fixture_only": False,
            "observation_evidence_digest": EVIDENCE_DIGEST, "production_origin_verified": True,
            "raw_observation_digest": self.raw_observation_digest, "runtime_capture_verified": True,
            "runtime_provenance_verification_digest": self.runtime_provenance_verification_digest,
            "source_contract_id": SOURCE_CONTRACT_ID, "source_family": SOURCE_FAMILY,
            "source_instance_id": SOURCE_INSTANCE_ID, "source_integrity_digest": self.source_integrity_digest,
            "synthetic": False, "verified_logical_tick": self.verified_logical_tick,
            "verifier_id": SOURCE_VERIFIER_ID, "verifier_trace_digest": self.verifier_trace_digest,
            "verifier_version": VERIFIER_VERSION,
        }

    @property
    def verification_digest(self) -> str:
        return _digest(self.to_mapping(), "c2_fatigue_pressure_production_source_verification")


def verify_fatigue_pressure_source(public_review: Mapping[str, Any], runtime: FatigueRuntimeVerification | None = None) -> FatigueSourceVerification:
    _, evidence = verify_public_review(public_review)
    runtime = runtime or verify_runtime_provenance(public_review)
    if type(runtime) is not FatigueRuntimeVerification or not runtime.counts_as_production:
        raise C2ReviewedFatiguePressureIntegrationError("fatigue-pressure source verifier requires issued runtime provenance")
    if SOURCE_CONTRACT_ID not in C2_FATIGUE_PRODUCTION_SOURCE_VERIFIERS:
        raise C2ReviewedFatiguePressureIntegrationError("fatigue-pressure production verifier is not registered")
    trace = _digest({
        "evidence_digest": EVIDENCE_DIGEST,
        "public_review_digest": PUBLIC_REVIEW_DIGEST,
        "runtime_provenance_verification_digest": runtime.verification_digest,
        "source_contract_id": SOURCE_CONTRACT_ID,
        "source_integrity_digest": evidence.source_integrity_digest,
        "verifier_id": SOURCE_VERIFIER_ID,
        "verifier_version": VERIFIER_VERSION,
    }, "c2_fatigue_pressure_source_verifier_trace")
    return FatigueSourceVerification(
        evidence.source_integrity_digest, evidence.raw_observation_digest,
        runtime.verification_digest, trace, evidence.observed_tick, _issuance_token=_SOURCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class C2ReviewedFatiguePressureCapture:
    capture_id: str
    capture_tick: int
    evidence: RegistryAxisPositiveConfidenceEvidence
    runtime_verification: FatigueRuntimeVerification
    source_verification: FatigueSourceVerification
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
            raise C2ReviewedFatiguePressureIntegrationError("fatigue-pressure capture must be issued by reviewed integration")
        if (
            self.evidence.evidence_digest != EVIDENCE_DIGEST
            or self.runtime_verification.verification_digest != self.source_verification.runtime_provenance_verification_digest
            or not self.runtime_verification.counts_as_production
            or not self.source_verification.counts_as_real
            or not self.retained_real_observation_eligible
            or self.authority != SHADOW_AUTHORITY
        ):
            raise C2ReviewedFatiguePressureIntegrationError("fatigue-pressure capture binding is incomplete")
        if any((self.retained_real_observation, self.observation_window_started, self.m3_b_complete, self.m3_c_open, self.m3_e_authority_open, self.cutover_authorized)):
            raise C2ReviewedFatiguePressureIntegrationError("pre-retention fatigue-pressure capture cannot claim later authority")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority, "capture_id": self.capture_id, "capture_tick": self.capture_tick,
            "cutover_authorized": self.cutover_authorized, "evidence": self.evidence.to_mapping(),
            "m3_b_complete": self.m3_b_complete, "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open, "observation_window_started": self.observation_window_started,
            "retained_real_observation": self.retained_real_observation,
            "retained_real_observation_eligible": self.retained_real_observation_eligible,
            "runtime_verification": self.runtime_verification.to_mapping(),
            "source_verification": self.source_verification.to_mapping(),
        }

    @property
    def capture_digest(self) -> str:
        return _digest(self.to_mapping(), "c2_reviewed_fatigue_pressure_production_capture")


def build_reviewed_capture(public_review: Mapping[str, Any]) -> C2ReviewedFatiguePressureCapture:
    _, evidence = verify_public_review(public_review)
    runtime = verify_runtime_provenance(public_review)
    source = verify_fatigue_pressure_source(public_review, runtime)
    return C2ReviewedFatiguePressureCapture(
        capture_id=f"c2:fatigue-pressure:{evidence.observation_id}", capture_tick=evidence.observed_tick,
        evidence=evidence, runtime_verification=runtime, source_verification=source, _issuance_token=_CAPTURE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class C2FatigueIntegrationStatus:
    reviewed_real_operator_attestation_count: int = 3
    registered_runtime_provenance_verifier_count: int = 3
    verified_production_runtime_anchor_count: int = 3
    registered_production_source_verifier_count: int = 3
    verified_positive_confidence_candidate_count: int = 3
    retained_real_observation_count: int = 2
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False


def integration_status(public_review: Mapping[str, Any]) -> C2FatigueIntegrationStatus:
    runtime = verify_runtime_provenance(public_review)
    source = verify_fatigue_pressure_source(public_review, runtime)
    if not runtime.counts_as_production or not source.counts_as_real:
        raise C2ReviewedFatiguePressureIntegrationError("fatigue-pressure reviewed status requires exact issued verifications")
    return C2FatigueIntegrationStatus()
