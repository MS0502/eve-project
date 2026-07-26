"""Reviewed C2 integration for the exact real phone ``recovery_need`` witness.

This pins the operator-reviewed public witness produced from merged PR #208 and
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

C2_SCHEMA_VERSION = "eve.m3-b.c2-reviewed-recovery-need-integration.v1"
PUBLIC_REVIEW_SCHEMA_VERSION = "eve.m3-b.phone-recovery-need-public-review.v1"
MEASUREMENT_POLICY_VERSION = "eve.m3-b.phone-recovery-need-measurement-policy.v1"
ATTESTATION_DIGEST = "ce8cda9955a415ed05200a83fd6b3e8d4cd4028bef29f73b8b17a1d5e3ad25e1"
PUBLIC_REVIEW_DIGEST = "e46df034d01b13e768ce37d14261b8ed20fdec30101945bea492d97e482e4c33"
LOCAL_VERIFICATION_TRACE_DIGEST = "e5e59447549fed22e2ac2aba0ca093ec35d0c10489805c55b0e1945a092420b9"
EVIDENCE_DIGEST = "535495759c0140d875da628d2fe5cc9ffc0904d5f91fa9546a784dd51b3baa4b"
PRIVATE_MATERIAL_DIGEST = "30a341632cc016da9fea6eb32822ac48a49216ccbfe160bf10e07bd567f36cd3"
PINNED_WITNESS_HEAD = "f0edb05201671814fed131ccbb73d2cb3b8d3f59"
TRUST_DOMAIN = "eve.operator-attestation.primary.v1"
OPERATOR_ID = "primary-operator"
RUNTIME_INSTANCE_ID = "runtime:phone:primary:f0edb0520167"
SOURCE_INSTANCE_ID = "runtime:phone-operational-recovery:primary"
ENTRYPOINT_ID = "scripts/operator/m3_b_phone_recovery_need_witness.py:main"
LAUNCH_ATTESTATION_ID = "operator-attestation:phone:f0edb0520167"
SOURCE_CONTRACT_ID = "eve:m3-b:registry-source:recovery_need:v1"
SOURCE_FAMILY = "operational_metrics_or_appraised_load_trace"
AXIS = "recovery_need"
RUNTIME_VERIFIER_ID = "eve.m3-b.c2.recovery-need-runtime-verifier"
SOURCE_VERIFIER_ID = "eve.m3-b.c2.recovery-need-production-verifier"
VERIFIER_VERSION = "v1"
PROCESS_CPU_METHODS = ("os_times_process_cpu_v1",)
QUEUE_METHODS = ("kernel_loadavg_1m_capacity_comparison_v1",)
COOLDOWN_METHODS = ("fixed_post_interaction_quiet_window_1s_v1",)
OVERLOAD_COUNT_METHODS = ("loadavg_visible_cpu_capacity_breach_count_v1",)
RECOVERY_COUNT_METHODS = ("cpu_and_queue_nonincrease_indicator_count_v1",)
SNAPSHOT_INTEGRITY_DIGESTS = (
    "454d617c4a96c7ff15e047d186f961fcf3b9ecb94a83ad5edc6f993eb6744276",
    "39d8e1f91ae0fa3ecd84c1bf7cc55a629917b03c29ca3ffda37d619b3499b043",
    "3d11943074f5cf4621cc517b5dc5e5cef5e309f244e9e476b5c0725cba357655",
)
PRIVATE_NONCE_COMMITMENT_DIGEST = "66b676d5d951629a0dbb9348b63ae4e5710bae182abdf1a0587797d1b31d786d"
NONCE_BINDING_DIGEST = "f0c659bd7288ac2035a4d922b782873d7c027de3da5a537657c6d2482f481d62"
ZERO_DIGEST = "0" * 64
_RUNTIME_TOKEN = object()
_SOURCE_TOKEN = object()
_CAPTURE_TOKEN = object()


class C2ReviewedRecoveryNeedIntegrationError(ValueError):
    pass


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
        raise C2ReviewedRecoveryNeedIntegrationError(
            f"{field} is not canonical JSON material"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        or value == ZERO_DIGEST
    ):
        raise C2ReviewedRecoveryNeedIntegrationError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
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
                raise C2ReviewedRecoveryNeedIntegrationError(
                    "recovery-need source contract drift"
                )
            return entry
    raise C2ReviewedRecoveryNeedIntegrationError(
        "recovery-need axis missing from source manifest"
    )


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
            raise C2ReviewedRecoveryNeedIntegrationError(
                "review registration must remain shadow_only"
            )


C2_RECOVERY_REVIEWED_OPERATOR_ATTESTATIONS = MappingProxyType(
    {
        ATTESTATION_DIGEST: ReviewedRegistration(
            ATTESTATION_DIGEST,
            PUBLIC_REVIEW_DIGEST,
            LOCAL_VERIFICATION_TRACE_DIGEST,
        )
    }
)
C2_RECOVERY_RUNTIME_PROVENANCE_VERIFIERS = MappingProxyType(
    {TRUST_DOMAIN: (RUNTIME_VERIFIER_ID, VERIFIER_VERSION)}
)
C2_RECOVERY_PRODUCTION_SOURCE_VERIFIERS = MappingProxyType(
    {SOURCE_CONTRACT_ID: (SOURCE_VERIFIER_ID, VERIFIER_VERSION)}
)


def verify_public_review(
    value: Mapping[str, Any],
) -> tuple[OperatorPublicLaunchAttestation, RegistryAxisPositiveConfidenceEvidence]:
    if not isinstance(value, Mapping):
        raise C2ReviewedRecoveryNeedIntegrationError("public review must be a mapping")
    review = dict(value)
    if review.get("public_review_digest") != PUBLIC_REVIEW_DIGEST:
        raise C2ReviewedRecoveryNeedIntegrationError(
            "public review digest is not the reviewed digest"
        )
    material = dict(review)
    material.pop("public_review_digest", None)
    if _digest(material, "phone_recovery_need_public_review") != PUBLIC_REVIEW_DIGEST:
        raise C2ReviewedRecoveryNeedIntegrationError(
            "public review canonical digest mismatch"
        )
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
        or tuple(review.get("cooldown_measurement_methods", ())) != COOLDOWN_METHODS
        or tuple(review.get("overload_count_methods", ())) != OVERLOAD_COUNT_METHODS
        or tuple(review.get("recovery_count_methods", ())) != RECOVERY_COUNT_METHODS
        or tuple(review.get("snapshot_integrity_digests", ())) != SNAPSHOT_INTEGRITY_DIGESTS
        or review.get("tick_hz") != 1_000_000
    ):
        raise C2ReviewedRecoveryNeedIntegrationError(
            "public review envelope does not match reviewed recovery-need contract"
        )
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
            raise C2ReviewedRecoveryNeedIntegrationError(
                f"phone witness cannot pre-claim {field}"
            )

    attestation = OperatorPublicLaunchAttestation.from_mapping(
        review.get("attestation", {})
    )
    if (
        attestation.attestation_digest != ATTESTATION_DIGEST
        or attestation.repository_head_sha != PINNED_WITNESS_HEAD
        or attestation.runtime_instance_id != RUNTIME_INSTANCE_ID
        or attestation.source_instance_id != SOURCE_INSTANCE_ID
        or attestation.entrypoint_id != ENTRYPOINT_ID
        or attestation.launch_attestation_id != LAUNCH_ATTESTATION_ID
        or attestation.private_nonce_commitment_digest
        != PRIVATE_NONCE_COMMITMENT_DIGEST
        or attestation.nonce_binding_digest != NONCE_BINDING_DIGEST
        or attestation.fixture_only
        or attestation.trust_domain != TRUST_DOMAIN
        or attestation.operator_id != OPERATOR_ID
    ):
        raise C2ReviewedRecoveryNeedIntegrationError(
            "public attestation does not match reviewed recovery-need registration"
        )
    if attestation.attestation_digest not in C2_RECOVERY_REVIEWED_OPERATOR_ATTESTATIONS:
        raise C2ReviewedRecoveryNeedIntegrationError(
            "attestation is not repository-reviewed for recovery_need"
        )

    local = review.get("attestation_local_review")
    if not isinstance(local, Mapping) or (
        local.get("attestation_digest") != ATTESTATION_DIGEST
        or local.get("local_verification_trace_digest")
        != LOCAL_VERIFICATION_TRACE_DIGEST
        or local.get("fixture_only") is not False
        or local.get("repository_head_sha") != PINNED_WITNESS_HEAD
        or local.get("runtime_instance_id") != RUNTIME_INSTANCE_ID
        or local.get("source_instance_id") != SOURCE_INSTANCE_ID
        or local.get("trust_domain") != TRUST_DOMAIN
        or local.get("private_nonce_commitment_digest")
        != PRIVATE_NONCE_COMMITMENT_DIGEST
        or local.get("launch_attestation_id") != LAUNCH_ATTESTATION_ID
    ):
        raise C2ReviewedRecoveryNeedIntegrationError(
            "local verification summary does not match reviewed recovery-need registration"
        )

    mapping = review.get("evidence")
    if not isinstance(mapping, Mapping):
        raise C2ReviewedRecoveryNeedIntegrationError(
            "public review lacks positive-confidence evidence"
        )
    try:
        evidence = RegistryAxisPositiveConfidenceEvidence(**dict(mapping))
    except (TypeError, ValueError) as exc:
        raise C2ReviewedRecoveryNeedIntegrationError(
            "public review evidence fails exact evidence schema"
        ) from exc
    if (
        evidence.axis != AXIS
        or evidence.source_family != SOURCE_FAMILY
        or evidence.source_instance_id != SOURCE_INSTANCE_ID
        or evidence.evidence_digest != EVIDENCE_DIGEST
        or review.get("evidence_digest") != EVIDENCE_DIGEST
        or review.get("evidence_observed_tick") != evidence.observed_tick
        or evidence.synthetic
        or evidence.proposal_only
        or evidence.verification_status != "verified"
    ):
        raise C2ReviewedRecoveryNeedIntegrationError(
            "public evidence does not match reviewed recovery-need pin"
        )
    if evidence.source_family != _source_entry().source_family:
        raise C2ReviewedRecoveryNeedIntegrationError(
            "evidence source family does not match recovery-need manifest"
        )
    return attestation, evidence


@dataclass(frozen=True, slots=True)
class RecoveryNeedRuntimeVerification:
    candidate_digest: str
    verifier_trace_digest: str
    verified_logical_tick: int
    authority: str = SHADOW_AUTHORITY
    fixture_only: bool = False
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RUNTIME_TOKEN:
            raise C2ReviewedRecoveryNeedIntegrationError(
                "runtime provenance must be issued by reviewed recovery-need verifier"
            )
        _sha256(self.candidate_digest, "candidate_digest")
        _sha256(self.verifier_trace_digest, "verifier_trace_digest")
        if self.authority != SHADOW_AUTHORITY or self.fixture_only:
            raise C2ReviewedRecoveryNeedIntegrationError(
                "recovery-need runtime verification cannot weaken production proof"
            )

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
        return _digest(
            self.to_mapping(),
            "c2_recovery_need_runtime_provenance_verification",
        )


def verify_runtime_provenance(
    public_review: Mapping[str, Any],
) -> RecoveryNeedRuntimeVerification:
    attestation, _ = verify_public_review(public_review)
    if attestation.trust_domain not in C2_RECOVERY_RUNTIME_PROVENANCE_VERIFIERS:
        raise C2ReviewedRecoveryNeedIntegrationError(
            "recovery-need runtime verifier is not registered"
        )
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
            "attestation_digest": ATTESTATION_DIGEST,
            "candidate_digest": candidate.candidate_digest,
            "local_verification_trace_digest": LOCAL_VERIFICATION_TRACE_DIGEST,
            "public_review_digest": PUBLIC_REVIEW_DIGEST,
            "review": "repository_reviewed_non_ci_phone_recovery_need_launch",
            "verifier_id": RUNTIME_VERIFIER_ID,
            "verifier_version": VERIFIER_VERSION,
        },
        "c2_recovery_need_runtime_provenance_verifier_trace",
    )
    return RecoveryNeedRuntimeVerification(
        candidate.candidate_digest,
        trace,
        attestation.logical_tick,
        _issuance_token=_RUNTIME_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class RecoveryNeedSourceVerification:
    source_integrity_digest: str
    raw_observation_digest: str
    runtime_provenance_verification_digest: str
    verifier_trace_digest: str
    verified_logical_tick: int
    authority: str = SHADOW_AUTHORITY
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SOURCE_TOKEN:
            raise C2ReviewedRecoveryNeedIntegrationError(
                "source verification must be issued by registered recovery-need verifier"
            )
        for field in (
            "source_integrity_digest",
            "raw_observation_digest",
            "runtime_provenance_verification_digest",
            "verifier_trace_digest",
        ):
            _sha256(getattr(self, field), field)
        if self.authority != SHADOW_AUTHORITY:
            raise C2ReviewedRecoveryNeedIntegrationError(
                "recovery-need source verification must remain shadow_only"
            )

    @property
    def counts_as_real(self) -> bool:
        return True

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "axis": AXIS,
            "fixture_only": False,
            "observation_evidence_digest": EVIDENCE_DIGEST,
            "production_origin_verified": True,
            "raw_observation_digest": self.raw_observation_digest,
            "runtime_capture_verified": True,
            "runtime_provenance_verification_digest": self.runtime_provenance_verification_digest,
            "source_contract_id": SOURCE_CONTRACT_ID,
            "source_family": SOURCE_FAMILY,
            "source_instance_id": SOURCE_INSTANCE_ID,
            "source_integrity_digest": self.source_integrity_digest,
            "synthetic": False,
            "verified_logical_tick": self.verified_logical_tick,
            "verifier_id": SOURCE_VERIFIER_ID,
            "verifier_trace_digest": self.verifier_trace_digest,
            "verifier_version": VERIFIER_VERSION,
        }

    @property
    def verification_digest(self) -> str:
        return _digest(
            self.to_mapping(),
            "c2_recovery_need_production_source_verification",
        )


def verify_recovery_need_source(
    public_review: Mapping[str, Any],
    runtime: RecoveryNeedRuntimeVerification | None = None,
) -> RecoveryNeedSourceVerification:
    _, evidence = verify_public_review(public_review)
    runtime = runtime or verify_runtime_provenance(public_review)
    if type(runtime) is not RecoveryNeedRuntimeVerification or not runtime.counts_as_production:
        raise C2ReviewedRecoveryNeedIntegrationError(
            "recovery-need source verifier requires issued runtime provenance"
        )
    if SOURCE_CONTRACT_ID not in C2_RECOVERY_PRODUCTION_SOURCE_VERIFIERS:
        raise C2ReviewedRecoveryNeedIntegrationError(
            "recovery-need production verifier is not registered"
        )
    trace = _digest(
        {
            "evidence_digest": EVIDENCE_DIGEST,
            "public_review_digest": PUBLIC_REVIEW_DIGEST,
            "runtime_provenance_verification_digest": runtime.verification_digest,
            "source_contract_id": SOURCE_CONTRACT_ID,
            "source_integrity_digest": evidence.source_integrity_digest,
            "verifier_id": SOURCE_VERIFIER_ID,
            "verifier_version": VERIFIER_VERSION,
        },
        "c2_recovery_need_source_verifier_trace",
    )
    return RecoveryNeedSourceVerification(
        evidence.source_integrity_digest,
        evidence.raw_observation_digest,
        runtime.verification_digest,
        trace,
        evidence.observed_tick,
        _issuance_token=_SOURCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class C2ReviewedRecoveryNeedCapture:
    capture_id: str
    capture_tick: int
    evidence: RegistryAxisPositiveConfidenceEvidence
    runtime_verification: RecoveryNeedRuntimeVerification
    source_verification: RecoveryNeedSourceVerification
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
            raise C2ReviewedRecoveryNeedIntegrationError(
                "recovery-need capture must be issued by reviewed integration"
            )
        if (
            self.evidence.evidence_digest != EVIDENCE_DIGEST
            or self.runtime_verification.verification_digest
            != self.source_verification.runtime_provenance_verification_digest
            or not self.runtime_verification.counts_as_production
            or not self.source_verification.counts_as_real
            or not self.retained_real_observation_eligible
            or self.authority != SHADOW_AUTHORITY
        ):
            raise C2ReviewedRecoveryNeedIntegrationError(
                "recovery-need capture binding is incomplete"
            )
        if any(
            (
                self.retained_real_observation,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise C2ReviewedRecoveryNeedIntegrationError(
                "pre-retention recovery-need capture cannot claim later authority"
            )

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
        return _digest(
            self.to_mapping(),
            "c2_reviewed_recovery_need_production_capture",
        )


def build_reviewed_capture(
    public_review: Mapping[str, Any],
) -> C2ReviewedRecoveryNeedCapture:
    _, evidence = verify_public_review(public_review)
    runtime = verify_runtime_provenance(public_review)
    source = verify_recovery_need_source(public_review, runtime)
    return C2ReviewedRecoveryNeedCapture(
        capture_id=f"c2:recovery-need:{evidence.observation_id}",
        capture_tick=evidence.observed_tick,
        evidence=evidence,
        runtime_verification=runtime,
        source_verification=source,
        _issuance_token=_CAPTURE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class C2RecoveryNeedIntegrationStatus:
    reviewed_real_operator_attestation_count: int = 4
    registered_runtime_provenance_verifier_count: int = 4
    verified_production_runtime_anchor_count: int = 4
    registered_production_source_verifier_count: int = 4
    verified_positive_confidence_candidate_count: int = 4
    retained_real_observation_count: int = 3
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False


def integration_status(
    public_review: Mapping[str, Any],
) -> C2RecoveryNeedIntegrationStatus:
    runtime = verify_runtime_provenance(public_review)
    source = verify_recovery_need_source(public_review, runtime)
    if not runtime.counts_as_production or not source.counts_as_real:
        raise C2ReviewedRecoveryNeedIntegrationError(
            "recovery-need reviewed status requires exact issued verifications"
        )
    return C2RecoveryNeedIntegrationStatus()
