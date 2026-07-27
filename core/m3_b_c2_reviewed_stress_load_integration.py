"""Reviewed C2 integration for the exact real phone ``stress_load`` witness.

This pins the operator-reviewed public witness produced from merged PR #211 and
creates immutable runtime/source verification plus a retention-eligible capture.
The witness provenance is explicitly two-stage: operator-private real runtime
metrics feed a deterministic appraisal bridge, while the canonical stress-load
record is a detached verified appraisal trace. Importing this module performs no
runtime I/O, persistence append, observation-window start, or authority promotion.
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

C2_SCHEMA_VERSION = "eve.m3-b.c2-reviewed-stress-load-integration.v1"
PUBLIC_REVIEW_SCHEMA_VERSION = "eve.m3-b.phone-stress-load-public-review.v1"
APPRAISAL_POLICY_VERSION = "eve.m3-b.phone-stress-load-appraisal-policy.v1"
APPRAISAL_VERSION = "eve.m3-b.survival-appraisal-trace.v1"
ATTESTATION_DIGEST = "7191e3493c582a191db3dcd488b2452dd3b0f29774b8a3a3ffeaff3b53c525fa"
PUBLIC_REVIEW_DIGEST = "1ec63bb54cfed398b0e5b93af25667474c3255d5ca50a47602974d363cf5e03a"
LOCAL_VERIFICATION_TRACE_DIGEST = "fba9d47b1cddf774a779b24f3b14b0c1bd970e0f73668fb2faaf9127a720d236"
EVIDENCE_DIGEST = "5bceb97155a5614de72b2b359b861db5c57eb6e892c259b56981d6003fc14680"
PRIVATE_MATERIAL_DIGEST = "c7911e841d12d7fc9449bf93b9788072fccdb00f6c205cacc36e029b5d845c24"
PINNED_WITNESS_HEAD = "3298d3b9911c79b1551a1d8bfe83bae756880840"
TRUST_DOMAIN = "eve.operator-attestation.primary.v1"
OPERATOR_ID = "primary-operator"
RUNTIME_INSTANCE_ID = "runtime:phone:primary:3298d3b9911c"
SOURCE_INSTANCE_ID = "runtime:phone-appraised-stress:primary"
ENTRYPOINT_ID = "scripts/operator/m3_b_phone_stress_load_witness.py:main"
LAUNCH_ATTESTATION_ID = "operator-attestation:phone:3298d3b9911c"
SOURCE_CONTRACT_ID = "eve:m3-b:registry-source:stress_load:v1"
SOURCE_FAMILY = "operational_metrics_or_appraised_load_trace"
AXIS = "stress_load"
RUNTIME_VERIFIER_ID = "eve.m3-b.c2.stress-load-runtime-verifier"
SOURCE_VERIFIER_ID = "eve.m3-b.c2.stress-load-production-verifier"
VERIFIER_VERSION = "v1"
PROCESS_CPU_METHODS = ("os_times_process_cpu_v1",)
QUEUE_METHODS = ("kernel_loadavg_1m_visible_cpu_ratio_v1",)
CONTROLLABILITY_METHODS = ("one_minus_mean_overload_and_queue_variability_v1",)
DEMAND_METHODS = ("mean_process_cpu_and_queue_ratio_v1",)
OVERLOAD_METHODS = ("max_process_cpu_and_queue_ratio_v1",)
UNCERTAINTY_METHODS = ("absolute_queue_ratio_delta_v1",)
APPRAISAL_INPUT_DIGESTS = (
    "d17acc92e3da94102293a0536f613cc82602e33c6467c78523118f5a7c7ee087",
    "0202672125629d533b35c67feced0f710e10e09f6c5e004353594c7a2bca3e63",
    "edb09c072e5104a3aa8fd734a33ce2106bf216b46b8bad8cd01ca190505bea33",
)
APPRAISAL_INTEGRITY_DIGESTS = (
    "3f11cddfc19fbf324dc2de1520e1ecd3172fd26057bb5b79d7c67f0cf76a0913",
    "cc14dbee92045eb8da1a48a157be32a63fe9badc73415733cf23eaa164809d6d",
    "9a13a8633021754a7c6b87bb97585d7e80b38d65b85780ad6bb68d23e1921ab5",
)
SNAPSHOT_INTEGRITY_DIGESTS = (
    "ca1a279c85058fce1cd45b528523c3e91bc2171dfd1ebaa61eee6f05b106e7d7",
    "78af39996508122f95e2450dab089e7bfefbbb0c650136617fc49eca57e505ac",
    "be3183b953b14a6ce16749ffdee4fdb393bef293027cd0bce976d0e4c010ebf3",
)
PROVENANCE_BOUNDARY = MappingProxyType(
    {
        "appraisal_bridge_output_detached": True,
        "appraisal_output_kind": "detached_verified_appraisal_trace",
        "canonical_appraised_record_hardware_direct_input": False,
        "canonical_appraised_record_runtime_polled": False,
        "raw_runtime_metrics_publicly_retained": False,
        "runtime_input_kind": "operator_private_real_runtime_metrics",
        "runtime_metrics_used_as_appraisal_input": True,
    }
)
PRIVATE_NONCE_COMMITMENT_DIGEST = "66b676d5d951629a0dbb9348b63ae4e5710bae182abdf1a0587797d1b31d786d"
NONCE_BINDING_DIGEST = "68ebced55adeab3ebb4549cc72a11c611f73d7a667c30954817f116093ef2663"
ZERO_DIGEST = "0" * 64
_RUNTIME_TOKEN = object()
_SOURCE_TOKEN = object()
_CAPTURE_TOKEN = object()


class C2ReviewedStressLoadIntegrationError(ValueError):
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
        raise C2ReviewedStressLoadIntegrationError(
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
        raise C2ReviewedStressLoadIntegrationError(
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
                or entry.appraisal_required is not True
                or entry.hardware_direct_input_allowed is not False
            ):
                raise C2ReviewedStressLoadIntegrationError(
                    "stress-load source contract drift"
                )
            return entry
    raise C2ReviewedStressLoadIntegrationError(
        "stress-load axis missing from source manifest"
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
            raise C2ReviewedStressLoadIntegrationError(
                "review registration must remain shadow_only"
            )


C2_STRESS_LOAD_REVIEWED_OPERATOR_ATTESTATIONS = MappingProxyType(
    {
        ATTESTATION_DIGEST: ReviewedRegistration(
            ATTESTATION_DIGEST,
            PUBLIC_REVIEW_DIGEST,
            LOCAL_VERIFICATION_TRACE_DIGEST,
        )
    }
)
C2_STRESS_LOAD_RUNTIME_PROVENANCE_VERIFIERS = MappingProxyType(
    {TRUST_DOMAIN: (RUNTIME_VERIFIER_ID, VERIFIER_VERSION)}
)
C2_STRESS_LOAD_PRODUCTION_SOURCE_VERIFIERS = MappingProxyType(
    {SOURCE_CONTRACT_ID: (SOURCE_VERIFIER_ID, VERIFIER_VERSION)}
)


def verify_public_review(
    value: Mapping[str, Any],
) -> tuple[OperatorPublicLaunchAttestation, RegistryAxisPositiveConfidenceEvidence]:
    if not isinstance(value, Mapping):
        raise C2ReviewedStressLoadIntegrationError("public review must be a mapping")
    review = dict(value)
    if review.get("public_review_digest") != PUBLIC_REVIEW_DIGEST:
        raise C2ReviewedStressLoadIntegrationError(
            "public review digest is not the reviewed digest"
        )
    material = dict(review)
    material.pop("public_review_digest", None)
    if _digest(material, "phone_stress_load_public_review") != PUBLIC_REVIEW_DIGEST:
        raise C2ReviewedStressLoadIntegrationError(
            "public review canonical digest mismatch"
        )
    if (
        review.get("schema_version") != PUBLIC_REVIEW_SCHEMA_VERSION
        or review.get("appraisal_policy_version") != APPRAISAL_POLICY_VERSION
        or review.get("appraisal_version") != APPRAISAL_VERSION
        or review.get("authority") != SHADOW_AUTHORITY
        or review.get("axis") != AXIS
        or review.get("source_instance_id") != SOURCE_INSTANCE_ID
        or review.get("fixture_only") is not False
        or review.get("raw_record_count") != 3
        or review.get("private_material_digest") != PRIVATE_MATERIAL_DIGEST
        or review.get("private_raw_location") != "operator_private_companion_only"
        or tuple(review.get("process_cpu_measurement_methods", ())) != PROCESS_CPU_METHODS
        or tuple(review.get("queue_measurement_methods", ())) != QUEUE_METHODS
        or tuple(review.get("controllability_methods", ())) != CONTROLLABILITY_METHODS
        or tuple(review.get("demand_methods", ())) != DEMAND_METHODS
        or tuple(review.get("overload_methods", ())) != OVERLOAD_METHODS
        or tuple(review.get("uncertainty_methods", ())) != UNCERTAINTY_METHODS
        or tuple(review.get("appraisal_input_digests", ())) != APPRAISAL_INPUT_DIGESTS
        or tuple(review.get("appraisal_integrity_digests", ())) != APPRAISAL_INTEGRITY_DIGESTS
        or tuple(review.get("snapshot_integrity_digests", ())) != SNAPSHOT_INTEGRITY_DIGESTS
        or dict(review.get("provenance_boundary", {})) != dict(PROVENANCE_BOUNDARY)
    ):
        raise C2ReviewedStressLoadIntegrationError(
            "public review envelope does not match reviewed stress-load contract"
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
            raise C2ReviewedStressLoadIntegrationError(
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
        raise C2ReviewedStressLoadIntegrationError(
            "public attestation does not match reviewed stress-load registration"
        )
    if attestation.attestation_digest not in C2_STRESS_LOAD_REVIEWED_OPERATOR_ATTESTATIONS:
        raise C2ReviewedStressLoadIntegrationError(
            "attestation is not repository-reviewed for stress_load"
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
        raise C2ReviewedStressLoadIntegrationError(
            "local verification summary does not match reviewed stress-load registration"
        )

    mapping = review.get("evidence")
    if not isinstance(mapping, Mapping):
        raise C2ReviewedStressLoadIntegrationError(
            "public review lacks positive-confidence evidence"
        )
    try:
        evidence = RegistryAxisPositiveConfidenceEvidence(**dict(mapping))
    except (TypeError, ValueError) as exc:
        raise C2ReviewedStressLoadIntegrationError(
            "public review evidence fails exact evidence schema"
        ) from exc
    if (
        evidence.axis != AXIS
        or evidence.source_family != SOURCE_FAMILY
        or evidence.source_instance_id != SOURCE_INSTANCE_ID
        or evidence.evidence_digest != EVIDENCE_DIGEST
        or review.get("evidence_digest") != EVIDENCE_DIGEST
        or review.get("evidence_observed_tick") != evidence.observed_tick
        or evidence.source_schema_version != "eve.m3-b.appraised-survival-raw-record.v1"
        or evidence.synthetic
        or evidence.proposal_only
        or evidence.verification_status != "verified"
    ):
        raise C2ReviewedStressLoadIntegrationError(
            "public evidence does not match reviewed stress-load pin"
        )
    entry = _source_entry()
    if evidence.source_family != entry.source_family:
        raise C2ReviewedStressLoadIntegrationError(
            "evidence source family does not match stress-load manifest"
        )
    return attestation, evidence


@dataclass(frozen=True, slots=True)
class StressLoadRuntimeVerification:
    candidate_digest: str
    verifier_trace_digest: str
    verified_logical_tick: int
    authority: str = SHADOW_AUTHORITY
    fixture_only: bool = False
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RUNTIME_TOKEN:
            raise C2ReviewedStressLoadIntegrationError(
                "runtime provenance must be issued by reviewed stress-load verifier"
            )
        _sha256(self.candidate_digest, "candidate_digest")
        _sha256(self.verifier_trace_digest, "verifier_trace_digest")
        if self.authority != SHADOW_AUTHORITY or self.fixture_only:
            raise C2ReviewedStressLoadIntegrationError(
                "stress-load runtime verification cannot weaken production proof"
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
            "c2_stress_load_runtime_provenance_verification",
        )


def verify_runtime_provenance(
    public_review: Mapping[str, Any],
) -> StressLoadRuntimeVerification:
    attestation, _ = verify_public_review(public_review)
    if attestation.trust_domain not in C2_STRESS_LOAD_RUNTIME_PROVENANCE_VERIFIERS:
        raise C2ReviewedStressLoadIntegrationError(
            "stress-load runtime verifier is not registered"
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
            "review": "repository_reviewed_non_ci_phone_stress_load_appraised_launch",
            "verifier_id": RUNTIME_VERIFIER_ID,
            "verifier_version": VERIFIER_VERSION,
        },
        "c2_stress_load_runtime_provenance_verifier_trace",
    )
    return StressLoadRuntimeVerification(
        candidate.candidate_digest,
        trace,
        attestation.logical_tick,
        _issuance_token=_RUNTIME_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class StressLoadSourceVerification:
    source_integrity_digest: str
    raw_observation_digest: str
    runtime_provenance_verification_digest: str
    verifier_trace_digest: str
    verified_logical_tick: int
    authority: str = SHADOW_AUTHORITY
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SOURCE_TOKEN:
            raise C2ReviewedStressLoadIntegrationError(
                "source verification must be issued by registered stress-load verifier"
            )
        for field in (
            "source_integrity_digest",
            "raw_observation_digest",
            "runtime_provenance_verification_digest",
            "verifier_trace_digest",
        ):
            _sha256(getattr(self, field), field)
        if self.authority != SHADOW_AUTHORITY:
            raise C2ReviewedStressLoadIntegrationError(
                "stress-load source verification must remain shadow_only"
            )

    @property
    def counts_as_real(self) -> bool:
        return True

    def to_mapping(self) -> dict[str, Any]:
        return {
            "appraisal_bridge_output_detached": True,
            "authority": self.authority,
            "axis": AXIS,
            "fixture_only": False,
            "observation_evidence_digest": EVIDENCE_DIGEST,
            "production_origin_verified": True,
            "raw_observation_digest": self.raw_observation_digest,
            "runtime_capture_verified": True,
            "runtime_metrics_used_as_appraisal_input": True,
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
            "c2_stress_load_production_source_verification",
        )


def verify_stress_load_source(
    public_review: Mapping[str, Any],
    runtime: StressLoadRuntimeVerification | None = None,
) -> StressLoadSourceVerification:
    _, evidence = verify_public_review(public_review)
    runtime = runtime or verify_runtime_provenance(public_review)
    if type(runtime) is not StressLoadRuntimeVerification or not runtime.counts_as_production:
        raise C2ReviewedStressLoadIntegrationError(
            "stress-load source verifier requires issued runtime provenance"
        )
    if SOURCE_CONTRACT_ID not in C2_STRESS_LOAD_PRODUCTION_SOURCE_VERIFIERS:
        raise C2ReviewedStressLoadIntegrationError(
            "stress-load production verifier is not registered"
        )
    trace = _digest(
        {
            "appraisal_policy_version": APPRAISAL_POLICY_VERSION,
            "evidence_digest": EVIDENCE_DIGEST,
            "public_review_digest": PUBLIC_REVIEW_DIGEST,
            "runtime_provenance_verification_digest": runtime.verification_digest,
            "source_contract_id": SOURCE_CONTRACT_ID,
            "source_integrity_digest": evidence.source_integrity_digest,
            "verifier_id": SOURCE_VERIFIER_ID,
            "verifier_version": VERIFIER_VERSION,
        },
        "c2_stress_load_source_verifier_trace",
    )
    return StressLoadSourceVerification(
        evidence.source_integrity_digest,
        evidence.raw_observation_digest,
        runtime.verification_digest,
        trace,
        evidence.observed_tick,
        _issuance_token=_SOURCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class C2ReviewedStressLoadCapture:
    capture_id: str
    capture_tick: int
    evidence: RegistryAxisPositiveConfidenceEvidence
    runtime_verification: StressLoadRuntimeVerification
    source_verification: StressLoadSourceVerification
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
            raise C2ReviewedStressLoadIntegrationError(
                "stress-load capture must be issued by reviewed integration"
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
            raise C2ReviewedStressLoadIntegrationError(
                "stress-load capture binding is incomplete"
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
            raise C2ReviewedStressLoadIntegrationError(
                "pre-retention stress-load capture cannot claim later authority"
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
            "c2_reviewed_stress_load_production_capture",
        )


def build_reviewed_capture(
    public_review: Mapping[str, Any],
) -> C2ReviewedStressLoadCapture:
    _, evidence = verify_public_review(public_review)
    runtime = verify_runtime_provenance(public_review)
    source = verify_stress_load_source(public_review, runtime)
    return C2ReviewedStressLoadCapture(
        capture_id=f"c2:stress-load:{evidence.observation_id}",
        capture_tick=evidence.observed_tick,
        evidence=evidence,
        runtime_verification=runtime,
        source_verification=source,
        _issuance_token=_CAPTURE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class C2StressLoadIntegrationStatus:
    reviewed_real_operator_attestation_count: int = 5
    registered_runtime_provenance_verifier_count: int = 5
    verified_production_runtime_anchor_count: int = 5
    registered_production_source_verifier_count: int = 5
    verified_positive_confidence_candidate_count: int = 5
    retained_real_observation_count: int = 4
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False


def integration_status(
    public_review: Mapping[str, Any],
) -> C2StressLoadIntegrationStatus:
    runtime = verify_runtime_provenance(public_review)
    source = verify_stress_load_source(public_review, runtime)
    if not runtime.counts_as_production or not source.counts_as_real:
        raise C2ReviewedStressLoadIntegrationError(
            "stress-load reviewed status requires exact issued verifications"
        )
    return C2StressLoadIntegrationStatus()
