"""Reviewed C2 integration for the exact non-CI phone energy-budget witness.

This is the second reviewed real-observation activation layer.  It pins the
operator-reviewed Android phone witness produced after PR #202 and installs
immutable, source-contract-specific verification machinery without modifying
historical C1/C2 preflight registries.

Importing this module performs no runtime I/O, durable append, registry-owner
mutation, observation-window transition, cutover, or M3-E authority promotion.
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

C2_SCHEMA_VERSION = "eve.m3-b.c2-reviewed-energy-budget-integration.v1"
PUBLIC_REVIEW_SCHEMA_VERSION = "eve.m3-b.phone-energy-budget-public-review.v2"
MEASUREMENT_POLICY_VERSION = "eve.m3-b.phone-energy-budget-measurement-policy.v2"
ATTESTATION_DIGEST = "5413c35e912f95d90a1c0a5b0b8731a243bffc00e7b6338c1b7d9e4056e1c07f"
PUBLIC_REVIEW_DIGEST = "a2ce3d84111224e2009bf22d1e03a8f92acab0506e42515aac185ae05ff54ab4"
LOCAL_VERIFICATION_TRACE_DIGEST = "2aed3bffd9a36b2a3db9a9c7d7ecbeb4c0752485eeaaf7bb06df40ce25a40275"
EVIDENCE_DIGEST = "9d814295e3b59fb42294f3ba661866aa29c512866b946e80a3f397864974af13"
PRIVATE_MATERIAL_DIGEST = "0778662a1d06f052c8a7d755ba3fc64a7f9d6e251615a0f3df98c1e597b376d6"
PINNED_WITNESS_HEAD = "1161bb15d7bba0629d4862c05e8a61126cdb12c0"
TRUST_DOMAIN = "eve.operator-attestation.primary.v1"
OPERATOR_ID = "primary-operator"
RUNTIME_INSTANCE_ID = "runtime:phone:primary:1161bb15d7bb"
SOURCE_INSTANCE_ID = "runtime:phone-operational-energy:primary"
ENTRYPOINT_ID = "scripts/operator/m3_b_phone_energy_budget_witness_v2.py:main"
LAUNCH_ATTESTATION_ID = "operator-attestation:phone:1161bb15d7bb"
SOURCE_CONTRACT_ID = "eve:m3-b:registry-source:energy_budget:v1"
SOURCE_FAMILY = "operational_metrics_or_appraised_load_trace"
AXIS = "energy_budget"
RUNTIME_VERIFIER_ID = "eve.m3-b.c2.energy-budget-runtime-verifier"
RUNTIME_VERIFIER_VERSION = "v1"
SOURCE_VERIFIER_ID = "eve.m3-b.c2.energy-budget-production-verifier"
SOURCE_VERIFIER_VERSION = "v1"
CPU_METHODS = ("kernel_loadavg_1m_headroom_v1",)
MEMORY_METHODS = ("proc_meminfo_available_v1",)
BATTERY_METHODS = ("termux_api_battery_status_v1",)
SNAPSHOT_INTEGRITY_DIGESTS = (
    "a4ac657f73615597d434b73846ed1fad6839c05e7ef7c7f96b1b5df21a9300cc",
    "00ff0a6085ba5a3d7d5a4bec00ff836ce9b7dfea7780c5ebd7678a5a982a85e8",
    "e93e29ff77edb9a3a3a50a86e98c0913c2add7965c054262ea81b61b2c6ddd4d",
)
ZERO_DIGEST = "0" * 64
_RUNTIME_TOKEN = object()
_SOURCE_TOKEN = object()
_CAPTURE_TOKEN = object()


class C2ReviewedEnergyBudgetIntegrationError(ValueError):
    """Raised when material deviates from the reviewed energy-budget witness."""


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
        raise C2ReviewedEnergyBudgetIntegrationError(
            f"{field} is not canonical JSON material"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == ZERO_DIGEST
    ):
        raise C2ReviewedEnergyBudgetIntegrationError(
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
                raise C2ReviewedEnergyBudgetIntegrationError(
                    "energy-budget source contract drift"
                )
            return entry
    raise C2ReviewedEnergyBudgetIntegrationError(
        "energy-budget axis missing from source manifest"
    )


@dataclass(frozen=True, slots=True)
class C2EnergyReviewedAttestationRegistration:
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
            raise C2ReviewedEnergyBudgetIntegrationError(
                "reviewed energy-budget attestation registration drift"
            )


@dataclass(frozen=True, slots=True)
class C2EnergyVerifierRegistration:
    binding_id: str
    verifier_id: str
    verifier_version: str
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        for value in (self.binding_id, self.verifier_id, self.verifier_version):
            if not isinstance(value, str) or not value.strip():
                raise C2ReviewedEnergyBudgetIntegrationError(
                    "verifier registration identifiers must be non-empty"
                )
        if self.authority != SHADOW_AUTHORITY:
            raise C2ReviewedEnergyBudgetIntegrationError(
                "energy-budget verifier registration must remain shadow_only"
            )


C2_ENERGY_REVIEWED_OPERATOR_ATTESTATIONS: Mapping[
    str, C2EnergyReviewedAttestationRegistration
] = MappingProxyType(
    {
        ATTESTATION_DIGEST: C2EnergyReviewedAttestationRegistration(
            attestation_digest=ATTESTATION_DIGEST,
            public_review_digest=PUBLIC_REVIEW_DIGEST,
            local_verification_trace_digest=LOCAL_VERIFICATION_TRACE_DIGEST,
            runtime_instance_id=RUNTIME_INSTANCE_ID,
            source_instance_id=SOURCE_INSTANCE_ID,
            repository_head_sha=PINNED_WITNESS_HEAD,
            entrypoint_id=ENTRYPOINT_ID,
            launch_attestation_id=LAUNCH_ATTESTATION_ID,
            private_nonce_commitment_digest="66b676d5d951629a0dbb9348b63ae4e5710bae182abdf1a0587797d1b31d786d",
            nonce_binding_digest="fcbb3ab7f4f1ec3510652569a1809663dad13b46084fcd1a89d056fc1bb2c2e8",
        )
    }
)
C2_ENERGY_RUNTIME_PROVENANCE_VERIFIERS: Mapping[
    str, C2EnergyVerifierRegistration
] = MappingProxyType(
    {
        TRUST_DOMAIN: C2EnergyVerifierRegistration(
            binding_id=TRUST_DOMAIN,
            verifier_id=RUNTIME_VERIFIER_ID,
            verifier_version=RUNTIME_VERIFIER_VERSION,
        )
    }
)
C2_ENERGY_PRODUCTION_SOURCE_VERIFIERS: Mapping[
    str, C2EnergyVerifierRegistration
] = MappingProxyType(
    {
        SOURCE_CONTRACT_ID: C2EnergyVerifierRegistration(
            binding_id=SOURCE_CONTRACT_ID,
            verifier_id=SOURCE_VERIFIER_ID,
            verifier_version=SOURCE_VERIFIER_VERSION,
        )
    }
)


def verify_public_review(
    value: Mapping[str, Any],
) -> tuple[OperatorPublicLaunchAttestation, RegistryAxisPositiveConfidenceEvidence]:
    """Validate the complete public-safe v2 witness against the reviewed pin."""

    if not isinstance(value, Mapping):
        raise C2ReviewedEnergyBudgetIntegrationError("public review must be a mapping")
    review = dict(value)
    if review.get("public_review_digest") != PUBLIC_REVIEW_DIGEST:
        raise C2ReviewedEnergyBudgetIntegrationError(
            "public review digest is not the reviewed digest"
        )
    material = dict(review)
    material.pop("public_review_digest", None)
    if _digest(material, "phone_energy_budget_public_review") != PUBLIC_REVIEW_DIGEST:
        raise C2ReviewedEnergyBudgetIntegrationError(
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
        or tuple(review.get("cpu_measurement_methods", ())) != CPU_METHODS
        or tuple(review.get("memory_measurement_methods", ())) != MEMORY_METHODS
        or tuple(review.get("battery_measurement_methods", ())) != BATTERY_METHODS
        or tuple(review.get("snapshot_integrity_digests", ()))
        != SNAPSHOT_INTEGRITY_DIGESTS
    ):
        raise C2ReviewedEnergyBudgetIntegrationError(
            "public review envelope does not match reviewed energy-budget contract"
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
            raise C2ReviewedEnergyBudgetIntegrationError(
                f"phone witness cannot pre-claim {field}"
            )

    attestation = OperatorPublicLaunchAttestation.from_mapping(
        review.get("attestation", {})
    )
    if attestation.attestation_digest != ATTESTATION_DIGEST:
        raise C2ReviewedEnergyBudgetIntegrationError(
            "public attestation digest mismatch"
        )
    registration = C2_ENERGY_REVIEWED_OPERATOR_ATTESTATIONS.get(
        attestation.attestation_digest
    )
    if type(registration) is not C2EnergyReviewedAttestationRegistration:
        raise C2ReviewedEnergyBudgetIntegrationError(
            "attestation is not repository-reviewed for energy_budget"
        )
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
        raise C2ReviewedEnergyBudgetIntegrationError(
            "public attestation does not match reviewed energy-budget registration"
        )

    local_review = review.get("attestation_local_review")
    if not isinstance(local_review, Mapping):
        raise C2ReviewedEnergyBudgetIntegrationError(
            "attestation local review is missing"
        )
    if (
        local_review.get("attestation_digest") != ATTESTATION_DIGEST
        or local_review.get("local_verification_trace_digest")
        != LOCAL_VERIFICATION_TRACE_DIGEST
        or local_review.get("fixture_only") is not False
        or local_review.get("repository_head_sha") != PINNED_WITNESS_HEAD
        or local_review.get("runtime_instance_id") != RUNTIME_INSTANCE_ID
        or local_review.get("source_instance_id") != SOURCE_INSTANCE_ID
        or local_review.get("trust_domain") != TRUST_DOMAIN
        or local_review.get("private_nonce_commitment_digest")
        != registration.private_nonce_commitment_digest
    ):
        raise C2ReviewedEnergyBudgetIntegrationError(
            "local verification summary does not match reviewed registration"
        )

    evidence_mapping = review.get("evidence")
    if not isinstance(evidence_mapping, Mapping):
        raise C2ReviewedEnergyBudgetIntegrationError(
            "public review lacks positive-confidence evidence"
        )
    try:
        evidence = RegistryAxisPositiveConfidenceEvidence(**dict(evidence_mapping))
    except (TypeError, ValueError) as exc:
        raise C2ReviewedEnergyBudgetIntegrationError(
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
        raise C2ReviewedEnergyBudgetIntegrationError(
            "public evidence does not match reviewed energy-budget pin"
        )
    entry = _source_entry()
    if evidence.source_family != entry.source_family:
        raise C2ReviewedEnergyBudgetIntegrationError(
            "evidence source family does not match energy-budget manifest"
        )
    return attestation, evidence


@dataclass(frozen=True, slots=True)
class C2EnergyRuntimeProvenanceVerification:
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
            raise C2ReviewedEnergyBudgetIntegrationError(
                "runtime provenance must be issued by reviewed energy-budget verifier"
            )
        for field in (
            "candidate_digest",
            "attestation_digest",
            "public_review_digest",
            "verifier_trace_digest",
        ):
            _sha256(getattr(self, field), field)
        if (
            self.attestation_digest != ATTESTATION_DIGEST
            or self.public_review_digest != PUBLIC_REVIEW_DIGEST
            or self.runtime_instance_id != RUNTIME_INSTANCE_ID
            or self.source_instance_id != SOURCE_INSTANCE_ID
            or self.repository_head_sha != PINNED_WITNESS_HEAD
            or self.fixture_only
            or not self.non_ci_runtime_verified
            or not self.production_launch_verified
            or not self.independent_trust_root_verified
            or self.authority != SHADOW_AUTHORITY
        ):
            raise C2ReviewedEnergyBudgetIntegrationError(
                "energy-budget runtime provenance cannot weaken reviewed production proof"
            )

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
        return _digest(
            self.to_mapping(), "c2_energy_budget_runtime_provenance_verification"
        )


def verify_runtime_provenance(
    public_review: Mapping[str, Any],
) -> C2EnergyRuntimeProvenanceVerification:
    attestation, _evidence = verify_public_review(public_review)
    registration = C2_ENERGY_RUNTIME_PROVENANCE_VERIFIERS.get(
        attestation.trust_domain
    )
    if type(registration) is not C2EnergyVerifierRegistration:
        raise C2ReviewedEnergyBudgetIntegrationError(
            "energy-budget runtime verifier is not registered"
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
            "attestation_digest": attestation.attestation_digest,
            "candidate_digest": candidate.candidate_digest,
            "local_verification_trace_digest": LOCAL_VERIFICATION_TRACE_DIGEST,
            "public_review_digest": PUBLIC_REVIEW_DIGEST,
            "review": "repository_reviewed_non_ci_phone_energy_budget_launch",
            "verifier_id": registration.verifier_id,
            "verifier_version": registration.verifier_version,
        },
        "c2_energy_budget_runtime_provenance_verifier_trace",
    )
    return C2EnergyRuntimeProvenanceVerification(
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
class C2EnergyProductionSourceVerification:
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
            raise C2ReviewedEnergyBudgetIntegrationError(
                "source verification must be issued by registered energy-budget verifier"
            )
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
            or self.source_family != SOURCE_FAMILY
            or self.source_instance_id != SOURCE_INSTANCE_ID
            or self.fixture_only
            or self.synthetic
            or not self.production_origin_verified
            or not self.runtime_capture_verified
            or self.authority != SHADOW_AUTHORITY
        ):
            raise C2ReviewedEnergyBudgetIntegrationError(
                "energy-budget source verification cannot weaken reviewed real-source proof"
            )

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
        return _digest(
            self.to_mapping(), "c2_energy_budget_production_source_verification"
        )


def verify_energy_budget_source(
    public_review: Mapping[str, Any],
    runtime_verification: C2EnergyRuntimeProvenanceVerification | None = None,
) -> C2EnergyProductionSourceVerification:
    _attestation, evidence = verify_public_review(public_review)
    runtime = runtime_verification or verify_runtime_provenance(public_review)
    if (
        type(runtime) is not C2EnergyRuntimeProvenanceVerification
        or not runtime.counts_as_production
    ):
        raise C2ReviewedEnergyBudgetIntegrationError(
            "energy-budget source verifier requires issued runtime provenance"
        )
    if (
        runtime.attestation_digest != ATTESTATION_DIGEST
        or runtime.public_review_digest != PUBLIC_REVIEW_DIGEST
        or runtime.source_instance_id != evidence.source_instance_id
        or runtime.repository_head_sha != PINNED_WITNESS_HEAD
    ):
        raise C2ReviewedEnergyBudgetIntegrationError(
            "runtime provenance does not bind reviewed energy-budget evidence"
        )
    entry = _source_entry()
    registration = C2_ENERGY_PRODUCTION_SOURCE_VERIFIERS.get(
        entry.source_contract_id
    )
    if type(registration) is not C2EnergyVerifierRegistration:
        raise C2ReviewedEnergyBudgetIntegrationError(
            "energy-budget production verifier is not registered"
        )
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
        "c2_energy_budget_source_verifier_trace",
    )
    return C2EnergyProductionSourceVerification(
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
class C2ReviewedEnergyBudgetCapture:
    capture_id: str
    capture_tick: int
    evidence: RegistryAxisPositiveConfidenceEvidence
    runtime_verification: C2EnergyRuntimeProvenanceVerification
    source_verification: C2EnergyProductionSourceVerification
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
            raise C2ReviewedEnergyBudgetIntegrationError(
                "energy-budget capture must be issued by reviewed integration"
            )
        if (
            self.evidence.evidence_digest != EVIDENCE_DIGEST
            or self.runtime_verification.verification_digest
            != self.source_verification.runtime_provenance_verification_digest
            or self.source_verification.observation_evidence_digest
            != self.evidence.evidence_digest
            or not self.runtime_verification.counts_as_production
            or not self.source_verification.counts_as_real
            or not self.retained_real_observation_eligible
            or self.authority != SHADOW_AUTHORITY
        ):
            raise C2ReviewedEnergyBudgetIntegrationError(
                "energy-budget capture binding is incomplete"
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
            raise C2ReviewedEnergyBudgetIntegrationError(
                "pre-retention energy-budget capture cannot claim later authority"
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
            self.to_mapping(), "c2_reviewed_energy_budget_production_capture"
        )


def build_reviewed_capture(
    public_review: Mapping[str, Any],
) -> C2ReviewedEnergyBudgetCapture:
    _attestation, evidence = verify_public_review(public_review)
    runtime = verify_runtime_provenance(public_review)
    source = verify_energy_budget_source(public_review, runtime)
    return C2ReviewedEnergyBudgetCapture(
        capture_id=f"c2:energy-budget:{evidence.observation_id}",
        capture_tick=evidence.observed_tick,
        evidence=evidence,
        runtime_verification=runtime,
        source_verification=source,
        _issuance_token=_CAPTURE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class C2EnergyIntegrationStatus:
    reviewed_real_operator_attestation_count: int = 2
    registered_runtime_provenance_verifier_count: int = 2
    verified_production_runtime_anchor_count: int = 2
    registered_production_source_verifier_count: int = 2
    verified_positive_confidence_candidate_count: int = 2
    retained_real_observation_count: int = 1
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
            self.reviewed_real_operator_attestation_count != 2
            or self.registered_runtime_provenance_verifier_count != 2
            or self.verified_production_runtime_anchor_count != 2
            or self.registered_production_source_verifier_count != 2
            or self.verified_positive_confidence_candidate_count != 2
            or self.retained_real_observation_count != 1
            or self.observation_window_eligible
            or self.observation_window_started
            or self.m3_b_complete
            or self.m3_c_open
            or self.m3_e_authority_open
            or self.cutover_authorized
            or self.authority != SHADOW_AUTHORITY
            or self.schema_version != C2_SCHEMA_VERSION
        ):
            raise C2ReviewedEnergyBudgetIntegrationError(
                "energy-budget C2 status cannot overstate reviewed repository boundary"
            )


def integration_status(public_review: Mapping[str, Any]) -> C2EnergyIntegrationStatus:
    runtime = verify_runtime_provenance(public_review)
    source = verify_energy_budget_source(public_review, runtime)
    if not runtime.counts_as_production or not source.counts_as_real:
        raise C2ReviewedEnergyBudgetIntegrationError(
            "energy-budget reviewed status requires exact issued verifications"
        )
    return C2EnergyIntegrationStatus()
