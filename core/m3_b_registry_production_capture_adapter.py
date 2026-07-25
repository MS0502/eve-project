"""Explicit M3-B production-origin verification boundary for registry observations.

The adapter is capability machinery only. Importing or constructing it performs no
runtime polling, source discovery, persistence, event append, scheduling, owner
mutation, observation-window transition, or authority promotion.

A production verification is accepted only when it was issued by executing the
registered source-contract-specific verifier over caller-supplied source material.
Callers cannot turn self-authored verification metadata into a real capture merely by
constructing ``ProductionSourceVerification``. No production verifier is registered
by this module; the closed registry remains intentionally empty until a later reviewed
runtime-source integration PR supplies an executable verifier.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_registry_observation_evidence import RegistryAxisPositiveConfidenceEvidence
from core.m3_b_registry_observation_source_manifest import (
    RegistryObservationSourceEntry,
    registry_observation_source_manifest,
)

VERIFICATION_SCHEMA_VERSION = "eve.m3-b.registry-production-source-verification.v1"
VERIFIER_RESULT_SCHEMA_VERSION = "eve.m3-b.registry-production-verifier-result.v1"
VERIFIER_REGISTRATION_SCHEMA_VERSION = "eve.m3-b.registry-production-verifier-registration.v1"
CAPTURE_RECORD_SCHEMA_VERSION = "eve.m3-b.registry-production-capture-record.v1"
CAPABILITY_SCHEMA_VERSION = "eve.m3-b.registry-production-capture-capability.v1"
ADAPTER_VERSION = "eve.m3-b.registry-production-capture-adapter.v1"
VERIFICATION_ENVIRONMENT = "production"
PRODUCTION_SOURCE_VERIFIER_BLOCKER = "REGISTRY_PRODUCTION_SOURCE_VERIFIER_COVERAGE_INCOMPLETE"
POSITIVE_CONFIDENCE_COVERAGE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
OBSERVATION_WINDOW_NOT_STARTED_BLOCKER = "REGISTRY_OBSERVATION_WINDOW_NOT_STARTED"
ZERO_DIGEST = "0" * 64
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_VERIFICATION_ISSUANCE_TOKEN = object()


class RegistryProductionCaptureError(ValueError):
    """Raised when production-origin evidence cannot be proven exactly."""


def _identifier(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise RegistryProductionCaptureError(f"{field} must be a bounded non-empty string")
    return value


def _digest_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None or value == ZERO_DIGEST:
        raise RegistryProductionCaptureError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RegistryProductionCaptureError(f"{field} must be a non-negative integer")
    return value


def _canonical(value: Mapping[str, Any], field: str) -> str:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RegistryProductionCaptureError(f"{field} is not canonical JSON") from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _manifest_entry(axis: str) -> RegistryObservationSourceEntry:
    for entry in registry_observation_source_manifest().entries:
        if entry.axis == axis:
            return entry
    raise RegistryProductionCaptureError("axis is absent from the registry source manifest")


@dataclass(frozen=True, slots=True)
class ProductionSourceVerifierResult:
    """Untrusted verifier output that must still be bound and issued by this module."""

    source_instance_id: str
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    raw_observation_digest: str
    observation_evidence_digest: str
    verifier_trace_digest: str
    verified_logical_tick: int
    verification_environment: str = VERIFICATION_ENVIRONMENT
    production_origin_verified: bool = True
    runtime_capture_verified: bool = True
    fixture_only: bool = False
    synthetic: bool = False
    proposal_only: bool = False
    registry_owner_source: bool = False
    schema_version: str = VERIFIER_RESULT_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        for field_name in (
            "source_instance_id",
            "source_snapshot_id",
            "source_schema_version",
        ):
            _identifier(getattr(self, field_name), field_name)
        for field_name in (
            "source_integrity_digest",
            "raw_observation_digest",
            "observation_evidence_digest",
            "verifier_trace_digest",
        ):
            _digest_string(getattr(self, field_name), field_name)
        _nonnegative_int(self.verified_logical_tick, "verified_logical_tick")
        if self.verification_environment not in {"production", "test_fixture"}:
            raise RegistryProductionCaptureError("unsupported verification environment")
        if self.verification_environment == "production" and self.fixture_only:
            raise RegistryProductionCaptureError(
                "production verification cannot simultaneously be a fixture"
            )
        if self.verification_environment == "test_fixture" and not self.fixture_only:
            raise RegistryProductionCaptureError(
                "test verification environment must remain fixture_only"
            )
        if not self.production_origin_verified or not self.runtime_capture_verified:
            raise RegistryProductionCaptureError(
                "verifier result requires explicit origin and runtime-capture proof"
            )
        if self.synthetic or self.proposal_only or self.registry_owner_source:
            raise RegistryProductionCaptureError(
                "synthetic, proposal-only, or registry-owner evidence cannot be verifier output"
            )
        if self.schema_version != VERIFIER_RESULT_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise RegistryProductionCaptureError(
                "production verifier result must remain exact shadow-only evidence"
            )


ProductionVerifierCallable = Callable[
    [RegistryAxisPositiveConfidenceEvidence, Mapping[str, Any]],
    ProductionSourceVerifierResult,
]


@dataclass(frozen=True, slots=True)
class ProductionSourceVerifierRegistration:
    """Reviewed executable verifier registration for exactly one source contract."""

    source_contract_id: str
    verifier_id: str
    verifier_version: str
    verifier: ProductionVerifierCallable = field(repr=False, compare=False)
    schema_version: str = VERIFIER_REGISTRATION_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        for field_name in ("source_contract_id", "verifier_id", "verifier_version"):
            _identifier(getattr(self, field_name), field_name)
        contracts = {
            entry.source_contract_id for entry in registry_observation_source_manifest().entries
        }
        if self.source_contract_id not in contracts:
            raise RegistryProductionCaptureError(
                "production verifier registration source contract is absent from manifest"
            )
        if not callable(self.verifier):
            raise RegistryProductionCaptureError("production verifier registration must be executable")
        if self.schema_version != VERIFIER_REGISTRATION_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise RegistryProductionCaptureError(
                "production verifier registration must remain exact shadow-only machinery"
            )


# Deliberately empty and immutable. Future source-integration PRs may change this
# constant only through reviewed repository code; runtime callers cannot inject a
# registration into the live mapping.
REGISTERED_PRODUCTION_SOURCE_VERIFIERS: Mapping[
    str, ProductionSourceVerifierRegistration
] = MappingProxyType({})


def _require_registered_verifier(verification: "ProductionSourceVerification") -> None:
    registration = REGISTERED_PRODUCTION_SOURCE_VERIFIERS.get(verification.source_contract_id)
    if (
        type(registration) is not ProductionSourceVerifierRegistration
        or registration.source_contract_id != verification.source_contract_id
        or registration.verifier_id != verification.verifier_id
        or registration.verifier_version != verification.verifier_version
    ):
        raise RegistryProductionCaptureError(
            "production verifier is not registered for this source contract"
        )


@dataclass(frozen=True, slots=True)
class ProductionSourceVerification:
    """Immutable production proof issued only by registered verifier execution."""

    axis: str
    source_contract_id: str
    source_family: str
    source_instance_id: str
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    raw_observation_digest: str
    observation_evidence_digest: str
    verifier_id: str
    verifier_version: str
    verifier_trace_digest: str
    verified_logical_tick: int
    verification_environment: str = VERIFICATION_ENVIRONMENT
    production_origin_verified: bool = True
    runtime_capture_verified: bool = True
    fixture_only: bool = False
    synthetic: bool = False
    proposal_only: bool = False
    registry_owner_source: bool = False
    schema_version: str = VERIFICATION_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _VERIFICATION_ISSUANCE_TOKEN:
            raise RegistryProductionCaptureError(
                "production verification must be issued by registered verifier execution"
            )
        entry = _manifest_entry(self.axis)
        for field_name in (
            "source_contract_id",
            "source_family",
            "source_instance_id",
            "source_snapshot_id",
            "source_schema_version",
            "verifier_id",
            "verifier_version",
        ):
            _identifier(getattr(self, field_name), field_name)
        for field_name in (
            "source_integrity_digest",
            "raw_observation_digest",
            "observation_evidence_digest",
            "verifier_trace_digest",
        ):
            _digest_string(getattr(self, field_name), field_name)
        _nonnegative_int(self.verified_logical_tick, "verified_logical_tick")
        if self.source_contract_id != entry.source_contract_id:
            raise RegistryProductionCaptureError(
                "production verification source contract does not match manifest"
            )
        if self.source_family != entry.source_family:
            raise RegistryProductionCaptureError(
                "production verification source family does not match manifest"
            )
        if self.verification_environment not in {"production", "test_fixture"}:
            raise RegistryProductionCaptureError("unsupported verification environment")
        if self.verification_environment == "production" and self.fixture_only:
            raise RegistryProductionCaptureError(
                "production verification cannot simultaneously be a fixture"
            )
        if self.verification_environment == "test_fixture" and not self.fixture_only:
            raise RegistryProductionCaptureError(
                "test verification environment must remain fixture_only"
            )
        if not self.production_origin_verified or not self.runtime_capture_verified:
            raise RegistryProductionCaptureError(
                "production verification requires explicit origin and runtime-capture proof"
            )
        if self.synthetic or self.proposal_only or self.registry_owner_source:
            raise RegistryProductionCaptureError(
                "synthetic, proposal-only, or registry-owner evidence cannot be production verification"
            )
        if self.schema_version != VERIFICATION_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise RegistryProductionCaptureError(
                "production verification must remain exact shadow-only evidence"
            )

    @property
    def counts_as_real(self) -> bool:
        return self.verification_environment == VERIFICATION_ENVIRONMENT and not self.fixture_only

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "axis": self.axis,
            "fixture_only": self.fixture_only,
            "observation_evidence_digest": self.observation_evidence_digest,
            "production_origin_verified": self.production_origin_verified,
            "proposal_only": self.proposal_only,
            "raw_observation_digest": self.raw_observation_digest,
            "registry_owner_source": self.registry_owner_source,
            "runtime_capture_verified": self.runtime_capture_verified,
            "schema_version": self.schema_version,
            "source_contract_id": self.source_contract_id,
            "source_family": self.source_family,
            "source_instance_id": self.source_instance_id,
            "source_integrity_digest": self.source_integrity_digest,
            "source_schema_version": self.source_schema_version,
            "source_snapshot_id": self.source_snapshot_id,
            "synthetic": self.synthetic,
            "verification_environment": self.verification_environment,
            "verified_logical_tick": self.verified_logical_tick,
            "verifier_id": self.verifier_id,
            "verifier_trace_digest": self.verifier_trace_digest,
            "verifier_version": self.verifier_version,
        }

    @property
    def verification_digest(self) -> str:
        return _digest(self.to_mapping(), "registry_production_source_verification")


def execute_registered_production_verifier(
    evidence: RegistryAxisPositiveConfidenceEvidence,
    source_material: Mapping[str, Any],
) -> ProductionSourceVerification:
    """Execute the exact registered verifier and issue a bound immutable proof."""

    if type(evidence) is not RegistryAxisPositiveConfidenceEvidence:
        raise RegistryProductionCaptureError(
            "registered verifier execution requires exact positive-confidence evidence"
        )
    if not isinstance(source_material, Mapping):
        raise RegistryProductionCaptureError("production verifier source material must be a mapping")
    entry = _manifest_entry(evidence.axis)
    registration = REGISTERED_PRODUCTION_SOURCE_VERIFIERS.get(entry.source_contract_id)
    if type(registration) is not ProductionSourceVerifierRegistration:
        raise RegistryProductionCaptureError(
            "production verifier is not registered for this source contract"
        )
    if registration.source_contract_id != entry.source_contract_id:
        raise RegistryProductionCaptureError(
            "registered verifier source contract does not match manifest"
        )
    result = registration.verifier(evidence, dict(source_material))
    if type(result) is not ProductionSourceVerifierResult:
        raise RegistryProductionCaptureError(
            "registered production verifier must return the exact immutable verifier result"
        )
    verification = ProductionSourceVerification(
        axis=evidence.axis,
        source_contract_id=entry.source_contract_id,
        source_family=entry.source_family,
        source_instance_id=result.source_instance_id,
        source_snapshot_id=result.source_snapshot_id,
        source_schema_version=result.source_schema_version,
        source_integrity_digest=result.source_integrity_digest,
        raw_observation_digest=result.raw_observation_digest,
        observation_evidence_digest=result.observation_evidence_digest,
        verifier_id=registration.verifier_id,
        verifier_version=registration.verifier_version,
        verifier_trace_digest=result.verifier_trace_digest,
        verified_logical_tick=result.verified_logical_tick,
        verification_environment=result.verification_environment,
        production_origin_verified=result.production_origin_verified,
        runtime_capture_verified=result.runtime_capture_verified,
        fixture_only=result.fixture_only,
        synthetic=result.synthetic,
        proposal_only=result.proposal_only,
        registry_owner_source=result.registry_owner_source,
        _issuance_token=_VERIFICATION_ISSUANCE_TOKEN,
    )
    expected = (
        evidence.source_family,
        evidence.source_instance_id,
        evidence.source_snapshot_id,
        evidence.source_schema_version,
        evidence.source_integrity_digest,
        evidence.raw_observation_digest,
        evidence.evidence_digest,
    )
    actual = (
        verification.source_family,
        verification.source_instance_id,
        verification.source_snapshot_id,
        verification.source_schema_version,
        verification.source_integrity_digest,
        verification.raw_observation_digest,
        verification.observation_evidence_digest,
    )
    if actual != expected:
        raise RegistryProductionCaptureError(
            "registered verifier output does not bind the exact observation evidence"
        )
    _require_registered_verifier(verification)
    return verification


@dataclass(frozen=True, slots=True)
class ProductionCaptureRecord:
    capture_id: str
    capture_tick: int
    evidence: RegistryAxisPositiveConfidenceEvidence
    verification: ProductionSourceVerification
    schema_version: str = CAPTURE_RECORD_SCHEMA_VERSION
    adapter_version: str = ADAPTER_VERSION
    authority: str = SHADOW_AUTHORITY
    retained_real_observation_eligible: bool = True
    persistence_accessed: bool = False
    event_append_performed: bool = False
    registry_owner_mutated: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        _identifier(self.capture_id, "capture_id")
        _nonnegative_int(self.capture_tick, "capture_tick")
        if type(self.evidence) is not RegistryAxisPositiveConfidenceEvidence:
            raise RegistryProductionCaptureError(
                "capture requires exact immutable positive-confidence evidence"
            )
        if type(self.verification) is not ProductionSourceVerification:
            raise RegistryProductionCaptureError(
                "capture requires exact immutable production verification"
            )
        evidence = self.evidence
        verification = self.verification
        entry = _manifest_entry(evidence.axis)
        expected = (
            evidence.axis,
            evidence.source_family,
            evidence.source_instance_id,
            evidence.source_snapshot_id,
            evidence.source_schema_version,
            evidence.source_integrity_digest,
            evidence.raw_observation_digest,
            evidence.evidence_digest,
        )
        actual = (
            verification.axis,
            verification.source_family,
            verification.source_instance_id,
            verification.source_snapshot_id,
            verification.source_schema_version,
            verification.source_integrity_digest,
            verification.raw_observation_digest,
            verification.observation_evidence_digest,
        )
        if actual != expected:
            raise RegistryProductionCaptureError(
                "production verification does not bind the exact observation evidence"
            )
        if verification.source_contract_id != entry.source_contract_id:
            raise RegistryProductionCaptureError("capture source contract does not match manifest")
        if verification.verified_logical_tick < evidence.observed_tick:
            raise RegistryProductionCaptureError(
                "production verification cannot precede the observed evidence tick"
            )
        if self.capture_tick < verification.verified_logical_tick:
            raise RegistryProductionCaptureError(
                "capture tick cannot precede production verification"
            )
        if not verification.counts_as_real:
            raise RegistryProductionCaptureError(
                "test fixtures cannot become retained-real-observation capture records"
            )
        _require_registered_verifier(verification)
        if self.schema_version != CAPTURE_RECORD_SCHEMA_VERSION:
            raise RegistryProductionCaptureError("unsupported production capture schema")
        if self.adapter_version != ADAPTER_VERSION or self.authority != SHADOW_AUTHORITY:
            raise RegistryProductionCaptureError(
                "production capture record must remain exact shadow-only machinery"
            )
        if self.retained_real_observation_eligible is not True:
            raise RegistryProductionCaptureError(
                "verified production capture record must be retention-eligible"
            )
        if any(
            (
                self.persistence_accessed,
                self.event_append_performed,
                self.registry_owner_mutated,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise RegistryProductionCaptureError(
                "capture adapter cannot claim persistence, mutation, window, or authority"
            )

    @property
    def axis(self) -> str:
        return self.evidence.axis

    def to_mapping(self) -> dict[str, Any]:
        return {
            "adapter_version": self.adapter_version,
            "authority": self.authority,
            "capture_id": self.capture_id,
            "capture_tick": self.capture_tick,
            "cutover_authorized": self.cutover_authorized,
            "event_append_performed": self.event_append_performed,
            "evidence": self.evidence.to_mapping(),
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "observation_window_started": self.observation_window_started,
            "persistence_accessed": self.persistence_accessed,
            "registry_owner_mutated": self.registry_owner_mutated,
            "retained_real_observation_eligible": self.retained_real_observation_eligible,
            "schema_version": self.schema_version,
            "verification": self.verification.to_mapping(),
        }

    @property
    def capture_digest(self) -> str:
        return _digest(self.to_mapping(), "registry_production_capture_record")


@dataclass(frozen=True, slots=True)
class ProductionCaptureCapabilityStatus:
    schema_version: str = CAPABILITY_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    production_capture_adapter_present: bool = True
    immutable_retention_sink_required: bool = True
    registered_production_source_verifier_count: int = 0
    retained_real_observation_count: int = 0
    positive_confidence_real_observation_count: int = 0
    observation_window_eligible: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise RegistryProductionCaptureError("unsupported production-capture capability status")
        if self.production_capture_adapter_present is not True or self.immutable_retention_sink_required is not True:
            raise RegistryProductionCaptureError("capture capability presence contract is incomplete")
        if self.registered_production_source_verifier_count != len(REGISTERED_PRODUCTION_SOURCE_VERIFIERS):
            raise RegistryProductionCaptureError(
                "capability verifier count disagrees with exact registration table"
            )
        if self.registered_production_source_verifier_count != 0:
            raise RegistryProductionCaptureError(
                "this capability-only module must not register production source verifiers"
            )
        if self.retained_real_observation_count != 0 or self.positive_confidence_real_observation_count != 0:
            raise RegistryProductionCaptureError(
                "capture machinery cannot fabricate retained real coverage"
            )
        if any(
            (
                self.observation_window_eligible,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise RegistryProductionCaptureError(
                "capture machinery cannot open window, completion, cutover, or authority"
            )

    @property
    def blockers(self) -> tuple[str, ...]:
        return (
            PRODUCTION_SOURCE_VERIFIER_BLOCKER,
            POSITIVE_CONFIDENCE_COVERAGE_BLOCKER,
            OBSERVATION_WINDOW_NOT_STARTED_BLOCKER,
        )


class RegistryProductionCaptureAdapter:
    """Pure explicit converter from registered verified production evidence."""

    def capture(
        self,
        evidence: RegistryAxisPositiveConfidenceEvidence,
        verification: ProductionSourceVerification,
        *,
        capture_id: str,
        capture_tick: int,
    ) -> ProductionCaptureRecord:
        return ProductionCaptureRecord(
            capture_id=capture_id,
            capture_tick=capture_tick,
            evidence=evidence,
            verification=verification,
        )


def production_capture_capability_status() -> ProductionCaptureCapabilityStatus:
    """Report code presence only; no production verifier or observation is implied."""
    return ProductionCaptureCapabilityStatus(
        registered_production_source_verifier_count=len(REGISTERED_PRODUCTION_SOURCE_VERIFIERS)
    )
