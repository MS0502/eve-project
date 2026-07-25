"""Explicit M3-B production-origin verification boundary for registry observations.

The adapter is capability machinery only. Importing or constructing it performs no
runtime polling, source discovery, persistence, event append, scheduling, owner
mutation, observation-window transition, or authority promotion. A caller must
supply already-derived positive-confidence evidence plus an exact immutable
production-source verification created by a separately reviewed source bridge.

No source bridge is registered by this module. Therefore the presence of this
adapter does not itself prove that any real production observation exists.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_registry_observation_evidence import (
    RegistryAxisPositiveConfidenceEvidence,
)
from core.m3_b_registry_observation_source_manifest import (
    RegistryObservationSourceEntry,
    registry_observation_source_manifest,
)

VERIFICATION_SCHEMA_VERSION = "eve.m3-b.registry-production-source-verification.v1"
CAPTURE_RECORD_SCHEMA_VERSION = "eve.m3-b.registry-production-capture-record.v1"
CAPABILITY_SCHEMA_VERSION = "eve.m3-b.registry-production-capture-capability.v1"
ADAPTER_VERSION = "eve.m3-b.registry-production-capture-adapter.v1"
VERIFICATION_ENVIRONMENT = "production"
PRODUCTION_SOURCE_VERIFIER_BLOCKER = "REGISTRY_PRODUCTION_SOURCE_VERIFIER_COVERAGE_INCOMPLETE"
POSITIVE_CONFIDENCE_COVERAGE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
OBSERVATION_WINDOW_NOT_STARTED_BLOCKER = "REGISTRY_OBSERVATION_WINDOW_NOT_STARTED"
ZERO_DIGEST = "0" * 64
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


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
class ProductionSourceVerification:
    """Immutable proof supplied by a separately reviewed production source bridge.

    This object cannot be inferred from registry state or from the derived value
    alone. Every source identity/digest must match the evidence exactly. Tests and
    synthetic callers must set ``fixture_only=True`` and are rejected by capture.
    """

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

    def __post_init__(self) -> None:
        entry = _manifest_entry(self.axis)
        for field in (
            "source_contract_id",
            "source_family",
            "source_instance_id",
            "source_snapshot_id",
            "source_schema_version",
            "verifier_id",
            "verifier_version",
        ):
            _identifier(getattr(self, field), field)
        for field in (
            "source_integrity_digest",
            "raw_observation_digest",
            "observation_evidence_digest",
            "verifier_trace_digest",
        ):
            _digest_string(getattr(self, field), field)
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
        if self.registered_production_source_verifier_count != 0:
            raise RegistryProductionCaptureError(
                "this module registers no production source verifier"
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
    """Pure explicit converter from verified production evidence to capture record."""

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
    return ProductionCaptureCapabilityStatus()
