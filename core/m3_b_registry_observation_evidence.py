"""Detached positive-confidence observation evidence for all 37 registry axes.

This module accepts only explicit, recalculable, verified current-value
observations. Registry genesis/default/baseline values, proposal metadata, and
synthetic inputs cannot masquerade as observation evidence. The module returns
new immutable shadow-only owner state and installs no runtime hook, scheduler,
persistence route, event append path, observation window, M3-C, cutover, or
M3-E authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

from adapters.affect_hormone_neural_rhythm_registry import affect_hormone_axis_registry
from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_registry_affect_owner import (
    REGISTRY_AXIS_ORDER,
    SOURCE_FAMILY as REGISTRY_OWNER_SOURCE_FAMILY,
    RegistryAffectOwnerState,
    RegistryAxisCurrentState,
)

EVIDENCE_SCHEMA_VERSION = "eve.m3-b.registry-axis-positive-confidence-evidence.v1"
BUNDLE_SCHEMA_VERSION = "eve.m3-b.registry-37-axis-positive-confidence-bundle.v1"
VERIFIED_OBSERVATION_KIND = "verified_current_value_observation"
VERIFIED_STATUS = "verified"
BUNDLE_ACQUISITION_MODE = "explicit_detached_verified_observation_bundle"
MATERIALIZED_TRANSITION_KIND = "detached_verified_observation_bundle"
ZERO_DIGEST = "0" * 64


class RegistryObservationEvidenceError(ValueError):
    """Raised when registry observation evidence fails closed."""


def _identifier(value: str, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise RegistryObservationEvidenceError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RegistryObservationEvidenceError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RegistryObservationEvidenceError(f"{field} must be finite")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RegistryObservationEvidenceError(
            f"{field} must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result == 0:
        raise RegistryObservationEvidenceError(f"{field} must be positive")
    return result


def _digest_string(value: str, field: str, *, allow_zero: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RegistryObservationEvidenceError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    if not allow_zero and value == ZERO_DIGEST:
        raise RegistryObservationEvidenceError(f"{field} cannot be a placeholder digest")
    return value


def _canonical(value: Mapping[str, Any], field: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RegistryObservationEvidenceError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _definitions() -> dict[str, dict[str, Any]]:
    definitions = affect_hormone_axis_registry()
    if tuple(definitions) != REGISTRY_AXIS_ORDER:
        raise RegistryObservationEvidenceError(
            "registry definitions must preserve exact 37-axis order"
        )
    if len(definitions) != 37 or len(set(definitions)) != 37:
        raise RegistryObservationEvidenceError(
            "registry definitions must contain 37 unique axes"
        )
    return definitions


@dataclass(frozen=True, slots=True)
class RegistryAxisPositiveConfidenceEvidence:
    axis: str
    value: float
    confidence: float
    observed_tick: int
    observation_id: str
    source_family: str
    source_instance_id: str
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    raw_observation_digest: str
    acquisition_method: str
    verification_method: str
    model_or_rule_version: str
    observation_kind: str = VERIFIED_OBSERVATION_KIND
    verification_status: str = VERIFIED_STATUS
    genesis_derived: bool = False
    baseline_derived: bool = False
    default_derived: bool = False
    proposal_only: bool = False
    synthetic: bool = False
    recalculable_reference_present: bool = True
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        definitions = _definitions()
        if self.axis not in REGISTRY_AXIS_ORDER:
            raise RegistryObservationEvidenceError(
                "evidence contains an unknown registry axis"
            )
        value = _finite(self.value, f"{self.axis}.value")
        confidence = _finite(self.confidence, f"{self.axis}.confidence")
        floor = float(definitions[self.axis]["min"])
        ceiling = float(definitions[self.axis]["max"])
        if not floor <= value <= ceiling:
            raise RegistryObservationEvidenceError(
                "observed registry value is outside declared bounds"
            )
        if not 0.0 < confidence <= 1.0:
            raise RegistryObservationEvidenceError(
                "observation confidence must be strictly positive and at most 1"
            )
        _nonnegative_int(self.observed_tick, "observed_tick")
        for field in (
            "observation_id",
            "source_family",
            "source_instance_id",
            "source_snapshot_id",
            "source_schema_version",
            "acquisition_method",
            "verification_method",
            "model_or_rule_version",
        ):
            _identifier(getattr(self, field), field)
        if self.source_family == REGISTRY_OWNER_SOURCE_FAMILY:
            raise RegistryObservationEvidenceError(
                "registry owner state cannot be its own observation source"
            )
        _digest_string(self.source_integrity_digest, "source_integrity_digest")
        _digest_string(self.raw_observation_digest, "raw_observation_digest")
        if self.observation_kind != VERIFIED_OBSERVATION_KIND:
            raise RegistryObservationEvidenceError(
                "observation_kind must identify verified current-value evidence"
            )
        if self.verification_status != VERIFIED_STATUS:
            raise RegistryObservationEvidenceError(
                "observation evidence must have verified status"
            )
        if any(
            (
                self.genesis_derived,
                self.baseline_derived,
                self.default_derived,
                self.proposal_only,
                self.synthetic,
            )
        ):
            raise RegistryObservationEvidenceError(
                "genesis/default/baseline/proposal/synthetic data is not observation evidence"
            )
        if self.recalculable_reference_present is not True:
            raise RegistryObservationEvidenceError(
                "observation evidence requires a recalculable raw reference"
            )
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise RegistryObservationEvidenceError(
                "unsupported registry observation evidence schema"
            )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "confidence", confidence)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "acquisition_method": self.acquisition_method,
            "axis": self.axis,
            "baseline_derived": self.baseline_derived,
            "confidence": self.confidence,
            "default_derived": self.default_derived,
            "genesis_derived": self.genesis_derived,
            "model_or_rule_version": self.model_or_rule_version,
            "observation_id": self.observation_id,
            "observation_kind": self.observation_kind,
            "observed_tick": self.observed_tick,
            "proposal_only": self.proposal_only,
            "raw_observation_digest": self.raw_observation_digest,
            "recalculable_reference_present": self.recalculable_reference_present,
            "schema_version": self.schema_version,
            "source_family": self.source_family,
            "source_instance_id": self.source_instance_id,
            "source_integrity_digest": self.source_integrity_digest,
            "source_schema_version": self.source_schema_version,
            "source_snapshot_id": self.source_snapshot_id,
            "synthetic": self.synthetic,
            "value": self.value,
            "verification_method": self.verification_method,
            "verification_status": self.verification_status,
        }

    @property
    def evidence_digest(self) -> str:
        return _digest(self.to_mapping(), "registry_axis_observation_evidence")


@dataclass(frozen=True, slots=True)
class RegistryPositiveConfidenceEvidenceBundle:
    bundle_id: str
    target_owner_instance_id: str
    expected_prior_owner_digest: str
    target_state_sequence: int
    logical_tick: int
    observations: tuple[RegistryAxisPositiveConfidenceEvidence, ...]
    source_manifest_schema_version: str
    source_manifest_digest: str
    verification_authorization_id: str
    acceptance_policy_version: str
    schema_version: str = BUNDLE_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    acquisition_mode: str = BUNDLE_ACQUISITION_MODE
    raw_observations_recalculable: bool = True
    runtime_hook_installed: bool = False
    scheduler_installed: bool = False
    persistence_accessed: bool = False
    event_append_performed: bool = False
    live_affect_mutated: bool = False
    live_drive_mutated: bool = False
    named_state_mutated: bool = False
    goal_memory_self_expression_mutated: bool = False
    observation_window_started: bool = False
    observation_window_satisfied: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        for field in (
            "bundle_id",
            "target_owner_instance_id",
            "source_manifest_schema_version",
            "verification_authorization_id",
            "acceptance_policy_version",
        ):
            _identifier(getattr(self, field), field)
        _digest_string(
            self.expected_prior_owner_digest,
            "expected_prior_owner_digest",
        )
        _digest_string(self.source_manifest_digest, "source_manifest_digest")
        _positive_int(self.target_state_sequence, "target_state_sequence")
        _nonnegative_int(self.logical_tick, "logical_tick")
        observations = tuple(self.observations)
        if len(observations) != 37:
            raise RegistryObservationEvidenceError(
                "evidence bundle must contain exactly 37 observations"
            )
        if any(
            type(item) is not RegistryAxisPositiveConfidenceEvidence
            for item in observations
        ):
            raise RegistryObservationEvidenceError(
                "evidence bundle observations must use the exact immutable evidence type"
            )
        if tuple(item.axis for item in observations) != REGISTRY_AXIS_ORDER:
            raise RegistryObservationEvidenceError(
                "evidence bundle must preserve exact canonical 37-axis order"
            )
        if len({item.axis for item in observations}) != 37:
            raise RegistryObservationEvidenceError(
                "evidence bundle axes must be unique"
            )
        if len({item.observation_id for item in observations}) != 37:
            raise RegistryObservationEvidenceError(
                "evidence bundle observation ids must be unique"
            )
        if any(item.observed_tick > self.logical_tick for item in observations):
            raise RegistryObservationEvidenceError(
                "axis observation tick cannot exceed bundle logical tick"
            )
        if self.schema_version != BUNDLE_SCHEMA_VERSION:
            raise RegistryObservationEvidenceError(
                "unsupported registry observation bundle schema"
            )
        if self.authority != SHADOW_AUTHORITY or self.acquisition_mode != BUNDLE_ACQUISITION_MODE:
            raise RegistryObservationEvidenceError(
                "registry observation bundle must remain detached shadow-only"
            )
        if self.raw_observations_recalculable is not True:
            raise RegistryObservationEvidenceError(
                "bundle must retain recalculable raw observation references"
            )
        if any(
            (
                self.runtime_hook_installed,
                self.scheduler_installed,
                self.persistence_accessed,
                self.event_append_performed,
                self.live_affect_mutated,
                self.live_drive_mutated,
                self.named_state_mutated,
                self.goal_memory_self_expression_mutated,
                self.observation_window_started,
                self.observation_window_satisfied,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise RegistryObservationEvidenceError(
                "evidence bundle cannot grant mutation, window, or authority"
            )
        object.__setattr__(self, "observations", observations)

    @property
    def positive_confidence_count(self) -> int:
        return sum(item.confidence > 0.0 for item in self.observations)

    @property
    def exact_positive_confidence_coverage(self) -> bool:
        return (
            len(self.observations) == 37
            and self.positive_confidence_count == 37
            and tuple(item.axis for item in self.observations) == REGISTRY_AXIS_ORDER
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "acceptance_policy_version": self.acceptance_policy_version,
            "acquisition_mode": self.acquisition_mode,
            "authority": self.authority,
            "bundle_id": self.bundle_id,
            "cutover_authorized": self.cutover_authorized,
            "event_append_performed": self.event_append_performed,
            "exact_positive_confidence_coverage": self.exact_positive_confidence_coverage,
            "expected_prior_owner_digest": self.expected_prior_owner_digest,
            "goal_memory_self_expression_mutated": self.goal_memory_self_expression_mutated,
            "live_affect_mutated": self.live_affect_mutated,
            "live_drive_mutated": self.live_drive_mutated,
            "logical_tick": self.logical_tick,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "named_state_mutated": self.named_state_mutated,
            "observation_window_satisfied": self.observation_window_satisfied,
            "observation_window_started": self.observation_window_started,
            "observations": [item.to_mapping() for item in self.observations],
            "persistence_accessed": self.persistence_accessed,
            "positive_confidence_count": self.positive_confidence_count,
            "raw_observations_recalculable": self.raw_observations_recalculable,
            "runtime_hook_installed": self.runtime_hook_installed,
            "scheduler_installed": self.scheduler_installed,
            "schema_version": self.schema_version,
            "source_manifest_digest": self.source_manifest_digest,
            "source_manifest_schema_version": self.source_manifest_schema_version,
            "target_owner_instance_id": self.target_owner_instance_id,
            "target_state_sequence": self.target_state_sequence,
            "verification_authorization_id": self.verification_authorization_id,
        }

    @property
    def bundle_digest(self) -> str:
        return _digest(self.to_mapping(), "registry_positive_confidence_evidence_bundle")


def build_registry_positive_confidence_evidence_bundle(
    owner: RegistryAffectOwnerState,
    observations: Sequence[RegistryAxisPositiveConfidenceEvidence],
    *,
    bundle_id: str,
    logical_tick: int,
    source_manifest_schema_version: str,
    source_manifest_digest: str,
    verification_authorization_id: str,
    acceptance_policy_version: str,
) -> RegistryPositiveConfidenceEvidenceBundle:
    """Bind exact 37-axis verified observations to one detached owner predecessor."""

    if type(owner) is not RegistryAffectOwnerState:
        raise RegistryObservationEvidenceError(
            "owner must be the exact RegistryAffectOwnerState type"
        )
    return RegistryPositiveConfidenceEvidenceBundle(
        bundle_id=bundle_id,
        target_owner_instance_id=owner.owner_instance_id,
        expected_prior_owner_digest=owner.state_digest,
        target_state_sequence=owner.state_sequence + 1,
        logical_tick=logical_tick,
        observations=tuple(observations),
        source_manifest_schema_version=source_manifest_schema_version,
        source_manifest_digest=source_manifest_digest,
        verification_authorization_id=verification_authorization_id,
        acceptance_policy_version=acceptance_policy_version,
    )


def materialize_registry_observed_owner(
    owner: RegistryAffectOwnerState,
    bundle: RegistryPositiveConfidenceEvidenceBundle,
) -> RegistryAffectOwnerState:
    """Return a new detached owner whose 37 values are bound to verified evidence.

    The predecessor remains unchanged. This function does not start an observation
    window and does not install any runtime or persistence authority.
    """

    if type(owner) is not RegistryAffectOwnerState:
        raise RegistryObservationEvidenceError(
            "owner must be the exact RegistryAffectOwnerState type"
        )
    if type(bundle) is not RegistryPositiveConfidenceEvidenceBundle:
        raise RegistryObservationEvidenceError(
            "bundle must be the exact immutable evidence-bundle type"
        )
    if bundle.target_owner_instance_id != owner.owner_instance_id:
        raise RegistryObservationEvidenceError(
            "evidence bundle target owner does not match current owner"
        )
    if bundle.expected_prior_owner_digest != owner.state_digest:
        raise RegistryObservationEvidenceError(
            "evidence bundle prior digest does not match current owner"
        )
    if bundle.target_state_sequence != owner.state_sequence + 1:
        raise RegistryObservationEvidenceError(
            "evidence bundle target sequence is not the next owner state"
        )
    if bundle.logical_tick < owner.logical_tick:
        raise RegistryObservationEvidenceError(
            "evidence bundle logical tick cannot move backward"
        )
    if any(item.observed_tick < owner.logical_tick for item in bundle.observations):
        raise RegistryObservationEvidenceError(
            "evidence bundle contains observations older than the current owner"
        )
    if not bundle.exact_positive_confidence_coverage:
        raise RegistryObservationEvidenceError(
            "evidence bundle lacks exact positive-confidence 37-axis coverage"
        )

    definitions = _definitions()
    prior_axes = {axis.axis: axis for axis in owner.axes}
    for axis in REGISTRY_AXIS_ORDER:
        prior = prior_axes[axis]
        definition = definitions[axis]
        if (
            prior.baseline != float(definition["baseline"])
            or prior.floor != float(definition["min"])
            or prior.ceiling != float(definition["max"])
        ):
            raise RegistryObservationEvidenceError(
                "current owner axis bounds do not match registry definitions"
            )

    evidence_by_axis = {item.axis: item for item in bundle.observations}
    axes = tuple(
        RegistryAxisCurrentState(
            axis=axis,
            value=evidence_by_axis[axis].value,
            baseline=prior_axes[axis].baseline,
            floor=prior_axes[axis].floor,
            ceiling=prior_axes[axis].ceiling,
            confidence=evidence_by_axis[axis].confidence,
            last_impulse_tick=prior_axes[axis].last_impulse_tick,
            update_count=prior_axes[axis].update_count + 1,
            last_source_kind=VERIFIED_OBSERVATION_KIND,
            last_source_id=evidence_by_axis[axis].observation_id,
        )
        for axis in REGISTRY_AXIS_ORDER
    )
    return RegistryAffectOwnerState(
        owner_instance_id=owner.owner_instance_id,
        logical_tick=bundle.logical_tick,
        state_sequence=bundle.target_state_sequence,
        axes=axes,
        prior_state_digest=owner.state_digest,
        last_transition_digest=bundle.bundle_digest,
        last_transition_kind=MATERIALIZED_TRANSITION_KIND,
        last_transition_id=bundle.bundle_id,
        applied_proposal_ids=owner.applied_proposal_ids,
        genesis_source_id=owner.genesis_source_id,
    )
