"""Detached deterministic owner for the 37 read-only affect-registry values.

The registry definitions and event proposal map are not current observations.  This
module materializes an explicit shadow-only owner state from a versioned genesis,
then returns new immutable owner states for validated proposals or caller-invoked
cadence advancement.  It installs no runtime hook or scheduler, accesses no
persistence, appends no event, emits no speech, and grants no M3-C/M3-E/cutover
authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

from adapters.affect_event_proposal_validator import validate_affect_event_proposal
from adapters.affect_hormone_neural_rhythm_registry import (
    AXIS_GROUPS,
    affect_hormone_axis_registry,
)
from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_affect_projection import AxisObservation

OWNER_SCHEMA_VERSION = "eve.m3-b.registry-37-axis-owner.v1"
AXIS_STATE_SCHEMA_VERSION = "eve.m3-b.registry-axis-current-state.v1"
SOURCE_SCHEMA_VERSION = "eve.m3-b.registry-37-axis-source.v1"
GENESIS_SCHEMA_VERSION = "eve.m3-b.registry-baseline-genesis.v1"
TRANSITION_SCHEMA_VERSION = "eve.m3-b.registry-owner-transition.v1"
SOURCE_FAMILY = "read_only_affect_registry"
ZERO_DIGEST = "0" * 64
MAX_APPLIED_PROPOSAL_IDS = 1024

REGISTRY_AXIS_ORDER = tuple(
    axis for axes in AXIS_GROUPS.values() for axis in axes
)


class RegistryAffectOwnerError(ValueError):
    """Raised when owner construction or a detached transition fails closed."""


def _identifier(value: str, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise RegistryAffectOwnerError(f"{field} must be a bounded non-empty string")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RegistryAffectOwnerError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RegistryAffectOwnerError(f"{field} must be finite")
    return result


def _require_digest(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RegistryAffectOwnerError(f"{field} must be a lowercase SHA-256 digest")
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
        raise RegistryAffectOwnerError(f"{field} is not canonical JSON") from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _clip(value: float, floor: float, ceiling: float) -> float:
    return min(ceiling, max(floor, value))


def _definitions() -> dict[str, dict[str, Any]]:
    registry = affect_hormone_axis_registry()
    if tuple(registry) != REGISTRY_AXIS_ORDER:
        raise RegistryAffectOwnerError("registry definitions must preserve exact 37-axis order")
    if len(registry) != 37 or len(set(registry)) != 37:
        raise RegistryAffectOwnerError("registry definitions must contain 37 unique axes")
    return registry


@dataclass(frozen=True, slots=True)
class RegistryAxisCurrentState:
    axis: str
    value: float
    baseline: float
    floor: float
    ceiling: float
    confidence: float
    last_impulse_tick: int
    update_count: int
    last_source_kind: str
    last_source_id: str
    schema_version: str = AXIS_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.axis not in REGISTRY_AXIS_ORDER:
            raise RegistryAffectOwnerError("axis state contains an unknown registry axis")
        value = _finite(self.value, f"{self.axis}.value")
        baseline = _finite(self.baseline, f"{self.axis}.baseline")
        floor = _finite(self.floor, f"{self.axis}.floor")
        ceiling = _finite(self.ceiling, f"{self.axis}.ceiling")
        confidence = _finite(self.confidence, f"{self.axis}.confidence")
        if not floor < baseline < ceiling:
            raise RegistryAffectOwnerError("registry axis requires floor < baseline < ceiling")
        if not floor <= value <= ceiling:
            raise RegistryAffectOwnerError("registry current value is outside declared bounds")
        if not 0.0 <= confidence <= 1.0:
            raise RegistryAffectOwnerError("registry confidence must remain within [0,1]")
        if (
            isinstance(self.last_impulse_tick, bool)
            or not isinstance(self.last_impulse_tick, int)
            or self.last_impulse_tick < 0
        ):
            raise RegistryAffectOwnerError("last_impulse_tick must be a non-negative integer")
        if (
            isinstance(self.update_count, bool)
            or not isinstance(self.update_count, int)
            or self.update_count < 0
        ):
            raise RegistryAffectOwnerError("update_count must be a non-negative integer")
        _identifier(self.last_source_kind, "last_source_kind")
        _identifier(self.last_source_id, "last_source_id")
        if self.schema_version != AXIS_STATE_SCHEMA_VERSION:
            raise RegistryAffectOwnerError("unsupported registry axis state schema")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "floor", floor)
        object.__setattr__(self, "ceiling", ceiling)
        object.__setattr__(self, "confidence", confidence)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "baseline": self.baseline,
            "ceiling": self.ceiling,
            "confidence": self.confidence,
            "floor": self.floor,
            "last_impulse_tick": self.last_impulse_tick,
            "last_source_id": self.last_source_id,
            "last_source_kind": self.last_source_kind,
            "schema_version": self.schema_version,
            "update_count": self.update_count,
            "value": self.value,
        }

    @property
    def state_digest(self) -> str:
        return _digest(self.to_mapping(), "registry_axis_current_state")


@dataclass(frozen=True, slots=True)
class RegistryAffectOwnerState:
    owner_instance_id: str
    logical_tick: int
    state_sequence: int
    axes: tuple[RegistryAxisCurrentState, ...]
    prior_state_digest: str
    last_transition_digest: str
    last_transition_kind: str
    last_transition_id: str
    applied_proposal_ids: tuple[str, ...]
    genesis_source_id: str
    schema_version: str = OWNER_SCHEMA_VERSION
    source_schema_version: str = SOURCE_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    acquisition_mode: str = "explicit_detached_registry_owner"
    genesis_is_observation_evidence: bool = False
    proposal_metadata_is_current_state: bool = False
    runtime_hook_installed: bool = False
    scheduler_installed: bool = False
    persistence_accessed: bool = False
    event_append_performed: bool = False
    live_affect_mutated: bool = False
    live_drive_mutated: bool = False
    goal_memory_self_expression_mutated: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        _identifier(self.owner_instance_id, "owner_instance_id")
        _identifier(self.genesis_source_id, "genesis_source_id")
        _identifier(self.last_transition_kind, "last_transition_kind")
        _identifier(self.last_transition_id, "last_transition_id")
        if isinstance(self.logical_tick, bool) or not isinstance(self.logical_tick, int) or self.logical_tick < 0:
            raise RegistryAffectOwnerError("logical_tick must be a non-negative integer")
        if isinstance(self.state_sequence, bool) or not isinstance(self.state_sequence, int) or self.state_sequence < 0:
            raise RegistryAffectOwnerError("state_sequence must be a non-negative integer")
        axes = tuple(self.axes)
        if tuple(axis.axis for axis in axes) != REGISTRY_AXIS_ORDER:
            raise RegistryAffectOwnerError("owner must contain the exact canonical 37-axis order")
        if any(axis.last_impulse_tick > self.logical_tick for axis in axes):
            raise RegistryAffectOwnerError("axis impulse tick cannot exceed owner logical tick")
        proposal_ids = tuple(self.applied_proposal_ids)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise RegistryAffectOwnerError("applied proposal ids must be unique")
        if len(proposal_ids) > MAX_APPLIED_PROPOSAL_IDS:
            raise RegistryAffectOwnerError("applied proposal id ledger exceeds fixed bound")
        for proposal_id in proposal_ids:
            _identifier(proposal_id, "applied_proposal_id")
        _require_digest(self.prior_state_digest, "prior_state_digest")
        _require_digest(self.last_transition_digest, "last_transition_digest")
        if self.schema_version != OWNER_SCHEMA_VERSION or self.source_schema_version != SOURCE_SCHEMA_VERSION:
            raise RegistryAffectOwnerError("unsupported registry owner schema")
        if self.authority != SHADOW_AUTHORITY or self.acquisition_mode != "explicit_detached_registry_owner":
            raise RegistryAffectOwnerError("registry owner must remain detached shadow-only")
        if self.genesis_is_observation_evidence or self.proposal_metadata_is_current_state:
            raise RegistryAffectOwnerError("definitions/proposals cannot masquerade as observations")
        if any(
            (
                self.runtime_hook_installed,
                self.scheduler_installed,
                self.persistence_accessed,
                self.event_append_performed,
                self.live_affect_mutated,
                self.live_drive_mutated,
                self.goal_memory_self_expression_mutated,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise RegistryAffectOwnerError("registry owner cannot grant live or cutover authority")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "applied_proposal_ids", proposal_ids)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "acquisition_mode": self.acquisition_mode,
            "applied_proposal_ids": list(self.applied_proposal_ids),
            "authority": self.authority,
            "axes": [axis.to_mapping() for axis in self.axes],
            "cutover_authorized": self.cutover_authorized,
            "event_append_performed": self.event_append_performed,
            "genesis_is_observation_evidence": self.genesis_is_observation_evidence,
            "genesis_source_id": self.genesis_source_id,
            "goal_memory_self_expression_mutated": self.goal_memory_self_expression_mutated,
            "last_transition_digest": self.last_transition_digest,
            "last_transition_id": self.last_transition_id,
            "last_transition_kind": self.last_transition_kind,
            "live_affect_mutated": self.live_affect_mutated,
            "live_drive_mutated": self.live_drive_mutated,
            "logical_tick": self.logical_tick,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "observation_window_started": self.observation_window_started,
            "owner_instance_id": self.owner_instance_id,
            "persistence_accessed": self.persistence_accessed,
            "prior_state_digest": self.prior_state_digest,
            "proposal_metadata_is_current_state": self.proposal_metadata_is_current_state,
            "runtime_hook_installed": self.runtime_hook_installed,
            "scheduler_installed": self.scheduler_installed,
            "schema_version": self.schema_version,
            "source_schema_version": self.source_schema_version,
            "state_sequence": self.state_sequence,
        }

    @property
    def state_digest(self) -> str:
        return _digest(self.to_mapping(), "registry_affect_owner_state")

    @property
    def source_snapshot_id(self) -> str:
        return f"{self.owner_instance_id}:s{self.state_sequence}:{self.state_digest[:24]}"

    def value_for(self, axis: str) -> float:
        if axis not in REGISTRY_AXIS_ORDER:
            raise RegistryAffectOwnerError("requested axis is not in the registry")
        return self.axes[REGISTRY_AXIS_ORDER.index(axis)].value

    def to_axis_observations(self) -> tuple[AxisObservation, ...]:
        integrity = self.state_digest
        return tuple(
            AxisObservation(
                axis=axis.axis,
                source_family=SOURCE_FAMILY,
                value=axis.value,
                baseline=axis.baseline,
                floor=axis.floor,
                ceiling=axis.ceiling,
                confidence=axis.confidence,
                source_snapshot_id=self.source_snapshot_id,
                source_schema_version=self.source_schema_version,
                source_integrity_digest=integrity,
                source_metadata=(
                    ("axis_state_digest", axis.state_digest),
                    ("genesis_source_id", self.genesis_source_id),
                    ("last_source_id", axis.last_source_id),
                    ("last_source_kind", axis.last_source_kind),
                    ("logical_tick", str(self.logical_tick)),
                    ("owner_instance_id", self.owner_instance_id),
                    ("state_sequence", str(self.state_sequence)),
                    ("update_count", str(axis.update_count)),
                ),
            )
            for axis in self.axes
        )


def create_registry_affect_owner(
    *,
    owner_instance_id: str,
    genesis_source_id: str,
) -> RegistryAffectOwnerState:
    """Materialize deterministic baseline genesis as state, not observation evidence."""

    _identifier(owner_instance_id, "owner_instance_id")
    _identifier(genesis_source_id, "genesis_source_id")
    registry = _definitions()
    axes = tuple(
        RegistryAxisCurrentState(
            axis=axis,
            value=float(registry[axis]["baseline"]),
            baseline=float(registry[axis]["baseline"]),
            floor=float(registry[axis]["min"]),
            ceiling=float(registry[axis]["max"]),
            confidence=0.0,
            last_impulse_tick=0,
            update_count=0,
            last_source_kind="deterministic_registry_baseline_genesis_not_observation",
            last_source_id=genesis_source_id,
        )
        for axis in REGISTRY_AXIS_ORDER
    )
    genesis_digest = _digest(
        {
            "axes": [axis.to_mapping() for axis in axes],
            "genesis_source_id": genesis_source_id,
            "owner_instance_id": owner_instance_id,
            "schema_version": GENESIS_SCHEMA_VERSION,
        },
        "registry_owner_genesis",
    )
    return RegistryAffectOwnerState(
        owner_instance_id=owner_instance_id,
        logical_tick=0,
        state_sequence=0,
        axes=axes,
        prior_state_digest=ZERO_DIGEST,
        last_transition_digest=genesis_digest,
        last_transition_kind="deterministic_genesis_not_observation",
        last_transition_id=genesis_source_id,
        applied_proposal_ids=(),
        genesis_source_id=genesis_source_id,
    )


def _proposal_digest(
    *,
    owner: RegistryAffectOwnerState,
    event_category: str,
    proposed_axis_deltas: Mapping[str, float],
    proposal_id: str,
    proposal_sequence: int,
    proposal_confidence: float,
    operator_authorization_id: str,
    transition_payload: Mapping[str, Any],
) -> str:
    ordered_deltas = {
        axis: float(proposed_axis_deltas[axis])
        for axis in REGISTRY_AXIS_ORDER
        if axis in proposed_axis_deltas
    }
    return _digest(
        {
            "event_category": event_category,
            "expected_owner_digest": owner.state_digest,
            "operator_authorization_id": operator_authorization_id,
            "proposal_confidence": proposal_confidence,
            "proposal_id": proposal_id,
            "proposal_sequence": proposal_sequence,
            "proposed_axis_deltas": ordered_deltas,
            "schema_version": TRANSITION_SCHEMA_VERSION,
            "transition_payload": dict(transition_payload),
        },
        "registry_owner_proposal",
    )


def apply_validated_registry_proposal(
    owner: RegistryAffectOwnerState,
    *,
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any],
    proposal_id: str,
    proposal_sequence: int,
    proposal_confidence: float,
    expected_owner_digest: str,
    operator_authorization_id: str,
    transition_payload: Mapping[str, Any] | None = None,
) -> RegistryAffectOwnerState:
    """Return a new detached owner state after existing read-only validation.

    This is not a production apply permission.  The old owner remains unchanged,
    and the returned owner remains disconnected, shadow-only, and non-persistent.
    """

    if type(owner) is not RegistryAffectOwnerState:
        raise RegistryAffectOwnerError("owner must be the exact RegistryAffectOwnerState type")
    _identifier(event_category, "event_category")
    _identifier(proposal_id, "proposal_id")
    _identifier(operator_authorization_id, "operator_authorization_id")
    _require_digest(expected_owner_digest, "expected_owner_digest")
    if expected_owner_digest != owner.state_digest:
        raise RegistryAffectOwnerError("expected owner digest does not match current state")
    if (
        isinstance(proposal_sequence, bool)
        or not isinstance(proposal_sequence, int)
        or proposal_sequence != owner.state_sequence + 1
    ):
        raise RegistryAffectOwnerError("proposal sequence must be exactly current sequence + 1")
    if proposal_id in owner.applied_proposal_ids:
        raise RegistryAffectOwnerError("duplicate proposal id rejected")
    if len(owner.applied_proposal_ids) >= MAX_APPLIED_PROPOSAL_IDS:
        raise RegistryAffectOwnerError("proposal id ledger is full; checkpoint rollover required")
    if not isinstance(proposed_axis_deltas, Mapping) or not proposed_axis_deltas:
        raise RegistryAffectOwnerError("proposal must contain at least one axis delta")
    confidence = _finite(proposal_confidence, "proposal_confidence")
    if not 0.0 < confidence <= 1.0:
        raise RegistryAffectOwnerError("proposal confidence must remain within (0,1]")
    deltas: dict[str, float] = {}
    for axis, value in proposed_axis_deltas.items():
        if not isinstance(axis, str) or axis not in REGISTRY_AXIS_ORDER:
            raise RegistryAffectOwnerError("proposal contains an unknown registry axis")
        deltas[axis] = _finite(value, f"proposal_delta.{axis}")
    payload = dict(transition_payload or {})
    validation = validate_affect_event_proposal(event_category, deltas, payload)
    if validation.get("passed") is not True:
        reasons = ",".join(str(item) for item in validation.get("blocked_reasons", ()))
        raise RegistryAffectOwnerError(f"proposal failed existing validator: {reasons}")
    if validation.get("requires_operator_authorization_for_apply") is not True:
        raise RegistryAffectOwnerError("proposal validator did not preserve operator authorization boundary")
    digest = _proposal_digest(
        owner=owner,
        event_category=event_category,
        proposed_axis_deltas=deltas,
        proposal_id=proposal_id,
        proposal_sequence=proposal_sequence,
        proposal_confidence=confidence,
        operator_authorization_id=operator_authorization_id,
        transition_payload=payload,
    )
    updated: list[RegistryAxisCurrentState] = []
    for axis_state in owner.axes:
        if axis_state.axis not in deltas:
            updated.append(axis_state)
            continue
        value = _clip(
            axis_state.value + deltas[axis_state.axis],
            axis_state.floor,
            axis_state.ceiling,
        )
        updated.append(
            RegistryAxisCurrentState(
                axis=axis_state.axis,
                value=value,
                baseline=axis_state.baseline,
                floor=axis_state.floor,
                ceiling=axis_state.ceiling,
                confidence=confidence if axis_state.confidence == 0.0 else min(axis_state.confidence, confidence),
                last_impulse_tick=owner.logical_tick,
                update_count=axis_state.update_count + 1,
                last_source_kind="validated_detached_event_proposal",
                last_source_id=proposal_id,
            )
        )
    return RegistryAffectOwnerState(
        owner_instance_id=owner.owner_instance_id,
        logical_tick=owner.logical_tick,
        state_sequence=proposal_sequence,
        axes=tuple(updated),
        prior_state_digest=owner.state_digest,
        last_transition_digest=digest,
        last_transition_kind="validated_detached_event_proposal",
        last_transition_id=proposal_id,
        applied_proposal_ids=owner.applied_proposal_ids + (proposal_id,),
        genesis_source_id=owner.genesis_source_id,
    )


def advance_registry_affect_owner(
    owner: RegistryAffectOwnerState,
    *,
    target_tick: int,
    cadence_id: str,
    expected_owner_digest: str,
) -> RegistryAffectOwnerState:
    """Advance explicit logical cadence and deterministically decay toward baselines."""

    if type(owner) is not RegistryAffectOwnerState:
        raise RegistryAffectOwnerError("owner must be the exact RegistryAffectOwnerState type")
    _identifier(cadence_id, "cadence_id")
    _require_digest(expected_owner_digest, "expected_owner_digest")
    if expected_owner_digest != owner.state_digest:
        raise RegistryAffectOwnerError("expected owner digest does not match current state")
    if isinstance(target_tick, bool) or not isinstance(target_tick, int) or target_tick <= owner.logical_tick:
        raise RegistryAffectOwnerError("target_tick must be an integer greater than current logical_tick")
    registry = _definitions()
    updated: list[RegistryAxisCurrentState] = []
    for axis_state in owner.axes:
        definition = registry[axis_state.axis]
        refractory_until = axis_state.last_impulse_tick + int(definition["refractory_ticks"])
        decay_start = max(owner.logical_tick, refractory_until)
        active_ticks = max(0, target_tick - decay_start)
        value = axis_state.value
        if active_ticks:
            decay_rate = _finite(definition["decay_rate"], f"{axis_state.axis}.decay_rate")
            value = axis_state.baseline + (
                axis_state.value - axis_state.baseline
            ) * ((1.0 - decay_rate) ** active_ticks)
            value = _clip(value, axis_state.floor, axis_state.ceiling)
            if abs(value - axis_state.baseline) < 1e-15:
                value = axis_state.baseline
        changed = value != axis_state.value
        updated.append(
            RegistryAxisCurrentState(
                axis=axis_state.axis,
                value=value,
                baseline=axis_state.baseline,
                floor=axis_state.floor,
                ceiling=axis_state.ceiling,
                confidence=axis_state.confidence,
                last_impulse_tick=axis_state.last_impulse_tick,
                update_count=axis_state.update_count + (1 if changed else 0),
                last_source_kind=(
                    "deterministic_caller_invoked_cadence_decay"
                    if changed
                    else axis_state.last_source_kind
                ),
                last_source_id=cadence_id if changed else axis_state.last_source_id,
            )
        )
    transition_sequence = owner.state_sequence + 1
    transition_digest = _digest(
        {
            "cadence_id": cadence_id,
            "expected_owner_digest": owner.state_digest,
            "schema_version": TRANSITION_SCHEMA_VERSION,
            "target_tick": target_tick,
            "transition_sequence": transition_sequence,
        },
        "registry_owner_cadence",
    )
    return RegistryAffectOwnerState(
        owner_instance_id=owner.owner_instance_id,
        logical_tick=target_tick,
        state_sequence=transition_sequence,
        axes=tuple(updated),
        prior_state_digest=owner.state_digest,
        last_transition_digest=transition_digest,
        last_transition_kind="deterministic_caller_invoked_cadence",
        last_transition_id=cadence_id,
        applied_proposal_ids=owner.applied_proposal_ids,
        genesis_source_id=owner.genesis_source_id,
    )
