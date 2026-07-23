"""Explicit combined 63-axis M3-B read-only observation packet.

The packet combines one caller-invoked exact legacy 26-axis capture with one
already-existing detached registry 37-axis owner snapshot. It installs no hook,
observer, scheduler, persistence route, event append path, projection mutation,
observation window, M3-C, cutover, or M3-E authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from hormone_system import HormoneSystem

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_affect_projection import AffectProjectionError, AxisObservation
from core.m3_b_legacy_affect_capture import (
    LEGACY_AXIS_ORDER,
    LegacyAffectCaptureError,
    capture_legacy_hormone_state,
)
from core.m3_b_registry_affect_owner import (
    REGISTRY_AXIS_ORDER,
    RegistryAffectOwnerState,
)

PACKET_SCHEMA_VERSION = "eve.m3-b.combined-63-axis-observation-packet.v1"
SOURCE_SET_SCHEMA_VERSION = "eve.m3-b.combined-observation-source-set.v1"
EXPECTED_AXIS_ORDER = LEGACY_AXIS_ORDER + REGISTRY_AXIS_ORDER
EXPECTED_AXIS_COUNT = 63
EXPECTED_LEGACY_COUNT = 26
EXPECTED_REGISTRY_COUNT = 37
WINDOW_BLOCKER_REGISTRY_CONFIDENCE = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
WINDOW_BLOCKER_LEGACY_CONFIDENCE = "LEGACY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"


class M3BObservationPacketError(ValueError):
    """Raised when a combined packet cannot be constructed exactly."""


def _identifier(value: str, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise M3BObservationPacketError(f"{field} must be a bounded non-empty string")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M3BObservationPacketError(f"{field} must be a non-negative integer")
    return value


def _digest_string(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3BObservationPacketError(f"{field} must be a lowercase SHA-256 digest")
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
        raise M3BObservationPacketError(f"{field} is not canonical JSON") from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CombinedObservationSourceSet:
    legacy_source_instance_id: str
    legacy_source_snapshot_id: str
    legacy_capture_digest: str
    legacy_source_integrity_digest: str
    registry_owner_instance_id: str
    registry_source_snapshot_id: str
    registry_owner_state_digest: str
    schema_version: str = SOURCE_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "legacy_source_instance_id",
            "legacy_source_snapshot_id",
            "registry_owner_instance_id",
            "registry_source_snapshot_id",
        ):
            _identifier(getattr(self, field), field)
        for field in (
            "legacy_capture_digest",
            "legacy_source_integrity_digest",
            "registry_owner_state_digest",
        ):
            _digest_string(getattr(self, field), field)
        if self.schema_version != SOURCE_SET_SCHEMA_VERSION:
            raise M3BObservationPacketError("unsupported source-set schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "legacy_capture_digest": self.legacy_capture_digest,
            "legacy_source_instance_id": self.legacy_source_instance_id,
            "legacy_source_integrity_digest": self.legacy_source_integrity_digest,
            "legacy_source_snapshot_id": self.legacy_source_snapshot_id,
            "registry_owner_instance_id": self.registry_owner_instance_id,
            "registry_owner_state_digest": self.registry_owner_state_digest,
            "registry_source_snapshot_id": self.registry_source_snapshot_id,
            "schema_version": self.schema_version,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping(), "combined_observation_source_set")


@dataclass(frozen=True, slots=True)
class M3BCombinedObservationPacket:
    packet_id: str
    packet_sequence: int
    logical_tick: int
    source_set: CombinedObservationSourceSet
    observations: tuple[AxisObservation, ...]
    positive_confidence_axes: tuple[str, ...]
    zero_confidence_axes: tuple[str, ...]
    window_blockers: tuple[str, ...]
    schema_version: str = PACKET_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    acquisition_mode: str = "explicit_combined_read_only"
    exact_axis_order_verified: bool = True
    source_integrity_verified: bool = True
    source_ownership_verified: bool = True
    legacy_capture_no_mutation_verified: bool = True
    registry_owner_unchanged_verified: bool = True
    projection_performed: bool = False
    observation_window_started: bool = False
    observation_window_satisfied: bool = False
    persistence_accessed: bool = False
    event_append_performed: bool = False
    live_affect_mutated: bool = False
    live_drive_mutated: bool = False
    named_state_mutated: bool = False
    goal_memory_self_expression_mutated: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        _identifier(self.packet_id, "packet_id")
        _nonnegative_int(self.packet_sequence, "packet_sequence")
        _nonnegative_int(self.logical_tick, "logical_tick")
        if type(self.source_set) is not CombinedObservationSourceSet:
            raise M3BObservationPacketError("source_set must be the exact immutable source-set type")
        observations = tuple(self.observations)
        if len(observations) != EXPECTED_AXIS_COUNT:
            raise M3BObservationPacketError("packet must contain exactly 63 observations")
        if any(type(item) is not AxisObservation for item in observations):
            raise M3BObservationPacketError("packet observations must be exact AxisObservation values")
        if tuple(item.axis for item in observations) != EXPECTED_AXIS_ORDER:
            raise M3BObservationPacketError("packet must preserve exact canonical 63-axis order")
        if len({item.axis for item in observations}) != EXPECTED_AXIS_COUNT:
            raise M3BObservationPacketError("packet axes must be unique")
        if any(item.source_family != "legacy_mutable_hormone" for item in observations[:EXPECTED_LEGACY_COUNT]):
            raise M3BObservationPacketError("first 26 observations must be legacy source values")
        if any(item.source_family != "read_only_affect_registry" for item in observations[EXPECTED_LEGACY_COUNT:]):
            raise M3BObservationPacketError("final 37 observations must be registry owner values")
        positive = tuple(item.axis for item in observations if item.confidence > 0.0)
        zero = tuple(item.axis for item in observations if item.confidence == 0.0)
        if tuple(self.positive_confidence_axes) != positive:
            raise M3BObservationPacketError("positive-confidence axis catalog is not derived from observations")
        if tuple(self.zero_confidence_axes) != zero:
            raise M3BObservationPacketError("zero-confidence axis catalog is not derived from observations")
        expected_blockers: list[str] = []
        if any(axis in LEGACY_AXIS_ORDER for axis in zero):
            expected_blockers.append(WINDOW_BLOCKER_LEGACY_CONFIDENCE)
        if any(axis in REGISTRY_AXIS_ORDER for axis in zero):
            expected_blockers.append(WINDOW_BLOCKER_REGISTRY_CONFIDENCE)
        if tuple(self.window_blockers) != tuple(expected_blockers):
            raise M3BObservationPacketError("window blockers are not derived from packet evidence")
        if self.schema_version != PACKET_SCHEMA_VERSION:
            raise M3BObservationPacketError("unsupported combined packet schema")
        if self.authority != SHADOW_AUTHORITY or self.acquisition_mode != "explicit_combined_read_only":
            raise M3BObservationPacketError("combined packet must remain explicit shadow-only evidence")
        if not all(
            (
                self.exact_axis_order_verified,
                self.source_integrity_verified,
                self.source_ownership_verified,
                self.legacy_capture_no_mutation_verified,
                self.registry_owner_unchanged_verified,
            )
        ):
            raise M3BObservationPacketError("packet structural/source proof is incomplete")
        if any(
            (
                self.projection_performed,
                self.observation_window_started,
                self.observation_window_satisfied,
                self.persistence_accessed,
                self.event_append_performed,
                self.live_affect_mutated,
                self.live_drive_mutated,
                self.named_state_mutated,
                self.goal_memory_self_expression_mutated,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise M3BObservationPacketError("packet cannot perform projection, mutation, window, or authority promotion")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "positive_confidence_axes", positive)
        object.__setattr__(self, "zero_confidence_axes", zero)
        object.__setattr__(self, "window_blockers", tuple(expected_blockers))

    @property
    def axis_count(self) -> int:
        return len(self.observations)

    @property
    def legacy_axis_count(self) -> int:
        return EXPECTED_LEGACY_COUNT

    @property
    def registry_axis_count(self) -> int:
        return EXPECTED_REGISTRY_COUNT

    @property
    def positive_confidence_count(self) -> int:
        return len(self.positive_confidence_axes)

    @property
    def zero_confidence_count(self) -> int:
        return len(self.zero_confidence_axes)

    @property
    def structurally_complete(self) -> bool:
        return all(
            (
                self.axis_count == EXPECTED_AXIS_COUNT,
                self.exact_axis_order_verified,
                self.source_integrity_verified,
                self.source_ownership_verified,
                self.legacy_capture_no_mutation_verified,
                self.registry_owner_unchanged_verified,
            )
        )

    @property
    def strict_projection_input_ready(self) -> bool:
        return self.structurally_complete

    @property
    def observation_window_start_eligible(self) -> bool:
        return self.structurally_complete and not self.window_blockers and self.zero_confidence_count == 0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "acquisition_mode": self.acquisition_mode,
            "authority": self.authority,
            "axis_count": self.axis_count,
            "cutover_authorized": self.cutover_authorized,
            "event_append_performed": self.event_append_performed,
            "exact_axis_order_verified": self.exact_axis_order_verified,
            "goal_memory_self_expression_mutated": self.goal_memory_self_expression_mutated,
            "legacy_axis_count": self.legacy_axis_count,
            "legacy_capture_no_mutation_verified": self.legacy_capture_no_mutation_verified,
            "live_affect_mutated": self.live_affect_mutated,
            "live_drive_mutated": self.live_drive_mutated,
            "logical_tick": self.logical_tick,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "named_state_mutated": self.named_state_mutated,
            "observation_window_satisfied": self.observation_window_satisfied,
            "observation_window_start_eligible": self.observation_window_start_eligible,
            "observation_window_started": self.observation_window_started,
            "observations": [item.to_mapping() for item in self.observations],
            "packet_id": self.packet_id,
            "packet_sequence": self.packet_sequence,
            "persistence_accessed": self.persistence_accessed,
            "positive_confidence_axes": list(self.positive_confidence_axes),
            "positive_confidence_count": self.positive_confidence_count,
            "projection_performed": self.projection_performed,
            "registry_axis_count": self.registry_axis_count,
            "registry_owner_unchanged_verified": self.registry_owner_unchanged_verified,
            "schema_version": self.schema_version,
            "source_integrity_verified": self.source_integrity_verified,
            "source_ownership_verified": self.source_ownership_verified,
            "source_set": self.source_set.to_mapping(),
            "strict_projection_input_ready": self.strict_projection_input_ready,
            "structurally_complete": self.structurally_complete,
            "window_blockers": list(self.window_blockers),
            "zero_confidence_axes": list(self.zero_confidence_axes),
            "zero_confidence_count": self.zero_confidence_count,
        }

    @property
    def packet_digest(self) -> str:
        return _digest(self.to_mapping(), "m3_b_combined_observation_packet")


def build_m3_b_observation_packet(
    legacy_source: HormoneSystem,
    registry_owner: RegistryAffectOwnerState,
    *,
    packet_id: str,
    packet_sequence: int,
    logical_tick: int,
    legacy_source_instance_id: str,
    legacy_source_snapshot_id: str,
) -> M3BCombinedObservationPacket:
    """Build one detached 63-axis packet from exact caller-supplied source owners."""

    _identifier(packet_id, "packet_id")
    _nonnegative_int(packet_sequence, "packet_sequence")
    _nonnegative_int(logical_tick, "logical_tick")
    _identifier(legacy_source_instance_id, "legacy_source_instance_id")
    _identifier(legacy_source_snapshot_id, "legacy_source_snapshot_id")
    if type(legacy_source) is not HormoneSystem:
        raise M3BObservationPacketError("legacy_source must be the exact HormoneSystem type")
    if type(registry_owner) is not RegistryAffectOwnerState:
        raise M3BObservationPacketError("registry_owner must be the exact RegistryAffectOwnerState type")

    registry_before_digest = registry_owner.state_digest
    try:
        legacy_capture = capture_legacy_hormone_state(
            legacy_source,
            source_instance_id=legacy_source_instance_id,
            source_snapshot_id=legacy_source_snapshot_id,
        )
        legacy_observations = legacy_capture.to_axis_observations()
    except (LegacyAffectCaptureError, AffectProjectionError) as exc:
        raise M3BObservationPacketError(
            "legacy source cannot produce the strict v1 observation envelope"
        ) from exc
    registry_observations = registry_owner.to_axis_observations()
    registry_after_digest = registry_owner.state_digest
    if registry_before_digest != registry_after_digest:
        raise M3BObservationPacketError("registry owner changed while packet was assembled")
    if len(legacy_observations) != EXPECTED_LEGACY_COUNT or len(registry_observations) != EXPECTED_REGISTRY_COUNT:
        raise M3BObservationPacketError("source owners did not supply exact 26+37 coverage")
    observations = tuple(legacy_observations) + tuple(registry_observations)
    positive = tuple(item.axis for item in observations if item.confidence > 0.0)
    zero = tuple(item.axis for item in observations if item.confidence == 0.0)
    blockers: list[str] = []
    if any(axis in LEGACY_AXIS_ORDER for axis in zero):
        blockers.append(WINDOW_BLOCKER_LEGACY_CONFIDENCE)
    if any(axis in REGISTRY_AXIS_ORDER for axis in zero):
        blockers.append(WINDOW_BLOCKER_REGISTRY_CONFIDENCE)
    source_set = CombinedObservationSourceSet(
        legacy_source_instance_id=legacy_capture.source_instance_id,
        legacy_source_snapshot_id=legacy_capture.source_snapshot_id,
        legacy_capture_digest=legacy_capture.capture_digest,
        legacy_source_integrity_digest=legacy_capture.source_integrity_digest,
        registry_owner_instance_id=registry_owner.owner_instance_id,
        registry_source_snapshot_id=registry_owner.source_snapshot_id,
        registry_owner_state_digest=registry_before_digest,
    )
    return M3BCombinedObservationPacket(
        packet_id=packet_id,
        packet_sequence=packet_sequence,
        logical_tick=logical_tick,
        source_set=source_set,
        observations=observations,
        positive_confidence_axes=positive,
        zero_confidence_axes=zero,
        window_blockers=tuple(blockers),
    )
