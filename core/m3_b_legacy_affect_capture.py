"""Caller-invoked immutable capture for the exact legacy 26-axis hormone source.

Importing this module performs no capture and constructs no runtime. Capture must
be invoked explicitly with an existing exact ``HormoneSystem`` object. It reads
that object twice, fails closed on any observed change or identity replacement,
and returns detached immutable evidence. It never calls update/stimulate, installs
no observer, accesses no persistence, appends no event, or grants live authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

from hormone_system import Hormone, HormoneSystem

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_affect_projection import AxisObservation

CAPTURE_SCHEMA_VERSION = "eve.m3-b.legacy-26-axis-capture.v1"
AXIS_SCHEMA_VERSION = "eve.m3-b.legacy-axis-evidence.v1"
SOURCE_SCHEMA_VERSION = "eve.legacy-hormone-system.v32.capture.v1"

LEGACY_AXIS_ORDER = (
    "glutamate",
    "gaba",
    "glycine",
    "dopamine",
    "serotonin",
    "norepinephrine",
    "histamine",
    "acetylcholine",
    "adenosine",
    "endorphin",
    "cortisol",
    "oxytocin",
    "vasopressin",
    "melatonin",
    "bdnf",
    "ngf",
    "estrogen",
    "testosterone",
    "insulin_brain",
    "thyroid",
    "leptin",
    "ghrelin",
    "prolactin",
    "dhea",
    "progesterone",
    "growth_hormone",
)


class LegacyAffectCaptureError(ValueError):
    """Raised when a source is inadmissible or changes during capture."""


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise LegacyAffectCaptureError(f"{field} must be a bounded non-empty string")
    return value


def _real(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LegacyAffectCaptureError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LegacyAffectCaptureError(f"{field} must be finite")
    return result


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
        raise LegacyAffectCaptureError(f"{field} is not canonical JSON") from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _read_source_state(source: HormoneSystem) -> dict[str, Any]:
    if type(source) is not HormoneSystem:
        raise LegacyAffectCaptureError("source must be the exact HormoneSystem type")
    if type(source.hormones) is not dict:
        raise LegacyAffectCaptureError("source hormones must be the exact legacy dict container")
    if tuple(source.hormones) != LEGACY_AXIS_ORDER:
        raise LegacyAffectCaptureError("source must contain the exact 26 axes in canonical order")

    phase = source.phase
    if isinstance(phase, bool) or not isinstance(phase, int) or not 1 <= phase <= 4:
        raise LegacyAffectCaptureError("source phase must be an integer in [1,4]")
    stage = _identifier(source.stage, "source.stage")
    source_time = _real(source.time, "source.time")
    sim_hour = _real(source.sim_hour, "source.sim_hour")
    if source_time < 0.0 or not 0.0 <= sim_hour < 24.0:
        raise LegacyAffectCaptureError("source time fields are outside legacy bounds")

    axes: list[dict[str, Any]] = []
    for axis in LEGACY_AXIS_ORDER:
        hormone = source.hormones[axis]
        if type(hormone) is not Hormone or hormone.name != axis:
            raise LegacyAffectCaptureError(f"{axis}: exact Hormone identity contract failed")
        level = _real(hormone.level, f"{axis}.level")
        baseline = _real(hormone.baseline, f"{axis}.baseline")
        reactivity = _real(hormone.reactivity, f"{axis}.reactivity")
        decay_rate = _real(hormone.decay_rate, f"{axis}.decay_rate")
        if not 0.0 <= level <= 1.0 or not 0.0 <= baseline <= 1.0:
            raise LegacyAffectCaptureError(f"{axis}: level/baseline outside [0,1]")
        if reactivity < 0.0 or decay_rate < 0.0:
            raise LegacyAffectCaptureError(f"{axis}: dynamics must be non-negative")
        if hormone.tier not in {"A", "B", "C"}:
            raise LegacyAffectCaptureError(f"{axis}: invalid tier")
        if isinstance(hormone.phase, bool) or not isinstance(hormone.phase, int):
            raise LegacyAffectCaptureError(f"{axis}: invalid phase")
        if not 1 <= hormone.phase <= 4:
            raise LegacyAffectCaptureError(f"{axis}: phase outside [1,4]")
        axes.append(
            {
                "axis": axis,
                "baseline": baseline,
                "ceiling": 1.0,
                "decay_rate": decay_rate,
                "floor": 0.0,
                "level": level,
                "phase": hormone.phase,
                "reactivity": reactivity,
                "tier": hormone.tier,
            }
        )

    active = tuple(source.active_hormones)
    expected_active = tuple(
        row["axis"] for row in axes if int(row["phase"]) <= phase
    )
    if active != expected_active:
        raise LegacyAffectCaptureError("active_hormones does not match source phase")
    return {
        "active_hormones": list(active),
        "axes": axes,
        "phase": phase,
        "sim_hour": sim_hour,
        "stage": stage,
        "time": source_time,
        "type": f"{type(source).__module__}.{type(source).__qualname__}",
    }


@dataclass(frozen=True, slots=True)
class LegacyAxisEvidence:
    axis: str
    value: float
    baseline: float
    floor: float
    ceiling: float
    reactivity: float
    decay_rate: float
    tier: str
    phase: int
    source_integrity_digest: str
    schema_version: str = AXIS_SCHEMA_VERSION
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.axis not in LEGACY_AXIS_ORDER:
            raise LegacyAffectCaptureError("axis evidence contains an unknown axis")
        for field in (
            "value",
            "baseline",
            "floor",
            "ceiling",
            "reactivity",
            "decay_rate",
            "confidence",
        ):
            object.__setattr__(self, field, _real(getattr(self, field), field))
        if not self.floor < self.ceiling:
            raise LegacyAffectCaptureError("axis evidence requires floor < ceiling")
        if not self.floor <= self.value <= self.ceiling:
            raise LegacyAffectCaptureError("axis value outside declared range")
        if not self.floor <= self.baseline <= self.ceiling:
            raise LegacyAffectCaptureError("axis baseline outside declared range")
        if self.reactivity < 0.0 or self.decay_rate < 0.0:
            raise LegacyAffectCaptureError("axis dynamics must be non-negative")
        if self.tier not in {"A", "B", "C"} or not 1 <= self.phase <= 4:
            raise LegacyAffectCaptureError("axis tier/phase is invalid")
        if self.schema_version != AXIS_SCHEMA_VERSION or self.confidence != 1.0:
            raise LegacyAffectCaptureError("legacy capture schema/confidence is fixed")
        if (
            len(self.source_integrity_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.source_integrity_digest)
        ):
            raise LegacyAffectCaptureError("source integrity digest must be lowercase SHA-256")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "baseline": self.baseline,
            "ceiling": self.ceiling,
            "confidence": self.confidence,
            "decay_rate": self.decay_rate,
            "floor": self.floor,
            "phase": self.phase,
            "reactivity": self.reactivity,
            "schema_version": self.schema_version,
            "source_integrity_digest": self.source_integrity_digest,
            "tier": self.tier,
            "value": self.value,
        }

    @property
    def axis_integrity_digest(self) -> str:
        return _digest(self.to_mapping(), "legacy_axis_evidence")


@dataclass(frozen=True, slots=True)
class LegacyHormoneCapture:
    source_instance_id: str
    source_snapshot_id: str
    source_integrity_digest: str
    source_type: str
    source_phase: int
    source_stage: str
    source_time: float
    source_sim_hour: float
    active_hormones: tuple[str, ...]
    axes: tuple[LegacyAxisEvidence, ...]
    schema_version: str = CAPTURE_SCHEMA_VERSION
    source_schema_version: str = SOURCE_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    acquisition_mode: str = "explicit_after_the_fact_read_only"
    exact_source_type_verified: bool = True
    source_container_identity_stable: bool = True
    source_axis_object_identities_stable: bool = True
    before_after_state_equal: bool = True
    source_mutated: bool = False
    persistence_accessed: bool = False
    event_append_performed: bool = False
    live_behavior_changed: bool = False
    observation_window_started: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        _identifier(self.source_instance_id, "source_instance_id")
        _identifier(self.source_snapshot_id, "source_snapshot_id")
        _identifier(self.source_type, "source_type")
        _identifier(self.source_stage, "source_stage")
        if self.schema_version != CAPTURE_SCHEMA_VERSION:
            raise LegacyAffectCaptureError("unsupported capture schema")
        if self.source_schema_version != SOURCE_SCHEMA_VERSION:
            raise LegacyAffectCaptureError("unsupported source schema")
        if self.authority != SHADOW_AUTHORITY:
            raise LegacyAffectCaptureError("capture authority must remain shadow_only")
        if self.acquisition_mode != "explicit_after_the_fact_read_only":
            raise LegacyAffectCaptureError("capture mode is fixed")
        if tuple(axis.axis for axis in self.axes) != LEGACY_AXIS_ORDER:
            raise LegacyAffectCaptureError("capture must contain exact canonical 26-axis order")
        if len(self.active_hormones) != len(set(self.active_hormones)):
            raise LegacyAffectCaptureError("active hormone identities must be unique")
        if any(axis not in LEGACY_AXIS_ORDER for axis in self.active_hormones):
            raise LegacyAffectCaptureError("active hormone identity is unknown")
        if not all(
            (
                self.exact_source_type_verified,
                self.source_container_identity_stable,
                self.source_axis_object_identities_stable,
                self.before_after_state_equal,
            )
        ):
            raise LegacyAffectCaptureError("capture identity/no-mutation proof is incomplete")
        if any(
            (
                self.source_mutated,
                self.persistence_accessed,
                self.event_append_performed,
                self.live_behavior_changed,
                self.observation_window_started,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise LegacyAffectCaptureError("capture cannot mutate or grant authority")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "acquisition_mode": self.acquisition_mode,
            "active_hormones": list(self.active_hormones),
            "authority": self.authority,
            "axes": [axis.to_mapping() for axis in self.axes],
            "before_after_state_equal": self.before_after_state_equal,
            "cutover_authorized": self.cutover_authorized,
            "event_append_performed": self.event_append_performed,
            "exact_source_type_verified": self.exact_source_type_verified,
            "live_behavior_changed": self.live_behavior_changed,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "observation_window_started": self.observation_window_started,
            "persistence_accessed": self.persistence_accessed,
            "schema_version": self.schema_version,
            "source_axis_object_identities_stable": self.source_axis_object_identities_stable,
            "source_container_identity_stable": self.source_container_identity_stable,
            "source_instance_id": self.source_instance_id,
            "source_integrity_digest": self.source_integrity_digest,
            "source_mutated": self.source_mutated,
            "source_phase": self.source_phase,
            "source_schema_version": self.source_schema_version,
            "source_sim_hour": self.source_sim_hour,
            "source_snapshot_id": self.source_snapshot_id,
            "source_stage": self.source_stage,
            "source_time": self.source_time,
            "source_type": self.source_type,
        }

    @property
    def capture_digest(self) -> str:
        return _digest(self.to_mapping(), "legacy_hormone_capture")

    def to_axis_observations(self) -> tuple[AxisObservation, ...]:
        return tuple(
            AxisObservation(
                axis=axis.axis,
                source_family="legacy_mutable_hormone",
                value=axis.value,
                baseline=axis.baseline,
                floor=axis.floor,
                ceiling=axis.ceiling,
                confidence=axis.confidence,
                source_snapshot_id=self.source_snapshot_id,
                source_schema_version=self.schema_version,
                source_integrity_digest=self.source_integrity_digest,
                source_metadata=(
                    ("axis_integrity_digest", axis.axis_integrity_digest),
                    ("capture_digest", self.capture_digest),
                    ("decay_rate", repr(axis.decay_rate)),
                    ("phase", str(axis.phase)),
                    ("reactivity", repr(axis.reactivity)),
                    ("source_instance_id", self.source_instance_id),
                    ("tier", axis.tier),
                ),
            )
            for axis in self.axes
        )


def capture_legacy_hormone_state(
    source: HormoneSystem,
    *,
    source_instance_id: str,
    source_snapshot_id: str,
) -> LegacyHormoneCapture:
    """Capture exact 26-axis legacy state without calling any mutation surface."""
    _identifier(source_instance_id, "source_instance_id")
    _identifier(source_snapshot_id, "source_snapshot_id")
    if type(source) is not HormoneSystem:
        raise LegacyAffectCaptureError("source must be the exact HormoneSystem type")

    container_reference = source.hormones
    if type(container_reference) is not dict:
        raise LegacyAffectCaptureError("source hormones must use the exact legacy dict")
    axis_references = tuple(container_reference.get(axis) for axis in LEGACY_AXIS_ORDER)
    before = _read_source_state(source)
    source_integrity_digest = _digest(before, "legacy_hormone_source_state")
    axis_evidence = tuple(
        LegacyAxisEvidence(
            axis=str(row["axis"]),
            value=float(row["level"]),
            baseline=float(row["baseline"]),
            floor=float(row["floor"]),
            ceiling=float(row["ceiling"]),
            reactivity=float(row["reactivity"]),
            decay_rate=float(row["decay_rate"]),
            tier=str(row["tier"]),
            phase=int(row["phase"]),
            source_integrity_digest=source_integrity_digest,
        )
        for row in before["axes"]
    )
    after = _read_source_state(source)

    if source.hormones is not container_reference:
        raise LegacyAffectCaptureError("source hormone container identity changed during capture")
    if any(source.hormones[axis] is not reference for axis, reference in zip(LEGACY_AXIS_ORDER, axis_references)):
        raise LegacyAffectCaptureError("source axis object identity changed during capture")
    if before != after:
        raise LegacyAffectCaptureError("source state changed during capture")

    return LegacyHormoneCapture(
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_integrity_digest=source_integrity_digest,
        source_type=str(before["type"]),
        source_phase=int(before["phase"]),
        source_stage=str(before["stage"]),
        source_time=float(before["time"]),
        source_sim_hour=float(before["sim_hour"]),
        active_hormones=tuple(str(axis) for axis in before["active_hormones"]),
        axes=axis_evidence,
    )
