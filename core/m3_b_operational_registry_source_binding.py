"""Detached source binding for the four registry axes that allow operational input.

The module accepts caller-supplied immutable raw records and deterministically
produces positive-confidence evidence for energy, fatigue, recovery, and
overload. It performs no hardware polling, runtime hook, scheduling,
persistence, event append, owner mutation, observation-window transition, or
authority promotion.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_registry_observation_evidence import (
    RegistryAxisPositiveConfidenceEvidence,
)
from core.m3_b_registry_observation_source_manifest import (
    RegistryObservationSourceEntry,
    registry_observation_source_manifest,
)

RAW_SCHEMA_VERSION = "eve.m3-b.operational-registry-raw-record.v1"
BINDING_SCHEMA_VERSION = "eve.m3-b.operational-registry-source-binding.v1"
BINDING_SET_SCHEMA_VERSION = "eve.m3-b.operational-registry-source-binding-set.v1"
SOURCE_FAMILY = "operational_metrics_or_appraised_load_trace"
ACQUISITION_METHOD = "explicit_caller_supplied_immutable_operational_record"
VERIFICATION_METHOD = "exact_schema_range_and_digest_verification"
RAW_MODEL_OR_RULE_VERSION = BINDING_SCHEMA_VERSION
OPERATIONAL_AXES = (
    "energy_budget",
    "fatigue_pressure",
    "recovery_need",
    "overload_risk",
)
REMAINING_BINDING_BLOCKER = "REGISTRY_APPRAISED_33_AXIS_SOURCE_BINDINGS_INCOMPLETE"
POSITIVE_CONFIDENCE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
ZERO_DIGEST = "0" * 64


class OperationalRegistrySourceBindingError(ValueError):
    """Raised when operational raw evidence or binding metadata is invalid."""


def _identifier(value: str, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise OperationalRegistrySourceBindingError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _digest_string(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == ZERO_DIGEST
    ):
        raise OperationalRegistrySourceBindingError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OperationalRegistrySourceBindingError(
            f"{field} must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result == 0:
        raise OperationalRegistrySourceBindingError(f"{field} must be positive")
    return result


def _unit(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OperationalRegistrySourceBindingError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise OperationalRegistrySourceBindingError(
            f"{field} must be finite and inside [0,1]"
        )
    return result


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
        raise OperationalRegistrySourceBindingError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _manifest_entry(axis: str) -> RegistryObservationSourceEntry:
    for entry in registry_observation_source_manifest().entries:
        if entry.axis == axis:
            return entry
    raise OperationalRegistrySourceBindingError(
        "operational axis missing from source manifest"
    )


def _raw_mapping(raw_values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {field: value for field, value in raw_values}


def operational_raw_observation_digest(
    *,
    axis: str,
    logical_tick: int,
    observation_id: str,
    source_instance_id: str,
    source_snapshot_id: str,
    source_schema_version: str,
    source_integrity_digest: str,
    raw_values: tuple[tuple[str, Any], ...],
) -> str:
    """Return the canonical digest that must accompany one raw record."""

    if axis not in OPERATIONAL_AXES:
        raise OperationalRegistrySourceBindingError("unsupported operational axis")
    _nonnegative_int(logical_tick, "logical_tick")
    for field, value in (
        ("observation_id", observation_id),
        ("source_instance_id", source_instance_id),
        ("source_snapshot_id", source_snapshot_id),
        ("source_schema_version", source_schema_version),
    ):
        _identifier(value, field)
    _digest_string(source_integrity_digest, "source_integrity_digest")
    values = tuple(raw_values)
    return _digest(
        {
            "acquisition_method": ACQUISITION_METHOD,
            "axis": axis,
            "logical_tick": logical_tick,
            "model_or_rule_version": RAW_MODEL_OR_RULE_VERSION,
            "observation_id": observation_id,
            "raw_values": [[field, value] for field, value in values],
            "schema_version": RAW_SCHEMA_VERSION,
            "source_family": SOURCE_FAMILY,
            "source_instance_id": source_instance_id,
            "source_integrity_digest": source_integrity_digest,
            "source_schema_version": source_schema_version,
            "source_snapshot_id": source_snapshot_id,
            "verification_method": VERIFICATION_METHOD,
        },
        "operational_raw_observation",
    )


def _validate_raw_values(axis: str, raw: Mapping[str, Any]) -> None:
    if axis == "energy_budget":
        for field in (
            "available_cpu_budget",
            "available_memory_budget",
            "battery_governor_band",
            "foreground_load",
        ):
            _unit(raw[field], field)
        _positive_int(raw["sampling_window_ticks"], "sampling_window_ticks")
        return
    if axis == "fatigue_pressure":
        window = _positive_int(raw["sampling_window_ticks"], "sampling_window_ticks")
        active = _nonnegative_int(
            raw["active_processing_ticks"], "active_processing_ticks"
        )
        recovery = _nonnegative_int(
            raw["recovery_interval_ticks"], "recovery_interval_ticks"
        )
        if active > window or recovery > window:
            raise OperationalRegistrySourceBindingError(
                "fatigue tick counts cannot exceed sampling window"
            )
        _unit(raw["queue_pressure"], "queue_pressure")
        _nonnegative_int(raw["task_switch_count"], "task_switch_count")
        return
    if axis == "recovery_need":
        window = _positive_int(raw["sampling_window_ticks"], "sampling_window_ticks")
        active = _nonnegative_int(
            raw["active_processing_ticks"], "active_processing_ticks"
        )
        cooldown = _nonnegative_int(raw["cooldown_ticks"], "cooldown_ticks")
        if active > window or cooldown > window:
            raise OperationalRegistrySourceBindingError(
                "recovery tick counts cannot exceed sampling window"
            )
        _nonnegative_int(raw["recent_overload_count"], "recent_overload_count")
        _nonnegative_int(
            raw["successful_recovery_count"], "successful_recovery_count"
        )
        return
    if axis == "overload_risk":
        _nonnegative_int(
            raw["concurrent_demand_count"], "concurrent_demand_count"
        )
        _unit(raw["latency_budget_ratio"], "latency_budget_ratio")
        _unit(raw["memory_pressure_ratio"], "memory_pressure_ratio")
        _nonnegative_int(raw["queue_depth"], "queue_depth")
        _unit(raw["thermal_governor_band"], "thermal_governor_band")
        return
    raise OperationalRegistrySourceBindingError("unsupported operational axis")


def _saturating_count(value: int) -> float:
    return float(value) / float(value + 1)


def _record_score(record: "OperationalRegistryRawRecord") -> float:
    raw = record.raw_mapping
    if record.axis == "energy_budget":
        values = (
            raw["available_cpu_budget"],
            raw["available_memory_budget"],
            raw["battery_governor_band"],
            1.0 - raw["foreground_load"],
        )
    elif record.axis == "fatigue_pressure":
        window = raw["sampling_window_ticks"]
        values = (
            raw["active_processing_ticks"] / window,
            raw["queue_pressure"],
            1.0 - raw["recovery_interval_ticks"] / window,
            min(1.0, raw["task_switch_count"] / window),
        )
    elif record.axis == "recovery_need":
        window = raw["sampling_window_ticks"]
        values = (
            raw["active_processing_ticks"] / window,
            1.0 - raw["cooldown_ticks"] / window,
            min(1.0, raw["recent_overload_count"] / window),
            1.0 - min(1.0, raw["successful_recovery_count"] / window),
        )
    elif record.axis == "overload_risk":
        values = (
            _saturating_count(raw["concurrent_demand_count"]),
            raw["latency_budget_ratio"],
            raw["memory_pressure_ratio"],
            _saturating_count(raw["queue_depth"]),
            raw["thermal_governor_band"],
        )
    else:
        raise OperationalRegistrySourceBindingError("unsupported operational axis")
    return float(sum(values) / len(values))


@dataclass(frozen=True, slots=True)
class OperationalRegistryRawRecord:
    axis: str
    logical_tick: int
    observation_id: str
    source_instance_id: str
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    raw_observation_digest: str
    raw_values: tuple[tuple[str, Any], ...]
    acquisition_method: str = ACQUISITION_METHOD
    verification_method: str = VERIFICATION_METHOD
    model_or_rule_version: str = RAW_MODEL_OR_RULE_VERSION
    source_family: str = SOURCE_FAMILY
    synthetic: bool = False
    proposal_only: bool = False
    registry_owner_source: bool = False
    runtime_polled: bool = False
    schema_version: str = RAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.axis not in OPERATIONAL_AXES:
            raise OperationalRegistrySourceBindingError("unsupported operational axis")
        _nonnegative_int(self.logical_tick, "logical_tick")
        for field in (
            "observation_id",
            "source_instance_id",
            "source_snapshot_id",
            "source_schema_version",
        ):
            _identifier(getattr(self, field), field)
        expected_provenance = {
            "acquisition_method": ACQUISITION_METHOD,
            "verification_method": VERIFICATION_METHOD,
            "model_or_rule_version": RAW_MODEL_OR_RULE_VERSION,
            "source_family": SOURCE_FAMILY,
        }
        for field, expected in expected_provenance.items():
            if getattr(self, field) != expected:
                raise OperationalRegistrySourceBindingError(
                    f"raw record {field} does not match the canonical operational provenance contract"
                )
        _digest_string(self.source_integrity_digest, "source_integrity_digest")
        _digest_string(self.raw_observation_digest, "raw_observation_digest")
        values = tuple(self.raw_values)
        fields = tuple(field for field, _ in values)
        expected_fields = _manifest_entry(self.axis).required_raw_fields
        if fields != expected_fields or len(set(fields)) != len(fields):
            raise OperationalRegistrySourceBindingError(
                "raw record fields do not match the canonical axis source plan"
            )
        _validate_raw_values(self.axis, _raw_mapping(values))
        expected_digest = operational_raw_observation_digest(
            axis=self.axis,
            logical_tick=self.logical_tick,
            observation_id=self.observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version,
            source_integrity_digest=self.source_integrity_digest,
            raw_values=values,
        )
        if self.raw_observation_digest != expected_digest:
            raise OperationalRegistrySourceBindingError(
                "raw observation digest does not match identity, time, provenance, and values"
            )
        if any(
            (
                self.synthetic,
                self.proposal_only,
                self.registry_owner_source,
                self.runtime_polled,
            )
        ):
            raise OperationalRegistrySourceBindingError(
                "operational evidence cannot be synthetic, proposal-only, circular, or runtime-polled here"
            )
        if self.schema_version != RAW_SCHEMA_VERSION:
            raise OperationalRegistrySourceBindingError(
                "unsupported operational raw schema"
            )
        object.__setattr__(self, "raw_values", values)

    @property
    def raw_mapping(self) -> dict[str, Any]:
        return _raw_mapping(self.raw_values)

    @property
    def recalculated_raw_observation_digest(self) -> str:
        return operational_raw_observation_digest(
            axis=self.axis,
            logical_tick=self.logical_tick,
            observation_id=self.observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version,
            source_integrity_digest=self.source_integrity_digest,
            raw_values=self.raw_values,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "acquisition_method": self.acquisition_method,
            "axis": self.axis,
            "logical_tick": self.logical_tick,
            "model_or_rule_version": self.model_or_rule_version,
            "observation_id": self.observation_id,
            "proposal_only": self.proposal_only,
            "raw_observation_digest": self.raw_observation_digest,
            "raw_values": [[field, value] for field, value in self.raw_values],
            "registry_owner_source": self.registry_owner_source,
            "runtime_polled": self.runtime_polled,
            "schema_version": self.schema_version,
            "source_family": self.source_family,
            "source_instance_id": self.source_instance_id,
            "source_integrity_digest": self.source_integrity_digest,
            "source_schema_version": self.source_schema_version,
            "source_snapshot_id": self.source_snapshot_id,
            "synthetic": self.synthetic,
            "verification_method": self.verification_method,
        }


@dataclass(frozen=True, slots=True)
class OperationalRegistrySourceBinding:
    axis: str
    source_contract_id: str
    binding_id: str
    raw_schema_version: str
    required_raw_fields: tuple[str, ...]
    minimum_raw_record_count: int
    minimum_logical_span_ticks: int
    derivation_rule_id: str
    confidence_rule_id: str
    schema_version: str = BINDING_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    binding_implemented: bool = True
    production_capture_present: bool = False
    runtime_capture_installed: bool = False
    hardware_polling_installed: bool = False
    persistence_accessed: bool = False
    event_append_performed: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        if self.axis not in OPERATIONAL_AXES:
            raise OperationalRegistrySourceBindingError("unsupported binding axis")
        entry = _manifest_entry(self.axis)
        expected = {
            "source_contract_id": entry.source_contract_id,
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "required_raw_fields": entry.required_raw_fields,
            "minimum_raw_record_count": entry.minimum_raw_record_count,
            "minimum_logical_span_ticks": entry.minimum_logical_span_ticks,
            "derivation_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:mean.v1",
            "confidence_rule_id": (
                f"{BINDING_SCHEMA_VERSION}:{self.axis}:coverage-variance.v1"
            ),
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise OperationalRegistrySourceBindingError(
                    f"binding {field} does not match the canonical plan"
                )
        _identifier(self.binding_id, "binding_id")
        if (
            self.schema_version != BINDING_SCHEMA_VERSION
            or self.authority != SHADOW_AUTHORITY
        ):
            raise OperationalRegistrySourceBindingError(
                "operational binding must use the exact shadow-only schema"
            )
        if self.binding_implemented is not True:
            raise OperationalRegistrySourceBindingError(
                "binding implementation flag must be true"
            )
        if any(
            (
                self.production_capture_present,
                self.runtime_capture_installed,
                self.hardware_polling_installed,
                self.persistence_accessed,
                self.event_append_performed,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise OperationalRegistrySourceBindingError(
                "binding cannot claim production capture, runtime, window, or authority"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "axis": self.axis,
            "binding_id": self.binding_id,
            "binding_implemented": self.binding_implemented,
            "confidence_rule_id": self.confidence_rule_id,
            "cutover_authorized": self.cutover_authorized,
            "derivation_rule_id": self.derivation_rule_id,
            "event_append_performed": self.event_append_performed,
            "hardware_polling_installed": self.hardware_polling_installed,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "minimum_logical_span_ticks": self.minimum_logical_span_ticks,
            "minimum_raw_record_count": self.minimum_raw_record_count,
            "observation_window_started": self.observation_window_started,
            "persistence_accessed": self.persistence_accessed,
            "production_capture_present": self.production_capture_present,
            "raw_schema_version": self.raw_schema_version,
            "required_raw_fields": list(self.required_raw_fields),
            "runtime_capture_installed": self.runtime_capture_installed,
            "schema_version": self.schema_version,
            "source_contract_id": self.source_contract_id,
        }

    @property
    def binding_digest(self) -> str:
        return _digest(self.to_mapping(), "operational_registry_source_binding")


@dataclass(frozen=True, slots=True)
class OperationalRegistrySourceBindingSet:
    bindings: tuple[OperationalRegistrySourceBinding, ...]
    schema_version: str = BINDING_SET_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    production_capture_present: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        if any(type(item) is not OperationalRegistrySourceBinding for item in bindings):
            raise OperationalRegistrySourceBindingError(
                "binding set requires exact immutable binding types"
            )
        if len(bindings) != 4 or tuple(item.axis for item in bindings) != OPERATIONAL_AXES:
            raise OperationalRegistrySourceBindingError(
                "operational binding set must preserve exact four-axis order"
            )
        if (
            self.schema_version != BINDING_SET_SCHEMA_VERSION
            or self.authority != SHADOW_AUTHORITY
        ):
            raise OperationalRegistrySourceBindingError(
                "unsupported binding-set schema"
            )
        if any(
            (
                self.production_capture_present,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise OperationalRegistrySourceBindingError(
                "binding set cannot claim production capture, window, or authority"
            )
        object.__setattr__(self, "bindings", bindings)

    @property
    def binding_count(self) -> int:
        return len(self.bindings)

    @property
    def remaining_axis_count(self) -> int:
        return 37 - self.binding_count

    @property
    def blockers(self) -> tuple[str, ...]:
        return (REMAINING_BINDING_BLOCKER, POSITIVE_CONFIDENCE_BLOCKER)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "binding_count": self.binding_count,
            "bindings": [item.to_mapping() for item in self.bindings],
            "blockers": list(self.blockers),
            "cutover_authorized": self.cutover_authorized,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "observation_window_started": self.observation_window_started,
            "production_capture_present": self.production_capture_present,
            "remaining_axis_count": self.remaining_axis_count,
            "schema_version": self.schema_version,
        }

    @property
    def binding_set_digest(self) -> str:
        return _digest(
            self.to_mapping(), "operational_registry_source_binding_set"
        )


def operational_registry_source_bindings() -> OperationalRegistrySourceBindingSet:
    bindings = []
    for axis in OPERATIONAL_AXES:
        entry = _manifest_entry(axis)
        bindings.append(
            OperationalRegistrySourceBinding(
                axis=axis,
                source_contract_id=entry.source_contract_id,
                binding_id=f"eve:m3-b:operational-binding:{axis}:v1",
                raw_schema_version=RAW_SCHEMA_VERSION,
                required_raw_fields=entry.required_raw_fields,
                minimum_raw_record_count=entry.minimum_raw_record_count,
                minimum_logical_span_ticks=entry.minimum_logical_span_ticks,
                derivation_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1",
                confidence_rule_id=(
                    f"{BINDING_SCHEMA_VERSION}:{axis}:coverage-variance.v1"
                ),
            )
        )
    return OperationalRegistrySourceBindingSet(bindings=tuple(bindings))


def derive_operational_axis_evidence(
    records: Sequence[OperationalRegistryRawRecord],
) -> RegistryAxisPositiveConfidenceEvidence:
    """Derive one detached evidence record from exact operational raw records."""

    items = tuple(records)
    if not items or any(type(item) is not OperationalRegistryRawRecord for item in items):
        raise OperationalRegistrySourceBindingError(
            "records must contain exact immutable operational raw records"
        )
    axis = items[0].axis
    if any(item.axis != axis for item in items):
        raise OperationalRegistrySourceBindingError("records cannot mix axes")
    ticks = tuple(item.logical_tick for item in items)
    if ticks != tuple(sorted(ticks)):
        raise OperationalRegistrySourceBindingError("record ticks must be sorted")
    if len(set(ticks)) != len(ticks):
        raise OperationalRegistrySourceBindingError("record ticks must be unique")
    if len({item.observation_id for item in items}) != len(items):
        raise OperationalRegistrySourceBindingError(
            "record observation ids must be unique"
        )
    if len({item.source_snapshot_id for item in items}) != len(items):
        raise OperationalRegistrySourceBindingError(
            "record snapshots must be unique"
        )
    shared_fields = (
        "source_instance_id",
        "source_schema_version",
        "acquisition_method",
        "verification_method",
        "model_or_rule_version",
    )
    for field in shared_fields:
        if len({getattr(item, field) for item in items}) != 1:
            raise OperationalRegistrySourceBindingError(
                f"records must share one {field}"
            )
    binding = next(
        item
        for item in operational_registry_source_bindings().bindings
        if item.axis == axis
    )
    span = items[-1].logical_tick - items[0].logical_tick
    if len(items) < binding.minimum_raw_record_count:
        raise OperationalRegistrySourceBindingError(
            "insufficient raw record count"
        )
    if span < binding.minimum_logical_span_ticks:
        raise OperationalRegistrySourceBindingError(
            "insufficient logical observation span"
        )
    scores = tuple(_record_score(item) for item in items)
    value = float(sum(scores) / len(scores))
    variance = float(
        sum((score - value) ** 2 for score in scores) / len(scores)
    )
    confidence = float(max(0.5, min(1.0, 1.0 - variance)))
    raw_bundle_digest = _digest(
        {
            "axis": axis,
            "binding_digest": binding.binding_digest,
            "records": [item.to_mapping() for item in items],
        },
        "operational_raw_bundle",
    )
    source_integrity_digest = _digest(
        {
            "binding_digest": binding.binding_digest,
            "raw_bundle_digest": raw_bundle_digest,
            "source_instance_id": items[0].source_instance_id,
        },
        "operational_source_integrity",
    )
    return RegistryAxisPositiveConfidenceEvidence(
        axis=axis,
        value=value,
        confidence=confidence,
        observed_tick=items[-1].logical_tick,
        observation_id=f"operational:{axis}:{raw_bundle_digest[:24]}",
        source_family=SOURCE_FAMILY,
        source_instance_id=items[0].source_instance_id,
        source_snapshot_id=(
            f"operational:{axis}:{items[0].logical_tick}:"
            f"{items[-1].logical_tick}:{raw_bundle_digest[:16]}"
        ),
        source_schema_version=RAW_SCHEMA_VERSION,
        source_integrity_digest=source_integrity_digest,
        raw_observation_digest=raw_bundle_digest,
        acquisition_method=ACQUISITION_METHOD,
        verification_method=VERIFICATION_METHOD,
        model_or_rule_version=binding.derivation_rule_id,
    )
