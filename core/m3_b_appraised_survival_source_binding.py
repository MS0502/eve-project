"""Detached appraised source bindings for the remaining survival-stability axes.

This module accepts caller-supplied immutable, verified appraisal traces for
``stress_load`` and ``stability_need``. It deterministically derives detached
positive-confidence registry evidence. It performs no hardware polling, raw
social-feedback ingestion, runtime hook, scheduling, persistence, event append,
owner mutation, observation-window transition, or authority promotion.
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

RAW_SCHEMA_VERSION = "eve.m3-b.appraised-survival-raw-record.v1"
APPRAISAL_SCHEMA_VERSION = "eve.m3-b.survival-appraisal-trace.v1"
BINDING_SCHEMA_VERSION = "eve.m3-b.appraised-survival-source-binding.v1"
BINDING_SET_SCHEMA_VERSION = "eve.m3-b.appraised-survival-source-binding-set.v1"
SOURCE_FAMILY = "operational_metrics_or_appraised_load_trace"
ACQUISITION_METHOD = "explicit_caller_supplied_immutable_appraised_survival_record"
VERIFICATION_METHOD = "exact_appraisal_schema_range_identity_and_digest_verification"
APPRAISAL_METHOD = "deterministic_bounded_survival_load_appraisal"
APPRAISAL_OUTCOME = "accepted_bounded_survival_appraisal"
QUARANTINE_STATUS = "not_applicable_non_social_survival_trace"
RAW_MODEL_OR_RULE_VERSION = BINDING_SCHEMA_VERSION
APPRAISED_SURVIVAL_AXES = ("stress_load", "stability_need")
TOTAL_BOUND_AXIS_COUNT = 6
REMAINING_BINDING_BLOCKER = "REGISTRY_APPRAISED_31_AXIS_SOURCE_BINDINGS_INCOMPLETE"
POSITIVE_CONFIDENCE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
ZERO_DIGEST = "0" * 64


class AppraisedSurvivalSourceBindingError(ValueError):
    """Raised when an appraised survival trace or binding is invalid."""


def _identifier(value: str, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise AppraisedSurvivalSourceBindingError(
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
        raise AppraisedSurvivalSourceBindingError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AppraisedSurvivalSourceBindingError(
            f"{field} must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result == 0:
        raise AppraisedSurvivalSourceBindingError(f"{field} must be positive")
    return result


def _unit(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AppraisedSurvivalSourceBindingError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise AppraisedSurvivalSourceBindingError(
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
        raise AppraisedSurvivalSourceBindingError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _manifest_entry(axis: str) -> RegistryObservationSourceEntry:
    for entry in registry_observation_source_manifest().entries:
        if entry.axis == axis:
            return entry
    raise AppraisedSurvivalSourceBindingError(
        "appraised survival axis missing from source manifest"
    )


def _raw_mapping(raw_values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {field: value for field, value in raw_values}


def appraised_survival_raw_observation_digest(
    *,
    axis: str,
    logical_tick: int,
    observation_id: str,
    source_instance_id: str,
    source_snapshot_id: str,
    source_schema_version: str,
    source_integrity_digest: str,
    appraisal_trace_id: str,
    appraisal_input_digest: str,
    appraisal_integrity_digest: str,
    raw_values: tuple[tuple[str, Any], ...],
) -> str:
    """Return the canonical digest required by one appraised survival record."""

    if axis not in APPRAISED_SURVIVAL_AXES:
        raise AppraisedSurvivalSourceBindingError("unsupported appraised survival axis")
    _nonnegative_int(logical_tick, "logical_tick")
    for field, value in (
        ("observation_id", observation_id),
        ("source_instance_id", source_instance_id),
        ("source_snapshot_id", source_snapshot_id),
        ("source_schema_version", source_schema_version),
        ("appraisal_trace_id", appraisal_trace_id),
    ):
        _identifier(value, field)
    for field, value in (
        ("source_integrity_digest", source_integrity_digest),
        ("appraisal_input_digest", appraisal_input_digest),
        ("appraisal_integrity_digest", appraisal_integrity_digest),
    ):
        _digest_string(value, field)
    values = tuple(raw_values)
    return _digest(
        {
            "acquisition_method": ACQUISITION_METHOD,
            "appraisal_input_digest": appraisal_input_digest,
            "appraisal_integrity_digest": appraisal_integrity_digest,
            "appraisal_method": APPRAISAL_METHOD,
            "appraisal_outcome": APPRAISAL_OUTCOME,
            "appraisal_schema_version": APPRAISAL_SCHEMA_VERSION,
            "appraisal_trace_id": appraisal_trace_id,
            "appraisal_verified": True,
            "axis": axis,
            "hardware_direct_input": False,
            "logical_tick": logical_tick,
            "model_or_rule_version": RAW_MODEL_OR_RULE_VERSION,
            "observation_id": observation_id,
            "quarantine_status": QUARANTINE_STATUS,
            "raw_social_feedback_source": False,
            "raw_values": [[field, value] for field, value in values],
            "schema_version": RAW_SCHEMA_VERSION,
            "source_family": SOURCE_FAMILY,
            "source_instance_id": source_instance_id,
            "source_integrity_digest": source_integrity_digest,
            "source_schema_version": source_schema_version,
            "source_snapshot_id": source_snapshot_id,
            "verification_method": VERIFICATION_METHOD,
        },
        "appraised_survival_raw_observation",
    )


def _validate_raw_values(axis: str, raw: Mapping[str, Any]) -> None:
    if axis == "stress_load":
        if raw["appraisal_version"] != APPRAISAL_SCHEMA_VERSION:
            raise AppraisedSurvivalSourceBindingError(
                "stress appraisal_version must match the canonical appraisal schema"
            )
        for field in (
            "controllability_score",
            "demand_score",
            "overload_score",
            "uncertainty_score",
        ):
            _unit(raw[field], field)
        return
    if axis == "stability_need":
        window = _positive_int(raw["sampling_window_ticks"], "sampling_window_ticks")
        for field in (
            "invariant_failure_count",
            "pending_migration_count",
            "replay_divergence_count",
        ):
            count = _nonnegative_int(raw[field], field)
            if count > window:
                raise AppraisedSurvivalSourceBindingError(
                    f"{field} cannot exceed sampling_window_ticks"
                )
        _unit(raw["rollback_readiness_score"], "rollback_readiness_score")
        return
    raise AppraisedSurvivalSourceBindingError("unsupported appraised survival axis")


def _record_score(record: "AppraisedSurvivalRawRecord") -> float:
    raw = record.raw_mapping
    if record.axis == "stress_load":
        values = (
            1.0 - raw["controllability_score"],
            raw["demand_score"],
            raw["overload_score"],
            raw["uncertainty_score"],
        )
    elif record.axis == "stability_need":
        window = raw["sampling_window_ticks"]
        values = (
            raw["invariant_failure_count"] / window,
            raw["pending_migration_count"] / window,
            raw["replay_divergence_count"] / window,
            1.0 - raw["rollback_readiness_score"],
        )
    else:
        raise AppraisedSurvivalSourceBindingError("unsupported appraised survival axis")
    return float(sum(values) / len(values))


@dataclass(frozen=True, slots=True)
class AppraisedSurvivalRawRecord:
    axis: str
    logical_tick: int
    observation_id: str
    source_instance_id: str
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    appraisal_trace_id: str
    appraisal_input_digest: str
    appraisal_integrity_digest: str
    raw_observation_digest: str
    raw_values: tuple[tuple[str, Any], ...]
    acquisition_method: str = ACQUISITION_METHOD
    verification_method: str = VERIFICATION_METHOD
    appraisal_schema_version: str = APPRAISAL_SCHEMA_VERSION
    appraisal_method: str = APPRAISAL_METHOD
    appraisal_outcome: str = APPRAISAL_OUTCOME
    quarantine_status: str = QUARANTINE_STATUS
    model_or_rule_version: str = RAW_MODEL_OR_RULE_VERSION
    source_family: str = SOURCE_FAMILY
    appraisal_verified: bool = True
    raw_social_feedback_source: bool = False
    hardware_direct_input: bool = False
    synthetic: bool = False
    proposal_only: bool = False
    registry_owner_source: bool = False
    runtime_polled: bool = False
    schema_version: str = RAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.axis not in APPRAISED_SURVIVAL_AXES:
            raise AppraisedSurvivalSourceBindingError(
                "unsupported appraised survival axis"
            )
        _nonnegative_int(self.logical_tick, "logical_tick")
        for field in (
            "observation_id",
            "source_instance_id",
            "source_snapshot_id",
            "source_schema_version",
            "appraisal_trace_id",
        ):
            _identifier(getattr(self, field), field)
        for field in (
            "source_integrity_digest",
            "appraisal_input_digest",
            "appraisal_integrity_digest",
            "raw_observation_digest",
        ):
            _digest_string(getattr(self, field), field)
        expected_provenance = {
            "acquisition_method": ACQUISITION_METHOD,
            "verification_method": VERIFICATION_METHOD,
            "appraisal_schema_version": APPRAISAL_SCHEMA_VERSION,
            "appraisal_method": APPRAISAL_METHOD,
            "appraisal_outcome": APPRAISAL_OUTCOME,
            "quarantine_status": QUARANTINE_STATUS,
            "model_or_rule_version": RAW_MODEL_OR_RULE_VERSION,
            "source_family": SOURCE_FAMILY,
        }
        for field, expected in expected_provenance.items():
            if getattr(self, field) != expected:
                raise AppraisedSurvivalSourceBindingError(
                    f"raw record {field} does not match the canonical appraised-survival provenance contract"
                )
        if self.appraisal_verified is not True:
            raise AppraisedSurvivalSourceBindingError(
                "appraised survival evidence requires an exactly verified appraisal"
            )
        if any(
            (
                self.raw_social_feedback_source,
                self.hardware_direct_input,
                self.synthetic,
                self.proposal_only,
                self.registry_owner_source,
                self.runtime_polled,
            )
        ):
            raise AppraisedSurvivalSourceBindingError(
                "appraised survival evidence cannot use raw social feedback, direct hardware, synthetic, proposal-only, circular, or runtime-polled input"
            )
        values = tuple(self.raw_values)
        fields = tuple(field for field, _ in values)
        expected_fields = _manifest_entry(self.axis).required_raw_fields
        if fields != expected_fields or len(set(fields)) != len(fields):
            raise AppraisedSurvivalSourceBindingError(
                "raw record fields do not match the canonical axis source plan"
            )
        _validate_raw_values(self.axis, _raw_mapping(values))
        expected_digest = appraised_survival_raw_observation_digest(
            axis=self.axis,
            logical_tick=self.logical_tick,
            observation_id=self.observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version,
            source_integrity_digest=self.source_integrity_digest,
            appraisal_trace_id=self.appraisal_trace_id,
            appraisal_input_digest=self.appraisal_input_digest,
            appraisal_integrity_digest=self.appraisal_integrity_digest,
            raw_values=values,
        )
        if self.raw_observation_digest != expected_digest:
            raise AppraisedSurvivalSourceBindingError(
                "raw observation digest does not match identity, time, appraisal provenance, and values"
            )
        if self.schema_version != RAW_SCHEMA_VERSION:
            raise AppraisedSurvivalSourceBindingError(
                "unsupported appraised-survival raw schema"
            )
        object.__setattr__(self, "raw_values", values)

    @property
    def raw_mapping(self) -> dict[str, Any]:
        return _raw_mapping(self.raw_values)

    @property
    def recalculated_raw_observation_digest(self) -> str:
        return appraised_survival_raw_observation_digest(
            axis=self.axis,
            logical_tick=self.logical_tick,
            observation_id=self.observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version,
            source_integrity_digest=self.source_integrity_digest,
            appraisal_trace_id=self.appraisal_trace_id,
            appraisal_input_digest=self.appraisal_input_digest,
            appraisal_integrity_digest=self.appraisal_integrity_digest,
            raw_values=self.raw_values,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "acquisition_method": self.acquisition_method,
            "appraisal_input_digest": self.appraisal_input_digest,
            "appraisal_integrity_digest": self.appraisal_integrity_digest,
            "appraisal_method": self.appraisal_method,
            "appraisal_outcome": self.appraisal_outcome,
            "appraisal_schema_version": self.appraisal_schema_version,
            "appraisal_trace_id": self.appraisal_trace_id,
            "appraisal_verified": self.appraisal_verified,
            "axis": self.axis,
            "hardware_direct_input": self.hardware_direct_input,
            "logical_tick": self.logical_tick,
            "model_or_rule_version": self.model_or_rule_version,
            "observation_id": self.observation_id,
            "proposal_only": self.proposal_only,
            "quarantine_status": self.quarantine_status,
            "raw_observation_digest": self.raw_observation_digest,
            "raw_social_feedback_source": self.raw_social_feedback_source,
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
class AppraisedSurvivalSourceBinding:
    axis: str
    source_contract_id: str
    binding_id: str
    raw_schema_version: str
    appraisal_schema_version: str
    required_raw_fields: tuple[str, ...]
    minimum_raw_record_count: int
    minimum_logical_span_ticks: int
    derivation_rule_id: str
    confidence_rule_id: str
    schema_version: str = BINDING_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    binding_implemented: bool = True
    appraisal_required: bool = True
    quarantine_required: bool = False
    hardware_direct_input_allowed: bool = False
    production_capture_present: bool = False
    runtime_capture_installed: bool = False
    persistence_accessed: bool = False
    event_append_performed: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        if self.axis not in APPRAISED_SURVIVAL_AXES:
            raise AppraisedSurvivalSourceBindingError("unsupported binding axis")
        entry = _manifest_entry(self.axis)
        expected = {
            "source_contract_id": entry.source_contract_id,
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "appraisal_schema_version": APPRAISAL_SCHEMA_VERSION,
            "required_raw_fields": entry.required_raw_fields,
            "minimum_raw_record_count": entry.minimum_raw_record_count,
            "minimum_logical_span_ticks": entry.minimum_logical_span_ticks,
            "derivation_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:mean.v1",
            "confidence_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:coverage-variance.v1",
            "appraisal_required": True,
            "quarantine_required": entry.quarantine_required,
            "hardware_direct_input_allowed": False,
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise AppraisedSurvivalSourceBindingError(
                    f"binding {field} does not match the canonical source plan"
                )
        _identifier(self.binding_id, "binding_id")
        if self.schema_version != BINDING_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise AppraisedSurvivalSourceBindingError(
                "appraised survival binding must use the exact shadow-only schema"
            )
        if self.binding_implemented is not True:
            raise AppraisedSurvivalSourceBindingError(
                "binding implementation flag must be true"
            )
        if any(
            (
                self.production_capture_present,
                self.runtime_capture_installed,
                self.persistence_accessed,
                self.event_append_performed,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise AppraisedSurvivalSourceBindingError(
                "binding cannot claim production capture, runtime, window, or authority"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "appraisal_required": self.appraisal_required,
            "appraisal_schema_version": self.appraisal_schema_version,
            "authority": self.authority,
            "axis": self.axis,
            "binding_id": self.binding_id,
            "binding_implemented": self.binding_implemented,
            "confidence_rule_id": self.confidence_rule_id,
            "cutover_authorized": self.cutover_authorized,
            "derivation_rule_id": self.derivation_rule_id,
            "event_append_performed": self.event_append_performed,
            "hardware_direct_input_allowed": self.hardware_direct_input_allowed,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "minimum_logical_span_ticks": self.minimum_logical_span_ticks,
            "minimum_raw_record_count": self.minimum_raw_record_count,
            "observation_window_started": self.observation_window_started,
            "persistence_accessed": self.persistence_accessed,
            "production_capture_present": self.production_capture_present,
            "quarantine_required": self.quarantine_required,
            "raw_schema_version": self.raw_schema_version,
            "required_raw_fields": list(self.required_raw_fields),
            "runtime_capture_installed": self.runtime_capture_installed,
            "schema_version": self.schema_version,
            "source_contract_id": self.source_contract_id,
        }

    @property
    def binding_digest(self) -> str:
        return _digest(self.to_mapping(), "appraised_survival_source_binding")


@dataclass(frozen=True, slots=True)
class AppraisedSurvivalSourceBindingSet:
    bindings: tuple[AppraisedSurvivalSourceBinding, ...]
    schema_version: str = BINDING_SET_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    total_bound_axis_count: int = TOTAL_BOUND_AXIS_COUNT
    remaining_axis_count: int = 31
    production_capture_present: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        if any(type(item) is not AppraisedSurvivalSourceBinding for item in bindings):
            raise AppraisedSurvivalSourceBindingError(
                "binding set requires exact immutable binding types"
            )
        if len(bindings) != 2 or tuple(item.axis for item in bindings) != APPRAISED_SURVIVAL_AXES:
            raise AppraisedSurvivalSourceBindingError(
                "appraised survival binding set must preserve exact two-axis order"
            )
        if (
            self.schema_version != BINDING_SET_SCHEMA_VERSION
            or self.authority != SHADOW_AUTHORITY
            or self.total_bound_axis_count != TOTAL_BOUND_AXIS_COUNT
            or self.remaining_axis_count != 31
        ):
            raise AppraisedSurvivalSourceBindingError(
                "unsupported appraised-survival binding-set contract"
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
            raise AppraisedSurvivalSourceBindingError(
                "binding set cannot claim production capture, window, or authority"
            )
        object.__setattr__(self, "bindings", bindings)

    @property
    def appraised_binding_count(self) -> int:
        return len(self.bindings)

    @property
    def blockers(self) -> tuple[str, ...]:
        return (REMAINING_BINDING_BLOCKER, POSITIVE_CONFIDENCE_BLOCKER)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "appraised_binding_count": self.appraised_binding_count,
            "authority": self.authority,
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
            "total_bound_axis_count": self.total_bound_axis_count,
        }

    @property
    def binding_set_digest(self) -> str:
        return _digest(self.to_mapping(), "appraised_survival_source_binding_set")


def appraised_survival_source_bindings() -> AppraisedSurvivalSourceBindingSet:
    bindings = []
    for axis in APPRAISED_SURVIVAL_AXES:
        entry = _manifest_entry(axis)
        bindings.append(
            AppraisedSurvivalSourceBinding(
                axis=axis,
                source_contract_id=entry.source_contract_id,
                binding_id=f"eve:m3-b:appraised-survival-binding:{axis}:v1",
                raw_schema_version=RAW_SCHEMA_VERSION,
                appraisal_schema_version=APPRAISAL_SCHEMA_VERSION,
                required_raw_fields=entry.required_raw_fields,
                minimum_raw_record_count=entry.minimum_raw_record_count,
                minimum_logical_span_ticks=entry.minimum_logical_span_ticks,
                derivation_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1",
                confidence_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:coverage-variance.v1",
                quarantine_required=entry.quarantine_required,
            )
        )
    return AppraisedSurvivalSourceBindingSet(bindings=tuple(bindings))


def derive_appraised_survival_axis_evidence(
    records: Sequence[AppraisedSurvivalRawRecord],
) -> RegistryAxisPositiveConfidenceEvidence:
    """Derive one detached evidence record from verified appraisal traces."""

    items = tuple(records)
    if not items or any(type(item) is not AppraisedSurvivalRawRecord for item in items):
        raise AppraisedSurvivalSourceBindingError(
            "records must contain exact immutable appraised-survival raw records"
        )
    axis = items[0].axis
    if any(item.axis != axis for item in items):
        raise AppraisedSurvivalSourceBindingError("records cannot mix axes")
    ticks = tuple(item.logical_tick for item in items)
    if ticks != tuple(sorted(ticks)):
        raise AppraisedSurvivalSourceBindingError("record ticks must be sorted")
    if len(set(ticks)) != len(ticks):
        raise AppraisedSurvivalSourceBindingError("record ticks must be unique")
    for field in ("observation_id", "source_snapshot_id", "appraisal_trace_id"):
        if len({getattr(item, field) for item in items}) != len(items):
            raise AppraisedSurvivalSourceBindingError(
                f"record {field} values must be unique"
            )
    for field in (
        "source_instance_id",
        "source_schema_version",
        "acquisition_method",
        "verification_method",
        "appraisal_schema_version",
        "appraisal_method",
        "appraisal_outcome",
        "quarantine_status",
        "model_or_rule_version",
        "source_family",
    ):
        if len({getattr(item, field) for item in items}) != 1:
            raise AppraisedSurvivalSourceBindingError(
                f"records must share one {field}"
            )
    binding = next(
        item
        for item in appraised_survival_source_bindings().bindings
        if item.axis == axis
    )
    span = items[-1].logical_tick - items[0].logical_tick
    if len(items) < binding.minimum_raw_record_count:
        raise AppraisedSurvivalSourceBindingError("insufficient raw record count")
    if span < binding.minimum_logical_span_ticks:
        raise AppraisedSurvivalSourceBindingError(
            "insufficient logical observation span"
        )
    scores = tuple(_record_score(item) for item in items)
    value = float(sum(scores) / len(scores))
    variance = float(sum((score - value) ** 2 for score in scores) / len(scores))
    confidence = float(max(0.5, min(1.0, 1.0 - variance)))
    raw_bundle_digest = _digest(
        {
            "axis": axis,
            "binding_digest": binding.binding_digest,
            "records": [item.to_mapping() for item in items],
        },
        "appraised_survival_raw_bundle",
    )
    source_integrity_digest = _digest(
        {
            "binding_digest": binding.binding_digest,
            "raw_bundle_digest": raw_bundle_digest,
            "source_instance_id": items[0].source_instance_id,
        },
        "appraised_survival_source_integrity",
    )
    return RegistryAxisPositiveConfidenceEvidence(
        axis=axis,
        value=value,
        confidence=confidence,
        observed_tick=items[-1].logical_tick,
        observation_id=f"appraised-survival:{axis}:{raw_bundle_digest[:24]}",
        source_family=SOURCE_FAMILY,
        source_instance_id=items[0].source_instance_id,
        source_snapshot_id=(
            f"appraised-survival:{axis}:{items[0].logical_tick}:"
            f"{items[-1].logical_tick}:{raw_bundle_digest[:16]}"
        ),
        source_schema_version=RAW_SCHEMA_VERSION,
        source_integrity_digest=source_integrity_digest,
        raw_observation_digest=raw_bundle_digest,
        acquisition_method=ACQUISITION_METHOD,
        verification_method=VERIFICATION_METHOD,
        model_or_rule_version=binding.derivation_rule_id,
    )
