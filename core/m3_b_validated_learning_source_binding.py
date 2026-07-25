"""Detached validated source bindings for the six learning-exploration axes.

This module accepts caller-supplied immutable records that already carry an
explicit versioned learning/prediction validation trace and a separate bounded
appraisal trace. It deterministically derives detached positive-confidence
registry evidence. It performs no training, memory consolidation write,
prediction update, source acquisition, raw social-feedback ingestion, hardware
polling, scheduling, persistence, event append, registry-owner mutation,
observation-window transition, or authority promotion.
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

RAW_SCHEMA_VERSION = "eve.m3-b.validated-learning-raw-record.v1"
VALIDATION_SCHEMA_VERSION = "eve.m3-b.learning-validation-trace.v1"
APPRAISAL_SCHEMA_VERSION = "eve.m3-b.learning-appraisal-trace.v1"
BINDING_SCHEMA_VERSION = "eve.m3-b.validated-learning-source-binding.v1"
BINDING_SET_SCHEMA_VERSION = "eve.m3-b.validated-learning-source-binding-set.v1"
SOURCE_FAMILY = "validated_learning_and_prediction_trace"
ACQUISITION_METHOD = "explicit_caller_supplied_immutable_validated_learning_record"
VERIFICATION_METHOD = "exact_learning_validation_appraisal_schema_range_identity_and_digest_verification"
VALIDATION_METHOD = "explicit_versioned_learning_or_prediction_validation"
VALIDATION_OUTCOME = "accepted_validated_learning_or_prediction_signal"
APPRAISAL_METHOD = "deterministic_bounded_learning_exploration_appraisal"
APPRAISAL_OUTCOME = "accepted_bounded_learning_exploration_appraisal"
RAW_MODEL_OR_RULE_VERSION = BINDING_SCHEMA_VERSION
LEARNING_EXPLORATION_AXES = (
    "curiosity_drive",
    "novelty_seeking",
    "learning_pressure",
    "memory_consolidation_pressure",
    "prediction_error_pressure",
    "competence_drive",
)
TOTAL_BOUND_AXIS_COUNT = 25
REMAINING_BINDING_BLOCKER = "REGISTRY_APPRAISED_12_AXIS_SOURCE_BINDINGS_INCOMPLETE"
POSITIVE_CONFIDENCE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
ZERO_DIGEST = "0" * 64
_VALIDATION_STATUS_ALLOWED = {"verified", "operator_validated"}
_VERIFICATION_STATUS_ALLOWED = {"verified"}


class ValidatedLearningSourceBindingError(ValueError):
    """Raised when a validated learning trace or binding fails closed."""


def _identifier(value: str, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValidatedLearningSourceBindingError(
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
        raise ValidatedLearningSourceBindingError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidatedLearningSourceBindingError(
            f"{field} must be a non-negative integer"
        )
    return value


def _unit(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidatedLearningSourceBindingError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValidatedLearningSourceBindingError(
            f"{field} must be finite and inside [0,1]"
        )
    return result


def _status(value: Any, field: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValidatedLearningSourceBindingError(
            f"{field} must be one of {tuple(sorted(allowed))}"
        )
    return value


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
        raise ValidatedLearningSourceBindingError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _manifest_entry(axis: str) -> RegistryObservationSourceEntry:
    for entry in registry_observation_source_manifest().entries:
        if entry.axis == axis:
            return entry
    raise ValidatedLearningSourceBindingError(
        "learning-exploration axis missing from source manifest"
    )


def _raw_mapping(raw_values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {field: value for field, value in raw_values}


def _count_pressure(value: Any, field: str) -> float:
    count = _nonnegative_int(value, field)
    return float(count / (count + 1.0))


def _span_support(value: Any, field: str) -> float:
    span = _nonnegative_int(value, field)
    return float(span / (span + 4.0))


def validated_learning_raw_observation_digest(
    *,
    axis: str,
    logical_tick: int,
    observation_id: str,
    source_instance_id: str,
    source_snapshot_id: str,
    source_schema_version: str,
    source_integrity_digest: str,
    validation_trace_id: str,
    validation_input_digest: str,
    validation_integrity_digest: str,
    appraisal_trace_id: str,
    appraisal_input_digest: str,
    appraisal_integrity_digest: str,
    raw_values: tuple[tuple[str, Any], ...],
) -> str:
    """Return the canonical digest for one validated learning record."""

    if axis not in LEARNING_EXPLORATION_AXES:
        raise ValidatedLearningSourceBindingError("unsupported learning axis")
    _nonnegative_int(logical_tick, "logical_tick")
    for field, value in (
        ("observation_id", observation_id),
        ("source_instance_id", source_instance_id),
        ("source_snapshot_id", source_snapshot_id),
        ("source_schema_version", source_schema_version),
        ("validation_trace_id", validation_trace_id),
        ("appraisal_trace_id", appraisal_trace_id),
    ):
        _identifier(value, field)
    for field, value in (
        ("source_integrity_digest", source_integrity_digest),
        ("validation_input_digest", validation_input_digest),
        ("validation_integrity_digest", validation_integrity_digest),
        ("appraisal_input_digest", appraisal_input_digest),
        ("appraisal_integrity_digest", appraisal_integrity_digest),
    ):
        _digest_string(value, field)
    if appraisal_input_digest != validation_integrity_digest:
        raise ValidatedLearningSourceBindingError(
            "learning appraisal input must be the exact verified validation output"
        )
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
            "raw_social_feedback_source": False,
            "raw_values": [[field, value] for field, value in values],
            "schema_version": RAW_SCHEMA_VERSION,
            "source_family": SOURCE_FAMILY,
            "source_instance_id": source_instance_id,
            "source_integrity_digest": source_integrity_digest,
            "source_schema_version": source_schema_version,
            "source_snapshot_id": source_snapshot_id,
            "validation_input_digest": validation_input_digest,
            "validation_integrity_digest": validation_integrity_digest,
            "validation_method": VALIDATION_METHOD,
            "validation_outcome": VALIDATION_OUTCOME,
            "validation_schema_version": VALIDATION_SCHEMA_VERSION,
            "validation_trace_id": validation_trace_id,
            "validation_verified": True,
            "verification_method": VERIFICATION_METHOD,
        },
        "validated_learning_raw_observation",
    )


def _validate_raw_values(axis: str, raw: Mapping[str, Any]) -> None:
    if axis == "curiosity_drive":
        _unit(raw["exploration_cost"], "exploration_cost")
        _unit(raw["information_gain_estimate"], "information_gain_estimate")
        _unit(raw["relevance_score"], "relevance_score")
        _nonnegative_int(raw["sampling_window_ticks"], "sampling_window_ticks")
        _nonnegative_int(raw["unknown_count"], "unknown_count")
        return
    if axis == "novelty_seeking":
        if raw["appraisal_version"] != APPRAISAL_SCHEMA_VERSION:
            raise ValidatedLearningSourceBindingError(
                "appraisal_version must match the canonical learning appraisal schema"
            )
        for field in (
            "expected_information_gain",
            "novelty_score",
            "reversibility",
            "safety_score",
        ):
            _unit(raw[field], field)
        return
    if axis == "learning_pressure":
        _unit(raw["available_training_signal"], "available_training_signal")
        _unit(raw["competence_gap"], "competence_gap")
        _nonnegative_int(raw["error_recurrence"], "error_recurrence")
        _unit(raw["task_relevance"], "task_relevance")
        _status(raw["validation_status"], "validation_status", _VALIDATION_STATUS_ALLOWED)
        return
    if axis == "memory_consolidation_pressure":
        for field in (
            "causal_relevance",
            "emotional_relevance",
            "provenance_completeness",
            "salience_score",
        ):
            _unit(raw[field], field)
        _nonnegative_int(raw["recurrence_count"], "recurrence_count")
        return
    if axis == "prediction_error_pressure":
        _identifier(raw["model_version"], "model_version")
        _unit(raw["normalized_error"], "normalized_error")
        _digest_string(raw["observed_value_digest"], "observed_value_digest")
        _digest_string(raw["predicted_value_digest"], "predicted_value_digest")
        _status(raw["verification_status"], "verification_status", _VERIFICATION_STATUS_ALLOWED)
        return
    if axis == "competence_drive":
        for field in (
            "calibrated_error_rate",
            "learning_progress",
            "skill_gap",
            "success_rate",
        ):
            _unit(raw[field], field)
        _identifier(raw["evaluation_version"], "evaluation_version")
        return
    raise ValidatedLearningSourceBindingError("unsupported learning axis")


def _record_score(record: "ValidatedLearningRawRecord") -> float:
    raw = record.raw_mapping
    if record.axis == "curiosity_drive":
        values = (
            1.0 - raw["exploration_cost"],
            raw["information_gain_estimate"],
            raw["relevance_score"],
            _span_support(raw["sampling_window_ticks"], "sampling_window_ticks"),
            _count_pressure(raw["unknown_count"], "unknown_count"),
        )
    elif record.axis == "novelty_seeking":
        values = (
            raw["expected_information_gain"],
            raw["novelty_score"],
            raw["reversibility"],
            raw["safety_score"],
        )
    elif record.axis == "learning_pressure":
        values = (
            raw["available_training_signal"],
            raw["competence_gap"],
            _count_pressure(raw["error_recurrence"], "error_recurrence"),
            raw["task_relevance"],
        )
    elif record.axis == "memory_consolidation_pressure":
        values = (
            raw["causal_relevance"],
            raw["emotional_relevance"],
            raw["provenance_completeness"],
            _count_pressure(raw["recurrence_count"], "recurrence_count"),
            raw["salience_score"],
        )
    elif record.axis == "prediction_error_pressure":
        values = (raw["normalized_error"],)
    elif record.axis == "competence_drive":
        values = (
            raw["calibrated_error_rate"],
            raw["learning_progress"],
            raw["skill_gap"],
            1.0 - raw["success_rate"],
        )
    else:
        raise ValidatedLearningSourceBindingError("unsupported learning axis")
    return float(sum(values) / len(values))


@dataclass(frozen=True, slots=True)
class ValidatedLearningRawRecord:
    axis: str
    logical_tick: int
    observation_id: str
    source_instance_id: str
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    validation_trace_id: str
    validation_input_digest: str
    validation_integrity_digest: str
    appraisal_trace_id: str
    appraisal_input_digest: str
    appraisal_integrity_digest: str
    raw_observation_digest: str
    raw_values: tuple[tuple[str, Any], ...]
    acquisition_method: str = ACQUISITION_METHOD
    verification_method: str = VERIFICATION_METHOD
    validation_schema_version: str = VALIDATION_SCHEMA_VERSION
    validation_method: str = VALIDATION_METHOD
    validation_outcome: str = VALIDATION_OUTCOME
    appraisal_schema_version: str = APPRAISAL_SCHEMA_VERSION
    appraisal_method: str = APPRAISAL_METHOD
    appraisal_outcome: str = APPRAISAL_OUTCOME
    model_or_rule_version: str = RAW_MODEL_OR_RULE_VERSION
    source_family: str = SOURCE_FAMILY
    validation_verified: bool = True
    appraisal_verified: bool = True
    raw_social_feedback_source: bool = False
    hardware_direct_input: bool = False
    synthetic: bool = False
    proposal_only: bool = False
    registry_owner_source: bool = False
    runtime_polled: bool = False
    learning_mutation_performed: bool = False
    memory_write_performed: bool = False
    schema_version: str = RAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.axis not in LEARNING_EXPLORATION_AXES:
            raise ValidatedLearningSourceBindingError("unsupported learning axis")
        _nonnegative_int(self.logical_tick, "logical_tick")
        for field in (
            "observation_id",
            "source_instance_id",
            "source_snapshot_id",
            "source_schema_version",
            "validation_trace_id",
            "appraisal_trace_id",
        ):
            _identifier(getattr(self, field), field)
        for field in (
            "source_integrity_digest",
            "validation_input_digest",
            "validation_integrity_digest",
            "appraisal_input_digest",
            "appraisal_integrity_digest",
            "raw_observation_digest",
        ):
            _digest_string(getattr(self, field), field)
        expected_provenance = {
            "acquisition_method": ACQUISITION_METHOD,
            "verification_method": VERIFICATION_METHOD,
            "validation_schema_version": VALIDATION_SCHEMA_VERSION,
            "validation_method": VALIDATION_METHOD,
            "validation_outcome": VALIDATION_OUTCOME,
            "appraisal_schema_version": APPRAISAL_SCHEMA_VERSION,
            "appraisal_method": APPRAISAL_METHOD,
            "appraisal_outcome": APPRAISAL_OUTCOME,
            "model_or_rule_version": RAW_MODEL_OR_RULE_VERSION,
            "source_family": SOURCE_FAMILY,
        }
        for field, expected in expected_provenance.items():
            if getattr(self, field) != expected:
                raise ValidatedLearningSourceBindingError(
                    f"raw record {field} does not match the canonical learning provenance contract"
                )
        if self.validation_verified is not True or self.appraisal_verified is not True:
            raise ValidatedLearningSourceBindingError(
                "learning evidence requires exact validation and appraisal verification"
            )
        if self.appraisal_input_digest != self.validation_integrity_digest:
            raise ValidatedLearningSourceBindingError(
                "learning appraisal input must be the exact verified validation output"
            )
        if any(
            (
                self.raw_social_feedback_source,
                self.hardware_direct_input,
                self.synthetic,
                self.proposal_only,
                self.registry_owner_source,
                self.runtime_polled,
                self.learning_mutation_performed,
                self.memory_write_performed,
            )
        ):
            raise ValidatedLearningSourceBindingError(
                "learning evidence cannot use raw social feedback, direct hardware, synthetic, proposal-only, circular, runtime-polled, mutating, or memory-writing input"
            )
        values = tuple(self.raw_values)
        fields = tuple(field for field, _ in values)
        expected_fields = _manifest_entry(self.axis).required_raw_fields
        if fields != expected_fields or len(set(fields)) != len(fields):
            raise ValidatedLearningSourceBindingError(
                "raw record fields do not match the canonical learning source plan"
            )
        _validate_raw_values(self.axis, _raw_mapping(values))
        expected_digest = validated_learning_raw_observation_digest(
            axis=self.axis,
            logical_tick=self.logical_tick,
            observation_id=self.observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version,
            source_integrity_digest=self.source_integrity_digest,
            validation_trace_id=self.validation_trace_id,
            validation_input_digest=self.validation_input_digest,
            validation_integrity_digest=self.validation_integrity_digest,
            appraisal_trace_id=self.appraisal_trace_id,
            appraisal_input_digest=self.appraisal_input_digest,
            appraisal_integrity_digest=self.appraisal_integrity_digest,
            raw_values=values,
        )
        if self.raw_observation_digest != expected_digest:
            raise ValidatedLearningSourceBindingError(
                "raw observation digest does not match identity, time, validation/appraisal provenance, and values"
            )
        if self.schema_version != RAW_SCHEMA_VERSION:
            raise ValidatedLearningSourceBindingError(
                "unsupported validated-learning raw schema"
            )
        object.__setattr__(self, "raw_values", values)

    @property
    def raw_mapping(self) -> dict[str, Any]:
        return _raw_mapping(self.raw_values)

    @property
    def recalculated_raw_observation_digest(self) -> str:
        return validated_learning_raw_observation_digest(
            axis=self.axis,
            logical_tick=self.logical_tick,
            observation_id=self.observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version,
            source_integrity_digest=self.source_integrity_digest,
            validation_trace_id=self.validation_trace_id,
            validation_input_digest=self.validation_input_digest,
            validation_integrity_digest=self.validation_integrity_digest,
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
            "learning_mutation_performed": self.learning_mutation_performed,
            "logical_tick": self.logical_tick,
            "memory_write_performed": self.memory_write_performed,
            "model_or_rule_version": self.model_or_rule_version,
            "observation_id": self.observation_id,
            "proposal_only": self.proposal_only,
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
            "validation_input_digest": self.validation_input_digest,
            "validation_integrity_digest": self.validation_integrity_digest,
            "validation_method": self.validation_method,
            "validation_outcome": self.validation_outcome,
            "validation_schema_version": self.validation_schema_version,
            "validation_trace_id": self.validation_trace_id,
            "validation_verified": self.validation_verified,
            "verification_method": self.verification_method,
        }


@dataclass(frozen=True, slots=True)
class ValidatedLearningSourceBinding:
    axis: str
    source_contract_id: str
    binding_id: str
    raw_schema_version: str
    validation_schema_version: str
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
    quarantine_required_for_social_feedback: bool = True
    hardware_direct_input_allowed: bool = False
    production_capture_present: bool = False
    runtime_capture_installed: bool = False
    persistence_accessed: bool = False
    event_append_performed: bool = False
    learning_mutation_performed: bool = False
    memory_write_performed: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        if self.axis not in LEARNING_EXPLORATION_AXES:
            raise ValidatedLearningSourceBindingError("unsupported binding axis")
        entry = _manifest_entry(self.axis)
        expected = {
            "source_contract_id": entry.source_contract_id,
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "validation_schema_version": VALIDATION_SCHEMA_VERSION,
            "appraisal_schema_version": APPRAISAL_SCHEMA_VERSION,
            "required_raw_fields": entry.required_raw_fields,
            "minimum_raw_record_count": entry.minimum_raw_record_count,
            "minimum_logical_span_ticks": entry.minimum_logical_span_ticks,
            "derivation_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:mean.v1",
            "confidence_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:coverage-variance.v1",
            "appraisal_required": True,
            "quarantine_required_for_social_feedback": True,
            "hardware_direct_input_allowed": False,
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise ValidatedLearningSourceBindingError(
                    f"binding {field} does not match the canonical learning source plan"
                )
        _identifier(self.binding_id, "binding_id")
        if self.schema_version != BINDING_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise ValidatedLearningSourceBindingError(
                "validated learning binding must use the exact shadow-only schema"
            )
        if self.binding_implemented is not True:
            raise ValidatedLearningSourceBindingError(
                "binding implementation flag must be true"
            )
        if any(
            (
                self.production_capture_present,
                self.runtime_capture_installed,
                self.persistence_accessed,
                self.event_append_performed,
                self.learning_mutation_performed,
                self.memory_write_performed,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise ValidatedLearningSourceBindingError(
                "binding cannot claim capture, mutation, persistence, window, or authority"
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
            "learning_mutation_performed": self.learning_mutation_performed,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "memory_write_performed": self.memory_write_performed,
            "minimum_logical_span_ticks": self.minimum_logical_span_ticks,
            "minimum_raw_record_count": self.minimum_raw_record_count,
            "observation_window_started": self.observation_window_started,
            "persistence_accessed": self.persistence_accessed,
            "production_capture_present": self.production_capture_present,
            "quarantine_required_for_social_feedback": self.quarantine_required_for_social_feedback,
            "raw_schema_version": self.raw_schema_version,
            "required_raw_fields": list(self.required_raw_fields),
            "runtime_capture_installed": self.runtime_capture_installed,
            "schema_version": self.schema_version,
            "source_contract_id": self.source_contract_id,
            "validation_schema_version": self.validation_schema_version,
        }

    @property
    def binding_digest(self) -> str:
        return _digest(self.to_mapping(), "validated_learning_source_binding")


@dataclass(frozen=True, slots=True)
class ValidatedLearningSourceBindingSet:
    bindings: tuple[ValidatedLearningSourceBinding, ...]
    schema_version: str = BINDING_SET_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    total_bound_axis_count: int = TOTAL_BOUND_AXIS_COUNT
    remaining_axis_count: int = 12
    production_capture_present: bool = False
    learning_mutation_performed: bool = False
    memory_write_performed: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        if any(type(item) is not ValidatedLearningSourceBinding for item in bindings):
            raise ValidatedLearningSourceBindingError(
                "binding set requires exact immutable binding types"
            )
        if len(bindings) != 6 or tuple(item.axis for item in bindings) != LEARNING_EXPLORATION_AXES:
            raise ValidatedLearningSourceBindingError(
                "validated learning binding set must preserve exact six-axis order"
            )
        if (
            self.schema_version != BINDING_SET_SCHEMA_VERSION
            or self.authority != SHADOW_AUTHORITY
            or self.total_bound_axis_count != TOTAL_BOUND_AXIS_COUNT
            or self.remaining_axis_count != 12
        ):
            raise ValidatedLearningSourceBindingError(
                "unsupported validated-learning binding-set contract"
            )
        if any(
            (
                self.production_capture_present,
                self.learning_mutation_performed,
                self.memory_write_performed,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise ValidatedLearningSourceBindingError(
                "binding set cannot claim capture, mutation, window, or authority"
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
            "learning_mutation_performed": self.learning_mutation_performed,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "memory_write_performed": self.memory_write_performed,
            "observation_window_started": self.observation_window_started,
            "production_capture_present": self.production_capture_present,
            "remaining_axis_count": self.remaining_axis_count,
            "schema_version": self.schema_version,
            "total_bound_axis_count": self.total_bound_axis_count,
        }

    @property
    def binding_set_digest(self) -> str:
        return _digest(self.to_mapping(), "validated_learning_source_binding_set")


def validated_learning_source_bindings() -> ValidatedLearningSourceBindingSet:
    bindings = []
    for axis in LEARNING_EXPLORATION_AXES:
        entry = _manifest_entry(axis)
        bindings.append(
            ValidatedLearningSourceBinding(
                axis=axis,
                source_contract_id=entry.source_contract_id,
                binding_id=f"eve:m3-b:validated-learning-binding:{axis}:v1",
                raw_schema_version=RAW_SCHEMA_VERSION,
                validation_schema_version=VALIDATION_SCHEMA_VERSION,
                appraisal_schema_version=APPRAISAL_SCHEMA_VERSION,
                required_raw_fields=entry.required_raw_fields,
                minimum_raw_record_count=entry.minimum_raw_record_count,
                minimum_logical_span_ticks=entry.minimum_logical_span_ticks,
                derivation_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1",
                confidence_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:coverage-variance.v1",
            )
        )
    return ValidatedLearningSourceBindingSet(bindings=tuple(bindings))


def derive_validated_learning_axis_evidence(
    records: Sequence[ValidatedLearningRawRecord],
) -> RegistryAxisPositiveConfidenceEvidence:
    """Derive one detached evidence record from validated learning traces."""

    items = tuple(records)
    if not items or any(type(item) is not ValidatedLearningRawRecord for item in items):
        raise ValidatedLearningSourceBindingError(
            "records must contain exact immutable validated-learning raw records"
        )
    axis = items[0].axis
    if any(item.axis != axis for item in items):
        raise ValidatedLearningSourceBindingError("records cannot mix axes")
    ticks = tuple(item.logical_tick for item in items)
    if ticks != tuple(sorted(ticks)):
        raise ValidatedLearningSourceBindingError("record ticks must be sorted")
    if len(set(ticks)) != len(ticks):
        raise ValidatedLearningSourceBindingError("record ticks must be unique")
    for field in (
        "observation_id",
        "source_snapshot_id",
        "validation_trace_id",
        "appraisal_trace_id",
    ):
        if len({getattr(item, field) for item in items}) != len(items):
            raise ValidatedLearningSourceBindingError(
                f"record {field} values must be unique"
            )
    for field in (
        "source_instance_id",
        "source_schema_version",
        "acquisition_method",
        "verification_method",
        "validation_schema_version",
        "validation_method",
        "validation_outcome",
        "appraisal_schema_version",
        "appraisal_method",
        "appraisal_outcome",
        "model_or_rule_version",
        "source_family",
    ):
        if len({getattr(item, field) for item in items}) != 1:
            raise ValidatedLearningSourceBindingError(
                f"records must share one {field}"
            )
    binding = next(
        item
        for item in validated_learning_source_bindings().bindings
        if item.axis == axis
    )
    span = items[-1].logical_tick - items[0].logical_tick
    if len(items) < binding.minimum_raw_record_count:
        raise ValidatedLearningSourceBindingError("insufficient raw record count")
    if span < binding.minimum_logical_span_ticks:
        raise ValidatedLearningSourceBindingError(
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
        "validated_learning_raw_bundle",
    )
    source_integrity_digest = _digest(
        {
            "binding_digest": binding.binding_digest,
            "raw_bundle_digest": raw_bundle_digest,
            "source_instance_id": items[0].source_instance_id,
        },
        "validated_learning_source_integrity",
    )
    return RegistryAxisPositiveConfidenceEvidence(
        axis=axis,
        value=value,
        confidence=confidence,
        observed_tick=items[-1].logical_tick,
        observation_id=f"validated-learning:{axis}:{raw_bundle_digest[:24]}",
        source_family=SOURCE_FAMILY,
        source_instance_id=items[0].source_instance_id,
        source_snapshot_id=(
            f"validated-learning:{axis}:{items[0].logical_tick}:"
            f"{items[-1].logical_tick}:{raw_bundle_digest[:16]}"
        ),
        source_schema_version=RAW_SCHEMA_VERSION,
        source_integrity_digest=source_integrity_digest,
        raw_observation_digest=raw_bundle_digest,
        acquisition_method=ACQUISITION_METHOD,
        verification_method=VERIFICATION_METHOD,
        model_or_rule_version=binding.derivation_rule_id,
    )
