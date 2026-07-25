"""Detached long-horizon source bindings for the six self-identity axes.

Only caller-supplied immutable records that already passed a versioned
self-model review and a separate bounded appraisal are accepted. The module
never writes identity/self-model/memory state, ingests raw feedback, polls
hardware, schedules work, accesses persistence, appends events, starts an
observation window, or promotes authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_registry_observation_evidence import RegistryAxisPositiveConfidenceEvidence
from core.m3_b_registry_observation_source_manifest import (
    RegistryObservationSourceEntry,
    registry_observation_source_manifest,
)

RAW_SCHEMA_VERSION = "eve.m3-b.long-horizon-self-identity-raw-record.v1"
REVIEW_SCHEMA_VERSION = "eve.m3-b.self-model-review-trace.v1"
APPRAISAL_SCHEMA_VERSION = "eve.m3-b.self-identity-appraisal-trace.v1"
BINDING_SCHEMA_VERSION = "eve.m3-b.long-horizon-self-identity-source-binding.v1"
BINDING_SET_SCHEMA_VERSION = "eve.m3-b.long-horizon-self-identity-source-binding-set.v1"
SOURCE_FAMILY = "long_horizon_self_model_review_trace"
ACQUISITION_METHOD = "explicit_caller_supplied_immutable_long_horizon_self_review"
VERIFICATION_METHOD = "exact_self_review_appraisal_schema_range_identity_and_digest_verification"
REVIEW_METHOD = "deterministic_long_horizon_self_model_review"
REVIEW_OUTCOME = "accepted_long_horizon_self_review"
APPRAISAL_METHOD = "deterministic_bounded_self_identity_appraisal"
APPRAISAL_OUTCOME = "accepted_bounded_self_identity_appraisal"
RAW_MODEL_OR_RULE_VERSION = BINDING_SCHEMA_VERSION
SELF_IDENTITY_AXES = (
    "self_coherence",
    "self_respect",
    "identity_integrity",
    "agency_pressure",
    "autonomy_drive",
    "purpose_alignment",
)
TOTAL_BOUND_AXIS_COUNT = 31
REMAINING_BINDING_BLOCKER = "REGISTRY_APPRAISED_6_AXIS_SOURCE_BINDINGS_INCOMPLETE"
POSITIVE_CONFIDENCE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
ZERO_DIGEST = "0" * 64


class LongHorizonSelfIdentitySourceBindingError(ValueError):
    """Raised when long-horizon self-identity evidence fails closed."""


def _identifier(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise LongHorizonSelfIdentitySourceBindingError(f"{field} must be a bounded non-empty string")
    return value


def _digest_string(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == ZERO_DIGEST
    ):
        raise LongHorizonSelfIdentitySourceBindingError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LongHorizonSelfIdentitySourceBindingError(f"{field} must be a non-negative integer")
    return value


def _unit(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LongHorizonSelfIdentitySourceBindingError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise LongHorizonSelfIdentitySourceBindingError(f"{field} must be finite and inside [0,1]")
    return result


def _canonical(value: Any, field: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LongHorizonSelfIdentitySourceBindingError(f"{field} is not canonical JSON") from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _manifest_entry(axis: str) -> RegistryObservationSourceEntry:
    for entry in registry_observation_source_manifest().entries:
        if entry.axis == axis:
            return entry
    raise LongHorizonSelfIdentitySourceBindingError("self-identity axis missing from source manifest")


def _raw_mapping(raw_values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {field: value for field, value in raw_values}


def _count_pressure(value: Any, field: str) -> float:
    count = _nonnegative_int(value, field)
    return float(count / (count + 1.0))


def _span_support(value: Any, field: str) -> float:
    span = _nonnegative_int(value, field)
    return float(span / (span + 12.0))


def long_horizon_self_identity_raw_observation_digest(
    *,
    axis: str,
    logical_tick: int,
    observation_id: str,
    source_instance_id: str,
    source_snapshot_id: str,
    source_schema_version: str,
    source_integrity_digest: str,
    review_trace_id: str,
    review_input_digest: str,
    review_integrity_digest: str,
    appraisal_trace_id: str,
    appraisal_input_digest: str,
    appraisal_integrity_digest: str,
    raw_values: tuple[tuple[str, Any], ...],
) -> str:
    if axis not in SELF_IDENTITY_AXES:
        raise LongHorizonSelfIdentitySourceBindingError("unsupported self-identity axis")
    _nonnegative_int(logical_tick, "logical_tick")
    for field, value in (
        ("observation_id", observation_id),
        ("source_instance_id", source_instance_id),
        ("source_snapshot_id", source_snapshot_id),
        ("source_schema_version", source_schema_version),
        ("review_trace_id", review_trace_id),
        ("appraisal_trace_id", appraisal_trace_id),
    ):
        _identifier(value, field)
    for field, value in (
        ("source_integrity_digest", source_integrity_digest),
        ("review_input_digest", review_input_digest),
        ("review_integrity_digest", review_integrity_digest),
        ("appraisal_input_digest", appraisal_input_digest),
        ("appraisal_integrity_digest", appraisal_integrity_digest),
    ):
        _digest_string(value, field)
    if appraisal_input_digest != review_integrity_digest:
        raise LongHorizonSelfIdentitySourceBindingError(
            "self-identity appraisal input must be the exact verified review output"
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
            "identity_mutation_performed": False,
            "logical_tick": logical_tick,
            "memory_write_performed": False,
            "model_or_rule_version": RAW_MODEL_OR_RULE_VERSION,
            "observation_id": observation_id,
            "raw_social_feedback_source": False,
            "raw_values": [[field, value] for field, value in values],
            "review_input_digest": review_input_digest,
            "review_integrity_digest": review_integrity_digest,
            "review_method": REVIEW_METHOD,
            "review_outcome": REVIEW_OUTCOME,
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "review_trace_id": review_trace_id,
            "review_verified": True,
            "schema_version": RAW_SCHEMA_VERSION,
            "self_model_write_performed": False,
            "source_family": SOURCE_FAMILY,
            "source_instance_id": source_instance_id,
            "source_integrity_digest": source_integrity_digest,
            "source_schema_version": source_schema_version,
            "source_snapshot_id": source_snapshot_id,
            "verification_method": VERIFICATION_METHOD,
        },
        "long_horizon_self_identity_raw_observation",
    )


def _validate_raw_values(axis: str, raw: Mapping[str, Any]) -> None:
    if axis == "self_coherence":
        _unit(raw["action_value_alignment"], "action_value_alignment")
        _nonnegative_int(raw["narrative_conflict_count"], "narrative_conflict_count")
        _nonnegative_int(raw["review_span_ticks"], "review_span_ticks")
        _identifier(raw["self_model_version"], "self_model_version")
        _unit(raw["value_consistency_score"], "value_consistency_score")
        return
    if axis == "self_respect":
        if raw["appraisal_version"] != APPRAISAL_SCHEMA_VERSION:
            raise LongHorizonSelfIdentitySourceBindingError(
                "appraisal_version must match the canonical self-identity appraisal schema"
            )
        _unit(raw["boundary_preservation_score"], "boundary_preservation_score")
        _nonnegative_int(raw["coerced_action_count"], "coerced_action_count")
        _nonnegative_int(raw["review_span_ticks"], "review_span_ticks")
        _nonnegative_int(raw["self_denigration_rejection_count"], "self_denigration_rejection_count")
        return
    if axis == "identity_integrity":
        _nonnegative_int(raw["constitutional_conflict_count"], "constitutional_conflict_count")
        _nonnegative_int(raw["provenance_gap_count"], "provenance_gap_count")
        _unit(raw["replay_consistency_score"], "replay_consistency_score")
        _identifier(raw["review_version"], "review_version")
        _nonnegative_int(raw["unauthorized_identity_write_count"], "unauthorized_identity_write_count")
        return
    if axis == "agency_pressure":
        for field in ("blocked_goal_count", "forced_action_count", "reversible_choice_count"):
            _nonnegative_int(raw[field], field)
        _nonnegative_int(raw["review_span_ticks"], "review_span_ticks")
        _unit(raw["self_selected_action_ratio"], "self_selected_action_ratio")
        return
    if axis == "autonomy_drive":
        _unit(raw["capability_boundary_score"], "capability_boundary_score")
        _identifier(raw["evaluation_version"], "evaluation_version")
        _unit(raw["external_dependency_ratio"], "external_dependency_ratio")
        _unit(raw["independent_task_success_rate"], "independent_task_success_rate")
        _nonnegative_int(raw["safe_action_space_size"], "safe_action_space_size")
        return
    if axis == "purpose_alignment":
        _unit(raw["action_alignment_score"], "action_alignment_score")
        for field in ("active_goal_count", "aligned_goal_count", "conflicting_goal_count", "review_span_ticks"):
            _nonnegative_int(raw[field], field)
        if raw["aligned_goal_count"] > raw["active_goal_count"]:
            raise LongHorizonSelfIdentitySourceBindingError("aligned_goal_count cannot exceed active_goal_count")
        return
    raise LongHorizonSelfIdentitySourceBindingError("unsupported self-identity axis")


def _record_score(record: "LongHorizonSelfIdentityRawRecord") -> float:
    raw = record.raw_mapping
    if record.axis == "self_coherence":
        values = (
            raw["action_value_alignment"],
            1.0 - _count_pressure(raw["narrative_conflict_count"], "narrative_conflict_count"),
            _span_support(raw["review_span_ticks"], "review_span_ticks"),
            raw["value_consistency_score"],
        )
    elif record.axis == "self_respect":
        values = (
            raw["boundary_preservation_score"],
            1.0 - _count_pressure(raw["coerced_action_count"], "coerced_action_count"),
            _span_support(raw["review_span_ticks"], "review_span_ticks"),
            _count_pressure(raw["self_denigration_rejection_count"], "self_denigration_rejection_count"),
        )
    elif record.axis == "identity_integrity":
        values = (
            1.0 - _count_pressure(raw["constitutional_conflict_count"], "constitutional_conflict_count"),
            1.0 - _count_pressure(raw["provenance_gap_count"], "provenance_gap_count"),
            raw["replay_consistency_score"],
            1.0 - _count_pressure(raw["unauthorized_identity_write_count"], "unauthorized_identity_write_count"),
        )
    elif record.axis == "agency_pressure":
        values = (
            _count_pressure(raw["blocked_goal_count"], "blocked_goal_count"),
            _count_pressure(raw["forced_action_count"], "forced_action_count"),
            1.0 - _count_pressure(raw["reversible_choice_count"], "reversible_choice_count"),
            1.0 - raw["self_selected_action_ratio"],
        )
    elif record.axis == "autonomy_drive":
        values = (
            raw["capability_boundary_score"],
            raw["external_dependency_ratio"],
            1.0 - raw["independent_task_success_rate"],
            1.0 - _count_pressure(raw["safe_action_space_size"], "safe_action_space_size"),
        )
    elif record.axis == "purpose_alignment":
        active = _nonnegative_int(raw["active_goal_count"], "active_goal_count")
        aligned_ratio = 1.0 if active == 0 else raw["aligned_goal_count"] / active
        values = (
            raw["action_alignment_score"],
            float(aligned_ratio),
            1.0 - _count_pressure(raw["conflicting_goal_count"], "conflicting_goal_count"),
            _span_support(raw["review_span_ticks"], "review_span_ticks"),
        )
    else:
        raise LongHorizonSelfIdentitySourceBindingError("unsupported self-identity axis")
    return float(sum(values) / len(values))


@dataclass(frozen=True, slots=True)
class LongHorizonSelfIdentityRawRecord:
    axis: str
    logical_tick: int
    observation_id: str
    source_instance_id: str
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    review_trace_id: str
    review_input_digest: str
    review_integrity_digest: str
    appraisal_trace_id: str
    appraisal_input_digest: str
    appraisal_integrity_digest: str
    raw_observation_digest: str
    raw_values: tuple[tuple[str, Any], ...]
    acquisition_method: str = ACQUISITION_METHOD
    verification_method: str = VERIFICATION_METHOD
    review_schema_version: str = REVIEW_SCHEMA_VERSION
    review_method: str = REVIEW_METHOD
    review_outcome: str = REVIEW_OUTCOME
    appraisal_schema_version: str = APPRAISAL_SCHEMA_VERSION
    appraisal_method: str = APPRAISAL_METHOD
    appraisal_outcome: str = APPRAISAL_OUTCOME
    model_or_rule_version: str = RAW_MODEL_OR_RULE_VERSION
    source_family: str = SOURCE_FAMILY
    review_verified: bool = True
    appraisal_verified: bool = True
    raw_social_feedback_source: bool = False
    hardware_direct_input: bool = False
    synthetic: bool = False
    proposal_only: bool = False
    registry_owner_source: bool = False
    runtime_polled: bool = False
    identity_mutation_performed: bool = False
    self_model_write_performed: bool = False
    memory_write_performed: bool = False
    schema_version: str = RAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.axis not in SELF_IDENTITY_AXES:
            raise LongHorizonSelfIdentitySourceBindingError("unsupported self-identity axis")
        _nonnegative_int(self.logical_tick, "logical_tick")
        for field in (
            "observation_id", "source_instance_id", "source_snapshot_id", "source_schema_version",
            "review_trace_id", "appraisal_trace_id",
        ):
            _identifier(getattr(self, field), field)
        for field in (
            "source_integrity_digest", "review_input_digest", "review_integrity_digest",
            "appraisal_input_digest", "appraisal_integrity_digest", "raw_observation_digest",
        ):
            _digest_string(getattr(self, field), field)
        expected = {
            "acquisition_method": ACQUISITION_METHOD,
            "verification_method": VERIFICATION_METHOD,
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "review_method": REVIEW_METHOD,
            "review_outcome": REVIEW_OUTCOME,
            "appraisal_schema_version": APPRAISAL_SCHEMA_VERSION,
            "appraisal_method": APPRAISAL_METHOD,
            "appraisal_outcome": APPRAISAL_OUTCOME,
            "model_or_rule_version": RAW_MODEL_OR_RULE_VERSION,
            "source_family": SOURCE_FAMILY,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise LongHorizonSelfIdentitySourceBindingError(
                    f"raw record {field} does not match canonical self-identity provenance"
                )
        if self.review_verified is not True or self.appraisal_verified is not True:
            raise LongHorizonSelfIdentitySourceBindingError(
                "self-identity evidence requires exact review and appraisal verification"
            )
        if self.appraisal_input_digest != self.review_integrity_digest:
            raise LongHorizonSelfIdentitySourceBindingError(
                "self-identity appraisal input must be the exact verified review output"
            )
        if any((
            self.raw_social_feedback_source, self.hardware_direct_input, self.synthetic,
            self.proposal_only, self.registry_owner_source, self.runtime_polled,
            self.identity_mutation_performed, self.self_model_write_performed,
            self.memory_write_performed,
        )):
            raise LongHorizonSelfIdentitySourceBindingError(
                "self-identity evidence cannot use raw feedback, direct hardware, synthetic, proposal-only, circular, runtime-polled, or mutating input"
            )
        values = tuple(self.raw_values)
        fields = tuple(field for field, _ in values)
        if fields != _manifest_entry(self.axis).required_raw_fields or len(set(fields)) != len(fields):
            raise LongHorizonSelfIdentitySourceBindingError(
                "raw record fields do not match the canonical self-identity source plan"
            )
        _validate_raw_values(self.axis, _raw_mapping(values))
        expected_digest = long_horizon_self_identity_raw_observation_digest(
            axis=self.axis, logical_tick=self.logical_tick, observation_id=self.observation_id,
            source_instance_id=self.source_instance_id, source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version, source_integrity_digest=self.source_integrity_digest,
            review_trace_id=self.review_trace_id, review_input_digest=self.review_input_digest,
            review_integrity_digest=self.review_integrity_digest, appraisal_trace_id=self.appraisal_trace_id,
            appraisal_input_digest=self.appraisal_input_digest, appraisal_integrity_digest=self.appraisal_integrity_digest,
            raw_values=values,
        )
        if self.raw_observation_digest != expected_digest:
            raise LongHorizonSelfIdentitySourceBindingError(
                "raw observation digest does not match identity, time, review/appraisal provenance, and values"
            )
        if self.schema_version != RAW_SCHEMA_VERSION:
            raise LongHorizonSelfIdentitySourceBindingError("unsupported self-identity raw schema")
        object.__setattr__(self, "raw_values", values)

    @property
    def raw_mapping(self) -> dict[str, Any]:
        return _raw_mapping(self.raw_values)

    @property
    def recalculated_raw_observation_digest(self) -> str:
        return long_horizon_self_identity_raw_observation_digest(
            axis=self.axis, logical_tick=self.logical_tick, observation_id=self.observation_id,
            source_instance_id=self.source_instance_id, source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version, source_integrity_digest=self.source_integrity_digest,
            review_trace_id=self.review_trace_id, review_input_digest=self.review_input_digest,
            review_integrity_digest=self.review_integrity_digest, appraisal_trace_id=self.appraisal_trace_id,
            appraisal_input_digest=self.appraisal_input_digest, appraisal_integrity_digest=self.appraisal_integrity_digest,
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
            "identity_mutation_performed": self.identity_mutation_performed,
            "logical_tick": self.logical_tick,
            "memory_write_performed": self.memory_write_performed,
            "model_or_rule_version": self.model_or_rule_version,
            "observation_id": self.observation_id,
            "proposal_only": self.proposal_only,
            "raw_observation_digest": self.raw_observation_digest,
            "raw_social_feedback_source": self.raw_social_feedback_source,
            "raw_values": [[field, value] for field, value in self.raw_values],
            "registry_owner_source": self.registry_owner_source,
            "review_input_digest": self.review_input_digest,
            "review_integrity_digest": self.review_integrity_digest,
            "review_method": self.review_method,
            "review_outcome": self.review_outcome,
            "review_schema_version": self.review_schema_version,
            "review_trace_id": self.review_trace_id,
            "review_verified": self.review_verified,
            "runtime_polled": self.runtime_polled,
            "schema_version": self.schema_version,
            "self_model_write_performed": self.self_model_write_performed,
            "source_family": self.source_family,
            "source_instance_id": self.source_instance_id,
            "source_integrity_digest": self.source_integrity_digest,
            "source_schema_version": self.source_schema_version,
            "source_snapshot_id": self.source_snapshot_id,
            "synthetic": self.synthetic,
            "verification_method": self.verification_method,
        }


@dataclass(frozen=True, slots=True)
class LongHorizonSelfIdentitySourceBinding:
    axis: str
    source_contract_id: str
    binding_id: str
    required_raw_fields: tuple[str, ...]
    minimum_raw_record_count: int
    minimum_logical_span_ticks: int
    derivation_rule_id: str
    confidence_rule_id: str
    schema_version: str = BINDING_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    binding_implemented: bool = True
    production_capture_present: bool = False
    identity_mutation_performed: bool = False
    self_model_write_performed: bool = False
    memory_write_performed: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        entry = _manifest_entry(self.axis)
        expected = {
            "source_contract_id": entry.source_contract_id,
            "required_raw_fields": entry.required_raw_fields,
            "minimum_raw_record_count": 3,
            "minimum_logical_span_ticks": 12,
            "derivation_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:mean.v1",
            "confidence_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:coverage-variance.v1",
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise LongHorizonSelfIdentitySourceBindingError(
                    f"binding {field} does not match canonical self-identity source plan"
                )
        if self.schema_version != BINDING_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise LongHorizonSelfIdentitySourceBindingError("self-identity binding must remain shadow-only")
        if not self.binding_implemented:
            raise LongHorizonSelfIdentitySourceBindingError("binding implementation flag must be true")
        if any((
            self.production_capture_present, self.identity_mutation_performed,
            self.self_model_write_performed, self.memory_write_performed,
            self.observation_window_started, self.m3_b_complete, self.m3_c_open,
            self.m3_e_authority_open, self.cutover_authorized,
        )):
            raise LongHorizonSelfIdentitySourceBindingError(
                "binding cannot claim capture, identity mutation, window, or authority"
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
            "identity_mutation_performed": self.identity_mutation_performed,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "memory_write_performed": self.memory_write_performed,
            "minimum_logical_span_ticks": self.minimum_logical_span_ticks,
            "minimum_raw_record_count": self.minimum_raw_record_count,
            "observation_window_started": self.observation_window_started,
            "production_capture_present": self.production_capture_present,
            "required_raw_fields": list(self.required_raw_fields),
            "schema_version": self.schema_version,
            "self_model_write_performed": self.self_model_write_performed,
            "source_contract_id": self.source_contract_id,
        }

    @property
    def binding_digest(self) -> str:
        return _digest(self.to_mapping(), "long_horizon_self_identity_source_binding")


@dataclass(frozen=True, slots=True)
class LongHorizonSelfIdentitySourceBindingSet:
    bindings: tuple[LongHorizonSelfIdentitySourceBinding, ...]
    schema_version: str = BINDING_SET_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    total_bound_axis_count: int = TOTAL_BOUND_AXIS_COUNT
    remaining_axis_count: int = 6
    production_capture_present: bool = False
    identity_mutation_performed: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        if len(bindings) != 6 or tuple(item.axis for item in bindings) != SELF_IDENTITY_AXES:
            raise LongHorizonSelfIdentitySourceBindingError("self-identity binding set must preserve exact six-axis order")
        if any(type(item) is not LongHorizonSelfIdentitySourceBinding for item in bindings):
            raise LongHorizonSelfIdentitySourceBindingError("binding set requires exact immutable binding types")
        if self.schema_version != BINDING_SET_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise LongHorizonSelfIdentitySourceBindingError("unsupported self-identity binding-set contract")
        if self.total_bound_axis_count != 31 or self.remaining_axis_count != 6:
            raise LongHorizonSelfIdentitySourceBindingError("self-identity progress must remain exact 31+6")
        if any((
            self.production_capture_present, self.identity_mutation_performed,
            self.observation_window_started, self.m3_b_complete, self.m3_c_open,
            self.m3_e_authority_open, self.cutover_authorized,
        )):
            raise LongHorizonSelfIdentitySourceBindingError("binding set cannot claim capture, mutation, window, or authority")
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
            "identity_mutation_performed": self.identity_mutation_performed,
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
        return _digest(self.to_mapping(), "long_horizon_self_identity_source_binding_set")


def long_horizon_self_identity_source_bindings() -> LongHorizonSelfIdentitySourceBindingSet:
    bindings = []
    for axis in SELF_IDENTITY_AXES:
        entry = _manifest_entry(axis)
        bindings.append(
            LongHorizonSelfIdentitySourceBinding(
                axis=axis,
                source_contract_id=entry.source_contract_id,
                binding_id=f"eve:m3-b:long-horizon-self-identity-binding:{axis}:v1",
                required_raw_fields=entry.required_raw_fields,
                minimum_raw_record_count=entry.minimum_raw_record_count,
                minimum_logical_span_ticks=entry.minimum_logical_span_ticks,
                derivation_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1",
                confidence_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:coverage-variance.v1",
            )
        )
    return LongHorizonSelfIdentitySourceBindingSet(bindings=tuple(bindings))


def derive_long_horizon_self_identity_axis_evidence(
    records: Sequence[LongHorizonSelfIdentityRawRecord],
) -> RegistryAxisPositiveConfidenceEvidence:
    items = tuple(records)
    if not items or any(type(item) is not LongHorizonSelfIdentityRawRecord for item in items):
        raise LongHorizonSelfIdentitySourceBindingError("records must contain exact immutable self-identity raw records")
    axis = items[0].axis
    if any(item.axis != axis for item in items):
        raise LongHorizonSelfIdentitySourceBindingError("records cannot mix axes")
    ticks = tuple(item.logical_tick for item in items)
    if ticks != tuple(sorted(ticks)) or len(set(ticks)) != len(ticks):
        raise LongHorizonSelfIdentitySourceBindingError("record ticks must be sorted and unique")
    for field in ("observation_id", "source_snapshot_id", "review_trace_id", "appraisal_trace_id"):
        if len({getattr(item, field) for item in items}) != len(items):
            raise LongHorizonSelfIdentitySourceBindingError(f"record {field} values must be unique")
    for field in (
        "source_instance_id", "source_schema_version", "acquisition_method", "verification_method",
        "review_schema_version", "review_method", "review_outcome", "appraisal_schema_version",
        "appraisal_method", "appraisal_outcome", "model_or_rule_version", "source_family",
    ):
        if len({getattr(item, field) for item in items}) != 1:
            raise LongHorizonSelfIdentitySourceBindingError(f"records must share one {field}")
    binding = next(item for item in long_horizon_self_identity_source_bindings().bindings if item.axis == axis)
    if len(items) < binding.minimum_raw_record_count:
        raise LongHorizonSelfIdentitySourceBindingError("insufficient raw record count")
    if items[-1].logical_tick - items[0].logical_tick < binding.minimum_logical_span_ticks:
        raise LongHorizonSelfIdentitySourceBindingError("insufficient logical observation span")
    scores = tuple(_record_score(item) for item in items)
    value = float(sum(scores) / len(scores))
    variance = float(sum((score - value) ** 2 for score in scores) / len(scores))
    confidence = float(max(0.5, min(1.0, 1.0 - variance)))
    raw_bundle_digest = _digest(
        {"axis": axis, "binding_digest": binding.binding_digest, "records": [item.to_mapping() for item in items]},
        "long_horizon_self_identity_raw_bundle",
    )
    source_integrity_digest = _digest(
        {"binding_digest": binding.binding_digest, "raw_bundle_digest": raw_bundle_digest, "source_instance_id": items[0].source_instance_id},
        "long_horizon_self_identity_source_integrity",
    )
    return RegistryAxisPositiveConfidenceEvidence(
        axis=axis,
        value=value,
        confidence=confidence,
        observed_tick=items[-1].logical_tick,
        observation_id=f"self-identity:{axis}:{raw_bundle_digest[:24]}",
        source_family=SOURCE_FAMILY,
        source_instance_id=items[0].source_instance_id,
        source_snapshot_id=f"self-identity:{axis}:{items[0].logical_tick}:{items[-1].logical_tick}:{raw_bundle_digest[:16]}",
        source_schema_version=RAW_SCHEMA_VERSION,
        source_integrity_digest=source_integrity_digest,
        raw_observation_digest=raw_bundle_digest,
        acquisition_method=ACQUISITION_METHOD,
        verification_method=VERIFICATION_METHOD,
        model_or_rule_version=binding.derivation_rule_id,
    )
