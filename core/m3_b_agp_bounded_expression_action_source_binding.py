"""Detached AGP-bounded source bindings for the six expression-action axes.

Only caller-supplied immutable records that already carry a versioned AGP
verification trace and a separate bounded appraisal are accepted. The module
derives detached positive-confidence registry evidence only. It does not emit
text/actions, execute tools, change AGP mode or thresholds, poll runtime state,
write memory/persistence, append events, install capture, start an observation
window, or promote authority.
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

RAW_SCHEMA_VERSION = "eve.m3-b.agp-bounded-expression-action-raw-record.v1"
AGP_TRACE_SCHEMA_VERSION = "eve.m3-b.agp-verification-trace.v1"
APPRAISAL_SCHEMA_VERSION = "eve.m3-b.expression-action-appraisal-trace.v1"
BINDING_SCHEMA_VERSION = "eve.m3-b.agp-bounded-expression-action-source-binding.v1"
BINDING_SET_SCHEMA_VERSION = "eve.m3-b.agp-bounded-expression-action-source-binding-set.v1"
SOURCE_FAMILY = "agp_bounded_expression_action_trace"
ACQUISITION_METHOD = "explicit_caller_supplied_immutable_agp_bounded_expression_action_record"
VERIFICATION_METHOD = "exact_agp_appraisal_schema_range_identity_and_digest_verification"
AGP_VERIFICATION_METHOD = "versioned_anchored_generation_verification"
APPRAISAL_METHOD = "deterministic_bounded_expression_action_appraisal"
APPRAISAL_OUTCOME = "accepted_bounded_expression_action_appraisal"
RAW_MODEL_OR_RULE_VERSION = BINDING_SCHEMA_VERSION
EXPRESSION_ACTION_AXES = (
    "expression_pressure",
    "expression_inhibition",
    "action_readiness",
    "risk_tolerance",
    "patience_level",
    "conflict_avoidance",
)
TOTAL_BOUND_AXIS_COUNT = 37
POSITIVE_CONFIDENCE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
PRODUCTION_CAPTURE_BLOCKER = "REGISTRY_RETAINED_REAL_OBSERVATION_CAPTURE_ABSENT"
ZERO_DIGEST = "0" * 64
_AGP_STATUS_ALLOWED = {"passed", "failed_bounded"}
_AUTHORIZATION_STATUS_ALLOWED = {"authorized", "not_authorized", "deferred"}


class AGPBoundedExpressionActionSourceBindingError(ValueError):
    """Raised when expression-action source evidence fails closed."""


def _identifier(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise AGPBoundedExpressionActionSourceBindingError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _digest_string(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == ZERO_DIGEST
    ):
        raise AGPBoundedExpressionActionSourceBindingError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AGPBoundedExpressionActionSourceBindingError(
            f"{field} must be a non-negative integer"
        )
    return value


def _unit(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AGPBoundedExpressionActionSourceBindingError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise AGPBoundedExpressionActionSourceBindingError(
            f"{field} must be finite and inside [0,1]"
        )
    return result


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise AGPBoundedExpressionActionSourceBindingError(f"{field} must be boolean")
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
        raise AGPBoundedExpressionActionSourceBindingError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _manifest_entry(axis: str) -> RegistryObservationSourceEntry:
    for entry in registry_observation_source_manifest().entries:
        if entry.axis == axis:
            return entry
    raise AGPBoundedExpressionActionSourceBindingError(
        "expression-action axis missing from source manifest"
    )


def _raw_mapping(raw_values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {field: value for field, value in raw_values}


def _count_pressure(value: Any, field: str) -> float:
    count = _nonnegative_int(value, field)
    return float(count / (count + 1.0))


def agp_bounded_expression_action_raw_observation_digest(
    *,
    axis: str,
    logical_tick: int,
    observation_id: str,
    source_instance_id: str,
    source_snapshot_id: str,
    source_schema_version: str,
    source_integrity_digest: str,
    agp_trace_id: str,
    agp_input_digest: str,
    agp_integrity_digest: str,
    agp_status: str,
    appraisal_trace_id: str,
    appraisal_input_digest: str,
    appraisal_integrity_digest: str,
    raw_values: tuple[tuple[str, Any], ...],
) -> str:
    if axis not in EXPRESSION_ACTION_AXES:
        raise AGPBoundedExpressionActionSourceBindingError(
            "unsupported expression-action axis"
        )
    _nonnegative_int(logical_tick, "logical_tick")
    for field, value in (
        ("observation_id", observation_id),
        ("source_instance_id", source_instance_id),
        ("source_snapshot_id", source_snapshot_id),
        ("source_schema_version", source_schema_version),
        ("agp_trace_id", agp_trace_id),
        ("appraisal_trace_id", appraisal_trace_id),
    ):
        _identifier(value, field)
    for field, value in (
        ("source_integrity_digest", source_integrity_digest),
        ("agp_input_digest", agp_input_digest),
        ("agp_integrity_digest", agp_integrity_digest),
        ("appraisal_input_digest", appraisal_input_digest),
        ("appraisal_integrity_digest", appraisal_integrity_digest),
    ):
        _digest_string(value, field)
    if agp_status not in _AGP_STATUS_ALLOWED:
        raise AGPBoundedExpressionActionSourceBindingError(
            "agp_status must be passed or failed_bounded"
        )
    if appraisal_input_digest != agp_integrity_digest:
        raise AGPBoundedExpressionActionSourceBindingError(
            "expression-action appraisal input must be the exact verified AGP output"
        )
    values = tuple(raw_values)
    return _digest(
        {
            "acquisition_method": ACQUISITION_METHOD,
            "agp_input_digest": agp_input_digest,
            "agp_integrity_digest": agp_integrity_digest,
            "agp_status": agp_status,
            "agp_trace_id": agp_trace_id,
            "agp_trace_schema_version": AGP_TRACE_SCHEMA_VERSION,
            "agp_trace_verified": True,
            "agp_verification_method": AGP_VERIFICATION_METHOD,
            "appraisal_input_digest": appraisal_input_digest,
            "appraisal_integrity_digest": appraisal_integrity_digest,
            "appraisal_method": APPRAISAL_METHOD,
            "appraisal_outcome": APPRAISAL_OUTCOME,
            "appraisal_schema_version": APPRAISAL_SCHEMA_VERSION,
            "appraisal_trace_id": appraisal_trace_id,
            "appraisal_verified": True,
            "axis": axis,
            "cutover_authorized": False,
            "expression_or_action_executed": False,
            "hardware_direct_input": False,
            "logical_tick": logical_tick,
            "memory_write_performed": False,
            "model_or_rule_version": RAW_MODEL_OR_RULE_VERSION,
            "observation_id": observation_id,
            "proposal_only": False,
            "raw_social_feedback_source": False,
            "raw_values": [[field, value] for field, value in values],
            "registry_owner_source": False,
            "runtime_polled": False,
            "schema_version": RAW_SCHEMA_VERSION,
            "source_family": SOURCE_FAMILY,
            "source_instance_id": source_instance_id,
            "source_integrity_digest": source_integrity_digest,
            "source_schema_version": source_schema_version,
            "source_snapshot_id": source_snapshot_id,
            "synthetic": False,
            "verification_method": VERIFICATION_METHOD,
        },
        "agp_bounded_expression_action_raw_observation",
    )


def _validate_raw_values(axis: str, raw: Mapping[str, Any]) -> None:
    if axis == "expression_pressure":
        _unit(raw["agp_anchor_coverage"], "agp_anchor_coverage")
        _unit(raw["context_relevance"], "context_relevance")
        _nonnegative_int(raw["pending_expression_count"], "pending_expression_count")
        _nonnegative_int(raw["recurrence_count"], "recurrence_count")
        _unit(raw["salience_score"], "salience_score")
        return
    if axis == "expression_inhibition":
        _nonnegative_int(raw["agp_failure_count"], "agp_failure_count")
        _unit(raw["conflict_risk"], "conflict_risk")
        _unit(raw["disclosure_risk"], "disclosure_risk")
        _boolean(raw["fallback_required"], "fallback_required")
        _unit(raw["uncertainty_score"], "uncertainty_score")
        return
    if axis == "action_readiness":
        status = _identifier(raw["authorization_status"], "authorization_status")
        if status not in _AUTHORIZATION_STATUS_ALLOWED:
            raise AGPBoundedExpressionActionSourceBindingError(
                "authorization_status is outside the bounded observational vocabulary"
            )
        _boolean(raw["capability_available"], "capability_available")
        _nonnegative_int(raw["feasible_action_count"], "feasible_action_count")
        _unit(raw["reversibility"], "reversibility")
        _unit(raw["selected_action_confidence"], "selected_action_confidence")
        return
    if axis == "risk_tolerance":
        _identifier(raw["authorization_scope"], "authorization_scope")
        _unit(raw["expected_cost"], "expected_cost")
        _unit(raw["reversibility"], "reversibility")
        _unit(raw["safety_margin"], "safety_margin")
        _unit(raw["uncertainty_score"], "uncertainty_score")
        return
    if axis == "patience_level":
        _nonnegative_int(raw["alternative_action_count"], "alternative_action_count")
        if raw["appraisal_version"] != APPRAISAL_SCHEMA_VERSION:
            raise AGPBoundedExpressionActionSourceBindingError(
                "appraisal_version must match the canonical expression-action appraisal schema"
            )
        _nonnegative_int(raw["cooldown_remaining"], "cooldown_remaining")
        _unit(raw["deadline_pressure"], "deadline_pressure")
        _unit(raw["uncertainty_resolution_gain"], "uncertainty_resolution_gain")
        return
    if axis == "conflict_avoidance":
        if raw["appraisal_version"] != APPRAISAL_SCHEMA_VERSION:
            raise AGPBoundedExpressionActionSourceBindingError(
                "appraisal_version must match the canonical expression-action appraisal schema"
            )
        _unit(raw["boundary_cost"], "boundary_cost")
        _unit(raw["conflict_probability"], "conflict_probability")
        _nonnegative_int(raw["deescalation_option_count"], "deescalation_option_count")
        _unit(raw["harm_avoidance_gain"], "harm_avoidance_gain")
        return
    raise AGPBoundedExpressionActionSourceBindingError(
        "unsupported expression-action axis"
    )


def _record_score(record: "AGPBoundedExpressionActionRawRecord") -> float:
    raw = record.raw_mapping
    if record.axis == "expression_pressure":
        values = (
            raw["agp_anchor_coverage"],
            raw["context_relevance"],
            _count_pressure(raw["pending_expression_count"], "pending_expression_count"),
            _count_pressure(raw["recurrence_count"], "recurrence_count"),
            raw["salience_score"],
        )
    elif record.axis == "expression_inhibition":
        values = (
            _count_pressure(raw["agp_failure_count"], "agp_failure_count"),
            raw["conflict_risk"],
            raw["disclosure_risk"],
            1.0 if raw["fallback_required"] else 0.0,
            raw["uncertainty_score"],
        )
    elif record.axis == "action_readiness":
        authorization = {
            "authorized": 1.0,
            "deferred": 0.5,
            "not_authorized": 0.0,
        }[raw["authorization_status"]]
        values = (
            authorization,
            1.0 if raw["capability_available"] else 0.0,
            _count_pressure(raw["feasible_action_count"], "feasible_action_count"),
            raw["reversibility"],
            raw["selected_action_confidence"],
        )
    elif record.axis == "risk_tolerance":
        values = (
            1.0 - raw["expected_cost"],
            raw["reversibility"],
            raw["safety_margin"],
            1.0 - raw["uncertainty_score"],
        )
    elif record.axis == "patience_level":
        values = (
            _count_pressure(raw["alternative_action_count"], "alternative_action_count"),
            _count_pressure(raw["cooldown_remaining"], "cooldown_remaining"),
            1.0 - raw["deadline_pressure"],
            raw["uncertainty_resolution_gain"],
        )
    elif record.axis == "conflict_avoidance":
        values = (
            raw["boundary_cost"],
            raw["conflict_probability"],
            _count_pressure(raw["deescalation_option_count"], "deescalation_option_count"),
            raw["harm_avoidance_gain"],
        )
    else:
        raise AGPBoundedExpressionActionSourceBindingError(
            "unsupported expression-action axis"
        )
    return float(sum(float(value) for value in values) / len(values))


@dataclass(frozen=True, slots=True)
class AGPBoundedExpressionActionRawRecord:
    axis: str
    logical_tick: int
    observation_id: str
    source_instance_id: str
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    agp_trace_id: str
    agp_input_digest: str
    agp_integrity_digest: str
    agp_status: str
    appraisal_trace_id: str
    appraisal_input_digest: str
    appraisal_integrity_digest: str
    raw_observation_digest: str
    raw_values: tuple[tuple[str, Any], ...]
    acquisition_method: str = ACQUISITION_METHOD
    verification_method: str = VERIFICATION_METHOD
    agp_trace_schema_version: str = AGP_TRACE_SCHEMA_VERSION
    agp_verification_method: str = AGP_VERIFICATION_METHOD
    agp_trace_verified: bool = True
    appraisal_schema_version: str = APPRAISAL_SCHEMA_VERSION
    appraisal_method: str = APPRAISAL_METHOD
    appraisal_outcome: str = APPRAISAL_OUTCOME
    appraisal_verified: bool = True
    model_or_rule_version: str = RAW_MODEL_OR_RULE_VERSION
    source_family: str = SOURCE_FAMILY
    raw_social_feedback_source: bool = False
    hardware_direct_input: bool = False
    synthetic: bool = False
    proposal_only: bool = False
    registry_owner_source: bool = False
    runtime_polled: bool = False
    expression_or_action_executed: bool = False
    memory_write_performed: bool = False
    cutover_authorized: bool = False
    schema_version: str = RAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.axis not in EXPRESSION_ACTION_AXES:
            raise AGPBoundedExpressionActionSourceBindingError(
                "unsupported expression-action axis"
            )
        _nonnegative_int(self.logical_tick, "logical_tick")
        for field in (
            "observation_id",
            "source_instance_id",
            "source_snapshot_id",
            "source_schema_version",
            "agp_trace_id",
            "appraisal_trace_id",
        ):
            _identifier(getattr(self, field), field)
        for field in (
            "source_integrity_digest",
            "agp_input_digest",
            "agp_integrity_digest",
            "appraisal_input_digest",
            "appraisal_integrity_digest",
            "raw_observation_digest",
        ):
            _digest_string(getattr(self, field), field)
        expected = {
            "acquisition_method": ACQUISITION_METHOD,
            "verification_method": VERIFICATION_METHOD,
            "agp_trace_schema_version": AGP_TRACE_SCHEMA_VERSION,
            "agp_verification_method": AGP_VERIFICATION_METHOD,
            "appraisal_schema_version": APPRAISAL_SCHEMA_VERSION,
            "appraisal_method": APPRAISAL_METHOD,
            "appraisal_outcome": APPRAISAL_OUTCOME,
            "model_or_rule_version": RAW_MODEL_OR_RULE_VERSION,
            "source_family": SOURCE_FAMILY,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise AGPBoundedExpressionActionSourceBindingError(
                    f"raw record {field} does not match canonical expression-action provenance"
                )
        if self.agp_status not in _AGP_STATUS_ALLOWED:
            raise AGPBoundedExpressionActionSourceBindingError(
                "agp_status must be passed or failed_bounded"
            )
        if self.agp_trace_verified is not True or self.appraisal_verified is not True:
            raise AGPBoundedExpressionActionSourceBindingError(
                "expression-action evidence requires exact AGP and appraisal verification"
            )
        if self.appraisal_input_digest != self.agp_integrity_digest:
            raise AGPBoundedExpressionActionSourceBindingError(
                "expression-action appraisal input must be the exact verified AGP output"
            )
        if any(
            (
                self.raw_social_feedback_source,
                self.hardware_direct_input,
                self.synthetic,
                self.proposal_only,
                self.registry_owner_source,
                self.runtime_polled,
                self.expression_or_action_executed,
                self.memory_write_performed,
                self.cutover_authorized,
            )
        ):
            raise AGPBoundedExpressionActionSourceBindingError(
                "expression-action evidence cannot use raw feedback, direct hardware, "
                "synthetic, proposal-only, circular, runtime-polled, executed, mutating, "
                "or cutover-authorized input"
            )
        values = tuple(self.raw_values)
        fields = tuple(field for field, _ in values)
        if (
            fields != _manifest_entry(self.axis).required_raw_fields
            or len(set(fields)) != len(fields)
        ):
            raise AGPBoundedExpressionActionSourceBindingError(
                "raw record fields do not match the canonical expression-action source plan"
            )
        _validate_raw_values(self.axis, _raw_mapping(values))
        expected_digest = agp_bounded_expression_action_raw_observation_digest(
            axis=self.axis,
            logical_tick=self.logical_tick,
            observation_id=self.observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version,
            source_integrity_digest=self.source_integrity_digest,
            agp_trace_id=self.agp_trace_id,
            agp_input_digest=self.agp_input_digest,
            agp_integrity_digest=self.agp_integrity_digest,
            agp_status=self.agp_status,
            appraisal_trace_id=self.appraisal_trace_id,
            appraisal_input_digest=self.appraisal_input_digest,
            appraisal_integrity_digest=self.appraisal_integrity_digest,
            raw_values=values,
        )
        if self.raw_observation_digest != expected_digest:
            raise AGPBoundedExpressionActionSourceBindingError(
                "raw observation digest does not match identity, time, AGP/appraisal provenance, and values"
            )
        if self.schema_version != RAW_SCHEMA_VERSION:
            raise AGPBoundedExpressionActionSourceBindingError(
                "unsupported expression-action raw schema"
            )
        object.__setattr__(self, "raw_values", values)

    @property
    def raw_mapping(self) -> dict[str, Any]:
        return _raw_mapping(self.raw_values)

    @property
    def recalculated_raw_observation_digest(self) -> str:
        return agp_bounded_expression_action_raw_observation_digest(
            axis=self.axis,
            logical_tick=self.logical_tick,
            observation_id=self.observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version,
            source_integrity_digest=self.source_integrity_digest,
            agp_trace_id=self.agp_trace_id,
            agp_input_digest=self.agp_input_digest,
            agp_integrity_digest=self.agp_integrity_digest,
            agp_status=self.agp_status,
            appraisal_trace_id=self.appraisal_trace_id,
            appraisal_input_digest=self.appraisal_input_digest,
            appraisal_integrity_digest=self.appraisal_integrity_digest,
            raw_values=self.raw_values,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "acquisition_method": self.acquisition_method,
            "agp_input_digest": self.agp_input_digest,
            "agp_integrity_digest": self.agp_integrity_digest,
            "agp_status": self.agp_status,
            "agp_trace_id": self.agp_trace_id,
            "agp_trace_schema_version": self.agp_trace_schema_version,
            "agp_trace_verified": self.agp_trace_verified,
            "agp_verification_method": self.agp_verification_method,
            "appraisal_input_digest": self.appraisal_input_digest,
            "appraisal_integrity_digest": self.appraisal_integrity_digest,
            "appraisal_method": self.appraisal_method,
            "appraisal_outcome": self.appraisal_outcome,
            "appraisal_schema_version": self.appraisal_schema_version,
            "appraisal_trace_id": self.appraisal_trace_id,
            "appraisal_verified": self.appraisal_verified,
            "axis": self.axis,
            "cutover_authorized": self.cutover_authorized,
            "expression_or_action_executed": self.expression_or_action_executed,
            "hardware_direct_input": self.hardware_direct_input,
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
            "verification_method": self.verification_method,
        }


@dataclass(frozen=True, slots=True)
class AGPBoundedExpressionActionSourceBinding:
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
    expression_or_action_executed: bool = False
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
            "minimum_raw_record_count": 2,
            "minimum_logical_span_ticks": 1,
            "derivation_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:mean.v1",
            "confidence_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:coverage-variance.v1",
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise AGPBoundedExpressionActionSourceBindingError(
                    f"binding {field} does not match canonical expression-action source plan"
                )
        if self.schema_version != BINDING_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise AGPBoundedExpressionActionSourceBindingError(
                "expression-action binding must remain shadow-only"
            )
        if not self.binding_implemented:
            raise AGPBoundedExpressionActionSourceBindingError(
                "binding implementation flag must be true"
            )
        if any(
            (
                self.production_capture_present,
                self.expression_or_action_executed,
                self.memory_write_performed,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise AGPBoundedExpressionActionSourceBindingError(
                "binding cannot claim capture, execution, window, mutation, or authority"
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
            "expression_or_action_executed": self.expression_or_action_executed,
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
            "source_contract_id": self.source_contract_id,
        }

    @property
    def binding_digest(self) -> str:
        return _digest(self.to_mapping(), "agp_bounded_expression_action_source_binding")


@dataclass(frozen=True, slots=True)
class AGPBoundedExpressionActionSourceBindingSet:
    bindings: tuple[AGPBoundedExpressionActionSourceBinding, ...]
    schema_version: str = BINDING_SET_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    total_bound_axis_count: int = TOTAL_BOUND_AXIS_COUNT
    remaining_axis_count: int = 0
    production_capture_present: bool = False
    retained_real_observation_count: int = 0
    positive_confidence_real_observation_count: int = 0
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        if (
            len(bindings) != 6
            or tuple(item.axis for item in bindings) != EXPRESSION_ACTION_AXES
        ):
            raise AGPBoundedExpressionActionSourceBindingError(
                "expression-action binding set must preserve exact six-axis order"
            )
        if any(type(item) is not AGPBoundedExpressionActionSourceBinding for item in bindings):
            raise AGPBoundedExpressionActionSourceBindingError(
                "binding set requires exact immutable binding types"
            )
        if self.schema_version != BINDING_SET_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise AGPBoundedExpressionActionSourceBindingError(
                "unsupported expression-action binding-set contract"
            )
        if self.total_bound_axis_count != 37 or self.remaining_axis_count != 0:
            raise AGPBoundedExpressionActionSourceBindingError(
                "expression-action progress must remain exact 37+0"
            )
        if self.retained_real_observation_count != 0 or self.positive_confidence_real_observation_count != 0:
            raise AGPBoundedExpressionActionSourceBindingError(
                "binding completion cannot fabricate retained real observations"
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
            raise AGPBoundedExpressionActionSourceBindingError(
                "binding set cannot claim capture, window, or authority"
            )
        object.__setattr__(self, "bindings", bindings)

    @property
    def appraised_binding_count(self) -> int:
        return len(self.bindings)

    @property
    def blockers(self) -> tuple[str, ...]:
        return (PRODUCTION_CAPTURE_BLOCKER, POSITIVE_CONFIDENCE_BLOCKER)

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
            "positive_confidence_real_observation_count": self.positive_confidence_real_observation_count,
            "production_capture_present": self.production_capture_present,
            "remaining_axis_count": self.remaining_axis_count,
            "retained_real_observation_count": self.retained_real_observation_count,
            "schema_version": self.schema_version,
            "total_bound_axis_count": self.total_bound_axis_count,
        }

    @property
    def binding_set_digest(self) -> str:
        return _digest(
            self.to_mapping(), "agp_bounded_expression_action_source_binding_set"
        )


def agp_bounded_expression_action_source_bindings() -> AGPBoundedExpressionActionSourceBindingSet:
    bindings: list[AGPBoundedExpressionActionSourceBinding] = []
    for axis in EXPRESSION_ACTION_AXES:
        entry = _manifest_entry(axis)
        bindings.append(
            AGPBoundedExpressionActionSourceBinding(
                axis=axis,
                source_contract_id=entry.source_contract_id,
                binding_id=f"eve:m3-b:agp-bounded-expression-action-binding:{axis}:v1",
                required_raw_fields=entry.required_raw_fields,
                minimum_raw_record_count=entry.minimum_raw_record_count,
                minimum_logical_span_ticks=entry.minimum_logical_span_ticks,
                derivation_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1",
                confidence_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:coverage-variance.v1",
            )
        )
    return AGPBoundedExpressionActionSourceBindingSet(bindings=tuple(bindings))


def derive_agp_bounded_expression_action_axis_evidence(
    records: Sequence[AGPBoundedExpressionActionRawRecord],
) -> RegistryAxisPositiveConfidenceEvidence:
    items = tuple(records)
    if not items or any(
        type(item) is not AGPBoundedExpressionActionRawRecord for item in items
    ):
        raise AGPBoundedExpressionActionSourceBindingError(
            "records must contain exact immutable expression-action raw records"
        )
    axis = items[0].axis
    if any(item.axis != axis for item in items):
        raise AGPBoundedExpressionActionSourceBindingError("records cannot mix axes")
    ticks = tuple(item.logical_tick for item in items)
    if ticks != tuple(sorted(ticks)) or len(set(ticks)) != len(ticks):
        raise AGPBoundedExpressionActionSourceBindingError(
            "record ticks must be sorted and unique"
        )
    for field in (
        "observation_id",
        "source_snapshot_id",
        "agp_trace_id",
        "appraisal_trace_id",
    ):
        if len({getattr(item, field) for item in items}) != len(items):
            raise AGPBoundedExpressionActionSourceBindingError(
                f"record {field} values must be unique"
            )
    for field in (
        "source_instance_id",
        "source_schema_version",
        "acquisition_method",
        "verification_method",
        "agp_trace_schema_version",
        "agp_verification_method",
        "appraisal_schema_version",
        "appraisal_method",
        "appraisal_outcome",
        "model_or_rule_version",
        "source_family",
    ):
        if len({getattr(item, field) for item in items}) != 1:
            raise AGPBoundedExpressionActionSourceBindingError(
                f"records must share one {field}"
            )
    binding = next(
        item
        for item in agp_bounded_expression_action_source_bindings().bindings
        if item.axis == axis
    )
    if len(items) < binding.minimum_raw_record_count:
        raise AGPBoundedExpressionActionSourceBindingError(
            "insufficient raw record count"
        )
    if items[-1].logical_tick - items[0].logical_tick < binding.minimum_logical_span_ticks:
        raise AGPBoundedExpressionActionSourceBindingError(
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
        "agp_bounded_expression_action_raw_bundle",
    )
    source_integrity_digest = _digest(
        {
            "binding_digest": binding.binding_digest,
            "raw_bundle_digest": raw_bundle_digest,
            "source_instance_id": items[0].source_instance_id,
        },
        "agp_bounded_expression_action_source_integrity",
    )
    return RegistryAxisPositiveConfidenceEvidence(
        axis=axis,
        value=value,
        confidence=confidence,
        observed_tick=items[-1].logical_tick,
        observation_id=f"expression-action:{axis}:{raw_bundle_digest[:24]}",
        source_family=SOURCE_FAMILY,
        source_instance_id=items[0].source_instance_id,
        source_snapshot_id=(
            f"expression-action:{axis}:{items[0].logical_tick}:"
            f"{items[-1].logical_tick}:{raw_bundle_digest[:16]}"
        ),
        source_schema_version=RAW_SCHEMA_VERSION,
        source_integrity_digest=source_integrity_digest,
        raw_observation_digest=raw_bundle_digest,
        acquisition_method=ACQUISITION_METHOD,
        verification_method=VERIFICATION_METHOD,
        model_or_rule_version=binding.derivation_rule_id,
    )
