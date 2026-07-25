"""Detached quarantined source bindings for the seven social-relationship axes.

The module accepts caller-supplied immutable records that already passed an
explicit social-input quarantine and a verified bounded social appraisal. It
only derives detached positive-confidence registry evidence. It does not
capture social input, poll hardware, schedule work, access persistence, append
events, mutate the registry owner, start an observation window, or grant any
runtime/cutover authority.
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

RAW_SCHEMA_VERSION = "eve.m3-b.quarantined-social-raw-record.v1"
QUARANTINE_SCHEMA_VERSION = "eve.m3-b.social-quarantine-trace.v1"
APPRAISAL_SCHEMA_VERSION = "eve.m3-b.social-appraisal-trace.v1"
BINDING_SCHEMA_VERSION = "eve.m3-b.quarantined-social-source-binding.v1"
BINDING_SET_SCHEMA_VERSION = "eve.m3-b.quarantined-social-source-binding-set.v1"
SOURCE_FAMILY = "quarantined_social_appraisal_trace"
ACQUISITION_METHOD = "explicit_caller_supplied_immutable_quarantined_social_record"
VERIFICATION_METHOD = "exact_social_quarantine_appraisal_schema_range_identity_and_digest_verification"
QUARANTINE_METHOD = "deterministic_social_input_quarantine_before_appraisal"
QUARANTINE_OUTCOME = "accepted_for_bounded_social_appraisal"
APPRAISAL_METHOD = "deterministic_bounded_social_relationship_appraisal"
APPRAISAL_OUTCOME = "accepted_bounded_social_relationship_appraisal"
QUARANTINE_STATUS = "verified_social_input_quarantined_before_appraisal"
RAW_MODEL_OR_RULE_VERSION = BINDING_SCHEMA_VERSION
SOCIAL_RELATIONSHIP_AXES = (
    "social_pain",
    "social_trust",
    "attachment",
    "care_drive",
    "loneliness_pressure",
    "belonging_need",
    "rejection_sensitivity",
)
TOTAL_BOUND_AXIS_COUNT = 19
REMAINING_BINDING_BLOCKER = "REGISTRY_APPRAISED_18_AXIS_SOURCE_BINDINGS_INCOMPLETE"
POSITIVE_CONFIDENCE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
ZERO_DIGEST = "0" * 64

_CONSENT_ALLOWANCE = {
    "withheld": 0.0,
    "limited": 0.5,
    "granted": 1.0,
}


class QuarantinedSocialSourceBindingError(ValueError):
    """Raised when a social trace or detached binding fails closed."""


def _identifier(value: str, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise QuarantinedSocialSourceBindingError(
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
        raise QuarantinedSocialSourceBindingError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QuarantinedSocialSourceBindingError(
            f"{field} must be a non-negative integer"
        )
    return value


def _unit(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise QuarantinedSocialSourceBindingError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise QuarantinedSocialSourceBindingError(
            f"{field} must be finite and inside [0,1]"
        )
    return result


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise QuarantinedSocialSourceBindingError(f"{field} must be boolean")
    return value


def _enum(value: Any, field: str, allowed: Mapping[str, float]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise QuarantinedSocialSourceBindingError(
            f"{field} must be one of {tuple(allowed)}"
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
        raise QuarantinedSocialSourceBindingError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _manifest_entry(axis: str) -> RegistryObservationSourceEntry:
    for entry in registry_observation_source_manifest().entries:
        if entry.axis == axis:
            return entry
    raise QuarantinedSocialSourceBindingError(
        "social axis missing from source manifest"
    )


def _raw_mapping(raw_values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {field: value for field, value in raw_values}


def _count_pressure(value: Any, field: str) -> float:
    count = _nonnegative_int(value, field)
    return float(count / (count + 1.0))


def _span_support(value: Any, field: str) -> float:
    span = _nonnegative_int(value, field)
    return float(span / (span + 4.0))


def quarantined_social_raw_observation_digest(
    *,
    axis: str,
    logical_tick: int,
    observation_id: str,
    source_instance_id: str,
    source_snapshot_id: str,
    source_schema_version: str,
    source_integrity_digest: str,
    quarantine_trace_id: str,
    quarantine_input_digest: str,
    quarantine_integrity_digest: str,
    appraisal_trace_id: str,
    appraisal_input_digest: str,
    appraisal_integrity_digest: str,
    raw_values: tuple[tuple[str, Any], ...],
) -> str:
    """Return the canonical digest for one quarantined social record."""

    if axis not in SOCIAL_RELATIONSHIP_AXES:
        raise QuarantinedSocialSourceBindingError("unsupported social axis")
    _nonnegative_int(logical_tick, "logical_tick")
    for field, value in (
        ("observation_id", observation_id),
        ("source_instance_id", source_instance_id),
        ("source_snapshot_id", source_snapshot_id),
        ("source_schema_version", source_schema_version),
        ("quarantine_trace_id", quarantine_trace_id),
        ("appraisal_trace_id", appraisal_trace_id),
    ):
        _identifier(value, field)
    for field, value in (
        ("source_integrity_digest", source_integrity_digest),
        ("quarantine_input_digest", quarantine_input_digest),
        ("quarantine_integrity_digest", quarantine_integrity_digest),
        ("appraisal_input_digest", appraisal_input_digest),
        ("appraisal_integrity_digest", appraisal_integrity_digest),
    ):
        _digest_string(value, field)
    if appraisal_input_digest != quarantine_integrity_digest:
        raise QuarantinedSocialSourceBindingError(
            "social appraisal input must be the exact verified quarantine output"
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
            "quarantine_input_digest": quarantine_input_digest,
            "quarantine_integrity_digest": quarantine_integrity_digest,
            "quarantine_method": QUARANTINE_METHOD,
            "quarantine_outcome": QUARANTINE_OUTCOME,
            "quarantine_schema_version": QUARANTINE_SCHEMA_VERSION,
            "quarantine_status": QUARANTINE_STATUS,
            "quarantine_trace_id": quarantine_trace_id,
            "quarantine_verified": True,
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
        "quarantined_social_raw_observation",
    )


def _validate_raw_values(axis: str, raw: Mapping[str, Any]) -> None:
    if "appraisal_version" in raw and raw["appraisal_version"] != APPRAISAL_SCHEMA_VERSION:
        raise QuarantinedSocialSourceBindingError(
            "appraisal_version must match the canonical social appraisal schema"
        )
    if axis == "social_pain":
        for field in ("injury_evidence_score", "intent_confidence", "source_trust"):
            _unit(raw[field], field)
        _nonnegative_int(raw["recurrence_count"], "recurrence_count")
        return
    if axis == "social_trust":
        for field in (
            "contradiction_count",
            "fulfilled_commitment_count",
            "observation_span_ticks",
            "repair_count",
        ):
            _nonnegative_int(raw[field], field)
        _unit(raw["source_trust"], "source_trust")
        return
    if axis == "attachment":
        for field in (
            "interaction_continuity",
            "mutual_reliability",
            "separation_tolerance",
        ):
            _unit(raw[field], field)
        _nonnegative_int(raw["relationship_span_ticks"], "relationship_span_ticks")
        return
    if axis == "care_drive":
        _unit(raw["capability_to_help"], "capability_to_help")
        _enum(raw["consent_status"], "consent_status", _CONSENT_ALLOWANCE)
        _unit(raw["cost_boundary"], "cost_boundary")
        _unit(raw["welfare_need_score"], "welfare_need_score")
        return
    if axis == "loneliness_pressure":
        _unit(raw["available_relationship_context"], "available_relationship_context")
        _boolean(raw["chosen_solitude_flag"], "chosen_solitude_flag")
        _nonnegative_int(raw["meaningful_contact_gap_ticks"], "meaningful_contact_gap_ticks")
        _nonnegative_int(raw["unmet_connection_signal_count"], "unmet_connection_signal_count")
        return
    if axis == "belonging_need":
        _nonnegative_int(raw["context_span_ticks"], "context_span_ticks")
        _unit(raw["group_continuity"], "group_continuity")
        _nonnegative_int(raw["reciprocal_inclusion_count"], "reciprocal_inclusion_count")
        _unit(raw["role_clarity"], "role_clarity")
        return
    if axis == "rejection_sensitivity":
        for field in (
            "ambiguous_signal_count",
            "false_positive_count",
            "observation_span_ticks",
            "verified_rejection_count",
        ):
            _nonnegative_int(raw[field], field)
        _unit(raw["source_trust"], "source_trust")
        return
    raise QuarantinedSocialSourceBindingError("unsupported social axis")


def _record_score(record: "QuarantinedSocialRawRecord") -> float:
    raw = record.raw_mapping
    if record.axis == "social_pain":
        values = (
            raw["injury_evidence_score"],
            raw["intent_confidence"],
            _count_pressure(raw["recurrence_count"], "recurrence_count"),
            raw["source_trust"],
        )
    elif record.axis == "social_trust":
        values = (
            1.0 - _count_pressure(raw["contradiction_count"], "contradiction_count"),
            _count_pressure(raw["fulfilled_commitment_count"], "fulfilled_commitment_count"),
            _span_support(raw["observation_span_ticks"], "observation_span_ticks"),
            _count_pressure(raw["repair_count"], "repair_count"),
            raw["source_trust"],
        )
    elif record.axis == "attachment":
        values = (
            raw["interaction_continuity"],
            raw["mutual_reliability"],
            _span_support(raw["relationship_span_ticks"], "relationship_span_ticks"),
            raw["separation_tolerance"],
        )
    elif record.axis == "care_drive":
        values = (
            raw["capability_to_help"],
            _CONSENT_ALLOWANCE[raw["consent_status"]],
            1.0 - raw["cost_boundary"],
            raw["welfare_need_score"],
        )
    elif record.axis == "loneliness_pressure":
        values = (
            1.0 - raw["available_relationship_context"],
            0.0 if raw["chosen_solitude_flag"] else 1.0,
            _span_support(raw["meaningful_contact_gap_ticks"], "meaningful_contact_gap_ticks"),
            _count_pressure(raw["unmet_connection_signal_count"], "unmet_connection_signal_count"),
        )
    elif record.axis == "belonging_need":
        values = (
            1.0 - _span_support(raw["context_span_ticks"], "context_span_ticks"),
            1.0 - raw["group_continuity"],
            1.0 - _count_pressure(raw["reciprocal_inclusion_count"], "reciprocal_inclusion_count"),
            1.0 - raw["role_clarity"],
        )
    elif record.axis == "rejection_sensitivity":
        verified = _nonnegative_int(raw["verified_rejection_count"], "verified_rejection_count")
        ambiguous = _nonnegative_int(raw["ambiguous_signal_count"], "ambiguous_signal_count")
        false_positive = _nonnegative_int(raw["false_positive_count"], "false_positive_count")
        calibrated_signal = float((verified + 0.5 * ambiguous) / (verified + ambiguous + false_positive + 1.0))
        values = (
            calibrated_signal,
            raw["source_trust"],
            _span_support(raw["observation_span_ticks"], "observation_span_ticks"),
        )
    else:
        raise QuarantinedSocialSourceBindingError("unsupported social axis")
    return float(sum(values) / len(values))


@dataclass(frozen=True, slots=True)
class QuarantinedSocialRawRecord:
    axis: str
    logical_tick: int
    observation_id: str
    source_instance_id: str
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    quarantine_trace_id: str
    quarantine_input_digest: str
    quarantine_integrity_digest: str
    appraisal_trace_id: str
    appraisal_input_digest: str
    appraisal_integrity_digest: str
    raw_observation_digest: str
    raw_values: tuple[tuple[str, Any], ...]
    acquisition_method: str = ACQUISITION_METHOD
    verification_method: str = VERIFICATION_METHOD
    quarantine_schema_version: str = QUARANTINE_SCHEMA_VERSION
    quarantine_method: str = QUARANTINE_METHOD
    quarantine_outcome: str = QUARANTINE_OUTCOME
    quarantine_status: str = QUARANTINE_STATUS
    appraisal_schema_version: str = APPRAISAL_SCHEMA_VERSION
    appraisal_method: str = APPRAISAL_METHOD
    appraisal_outcome: str = APPRAISAL_OUTCOME
    model_or_rule_version: str = RAW_MODEL_OR_RULE_VERSION
    source_family: str = SOURCE_FAMILY
    quarantine_verified: bool = True
    appraisal_verified: bool = True
    raw_social_feedback_source: bool = False
    hardware_direct_input: bool = False
    synthetic: bool = False
    proposal_only: bool = False
    registry_owner_source: bool = False
    runtime_polled: bool = False
    schema_version: str = RAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.axis not in SOCIAL_RELATIONSHIP_AXES:
            raise QuarantinedSocialSourceBindingError("unsupported social axis")
        _nonnegative_int(self.logical_tick, "logical_tick")
        for field in (
            "observation_id",
            "source_instance_id",
            "source_snapshot_id",
            "source_schema_version",
            "quarantine_trace_id",
            "appraisal_trace_id",
        ):
            _identifier(getattr(self, field), field)
        for field in (
            "source_integrity_digest",
            "quarantine_input_digest",
            "quarantine_integrity_digest",
            "appraisal_input_digest",
            "appraisal_integrity_digest",
            "raw_observation_digest",
        ):
            _digest_string(getattr(self, field), field)
        expected_provenance = {
            "acquisition_method": ACQUISITION_METHOD,
            "verification_method": VERIFICATION_METHOD,
            "quarantine_schema_version": QUARANTINE_SCHEMA_VERSION,
            "quarantine_method": QUARANTINE_METHOD,
            "quarantine_outcome": QUARANTINE_OUTCOME,
            "quarantine_status": QUARANTINE_STATUS,
            "appraisal_schema_version": APPRAISAL_SCHEMA_VERSION,
            "appraisal_method": APPRAISAL_METHOD,
            "appraisal_outcome": APPRAISAL_OUTCOME,
            "model_or_rule_version": RAW_MODEL_OR_RULE_VERSION,
            "source_family": SOURCE_FAMILY,
        }
        for field, expected in expected_provenance.items():
            if getattr(self, field) != expected:
                raise QuarantinedSocialSourceBindingError(
                    f"raw record {field} does not match the canonical social provenance contract"
                )
        if self.quarantine_verified is not True or self.appraisal_verified is not True:
            raise QuarantinedSocialSourceBindingError(
                "social evidence requires exact quarantine and appraisal verification"
            )
        if self.appraisal_input_digest != self.quarantine_integrity_digest:
            raise QuarantinedSocialSourceBindingError(
                "social appraisal input must be the exact verified quarantine output"
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
            raise QuarantinedSocialSourceBindingError(
                "social evidence cannot use raw social feedback, direct hardware, synthetic, proposal-only, circular, or runtime-polled input"
            )
        values = tuple(self.raw_values)
        fields = tuple(field for field, _ in values)
        expected_fields = _manifest_entry(self.axis).required_raw_fields
        if fields != expected_fields or len(set(fields)) != len(fields):
            raise QuarantinedSocialSourceBindingError(
                "raw record fields do not match the canonical social source plan"
            )
        _validate_raw_values(self.axis, _raw_mapping(values))
        expected_digest = quarantined_social_raw_observation_digest(
            axis=self.axis,
            logical_tick=self.logical_tick,
            observation_id=self.observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version,
            source_integrity_digest=self.source_integrity_digest,
            quarantine_trace_id=self.quarantine_trace_id,
            quarantine_input_digest=self.quarantine_input_digest,
            quarantine_integrity_digest=self.quarantine_integrity_digest,
            appraisal_trace_id=self.appraisal_trace_id,
            appraisal_input_digest=self.appraisal_input_digest,
            appraisal_integrity_digest=self.appraisal_integrity_digest,
            raw_values=values,
        )
        if self.raw_observation_digest != expected_digest:
            raise QuarantinedSocialSourceBindingError(
                "raw observation digest does not match identity, time, quarantine/appraisal provenance, and values"
            )
        if self.schema_version != RAW_SCHEMA_VERSION:
            raise QuarantinedSocialSourceBindingError(
                "unsupported quarantined-social raw schema"
            )
        object.__setattr__(self, "raw_values", values)

    @property
    def raw_mapping(self) -> dict[str, Any]:
        return _raw_mapping(self.raw_values)

    @property
    def recalculated_raw_observation_digest(self) -> str:
        return quarantined_social_raw_observation_digest(
            axis=self.axis,
            logical_tick=self.logical_tick,
            observation_id=self.observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.source_schema_version,
            source_integrity_digest=self.source_integrity_digest,
            quarantine_trace_id=self.quarantine_trace_id,
            quarantine_input_digest=self.quarantine_input_digest,
            quarantine_integrity_digest=self.quarantine_integrity_digest,
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
            "quarantine_input_digest": self.quarantine_input_digest,
            "quarantine_integrity_digest": self.quarantine_integrity_digest,
            "quarantine_method": self.quarantine_method,
            "quarantine_outcome": self.quarantine_outcome,
            "quarantine_schema_version": self.quarantine_schema_version,
            "quarantine_status": self.quarantine_status,
            "quarantine_trace_id": self.quarantine_trace_id,
            "quarantine_verified": self.quarantine_verified,
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
class QuarantinedSocialSourceBinding:
    axis: str
    source_contract_id: str
    binding_id: str
    raw_schema_version: str
    quarantine_schema_version: str
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
    quarantine_required: bool = True
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
        if self.axis not in SOCIAL_RELATIONSHIP_AXES:
            raise QuarantinedSocialSourceBindingError("unsupported binding axis")
        entry = _manifest_entry(self.axis)
        expected = {
            "source_contract_id": entry.source_contract_id,
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "quarantine_schema_version": QUARANTINE_SCHEMA_VERSION,
            "appraisal_schema_version": APPRAISAL_SCHEMA_VERSION,
            "required_raw_fields": entry.required_raw_fields,
            "minimum_raw_record_count": entry.minimum_raw_record_count,
            "minimum_logical_span_ticks": entry.minimum_logical_span_ticks,
            "derivation_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:mean.v1",
            "confidence_rule_id": f"{BINDING_SCHEMA_VERSION}:{self.axis}:coverage-variance.v1",
            "appraisal_required": True,
            "quarantine_required": True,
            "hardware_direct_input_allowed": False,
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise QuarantinedSocialSourceBindingError(
                    f"binding {field} does not match the canonical social source plan"
                )
        _identifier(self.binding_id, "binding_id")
        if self.schema_version != BINDING_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise QuarantinedSocialSourceBindingError(
                "quarantined social binding must use the exact shadow-only schema"
            )
        if self.binding_implemented is not True:
            raise QuarantinedSocialSourceBindingError(
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
            raise QuarantinedSocialSourceBindingError(
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
            "quarantine_schema_version": self.quarantine_schema_version,
            "raw_schema_version": self.raw_schema_version,
            "required_raw_fields": list(self.required_raw_fields),
            "runtime_capture_installed": self.runtime_capture_installed,
            "schema_version": self.schema_version,
            "source_contract_id": self.source_contract_id,
        }

    @property
    def binding_digest(self) -> str:
        return _digest(self.to_mapping(), "quarantined_social_source_binding")


@dataclass(frozen=True, slots=True)
class QuarantinedSocialSourceBindingSet:
    bindings: tuple[QuarantinedSocialSourceBinding, ...]
    schema_version: str = BINDING_SET_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    total_bound_axis_count: int = TOTAL_BOUND_AXIS_COUNT
    remaining_axis_count: int = 18
    production_capture_present: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        if any(type(item) is not QuarantinedSocialSourceBinding for item in bindings):
            raise QuarantinedSocialSourceBindingError(
                "binding set requires exact immutable binding types"
            )
        if len(bindings) != 7 or tuple(item.axis for item in bindings) != SOCIAL_RELATIONSHIP_AXES:
            raise QuarantinedSocialSourceBindingError(
                "quarantined social binding set must preserve exact seven-axis order"
            )
        if (
            self.schema_version != BINDING_SET_SCHEMA_VERSION
            or self.authority != SHADOW_AUTHORITY
            or self.total_bound_axis_count != TOTAL_BOUND_AXIS_COUNT
            or self.remaining_axis_count != 18
        ):
            raise QuarantinedSocialSourceBindingError(
                "unsupported quarantined-social binding-set contract"
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
            raise QuarantinedSocialSourceBindingError(
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
        return _digest(self.to_mapping(), "quarantined_social_source_binding_set")


def quarantined_social_source_bindings() -> QuarantinedSocialSourceBindingSet:
    bindings = []
    for axis in SOCIAL_RELATIONSHIP_AXES:
        entry = _manifest_entry(axis)
        bindings.append(
            QuarantinedSocialSourceBinding(
                axis=axis,
                source_contract_id=entry.source_contract_id,
                binding_id=f"eve:m3-b:quarantined-social-binding:{axis}:v1",
                raw_schema_version=RAW_SCHEMA_VERSION,
                quarantine_schema_version=QUARANTINE_SCHEMA_VERSION,
                appraisal_schema_version=APPRAISAL_SCHEMA_VERSION,
                required_raw_fields=entry.required_raw_fields,
                minimum_raw_record_count=entry.minimum_raw_record_count,
                minimum_logical_span_ticks=entry.minimum_logical_span_ticks,
                derivation_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1",
                confidence_rule_id=f"{BINDING_SCHEMA_VERSION}:{axis}:coverage-variance.v1",
            )
        )
    return QuarantinedSocialSourceBindingSet(bindings=tuple(bindings))


def derive_quarantined_social_axis_evidence(
    records: Sequence[QuarantinedSocialRawRecord],
) -> RegistryAxisPositiveConfidenceEvidence:
    """Derive one detached social evidence record from quarantined appraisals."""

    items = tuple(records)
    if not items or any(type(item) is not QuarantinedSocialRawRecord for item in items):
        raise QuarantinedSocialSourceBindingError(
            "records must contain exact immutable quarantined-social raw records"
        )
    axis = items[0].axis
    if any(item.axis != axis for item in items):
        raise QuarantinedSocialSourceBindingError("records cannot mix axes")
    ticks = tuple(item.logical_tick for item in items)
    if ticks != tuple(sorted(ticks)):
        raise QuarantinedSocialSourceBindingError("record ticks must be sorted")
    if len(set(ticks)) != len(ticks):
        raise QuarantinedSocialSourceBindingError("record ticks must be unique")
    for field in (
        "observation_id",
        "source_snapshot_id",
        "quarantine_trace_id",
        "appraisal_trace_id",
    ):
        if len({getattr(item, field) for item in items}) != len(items):
            raise QuarantinedSocialSourceBindingError(
                f"record {field} values must be unique"
            )
    for field in (
        "source_instance_id",
        "source_schema_version",
        "acquisition_method",
        "verification_method",
        "quarantine_schema_version",
        "quarantine_method",
        "quarantine_outcome",
        "quarantine_status",
        "appraisal_schema_version",
        "appraisal_method",
        "appraisal_outcome",
        "model_or_rule_version",
        "source_family",
    ):
        if len({getattr(item, field) for item in items}) != 1:
            raise QuarantinedSocialSourceBindingError(
                f"records must share one {field}"
            )
    binding = next(
        item
        for item in quarantined_social_source_bindings().bindings
        if item.axis == axis
    )
    span = items[-1].logical_tick - items[0].logical_tick
    if len(items) < binding.minimum_raw_record_count:
        raise QuarantinedSocialSourceBindingError("insufficient raw record count")
    if span < binding.minimum_logical_span_ticks:
        raise QuarantinedSocialSourceBindingError(
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
        "quarantined_social_raw_bundle",
    )
    source_integrity_digest = _digest(
        {
            "binding_digest": binding.binding_digest,
            "raw_bundle_digest": raw_bundle_digest,
            "source_instance_id": items[0].source_instance_id,
        },
        "quarantined_social_source_integrity",
    )
    return RegistryAxisPositiveConfidenceEvidence(
        axis=axis,
        value=value,
        confidence=confidence,
        observed_tick=items[-1].logical_tick,
        observation_id=f"quarantined-social:{axis}:{raw_bundle_digest[:24]}",
        source_family=SOURCE_FAMILY,
        source_instance_id=items[0].source_instance_id,
        source_snapshot_id=(
            f"quarantined-social:{axis}:{items[0].logical_tick}:"
            f"{items[-1].logical_tick}:{raw_bundle_digest[:16]}"
        ),
        source_schema_version=RAW_SCHEMA_VERSION,
        source_integrity_digest=source_integrity_digest,
        raw_observation_digest=raw_bundle_digest,
        acquisition_method=ACQUISITION_METHOD,
        verification_method=VERIFICATION_METHOD,
        model_or_rule_version=binding.derivation_rule_id,
    )
