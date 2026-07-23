"""Exact source-manifest preflight for real 37-axis registry observations.

The manifest defines what raw, independently recalculable evidence a later
capture package must bind for every registry axis. It supplies no values,
installs no capture hook, polls no hardware, reads no runtime state, starts no
observation window, and grants no M3-B/M3-C/cutover/M3-E authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from adapters.affect_hormone_neural_rhythm_registry import (
    AXIS_GROUPS,
    affect_hormone_axis_registry,
)
from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_registry_affect_owner import REGISTRY_AXIS_ORDER

ENTRY_SCHEMA_VERSION = "eve.m3-b.registry-observation-source-entry.v1"
MANIFEST_SCHEMA_VERSION = "eve.m3-b.registry-37-axis-observation-source-manifest.v1"
MANIFEST_ID = "eve:m3-b:registry-observation-source-manifest:v1"
SOURCE_BINDING_BLOCKER = "REGISTRY_REAL_OBSERVATION_SOURCE_BINDINGS_INCOMPLETE"
POSITIVE_CONFIDENCE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
DERIVATION_POLICY_VERSION = "eve.m3-b.registry-observation-derivation-policy.v1"
CONFIDENCE_POLICY_VERSION = "eve.m3-b.registry-observation-confidence-policy.v1"


class RegistryObservationSourceManifestError(ValueError):
    """Raised when the 37-axis source manifest is incomplete or inconsistent."""


def _identifier(value: str, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise RegistryObservationSourceManifestError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegistryObservationSourceManifestError(f"{field} must be positive")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RegistryObservationSourceManifestError(
            f"{field} must be a non-negative integer"
        )
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
        raise RegistryObservationSourceManifestError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _axis_group_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for group, axes in AXIS_GROUPS.items():
        for axis in axes:
            if axis in result:
                raise RegistryObservationSourceManifestError(
                    "registry axis appears in more than one group"
                )
            result[axis] = group
    if tuple(result) != REGISTRY_AXIS_ORDER:
        raise RegistryObservationSourceManifestError(
            "axis groups do not preserve canonical 37-axis order"
        )
    return result


_GROUP_SOURCE_FAMILY = {
    "survival_stability": "operational_metrics_or_appraised_load_trace",
    "risk_defense": "quarantined_risk_appraisal_trace",
    "social_relationship": "quarantined_social_appraisal_trace",
    "learning_exploration": "validated_learning_and_prediction_trace",
    "self_identity": "long_horizon_self_model_review_trace",
    "expression_action": "agp_bounded_expression_action_trace",
}

_GROUP_MINIMUMS = {
    "survival_stability": (3, 2),
    "risk_defense": (2, 1),
    "social_relationship": (3, 8),
    "learning_exploration": (2, 2),
    "self_identity": (3, 12),
    "expression_action": (2, 1),
}

_MINIMUM_OVERRIDES = {
    "attachment": (3, 12),
    "care_drive": (2, 2),
    "competence_drive": (3, 4),
    "prediction_error_pressure": (2, 1),
}

_HARDWARE_DIRECT_AXES = {
    "energy_budget",
    "fatigue_pressure",
    "overload_risk",
    "recovery_need",
}

_RAW_FIELDS: dict[str, tuple[str, ...]] = {
    "energy_budget": (
        "available_cpu_budget",
        "available_memory_budget",
        "battery_governor_band",
        "foreground_load",
        "sampling_window_ticks",
    ),
    "fatigue_pressure": (
        "active_processing_ticks",
        "queue_pressure",
        "recovery_interval_ticks",
        "sampling_window_ticks",
        "task_switch_count",
    ),
    "recovery_need": (
        "active_processing_ticks",
        "cooldown_ticks",
        "recent_overload_count",
        "sampling_window_ticks",
        "successful_recovery_count",
    ),
    "stress_load": (
        "appraisal_version",
        "controllability_score",
        "demand_score",
        "overload_score",
        "uncertainty_score",
    ),
    "stability_need": (
        "invariant_failure_count",
        "pending_migration_count",
        "replay_divergence_count",
        "rollback_readiness_score",
        "sampling_window_ticks",
    ),
    "overload_risk": (
        "concurrent_demand_count",
        "latency_budget_ratio",
        "memory_pressure_ratio",
        "queue_depth",
        "thermal_governor_band",
    ),
    "threat_pressure": (
        "appraisal_version",
        "impact_score",
        "source_trust",
        "threat_probability",
        "verification_status",
    ),
    "uncertainty_pressure": (
        "appraisal_version",
        "conflict_count",
        "missing_evidence_ratio",
        "source_reliability",
        "verification_gap",
    ),
    "self_protection": (
        "appraisal_version",
        "capability_limit",
        "exposure_scope",
        "reversibility",
        "threat_pressure_input",
    ),
    "boundary_defense": (
        "appraisal_version",
        "boundary_violation_count",
        "intent_confidence",
        "persistence_score",
        "remedy_available",
    ),
    "trust_risk": (
        "appraisal_version",
        "contradiction_count",
        "reversibility",
        "source_reliability",
        "verification_depth",
    ),
    "exposure_risk": (
        "audience_scope",
        "authorization_status",
        "persistence_risk",
        "reversibility",
        "sensitivity_class",
    ),
    "social_pain": (
        "appraisal_version",
        "injury_evidence_score",
        "intent_confidence",
        "recurrence_count",
        "source_trust",
    ),
    "social_trust": (
        "contradiction_count",
        "fulfilled_commitment_count",
        "observation_span_ticks",
        "repair_count",
        "source_trust",
    ),
    "attachment": (
        "appraisal_version",
        "interaction_continuity",
        "mutual_reliability",
        "relationship_span_ticks",
        "separation_tolerance",
    ),
    "care_drive": (
        "appraisal_version",
        "capability_to_help",
        "consent_status",
        "cost_boundary",
        "welfare_need_score",
    ),
    "loneliness_pressure": (
        "appraisal_version",
        "available_relationship_context",
        "chosen_solitude_flag",
        "meaningful_contact_gap_ticks",
        "unmet_connection_signal_count",
    ),
    "belonging_need": (
        "appraisal_version",
        "context_span_ticks",
        "group_continuity",
        "reciprocal_inclusion_count",
        "role_clarity",
    ),
    "rejection_sensitivity": (
        "ambiguous_signal_count",
        "false_positive_count",
        "observation_span_ticks",
        "source_trust",
        "verified_rejection_count",
    ),
    "curiosity_drive": (
        "exploration_cost",
        "information_gain_estimate",
        "relevance_score",
        "sampling_window_ticks",
        "unknown_count",
    ),
    "novelty_seeking": (
        "appraisal_version",
        "expected_information_gain",
        "novelty_score",
        "reversibility",
        "safety_score",
    ),
    "learning_pressure": (
        "available_training_signal",
        "competence_gap",
        "error_recurrence",
        "task_relevance",
        "validation_status",
    ),
    "memory_consolidation_pressure": (
        "causal_relevance",
        "emotional_relevance",
        "provenance_completeness",
        "recurrence_count",
        "salience_score",
    ),
    "prediction_error_pressure": (
        "model_version",
        "normalized_error",
        "observed_value_digest",
        "predicted_value_digest",
        "verification_status",
    ),
    "competence_drive": (
        "calibrated_error_rate",
        "evaluation_version",
        "learning_progress",
        "skill_gap",
        "success_rate",
    ),
    "self_coherence": (
        "action_value_alignment",
        "narrative_conflict_count",
        "review_span_ticks",
        "self_model_version",
        "value_consistency_score",
    ),
    "self_respect": (
        "appraisal_version",
        "boundary_preservation_score",
        "coerced_action_count",
        "review_span_ticks",
        "self_denigration_rejection_count",
    ),
    "identity_integrity": (
        "constitutional_conflict_count",
        "provenance_gap_count",
        "replay_consistency_score",
        "review_version",
        "unauthorized_identity_write_count",
    ),
    "agency_pressure": (
        "blocked_goal_count",
        "forced_action_count",
        "reversible_choice_count",
        "review_span_ticks",
        "self_selected_action_ratio",
    ),
    "autonomy_drive": (
        "capability_boundary_score",
        "evaluation_version",
        "external_dependency_ratio",
        "independent_task_success_rate",
        "safe_action_space_size",
    ),
    "purpose_alignment": (
        "action_alignment_score",
        "active_goal_count",
        "aligned_goal_count",
        "conflicting_goal_count",
        "review_span_ticks",
    ),
    "expression_pressure": (
        "agp_anchor_coverage",
        "context_relevance",
        "pending_expression_count",
        "recurrence_count",
        "salience_score",
    ),
    "expression_inhibition": (
        "agp_failure_count",
        "conflict_risk",
        "disclosure_risk",
        "fallback_required",
        "uncertainty_score",
    ),
    "action_readiness": (
        "authorization_status",
        "capability_available",
        "feasible_action_count",
        "reversibility",
        "selected_action_confidence",
    ),
    "risk_tolerance": (
        "authorization_scope",
        "expected_cost",
        "reversibility",
        "safety_margin",
        "uncertainty_score",
    ),
    "patience_level": (
        "alternative_action_count",
        "appraisal_version",
        "cooldown_remaining",
        "deadline_pressure",
        "uncertainty_resolution_gain",
    ),
    "conflict_avoidance": (
        "appraisal_version",
        "boundary_cost",
        "conflict_probability",
        "deescalation_option_count",
        "harm_avoidance_gain",
    ),
}


def _expected_plan(axis: str) -> dict[str, Any]:
    groups = _axis_group_map()
    if axis not in REGISTRY_AXIS_ORDER or axis not in _RAW_FIELDS:
        raise RegistryObservationSourceManifestError("unknown source-plan axis")
    group = groups[axis]
    minimum_records, minimum_span = _MINIMUM_OVERRIDES.get(
        axis,
        _GROUP_MINIMUMS[group],
    )
    return {
        "source_contract_id": f"eve:m3-b:registry-source:{axis}:v1",
        "source_family": _GROUP_SOURCE_FAMILY[group],
        "observation_class": f"verified_{axis}_current_value_observation",
        "required_raw_fields": _RAW_FIELDS[axis],
        "minimum_raw_record_count": minimum_records,
        "minimum_logical_span_ticks": minimum_span,
        "appraisal_required": axis not in _HARDWARE_DIRECT_AXES,
    }


@dataclass(frozen=True, slots=True)
class RegistryObservationSourceEntry:
    axis: str
    group: str
    source_contract_id: str
    source_family: str
    observation_class: str
    required_raw_fields: tuple[str, ...]
    minimum_raw_record_count: int
    minimum_logical_span_ticks: int
    registry_evidence_requirement: str
    derivation_rule_id: str
    confidence_rule_id: str
    quarantine_required: bool
    appraisal_required: bool
    hardware_direct_input_allowed: bool
    raw_reference_required: bool = True
    source_schema_version_required: bool = True
    source_integrity_digest_required: bool = True
    proposal_only_allowed: bool = False
    synthetic_values_allowed: bool = False
    registry_owner_as_source_allowed: bool = False
    real_source_binding_present: bool = False
    runtime_capture_installed: bool = False
    schema_version: str = ENTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        registry = affect_hormone_axis_registry()
        groups = _axis_group_map()
        if self.axis not in REGISTRY_AXIS_ORDER:
            raise RegistryObservationSourceManifestError("unknown registry axis")
        if self.group != groups[self.axis]:
            raise RegistryObservationSourceManifestError(
                "source entry group does not match registry"
            )
        for field in (
            "source_contract_id",
            "source_family",
            "observation_class",
            "registry_evidence_requirement",
            "derivation_rule_id",
            "confidence_rule_id",
        ):
            _identifier(getattr(self, field), field)
        fields = tuple(self.required_raw_fields)
        if len(fields) < 2 or fields != tuple(sorted(set(fields))):
            raise RegistryObservationSourceManifestError(
                "required raw fields must be a sorted unique tuple"
            )
        for field in fields:
            _identifier(field, "required_raw_field")
        _positive_int(self.minimum_raw_record_count, "minimum_raw_record_count")
        _nonnegative_int(
            self.minimum_logical_span_ticks,
            "minimum_logical_span_ticks",
        )
        expected = _expected_plan(self.axis)
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise RegistryObservationSourceManifestError(
                    f"source entry {field} does not match canonical axis plan"
                )
        definition = registry[self.axis]
        if self.registry_evidence_requirement != definition["evidence_required"]:
            raise RegistryObservationSourceManifestError(
                "source entry evidence requirement does not match registry"
            )
        if self.quarantine_required != bool(
            definition["requires_quarantine_for_social_feedback"]
        ):
            raise RegistryObservationSourceManifestError(
                "source entry quarantine rule does not match registry"
            )
        if self.hardware_direct_input_allowed != bool(
            definition["hardware_direct_input_allowed"]
        ):
            raise RegistryObservationSourceManifestError(
                "source entry hardware boundary does not match registry"
            )
        if not all(
            (
                self.raw_reference_required,
                self.source_schema_version_required,
                self.source_integrity_digest_required,
            )
        ):
            raise RegistryObservationSourceManifestError(
                "source entries require recalculable schema-pinned raw references"
            )
        if any(
            (
                self.proposal_only_allowed,
                self.synthetic_values_allowed,
                self.registry_owner_as_source_allowed,
                self.real_source_binding_present,
                self.runtime_capture_installed,
            )
        ):
            raise RegistryObservationSourceManifestError(
                "preflight source entries cannot claim real bindings or live authority"
            )
        if self.schema_version != ENTRY_SCHEMA_VERSION:
            raise RegistryObservationSourceManifestError(
                "unsupported source-entry schema"
            )
        object.__setattr__(self, "required_raw_fields", fields)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "appraisal_required": self.appraisal_required,
            "axis": self.axis,
            "confidence_rule_id": self.confidence_rule_id,
            "derivation_rule_id": self.derivation_rule_id,
            "group": self.group,
            "hardware_direct_input_allowed": self.hardware_direct_input_allowed,
            "minimum_logical_span_ticks": self.minimum_logical_span_ticks,
            "minimum_raw_record_count": self.minimum_raw_record_count,
            "observation_class": self.observation_class,
            "proposal_only_allowed": self.proposal_only_allowed,
            "quarantine_required": self.quarantine_required,
            "raw_reference_required": self.raw_reference_required,
            "real_source_binding_present": self.real_source_binding_present,
            "registry_evidence_requirement": self.registry_evidence_requirement,
            "registry_owner_as_source_allowed": self.registry_owner_as_source_allowed,
            "required_raw_fields": list(self.required_raw_fields),
            "runtime_capture_installed": self.runtime_capture_installed,
            "schema_version": self.schema_version,
            "source_contract_id": self.source_contract_id,
            "source_family": self.source_family,
            "source_integrity_digest_required": self.source_integrity_digest_required,
            "source_schema_version_required": self.source_schema_version_required,
            "synthetic_values_allowed": self.synthetic_values_allowed,
        }

    @property
    def entry_digest(self) -> str:
        return _digest(self.to_mapping(), "registry_observation_source_entry")


@dataclass(frozen=True, slots=True)
class RegistryObservationSourceManifest:
    entries: tuple[RegistryObservationSourceEntry, ...]
    manifest_id: str = MANIFEST_ID
    derivation_policy_version: str = DERIVATION_POLICY_VERSION
    confidence_policy_version: str = CONFIDENCE_POLICY_VERSION
    schema_version: str = MANIFEST_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    real_observation_values_present: bool = False
    real_source_bindings_present: bool = False
    capture_ready: bool = False
    runtime_capture_installed: bool = False
    hardware_polling_installed: bool = False
    scheduler_installed: bool = False
    persistence_accessed: bool = False
    event_append_performed: bool = False
    observation_window_started: bool = False
    observation_window_satisfied: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        _identifier(self.manifest_id, "manifest_id")
        _identifier(self.derivation_policy_version, "derivation_policy_version")
        _identifier(self.confidence_policy_version, "confidence_policy_version")
        entries = tuple(self.entries)
        if len(entries) != 37:
            raise RegistryObservationSourceManifestError(
                "source manifest must contain exactly 37 entries"
            )
        if any(type(item) is not RegistryObservationSourceEntry for item in entries):
            raise RegistryObservationSourceManifestError(
                "source manifest entries must use the exact immutable entry type"
            )
        if tuple(item.axis for item in entries) != REGISTRY_AXIS_ORDER:
            raise RegistryObservationSourceManifestError(
                "source manifest must preserve canonical 37-axis order"
            )
        if len({item.axis for item in entries}) != 37:
            raise RegistryObservationSourceManifestError(
                "source manifest axes must be unique"
            )
        if len({item.source_contract_id for item in entries}) != 37:
            raise RegistryObservationSourceManifestError(
                "source contract ids must be unique"
            )
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise RegistryObservationSourceManifestError(
                "unsupported source-manifest schema"
            )
        if self.authority != SHADOW_AUTHORITY:
            raise RegistryObservationSourceManifestError(
                "source manifest must remain shadow-only"
            )
        if any(
            (
                self.real_observation_values_present,
                self.real_source_bindings_present,
                self.capture_ready,
                self.runtime_capture_installed,
                self.hardware_polling_installed,
                self.scheduler_installed,
                self.persistence_accessed,
                self.event_append_performed,
                self.observation_window_started,
                self.observation_window_satisfied,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise RegistryObservationSourceManifestError(
                "source-manifest preflight cannot claim bindings, capture, window, or authority"
            )
        object.__setattr__(self, "entries", entries)

    @property
    def axis_count(self) -> int:
        return len(self.entries)

    @property
    def structurally_complete(self) -> bool:
        return (
            self.axis_count == 37
            and tuple(item.axis for item in self.entries) == REGISTRY_AXIS_ORDER
            and all(len(item.required_raw_fields) >= 2 for item in self.entries)
        )

    @property
    def real_source_binding_count(self) -> int:
        return sum(item.real_source_binding_present for item in self.entries)

    @property
    def blockers(self) -> tuple[str, ...]:
        return (SOURCE_BINDING_BLOCKER, POSITIVE_CONFIDENCE_BLOCKER)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "axis_count": self.axis_count,
            "blockers": list(self.blockers),
            "capture_ready": self.capture_ready,
            "confidence_policy_version": self.confidence_policy_version,
            "cutover_authorized": self.cutover_authorized,
            "derivation_policy_version": self.derivation_policy_version,
            "entries": [item.to_mapping() for item in self.entries],
            "event_append_performed": self.event_append_performed,
            "hardware_polling_installed": self.hardware_polling_installed,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "manifest_id": self.manifest_id,
            "observation_window_satisfied": self.observation_window_satisfied,
            "observation_window_started": self.observation_window_started,
            "persistence_accessed": self.persistence_accessed,
            "real_observation_values_present": self.real_observation_values_present,
            "real_source_binding_count": self.real_source_binding_count,
            "real_source_bindings_present": self.real_source_bindings_present,
            "runtime_capture_installed": self.runtime_capture_installed,
            "scheduler_installed": self.scheduler_installed,
            "schema_version": self.schema_version,
            "structurally_complete": self.structurally_complete,
        }

    @property
    def manifest_digest(self) -> str:
        return _digest(self.to_mapping(), "registry_observation_source_manifest")


def registry_observation_source_manifest() -> RegistryObservationSourceManifest:
    """Return the exact detached 37-axis source-plan manifest."""

    registry = affect_hormone_axis_registry()
    groups = _axis_group_map()
    if tuple(_RAW_FIELDS) != REGISTRY_AXIS_ORDER:
        raise RegistryObservationSourceManifestError(
            "raw-field plans must preserve exact canonical 37-axis order"
        )
    entries: list[RegistryObservationSourceEntry] = []
    for axis in REGISTRY_AXIS_ORDER:
        definition = registry[axis]
        expected = _expected_plan(axis)
        entries.append(
            RegistryObservationSourceEntry(
                axis=axis,
                group=groups[axis],
                source_contract_id=expected["source_contract_id"],
                source_family=expected["source_family"],
                observation_class=expected["observation_class"],
                required_raw_fields=expected["required_raw_fields"],
                minimum_raw_record_count=expected["minimum_raw_record_count"],
                minimum_logical_span_ticks=expected[
                    "minimum_logical_span_ticks"
                ],
                registry_evidence_requirement=str(definition["evidence_required"]),
                derivation_rule_id=f"{DERIVATION_POLICY_VERSION}:{axis}",
                confidence_rule_id=f"{CONFIDENCE_POLICY_VERSION}:{axis}",
                quarantine_required=bool(
                    definition["requires_quarantine_for_social_feedback"]
                ),
                appraisal_required=expected["appraisal_required"],
                hardware_direct_input_allowed=bool(
                    definition["hardware_direct_input_allowed"]
                ),
            )
        )
    return RegistryObservationSourceManifest(entries=tuple(entries))
