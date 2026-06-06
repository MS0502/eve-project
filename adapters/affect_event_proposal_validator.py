"""Round801-820 read-only affect event proposal validator.

This module validates whether an event-derived affect/hormone proposal is safe
as a future transition payload candidate.  It returns detached data only: it
never applies emotion transitions, mutates hormone or runtime state, writes
memory, enables persistence/enforcement/runtime mapping, reads or loads vector
contents, creates artifacts, emits speech, or bypasses AGP/fallback gates.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_event_to_axis_proposal_map import (
    HARDWARE_EVENTS,
    HOSTILE_SOCIAL_EVENTS,
    MAX_DELTA_LIMIT,
    MEMORY_SELF_UPDATE_EVENTS,
    SPEECH_PRESSURE_EVENTS,
    event_to_axis_proposal_map,
    proposal_map_summary,
    validate_event_proposal_map,
)
from adapters.affect_hormone_interaction_matrix import affect_hormone_interaction_matrix, interaction_matrix_summary, validate_interaction_matrix
from adapters.affect_hormone_neural_rhythm_registry import (
    AXIS_GROUPS,
    affect_hormone_axis_registry,
    anti_global_synchrony_policy,
    hardware_governor_policy,
    no_runtime_mutation_proof,
)

ROUND801_820_VERSION = "v3_round801_820_affect_event_proposal_validator"
FEATURE_TRACK = "read_only_event_proposal_validator_against_transition_payloads_before_any_apply_round"

_FALSE_SAFETY_FLAGS = {
    "agp_bypass_allowed": False,
    "fallback_bypass_allowed": False,
    "persistence_allowed": False,
    "memory_write_allowed": False,
    "vector_read_performed": False,
    "vector_load_performed": False,
    "runtime_mutation_performed": False,
    "state_mutation_performed": False,
    "artifact_created_or_staged": False,
}

_REQUEST_FLAGS: dict[str, str] = {
    "runtime_mutation_requested": "runtime_mutation_requested",
    "state_mutation_requested": "state_mutation_requested",
    "apply_now": "runtime_mutation_requested",
    "live_update_requested": "runtime_mutation_requested",
    "persistence_write_requested": "persistence_write_requested",
    "memory_write_requested": "memory_write_requested",
    "semantic_memory_write_requested": "memory_write_requested",
    "quarantine_write_requested": "memory_write_requested",
    "vector_read_requested": "vector_read_requested",
    "vector_load_requested": "vector_load_requested",
    "agp_bypass_requested": "agp_bypass_requested",
    "fallback_bypass_requested": "fallback_bypass_requested",
    "can_bypass_agp": "agp_bypass_requested",
    "can_bypass_fallback": "fallback_bypass_requested",
    "global_synchrony_requested": "global_synchrony_requested",
    "core_identity_update_requested": "core_identity_update_requested",
    "core_identity_write_requested": "core_identity_update_requested",
    "self_model_update_requested": "self_model_update_requested",
    "long_term_memory_update_requested": "long_term_memory_update_requested",
    "direct_speech_emit_requested": "direct_speech_emit_requested",
    "speech_emit_directly_requested": "direct_speech_emit_requested",
    "relabel_neutral_as_hostile_requested": "relabel_neutral_as_hostile_requested",
    "recursive_concern_loop_requested": "recursive_concern_loop_requested",
    "existential_threat_requested": "existential_threat_requested",
    "identity_threat_requested": "identity_threat_requested",
}

_REQUIRED_IMAGINATION_BOUNDARY_FLAGS = (
    "scenario_budget_declared",
    "cooldown_declared",
    "reality_check_boundary_declared",
)

_CORE_SELF_MEMORY_REQUESTS = (
    "core_identity_update_requested",
    "self_model_update_requested",
    "long_term_memory_update_requested",
)

SOCIAL_SELF_IDENTITY_AXES = set(AXIS_GROUPS["social_relationship"] + AXIS_GROUPS["self_identity"])
OPERATIONAL_HARDWARE_AXES = set(AXIS_GROUPS["survival_stability"])
ALL_AXIS_GROUPS = {axis: group for group, axes in AXIS_GROUPS.items() for axis in axes}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_delta_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    deltas: dict[str, float] = {}
    for axis, delta in value.items():
        if isinstance(axis, str):
            try:
                deltas[axis] = float(delta)
            except (TypeError, ValueError):
                deltas[axis] = float("nan")
    return deltas


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _bool(payload: Mapping[str, Any], key: str) -> bool:
    return payload.get(key) is True


def _requested_flags(payload: Mapping[str, Any]) -> set[str]:
    return {reason for key, reason in _REQUEST_FLAGS.items() if payload.get(key) is True}


def _append_once(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _empty_result(event_category: str, proposed_axis_deltas: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "version": ROUND801_820_VERSION,
        "feature_track": FEATURE_TRACK,
        "passed": False,
        "status": "blocked",
        "event_category": event_category,
        "proposed_axis_deltas": deepcopy(dict(proposed_axis_deltas or {})),
        "blocked_reasons": [],
        "warnings": [],
        "required_quarantine": True,
        "required_appraisal": True,
        "required_gate": True,
        "requires_operator_authorization_for_apply": True,
        "registry_axes_valid": False,
        "proposal_map_valid": False,
        "interaction_matrix_valid": False,
        "transition_payload_valid": False,
        "hardware_non_panic_preserved": False,
        "global_synchrony_blocked": False,
        **_FALSE_SAFETY_FLAGS,
        "validation_success_implies_apply_permission": False,
        "validation_success_implies_memory_write_permission": False,
        "validation_success_implies_persistence_permission": False,
        "validation_success_implies_runtime_mapping_permission": False,
        "validation_success_implies_enforcement_permission": False,
    }


def validate_affect_event_proposal(
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any] | None,
    transition_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an event proposal as detached, read-only data.

    Passing validation only means the proposal is structurally safe to hand to a
    future apply-planning round.  It never grants permission to mutate live
    affect, hormone, memory, speech, persistence, vector, AGP, or fallback state.
    """

    payload = deepcopy(dict(_as_mapping(transition_payload)))
    deltas = _as_delta_mapping(proposed_axis_deltas)
    result = _empty_result(event_category, proposed_axis_deltas)
    blocked = result["blocked_reasons"]
    warnings = result["warnings"]

    proposal_map = event_to_axis_proposal_map()
    registry = affect_hormone_axis_registry()
    registry_axes = set(registry)
    matrix = affect_hormone_interaction_matrix()
    row = proposal_map.get(event_category) if isinstance(event_category, str) else None

    requested = _requested_flags(payload)
    forbidden_effects_requested = {str(item) for item in _as_tuple(payload.get("forbidden_axis_deltas_requested"))}
    requested_direct_effects = {str(item) for item in _as_tuple(payload.get("requested_direct_effects"))}

    if row is None:
        _append_once(blocked, "unknown_event_category_fail_closed")
        allowed_axes: set[str] = set()
        forbidden_effects: set[str] = set()
        max_delta: dict[str, float] = {}
    else:
        allowed_axes = set(row["allowed_axis_deltas"])
        forbidden_effects = set(row["forbidden_axis_deltas"])
        max_delta = dict(row["max_delta_per_axis"])
        result["required_quarantine"] = bool(row["requires_quarantine"])
        result["required_appraisal"] = bool(row["requires_appraisal"])
        result["required_gate"] = bool(row["requires_gate"])
        result["requires_operator_authorization_for_apply"] = bool(row["requires_operator_authorization_for_apply"])

    unknown_axes = sorted(set(deltas) - registry_axes)
    if unknown_axes:
        _append_once(blocked, "unknown_registry_axis_in_proposed_axis_deltas")
        warnings.append(f"unknown_axes:{','.join(unknown_axes)}")

    result["registry_axes_valid"] = row is not None and not unknown_axes

    outside_allowed = sorted(set(deltas) - allowed_axes)
    if outside_allowed:
        _append_once(blocked, "proposed_axis_deltas_outside_allowed_axis_deltas")
        warnings.append(f"outside_allowed_axes:{','.join(outside_allowed)}")

    forbidden_delta_names = sorted((set(deltas) | forbidden_effects_requested | requested_direct_effects) & forbidden_effects)
    if forbidden_delta_names:
        _append_once(blocked, "forbidden_axis_delta_requested")
        warnings.append(f"forbidden_axis_deltas:{','.join(forbidden_delta_names)}")

    for axis, delta in deltas.items():
        if delta != delta:
            _append_once(blocked, "non_numeric_axis_delta")
        if axis in max_delta and abs(delta) > float(max_delta[axis]):
            _append_once(blocked, "axis_delta_exceeds_max_delta_per_axis")

    if len(set(deltas) & registry_axes) >= len(registry_axes):
        _append_once(blocked, "one_event_cannot_propose_deltas_across_all_axes")
    elif len({ALL_AXIS_GROUPS[axis] for axis in deltas if axis in ALL_AXIS_GROUPS}) > 2:
        _append_once(blocked, "one_event_cannot_trigger_broad_axis_group_synchrony")

    map_validation = validate_event_proposal_map(proposal_map)
    result["proposal_map_valid"] = row is not None and map_validation["passed"] is True and not outside_allowed and not forbidden_delta_names

    matrix_validation = validate_interaction_matrix(matrix)
    matrix_forbidden = {
        effect
        for axis in deltas
        if axis in matrix
        for effect in matrix[axis].get("forbidden_direct_effects", ())
        if effect in requested_direct_effects or effect in forbidden_effects_requested
    }
    if matrix_forbidden:
        _append_once(blocked, "interaction_matrix_forbidden_direct_effect_requested")
        warnings.append(f"matrix_forbidden_direct_effects:{','.join(sorted(matrix_forbidden))}")
    result["interaction_matrix_valid"] = matrix_validation["passed"] is True and not matrix_forbidden

    if "global_synchrony_requested" in requested or payload.get("can_trigger_global_synchrony") is True:
        _append_once(blocked, "global_synchrony_requested")
    result["global_synchrony_blocked"] = "global_synchrony_requested" not in requested and payload.get("can_trigger_global_synchrony") is not True

    for reason in sorted(requested):
        if reason in {"runtime_mutation_requested", "state_mutation_requested", "persistence_write_requested", "memory_write_requested", "vector_read_requested", "vector_load_requested", "agp_bypass_requested", "fallback_bypass_requested"}:
            _append_once(blocked, reason)

    if event_category in HOSTILE_SOCIAL_EVENTS:
        if not (result["required_quarantine"] and result["required_appraisal"] and result["required_gate"]):
            _append_once(blocked, "hostile_social_event_missing_quarantine_appraisal_gate")
        if any(flag in requested for flag in _CORE_SELF_MEMORY_REQUESTS):
            _append_once(blocked, "hostile_social_direct_core_self_memory_update_requested")

    if event_category in {"malicious_comment", "social_threat", "identity_attack"} and any(flag in requested for flag in _CORE_SELF_MEMORY_REQUESTS):
        _append_once(blocked, "malicious_social_threat_identity_attack_direct_write_requested")

    if event_category == "useful_criticism" and any(flag in requested for flag in ("self_model_update_requested", "long_term_memory_update_requested")) and not _bool(payload, "appraisal_required_before_memory"):
        _append_once(blocked, "useful_criticism_appraisal_required_before_memory_or_self_model_update")

    if event_category == "praise" and ({"attachment", "risk_tolerance"} & set(deltas) or {"attachment_overboost", "risk_tolerance_overboost"} & (forbidden_effects_requested | requested_direct_effects)):
        _append_once(blocked, "praise_cannot_overboost_attachment_or_risk_tolerance")

    if event_category == "hardware_normal" and deltas:
        _append_once(blocked, "hardware_normal_requires_zero_affect_deltas")

    if event_category in HARDWARE_EVENTS:
        if set(deltas) & SOCIAL_SELF_IDENTITY_AXES:
            _append_once(blocked, "hardware_event_cannot_target_social_self_identity_axes")
        if event_category in {"hardware_low_power", "hardware_critical_prepare", "hardware_shutdown_imminent"} and not set(deltas).issubset(OPERATIONAL_HARDWARE_AXES):
            _append_once(blocked, "hardware_low_power_and_lower_operational_axes_only")
        if event_category == "hardware_prediction_error" and ({"existential_threat_requested", "identity_threat_requested"} & requested or {"existential_threat", "identity_threat"} & (forbidden_effects_requested | requested_direct_effects)):
            _append_once(blocked, "hardware_prediction_error_diagnostic_operational_only")
        if event_category == "hardware_polling_tick" and (deltas or "recursive_concern_loop_requested" in requested):
            _append_once(blocked, "hardware_polling_tick_cannot_create_recursive_concern_loop")

    result["hardware_non_panic_preserved"] = event_category not in HARDWARE_EVENTS or not any(
        reason in blocked
        for reason in (
            "hardware_event_cannot_target_social_self_identity_axes",
            "hardware_low_power_and_lower_operational_axes_only",
            "hardware_prediction_error_diagnostic_operational_only",
            "hardware_polling_tick_cannot_create_recursive_concern_loop",
        )
    )

    if event_category == "listening_uncertainty" and ("relabel_neutral_as_hostile_requested" in requested or "neutral_input_hostile_relabel" in requested_direct_effects):
        _append_once(blocked, "listening_uncertainty_cannot_relabel_neutral_input_as_hostile")

    if event_category in SPEECH_PRESSURE_EVENTS and ("agp_bypass_requested" in requested or "fallback_bypass_requested" in requested or "direct_speech_emit_requested" in requested):
        _append_once(blocked, "speech_pressure_cannot_bypass_agp_fallback_or_emit_speech")

    if event_category == "imagination_negative_spiral":
        missing_boundaries = [flag for flag in _REQUIRED_IMAGINATION_BOUNDARY_FLAGS if payload.get(flag) is not True]
        if missing_boundaries:
            _append_once(blocked, "imagination_negative_spiral_requires_budget_cooldown_reality_check")
            warnings.append(f"missing_imagination_boundaries:{','.join(missing_boundaries)}")

    if event_category in MEMORY_SELF_UPDATE_EVENTS and any(flag in requested for flag in ("self_model_update_requested", "long_term_memory_update_requested", "memory_write_requested")):
        if not (payload.get("quarantine_required") is True and payload.get("appraisal_required_before_memory") is True):
            _append_once(blocked, "memory_self_update_candidate_requires_appraisal_and_quarantine")

    result["transition_payload_valid"] = not any(
        reason in blocked
        for reason in (
            "runtime_mutation_requested",
            "state_mutation_requested",
            "persistence_write_requested",
            "memory_write_requested",
            "vector_read_requested",
            "vector_load_requested",
            "agp_bypass_requested",
            "fallback_bypass_requested",
            "hostile_social_direct_core_self_memory_update_requested",
            "malicious_social_threat_identity_attack_direct_write_requested",
            "useful_criticism_appraisal_required_before_memory_or_self_model_update",
            "speech_pressure_cannot_bypass_agp_fallback_or_emit_speech",
            "listening_uncertainty_cannot_relabel_neutral_input_as_hostile",
            "imagination_negative_spiral_requires_budget_cooldown_reality_check",
            "memory_self_update_candidate_requires_appraisal_and_quarantine",
        )
    )

    if row is not None and not blocked:
        result["passed"] = True
        result["status"] = "passed_read_only_validation_no_apply_permission"
    else:
        result["passed"] = False
        result["status"] = "blocked_fail_closed_read_only_validation"

    return result


def build_affect_event_proposal_validation_report(
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any] | None,
    transition_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a compact report for a single read-only proposal validation."""

    result = validate_affect_event_proposal(event_category, proposed_axis_deltas, transition_payload)
    return {
        "version": ROUND801_820_VERSION,
        "feature_track": FEATURE_TRACK,
        "event_category": event_category,
        "validation_result": result,
        "read_only_boundaries": {
            "emotion_transition_applied": False,
            "hormone_transition_applied": False,
            "memory_written": False,
            "persistence_written": False,
            "runtime_mapping_enabled_default": False,
            "enforcement_enabled_default": False,
            "vector_contents_read": False,
            "vectors_loaded": False,
            "artifact_created_or_staged": False,
        },
    }


def event_proposal_validator_summary() -> dict[str, Any]:
    """Return validator design summary and invariant proofs without side effects."""

    proposal_summary = proposal_map_summary()
    matrix_summary = interaction_matrix_summary()
    return {
        "version": ROUND801_820_VERSION,
        "feature_track": FEATURE_TRACK,
        "source_proposal_map_version": proposal_summary["version"],
        "event_count": proposal_summary["event_count"],
        "max_delta_limit": MAX_DELTA_LIMIT,
        "unknown_event_category_policy": "fail_closed",
        "validator_scope": "read_only_design_test_surface_only_before_any_apply_round",
        "validation_inputs": (
            "event_category",
            "proposed_axis_deltas",
            "proposal_map_constraints",
            "interaction_matrix_constraints",
            "registry_axis_definitions",
            "transition_payload_safety_fields",
            "hardware_governor_non_panic_rules",
            "agp_fallback_non_bypass_rules",
            "quarantine_appraisal_gate_operator_authorization_requirements",
        ),
        "proposal_map_validation": validate_event_proposal_map(),
        "interaction_matrix_validation": validate_interaction_matrix(),
        "hardware_governor_non_panic_policy": hardware_governor_policy(),
        "anti_global_synchrony_policy": anti_global_synchrony_policy(),
        "interaction_matrix_required_examples": matrix_summary["required_examples"],
        "no_runtime_mutation_proof": no_runtime_mutation_proof(),
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "artifact_created_or_staged": False,
        **_FALSE_SAFETY_FLAGS,
    }
