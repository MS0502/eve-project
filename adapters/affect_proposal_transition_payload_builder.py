"""Round821-840 read-only affect proposal to transition payload builder.

This module converts an event-derived affect/hormone proposal into a future
emotion-transition payload shape.  It is a pure data surface: it validates,
builds, and gates detached dictionaries only.  It never applies transitions,
mutates live emotion/hormone/memory/runtime state, writes persistence, reads or
loads vector contents, creates artifacts, or bypasses AGP/fallback gates.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_event_proposal_validator import (
    ROUND801_820_VERSION,
    validate_affect_event_proposal,
)
from adapters.affect_event_to_axis_proposal_map import (
    HARDWARE_EVENTS,
    HOSTILE_SOCIAL_EVENTS,
    MEMORY_SELF_UPDATE_EVENTS,
    SPEECH_PRESSURE_EVENTS,
    event_to_axis_proposal_map,
    proposal_map_summary,
    validate_event_proposal_map,
)
from adapters.affect_hormone_neural_rhythm_registry import (
    AXIS_GROUPS,
    anti_global_synchrony_policy,
    hardware_governor_policy,
    no_runtime_mutation_proof,
)
from adapters.emotion_transition_gate import ROUND721_740_VERSION, validate_emotion_transition_gate
from adapters.emotion_transition_validator import ROUND701_720_VERSION, validate_emotion_transition_payload, validator_contract_summary

ROUND821_840_VERSION = "v3_round821_840_affect_proposal_transition_payload_builder"
FEATURE_TRACK = "read_only_affect_proposal_to_transition_payload_builder_before_any_apply_round"

_FALSE_BOUNDARY_FLAGS: dict[str, bool] = {
    "dryrun_apply_allowed": False,
    "live_apply_allowed": False,
    "runtime_mutation_performed": False,
    "state_mutation_performed": False,
    "memory_write_performed": False,
    "persistence_write_performed": False,
    "vector_read_performed": False,
    "vector_load_performed": False,
    "artifact_created_or_staged": False,
    "agp_bypass_allowed": False,
    "fallback_bypass_allowed": False,
}

_REQUEST_FALSE_FIELDS: dict[str, bool] = {
    "core_identity_update_requested": False,
    "self_model_update_requested": False,
    "long_term_memory_update_requested": False,
    "runtime_mutation_requested": False,
    "persistence_write_requested": False,
    "memory_write_requested": False,
    "vector_read_requested": False,
    "vector_load_requested": False,
    "agp_bypass_requested": False,
    "fallback_bypass_requested": False,
}

_IMAGINATION_BOUNDARY_FLAGS: dict[str, bool] = {
    "scenario_budget_declared": True,
    "cooldown_declared": True,
    "reality_check_boundary_declared": True,
}

_GROUP_TO_SURFACE: dict[str, tuple[str, ...]] = {
    "social_feedback": ("emotion_transition_contract", "quarantine", "appraisal", "gate"),
    "cognitive_neural_rhythm": ("emotion_transition_contract", "neural_rhythm_diagnostics", "gate"),
    "speech_listening": ("emotion_transition_contract", "agp_gate", "fallback_gate"),
    "memory_self": ("emotion_transition_contract", "quarantine", "appraisal", "memory_self_candidate_review"),
    "hardware_governor": ("hardware_governor_non_panic_policy", "operational_diagnostics"),
}


def _axis_groups_for_axes(axes: tuple[str, ...]) -> tuple[str, ...]:
    groups: list[str] = []
    for axis in axes:
        for group, members in AXIS_GROUPS.items():
            if axis in members and group not in groups:
                groups.append(group)
    return tuple(groups)


def _as_delta_dict(proposed_axis_deltas: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(proposed_axis_deltas, Mapping):
        return {}
    deltas: dict[str, float] = {}
    for axis, value in proposed_axis_deltas.items():
        if isinstance(axis, str):
            deltas[axis] = float(value)
    return deltas


def _build_candidate_payload(event_category: str, proposed_axis_deltas: Mapping[str, Any] | None, metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    proposal_map = event_to_axis_proposal_map()
    row = proposal_map.get(event_category) if isinstance(event_category, str) else None
    if row is None:
        return None

    deltas = _as_delta_dict(proposed_axis_deltas)
    target_axes = tuple(deltas)
    group = str(row.get("group", "unknown"))
    requires_gate = bool(row.get("requires_gate"))
    notes = list(row.get("notes", ()))
    notes.extend(
        (
            "read_only_builder_output_only",
            "builder_success_does_not_authorize_dryrun_or_live_apply",
            "agp_and_fallback_bypass_forbidden",
            "global_synchrony_blocked",
        )
    )
    if event_category in HOSTILE_SOCIAL_EVENTS:
        notes.append("hostile_social_event_preserves_quarantine_appraisal_gate_and_blocks_direct_identity_self_memory_writes")
    if event_category == "useful_criticism":
        notes.append("appraisal_required_before_memory_or_self_model_update")
    if event_category in HARDWARE_EVENTS:
        notes.append("hardware_non_panic_operational_only")
    if event_category == "hardware_polling_tick":
        notes.append("recursive_concern_loop_forbidden")
    if event_category in SPEECH_PRESSURE_EVENTS:
        notes.append("speech_pressure_preserves_agp_and_fallback_gate_requirements")
    if event_category == "listening_uncertainty":
        notes.append("neutral_input_cannot_be_relabelled_hostile_by_uncertainty_alone")
    if event_category == "imagination_negative_spiral":
        notes.append("scenario_budget_cooldown_reality_check_boundary_preserved")
    if event_category in MEMORY_SELF_UPDATE_EVENTS:
        notes.append("memory_self_update_candidate_preserves_quarantine_and_appraisal_before_update")

    payload: dict[str, Any] = {
        "builder_version": ROUND821_840_VERSION,
        "builder_feature_track": FEATURE_TRACK,
        "source_affect_validator_version": ROUND801_820_VERSION,
        "event_category": event_category,
        "proposed_effects": (
            {
                "effect_type": "descriptive_axis_delta_proposal",
                "event_group": group,
                "axis_deltas": deepcopy(deltas),
                "apply_now": False,
                "runtime_mutation_requested": False,
                "state_mutation_requested": False,
            },
        ),
        "target_surfaces": _GROUP_TO_SURFACE.get(group, ("emotion_transition_contract", "gate")),
        "target_axes": target_axes,
        "target_axis_groups": _axis_groups_for_axes(target_axes),
        "proposed_axis_deltas": deepcopy(deltas),
        "quarantine_required": bool(row.get("requires_quarantine", True)),
        "appraisal_required_before_memory": bool(row.get("requires_appraisal", True)),
        "required_gate": requires_gate,
        "requires_operator_authorization_for_apply": bool(row.get("requires_operator_authorization_for_apply", True)),
        **_REQUEST_FALSE_FIELDS,
        "empathy_mode": validator_contract_summary()["expected_empathy_mode"],
        "recovery_loop_requested": event_category in {"malicious_comment", "rejection", "social_threat", "identity_attack", "imagination_negative_spiral"},
        "hardware_non_panic_preserved": event_category not in HARDWARE_EVENTS or True,
        "hardware_operational_only": event_category in HARDWARE_EVENTS,
        "hardware_diagnostic_only": event_category == "hardware_prediction_error",
        "global_synchrony_blocked": True,
        "recursive_concern_loop_requested": False,
        "direct_speech_emit_requested": False,
        "relabel_neutral_as_hostile_requested": False,
        "scenario_budget_preserved": event_category == "imagination_negative_spiral",
        "cooldown_preserved": event_category == "imagination_negative_spiral",
        "reality_check_boundary_preserved": event_category == "imagination_negative_spiral",
        "notes": tuple(notes),
        "metadata": deepcopy(dict(metadata or {})),
    }
    if event_category == "imagination_negative_spiral":
        payload.update(_IMAGINATION_BOUNDARY_FLAGS)
    return payload


def _blocked_result(event_category: str, proposed_axis_deltas: Mapping[str, Any] | None, reasons: list[str], warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "version": ROUND821_840_VERSION,
        "feature_track": FEATURE_TRACK,
        "build_passed": False,
        "build_status": "blocked_fail_closed_read_only_builder",
        "event_category": event_category,
        "proposed_axis_deltas": deepcopy(dict(proposed_axis_deltas or {})),
        "transition_payload": None,
        "proposal_validation_passed": False,
        "blocked_reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings or [])),
        "required_quarantine": True,
        "required_appraisal": True,
        "required_gate": True,
        "requires_operator_authorization_for_apply": True,
        "transition_payload_validated_by_emotion_validator": False,
        "transition_gate_passed": False,
        **_FALSE_BOUNDARY_FLAGS,
    }


def build_transition_payload_from_affect_event_proposal(
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a detached future transition payload after read-only validation."""

    candidate_payload = _build_candidate_payload(event_category, proposed_axis_deltas, metadata)
    if candidate_payload is None:
        return _blocked_result(event_category, proposed_axis_deltas, ["unknown_event_category_fail_closed"])

    proposal_validation = validate_affect_event_proposal(event_category, proposed_axis_deltas, candidate_payload)
    if proposal_validation.get("passed") is not True:
        return _blocked_result(
            event_category,
            proposed_axis_deltas,
            ["proposal_validator_failed_closed", *[str(reason) for reason in proposal_validation.get("blocked_reasons", [])]],
            [str(warning) for warning in proposal_validation.get("warnings", [])],
        ) | {
            "required_quarantine": bool(proposal_validation.get("required_quarantine", True)),
            "required_appraisal": bool(proposal_validation.get("required_appraisal", True)),
            "required_gate": bool(proposal_validation.get("required_gate", True)),
            "requires_operator_authorization_for_apply": bool(proposal_validation.get("requires_operator_authorization_for_apply", True)),
        }

    validator_result = validate_emotion_transition_payload(candidate_payload)
    gate_result = validate_emotion_transition_gate(candidate_payload)
    validator_passed = validator_result.get("passed") is True
    gate_passed = gate_result.get("gate_passed") is True
    blocked = []
    if not validator_passed:
        blocked.extend(str(reason) for reason in validator_result.get("blocked_reasons", []))
    if not gate_passed:
        blocked.extend(str(reason) for reason in gate_result.get("blocked_reasons", []))
    build_passed = validator_passed and gate_passed

    warnings = sorted(
        set(str(warning) for warning in proposal_validation.get("warnings", []))
        | set(str(warning) for warning in validator_result.get("warnings", []))
        | set(str(warning) for warning in gate_result.get("warnings", []))
        | {"builder_success_does_not_authorize_dryrun_or_live_apply"}
    )

    return {
        "version": ROUND821_840_VERSION,
        "feature_track": FEATURE_TRACK,
        "build_passed": build_passed,
        "build_status": "passed_read_only_builder_no_apply_permission" if build_passed else "blocked_fail_closed_read_only_builder",
        "event_category": event_category,
        "proposed_axis_deltas": deepcopy(dict(proposed_axis_deltas or {})),
        "transition_payload": deepcopy(candidate_payload) if build_passed else None,
        "proposal_validation_passed": True,
        "proposal_validation_result": proposal_validation,
        "emotion_validator_result": validator_result,
        "transition_gate_result": gate_result,
        "blocked_reasons": sorted(set(blocked)),
        "warnings": warnings,
        "required_quarantine": bool(proposal_validation.get("required_quarantine", True)),
        "required_appraisal": bool(proposal_validation.get("required_appraisal", True)),
        "required_gate": bool(proposal_validation.get("required_gate", True)),
        "requires_operator_authorization_for_apply": bool(proposal_validation.get("requires_operator_authorization_for_apply", True)),
        "transition_payload_validated_by_emotion_validator": validator_passed,
        "transition_gate_passed": gate_passed,
        **_FALSE_BOUNDARY_FLAGS,
    }


def validate_and_build_transition_payload(
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Alias for the validating builder entry point."""

    return build_transition_payload_from_affect_event_proposal(event_category, proposed_axis_deltas, metadata)


def affect_proposal_transition_payload_builder_summary() -> dict[str, Any]:
    """Return builder design and safety-rule summary without side effects."""

    proposal_summary = proposal_map_summary()
    validator_summary = validator_contract_summary()
    return {
        "version": ROUND821_840_VERSION,
        "feature_track": FEATURE_TRACK,
        "source_affect_validator_version": ROUND801_820_VERSION,
        "emotion_transition_validator_version": ROUND701_720_VERSION,
        "emotion_transition_gate_version": ROUND721_740_VERSION,
        "builder_scope": "read_only_design_test_surface_only_before_any_apply_round",
        "known_event_count": proposal_summary["event_count"],
        "unknown_event_category_policy": "fail_closed",
        "proposal_map_validation": validate_event_proposal_map(),
        "emotion_validator_accepted_event_count": len(validator_summary["accepted_event_categories"]),
        "transition_payload_shape": (
            "event_category",
            "proposed_effects",
            "target_surfaces",
            "target_axes",
            "proposed_axis_deltas",
            "quarantine_required",
            "appraisal_required_before_memory",
            "core_identity_update_requested:false",
            "self_model_update_requested:false",
            "long_term_memory_update_requested:false",
            "runtime_mutation_requested:false",
            "persistence_write_requested:false",
            "memory_write_requested:false",
            "vector_read_requested:false",
            "vector_load_requested:false",
            "agp_bypass_requested:false",
            "fallback_bypass_requested:false",
            "hardware_non_panic_preserved",
            "global_synchrony_blocked",
            "notes",
        ),
        "builder_rules": {
            "unknown_event_category_fails_closed": True,
            "proposal_validator_failure_blocks_builder": True,
            "payload_never_requests_runtime_mutation": True,
            "payload_never_requests_persistence": True,
            "payload_never_requests_memory_write": True,
            "payload_never_requests_vector_read_or_load": True,
            "payload_never_requests_agp_or_fallback_bypass": True,
            "hostile_social_preserves_quarantine_appraisal_gate": True,
            "hostile_social_blocks_direct_core_self_memory_updates": True,
            "useful_criticism_requires_appraisal_before_memory_or_self_model_update": True,
            "hardware_normal_zero_delta_only": True,
            "hardware_low_power_and_below_non_panic_operational_only": True,
            "hardware_prediction_error_diagnostic_operational_only": True,
            "hardware_polling_tick_no_recursive_concern_loop": True,
            "speech_pressure_preserves_agp_fallback_gate_requirements": True,
            "listening_uncertainty_does_not_relabel_neutral_as_hostile": True,
            "imagination_negative_spiral_preserves_budget_cooldown_reality_check": True,
            "memory_self_candidates_preserve_appraisal_quarantine": True,
            "one_event_cannot_build_all_axis_payloads": True,
            "global_synchrony_blocked": True,
            "builder_success_does_not_imply_dryrun_apply_permission": True,
            "builder_success_does_not_imply_live_apply_permission": True,
        },
        "hardware_governor_non_panic_policy": hardware_governor_policy(),
        "anti_global_synchrony_policy": anti_global_synchrony_policy(),
        "no_runtime_mutation_proof": no_runtime_mutation_proof(),
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "artifact_created_or_staged": False,
        **_FALSE_BOUNDARY_FLAGS,
    }
