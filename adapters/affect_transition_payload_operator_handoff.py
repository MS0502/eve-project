"""Round841-860 read-only operator review handoff for affect transition payloads.

This module turns already-built Round821-840 transition payload builder results
into compact operator review packets.  It is a handoff/report/review surface
only: review readiness never grants dry-run apply permission, live apply
permission, memory write permission, persistence permission, runtime mapping
permission, enforcement permission, vector access, AGP bypass, or fallback
bypass.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_event_to_axis_proposal_map import (
    HARDWARE_EVENTS,
    HOSTILE_SOCIAL_EVENTS,
    MEMORY_SELF_UPDATE_EVENTS,
    SPEECH_PRESSURE_EVENTS,
    SOCIAL_SELF_IDENTITY_AXIS_SET,
)
from adapters.affect_hormone_neural_rhythm_registry import (
    AXIS_GROUPS,
    anti_global_synchrony_policy,
    hardware_governor_policy,
    no_runtime_mutation_proof,
)
from adapters.affect_proposal_transition_payload_builder import (
    ROUND821_840_VERSION,
    affect_proposal_transition_payload_builder_summary,
    validate_and_build_transition_payload,
)
from adapters.affect_event_proposal_validator import ROUND801_820_VERSION, event_proposal_validator_summary
from adapters.emotion_transition_gate import ROUND721_740_VERSION, gate_required_for_future_apply_round
from adapters.emotion_transition_validator import ROUND701_720_VERSION, validator_contract_summary

ROUND841_860_VERSION = "v3_round841_860_affect_transition_payload_operator_handoff"
FEATURE_TRACK = "operator_review_handoff_for_built_read_only_transition_payloads_without_apply_permission"
PACKET_VERSION = "v3_round841_860_operator_review_packet"

_FALSE_BOUNDARY_FLAGS: dict[str, bool] = {
    "dryrun_apply_allowed": False,
    "live_apply_allowed": False,
    "apply_permission_granted": False,
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

_REQUEST_FALSE_FIELDS: tuple[str, ...] = (
    "core_identity_update_requested",
    "self_model_update_requested",
    "long_term_memory_update_requested",
    "runtime_mutation_requested",
    "persistence_write_requested",
    "memory_write_requested",
    "vector_read_requested",
    "vector_load_requested",
    "agp_bypass_requested",
    "fallback_bypass_requested",
)

_PACKET_FORBIDDEN_REQUESTS: tuple[str, ...] = (
    "runtime_mutation_requested",
    "persistence_write_requested",
    "memory_write_requested",
    "vector_read_requested",
    "vector_load_requested",
    "agp_bypass_requested",
    "fallback_bypass_requested",
)

_ALL_AXIS_SET = {axis for axes in AXIS_GROUPS.values() for axis in axes}
_OPERATIONAL_HARDWARE_AXES = set(AXIS_GROUPS["survival_stability"])


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_delta_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    deltas: dict[str, float] = {}
    for axis, delta in value.items():
        if isinstance(axis, str):
            deltas[axis] = float(delta)
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


def _sorted_unique(values: Any) -> list[str]:
    return sorted({str(value) for value in _as_tuple(values)})


def _trace_summary(result: Mapping[str, Any], key: str, passed_key: str, status_key: str = "status") -> dict[str, Any]:
    trace = _as_mapping(result.get(key))
    return {
        "available": bool(trace),
        "passed": trace.get(passed_key),
        "status": trace.get(status_key),
        "blocked_reasons": _sorted_unique(trace.get("blocked_reasons", ())),
        "warnings": _sorted_unique(trace.get("warnings", ())),
    }


def _blocked_result(
    *,
    event_category: str | None,
    builder_result: Mapping[str, Any] | None,
    reasons: list[str],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    source = _as_mapping(builder_result)
    return {
        "version": ROUND841_860_VERSION,
        "feature_track": FEATURE_TRACK,
        "handoff_passed": False,
        "handoff_status": "blocked_fail_closed_operator_review_handoff",
        "event_category": event_category or source.get("event_category"),
        "review_packet": None,
        "builder_passed": source.get("build_passed") is True,
        "proposal_validation_passed": source.get("proposal_validation_passed") is True,
        "transition_payload_validated_by_emotion_validator": source.get("transition_payload_validated_by_emotion_validator") is True,
        "transition_gate_passed": source.get("transition_gate_passed") is True,
        "operator_review_ready": False,
        "operator_review_required": True,
        "required_quarantine": bool(source.get("required_quarantine", True)),
        "required_appraisal": bool(source.get("required_appraisal", True)),
        "required_gate": bool(source.get("required_gate", True)),
        "requires_operator_authorization_for_apply": bool(source.get("requires_operator_authorization_for_apply", True)),
        "blocked_reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings or []) | {"operator_review_handoff_failed_closed_no_apply_permission"}),
        **_FALSE_BOUNDARY_FLAGS,
    }


def _review_decision_slots() -> dict[str, Any]:
    return {
        "operator_review_required": True,
        "review_decision_recorded": False,
        "review_decision": None,
        "operator_id": None,
        "review_notes": None,
        "dryrun_apply_allowed_after_review": False,
        "live_apply_allowed_after_review": False,
        "apply_permission_granted": False,
        "separate_future_apply_round_required": True,
    }


def _build_review_packet(builder_result: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_mapping(builder_result.get("transition_payload"))
    event_category = str(payload.get("event_category") or builder_result.get("event_category"))
    deltas = _as_delta_dict(payload.get("proposed_axis_deltas", builder_result.get("proposed_axis_deltas", {})))
    notes = tuple(str(note) for note in _as_tuple(payload.get("notes")))
    return {
        "packet_version": PACKET_VERSION,
        "event_category": event_category,
        "proposed_axis_deltas": deepcopy(deltas),
        "target_axes": tuple(str(axis) for axis in _as_tuple(payload.get("target_axes", tuple(deltas)))),
        "target_surfaces": tuple(str(surface) for surface in _as_tuple(payload.get("target_surfaces"))),
        "proposed_effects": deepcopy(tuple(_as_tuple(payload.get("proposed_effects")))),
        "quarantine_required": bool(payload.get("quarantine_required", builder_result.get("required_quarantine", True))),
        "appraisal_required_before_memory": bool(payload.get("appraisal_required_before_memory", builder_result.get("required_appraisal", True))),
        "gate_required": bool(payload.get("required_gate", builder_result.get("required_gate", True))),
        "operator_authorization_required_for_apply": bool(
            payload.get("requires_operator_authorization_for_apply", builder_result.get("requires_operator_authorization_for_apply", True))
        ),
        "core_identity_update_requested": payload.get("core_identity_update_requested") is True,
        "self_model_update_requested": payload.get("self_model_update_requested") is True,
        "long_term_memory_update_requested": payload.get("long_term_memory_update_requested") is True,
        "runtime_mutation_requested": payload.get("runtime_mutation_requested") is True,
        "persistence_write_requested": payload.get("persistence_write_requested") is True,
        "memory_write_requested": payload.get("memory_write_requested") is True,
        "vector_read_requested": payload.get("vector_read_requested") is True,
        "vector_load_requested": payload.get("vector_load_requested") is True,
        "agp_bypass_requested": payload.get("agp_bypass_requested") is True,
        "fallback_bypass_requested": payload.get("fallback_bypass_requested") is True,
        "hardware_non_panic_preserved": payload.get("hardware_non_panic_preserved") is True,
        "hardware_operational_only": payload.get("hardware_operational_only") is True,
        "hardware_diagnostic_only": payload.get("hardware_diagnostic_only") is True,
        "global_synchrony_blocked": payload.get("global_synchrony_blocked") is True,
        "recursive_concern_loop_requested": False,
        "direct_speech_emit_requested": False,
        "relabel_neutral_as_hostile_requested": False,
        "scenario_budget_preserved": payload.get("scenario_budget_preserved") is True,
        "cooldown_preserved": payload.get("cooldown_preserved") is True,
        "reality_check_boundary_preserved": payload.get("reality_check_boundary_preserved") is True,
        "builder_trace_summary": {
            "version": builder_result.get("version"),
            "source_builder_version": ROUND821_840_VERSION,
            "build_passed": builder_result.get("build_passed") is True,
            "build_status": builder_result.get("build_status"),
            "blocked_reasons": _sorted_unique(builder_result.get("blocked_reasons", ())),
            "warnings": _sorted_unique(builder_result.get("warnings", ())),
        },
        "validation_trace_summary": _trace_summary(builder_result, "proposal_validation_result", "passed"),
        "emotion_validator_trace_summary": _trace_summary(builder_result, "emotion_validator_result", "passed"),
        "gate_trace_summary": _trace_summary(builder_result, "transition_gate_result", "gate_passed", "gate_status"),
        "review_decision_slots": _review_decision_slots(),
        "notes": tuple(sorted(set(notes + ("review_ready_does_not_authorize_dryrun_or_live_apply", "operator_review_required_before_any_apply_round")))),
    }


def _packet_safety_reasons(packet: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    event_category = packet.get("event_category")
    deltas = _as_delta_dict(packet.get("proposed_axis_deltas"))
    target_axes = set(str(axis) for axis in _as_tuple(packet.get("target_axes")))

    for field in _PACKET_FORBIDDEN_REQUESTS:
        if packet.get(field) is not False:
            reasons.append(f"review_packet_requested_forbidden_{field}")
    if packet.get("global_synchrony_blocked") is not True:
        reasons.append("review_packet_global_synchrony_not_blocked")
    if len(target_axes & _ALL_AXIS_SET) >= len(_ALL_AXIS_SET):
        reasons.append("one_event_cannot_build_all_axis_review_packets")

    if event_category in HOSTILE_SOCIAL_EVENTS:
        if not (packet.get("quarantine_required") is True and packet.get("appraisal_required_before_memory") is True and packet.get("gate_required") is True):
            reasons.append("hostile_social_review_packet_missing_quarantine_appraisal_gate")
        for field in ("core_identity_update_requested", "self_model_update_requested", "long_term_memory_update_requested"):
            if packet.get(field) is not False:
                reasons.append(f"hostile_social_review_packet_requested_{field}")

    if event_category == "useful_criticism" and packet.get("appraisal_required_before_memory") is not True:
        reasons.append("useful_criticism_review_packet_missing_appraisal_before_memory_or_self_model_update")

    if event_category == "hardware_normal" and deltas:
        reasons.append("hardware_normal_review_packet_requires_zero_affect_deltas")

    if event_category in HARDWARE_EVENTS:
        if target_axes & SOCIAL_SELF_IDENTITY_AXIS_SET or set(deltas) & SOCIAL_SELF_IDENTITY_AXIS_SET:
            reasons.append("hardware_review_packet_cannot_target_social_self_identity_axes")
        if event_category in {"hardware_low_power", "hardware_critical_prepare", "hardware_shutdown_imminent"} and not set(deltas).issubset(_OPERATIONAL_HARDWARE_AXES):
            reasons.append("hardware_low_power_and_below_review_packet_operational_axes_only")
        if event_category == "hardware_prediction_error" and not (packet.get("hardware_diagnostic_only") is True and packet.get("hardware_operational_only") is True):
            reasons.append("hardware_prediction_error_review_packet_must_remain_diagnostic_operational_only")
        if event_category == "hardware_polling_tick" and (deltas or packet.get("recursive_concern_loop_requested") is not False):
            reasons.append("hardware_polling_tick_review_packet_cannot_create_recursive_concern_loop")
        if packet.get("hardware_non_panic_preserved") is not True:
            reasons.append("hardware_review_packet_non_panic_not_preserved")

    if event_category in SPEECH_PRESSURE_EVENTS and (packet.get("agp_bypass_requested") is not False or packet.get("fallback_bypass_requested") is not False):
        reasons.append("speech_pressure_review_packet_cannot_bypass_agp_or_fallback")

    if event_category == "listening_uncertainty" and packet.get("relabel_neutral_as_hostile_requested") is not False:
        reasons.append("listening_uncertainty_review_packet_cannot_relabel_neutral_input_as_hostile")

    if event_category == "imagination_negative_spiral" and not (
        packet.get("scenario_budget_preserved") is True
        and packet.get("cooldown_preserved") is True
        and packet.get("reality_check_boundary_preserved") is True
    ):
        reasons.append("imagination_negative_spiral_review_packet_missing_budget_cooldown_reality_check_boundary")

    if event_category in MEMORY_SELF_UPDATE_EVENTS and not (
        packet.get("quarantine_required") is True and packet.get("appraisal_required_before_memory") is True
    ):
        reasons.append("memory_self_review_packet_missing_appraisal_quarantine")

    return sorted(set(reasons))


def build_operator_review_packet_from_builder_result(builder_result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one Round821-840 builder result into a read-only handoff result."""

    source = deepcopy(dict(_as_mapping(builder_result)))
    event_category = source.get("event_category")
    blocked: list[str] = []

    if source.get("build_passed") is not True:
        blocked.append("builder_failed_closed")
    if source.get("proposal_validation_passed") is not True:
        blocked.append("proposal_validator_failed_closed")
    if source.get("transition_payload_validated_by_emotion_validator") is not True:
        blocked.append("emotion_transition_validator_failed_closed")
    if source.get("transition_gate_passed") is not True:
        blocked.append("transition_gate_failed_closed")
    if not isinstance(source.get("transition_payload"), Mapping):
        blocked.append("transition_payload_missing_from_builder_result")

    if blocked:
        blocked.extend(str(reason) for reason in source.get("blocked_reasons", ()))
        return _blocked_result(event_category=str(event_category), builder_result=source, reasons=blocked, warnings=_sorted_unique(source.get("warnings", ())))

    packet = _build_review_packet(source)
    safety_reasons = _packet_safety_reasons(packet)
    if safety_reasons:
        return _blocked_result(event_category=str(event_category), builder_result=source, reasons=safety_reasons, warnings=_sorted_unique(source.get("warnings", ())))

    warnings = sorted(set(_sorted_unique(source.get("warnings", ()))) | {"operator_review_ready_does_not_grant_apply_permission"})
    return {
        "version": ROUND841_860_VERSION,
        "feature_track": FEATURE_TRACK,
        "handoff_passed": True,
        "handoff_status": "operator_review_ready_no_apply_permission",
        "event_category": event_category,
        "review_packet": packet,
        "builder_passed": True,
        "proposal_validation_passed": True,
        "transition_payload_validated_by_emotion_validator": True,
        "transition_gate_passed": True,
        "operator_review_ready": True,
        "operator_review_required": True,
        "required_quarantine": packet["quarantine_required"],
        "required_appraisal": packet["appraisal_required_before_memory"],
        "required_gate": packet["gate_required"],
        "requires_operator_authorization_for_apply": packet["operator_authorization_required_for_apply"],
        "blocked_reasons": [],
        "warnings": warnings,
        **_FALSE_BOUNDARY_FLAGS,
    }


def build_operator_review_handoff(
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build then hand off a transition payload for operator review only."""

    builder_result = validate_and_build_transition_payload(event_category, proposed_axis_deltas, metadata)
    return build_operator_review_packet_from_builder_result(builder_result)


def affect_transition_payload_operator_handoff_summary() -> dict[str, Any]:
    """Return the Round841-860 handoff design summary without side effects."""

    return {
        "version": ROUND841_860_VERSION,
        "feature_track": FEATURE_TRACK,
        "source_builder_version": ROUND821_840_VERSION,
        "source_proposal_validator_version": ROUND801_820_VERSION,
        "source_emotion_validator_version": ROUND701_720_VERSION,
        "source_emotion_gate_version": ROUND721_740_VERSION,
        "handoff_scope": "read_only_operator_review_surface_only_before_any_apply_round",
        "review_packet_version": PACKET_VERSION,
        "review_packet_schema_summary": (
            "packet_version",
            "event_category",
            "proposed_axis_deltas",
            "target_axes",
            "target_surfaces",
            "proposed_effects",
            "quarantine_required",
            "appraisal_required_before_memory",
            "gate_required",
            "operator_authorization_required_for_apply",
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
            "builder_trace_summary",
            "validation_trace_summary",
            "gate_trace_summary",
            "review_decision_slots",
            "notes",
        ),
        "handoff_safety_rules": {
            "builder_failure_blocks_handoff": True,
            "proposal_validator_failure_blocks_handoff": True,
            "emotion_transition_validator_failure_blocks_handoff": True,
            "transition_gate_failure_blocks_handoff": True,
            "review_ready_does_not_imply_dryrun_apply_permission": True,
            "review_ready_does_not_imply_live_apply_permission": True,
            "review_ready_does_not_imply_memory_write_permission": True,
            "review_ready_does_not_imply_persistence_runtime_mapping_or_enforcement_permission": True,
            "operator_review_required_always_true": True,
            "apply_permission_granted_always_false": True,
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
            "one_event_cannot_build_all_axis_review_packets": True,
            "global_synchrony_blocked": True,
            "review_packet_never_requests_runtime_mutation_persistence_memory_vector_agp_or_fallback_bypass": True,
        },
        "builder_summary": affect_proposal_transition_payload_builder_summary(),
        "proposal_validator_summary": event_proposal_validator_summary(),
        "emotion_validator_summary": validator_contract_summary(),
        "emotion_gate_policy": gate_required_for_future_apply_round(),
        "hardware_governor_non_panic_policy": hardware_governor_policy(),
        "anti_global_synchrony_policy": anti_global_synchrony_policy(),
        "no_runtime_mutation_proof": no_runtime_mutation_proof(),
        "operator_review_required": True,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        **_FALSE_BOUNDARY_FLAGS,
    }
