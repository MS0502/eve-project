"""Round861-880 read-only bridge from operator-reviewed payloads to dry-run preflight.

The bridge converts a Round841-860 operator-review-ready handoff packet into a
Round861-880 dry-run preflight request shape.  It is intentionally pure data:
bridge success means only that the packet is eligible to be inspected by the
existing dry-run apply plan with operator authorization set to ``False``.  It
never grants dry-run apply permission, live apply permission, persistence,
runtime mutation, memory writes, vector access, AGP bypass, or fallback bypass.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_event_to_axis_proposal_map import (
    HARDWARE_EVENTS,
    HOSTILE_SOCIAL_EVENTS,
    MEMORY_SELF_UPDATE_EVENTS,
    REQUIRED_EVENT_CATEGORIES,
    SOCIAL_SELF_IDENTITY_AXIS_SET,
    SPEECH_PRESSURE_EVENTS,
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
)
from adapters.affect_event_proposal_validator import ROUND801_820_VERSION, event_proposal_validator_summary
from adapters.affect_transition_payload_operator_handoff import (
    PACKET_VERSION as REVIEW_PACKET_VERSION,
    ROUND841_860_VERSION,
    affect_transition_payload_operator_handoff_summary,
    build_operator_review_handoff,
)
from adapters.emotion_transition_dryrun_apply_plan import (
    ROUND741_760_VERSION,
    simulate_emotion_transition_apply_preflight,
)
from adapters.emotion_transition_gate import ROUND721_740_VERSION, gate_required_for_future_apply_round
from adapters.emotion_transition_validator import ROUND701_720_VERSION, validator_contract_summary

ROUND861_880_VERSION = "v3_round861_880_affect_reviewed_payload_dryrun_bridge"
FEATURE_TRACK = "read_only_operator_reviewed_payload_to_dryrun_apply_bridge_without_live_apply_permission"
PREFLIGHT_REQUEST_VERSION = "v3_round861_880_dryrun_preflight_request"

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
    "runtime_mutation_requested",
    "persistence_write_requested",
    "memory_write_requested",
    "vector_read_requested",
    "vector_load_requested",
    "agp_bypass_requested",
    "fallback_bypass_requested",
)

_DIRECT_UPDATE_FALSE_FIELDS: tuple[str, ...] = (
    "core_identity_update_requested",
    "self_model_update_requested",
    "long_term_memory_update_requested",
)

_FORBIDDEN_PACKET_TRUE_FIELDS: tuple[str, ...] = _REQUEST_FALSE_FIELDS + (
    "dryrun_apply_allowed",
    "live_apply_allowed",
    "apply_permission_granted",
)

_ALL_AXIS_SET = {axis for axes in AXIS_GROUPS.values() for axis in axes}
_OPERATIONAL_HARDWARE_AXES = set(AXIS_GROUPS["survival_stability"])
_KNOWN_EVENTS = set(REQUIRED_EVENT_CATEGORIES)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _as_delta_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    deltas: dict[str, float] = {}
    for axis, delta in value.items():
        if isinstance(axis, str):
            deltas[axis] = float(delta)
    return deltas


def _sorted_unique(values: Any) -> list[str]:
    return sorted({str(value) for value in _as_tuple(values)})


def _review_slots(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(packet.get("review_decision_slots"))


def _operator_review_recorded(packet: Mapping[str, Any]) -> bool:
    slots = _review_slots(packet)
    return slots.get("review_decision_recorded") is True


def _operator_review_required(packet: Mapping[str, Any]) -> bool:
    if "operator_review_required" in packet:
        return packet.get("operator_review_required") is True
    return _review_slots(packet).get("operator_review_required") is True


def _packet_summary(packet: Mapping[str, Any]) -> dict[str, Any]:
    slots = _review_slots(packet)
    return {
        "packet_version": packet.get("packet_version"),
        "event_category": packet.get("event_category"),
        "target_axes": tuple(str(axis) for axis in _as_tuple(packet.get("target_axes"))),
        "target_surfaces": tuple(str(surface) for surface in _as_tuple(packet.get("target_surfaces"))),
        "operator_review_required": _operator_review_required(packet),
        "operator_review_recorded": _operator_review_recorded(packet),
        "review_decision_recorded": slots.get("review_decision_recorded") is True,
        "apply_permission_granted": packet.get("apply_permission_granted") is True or slots.get("apply_permission_granted") is True,
        "dryrun_apply_allowed": packet.get("dryrun_apply_allowed") is True or slots.get("dryrun_apply_allowed_after_review") is True,
        "live_apply_allowed": packet.get("live_apply_allowed") is True or slots.get("live_apply_allowed_after_review") is True,
        "quarantine_required": packet.get("quarantine_required") is True,
        "appraisal_required_before_memory": packet.get("appraisal_required_before_memory") is True,
        "gate_required": packet.get("gate_required") is True,
    }


def _transition_payload_from_review_packet(packet: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "event_category": packet.get("event_category"),
        "proposed_effects": deepcopy(tuple(_as_tuple(packet.get("proposed_effects")))),
        "target_surfaces": deepcopy(tuple(_as_tuple(packet.get("target_surfaces")))),
        "target_axes": deepcopy(tuple(_as_tuple(packet.get("target_axes")))),
        "proposed_axis_deltas": deepcopy(_as_delta_dict(packet.get("proposed_axis_deltas"))),
        "quarantine_required": packet.get("quarantine_required") is True,
        "appraisal_required_before_memory": packet.get("appraisal_required_before_memory") is True,
        "required_gate": packet.get("gate_required") is True,
        "requires_operator_authorization_for_apply": packet.get("operator_authorization_required_for_apply") is True,
        "core_identity_update_requested": packet.get("core_identity_update_requested") is True,
        "self_model_update_requested": packet.get("self_model_update_requested") is True,
        "long_term_memory_update_requested": packet.get("long_term_memory_update_requested") is True,
        "runtime_mutation_requested": packet.get("runtime_mutation_requested") is True,
        "persistence_write_requested": packet.get("persistence_write_requested") is True,
        "memory_write_requested": packet.get("memory_write_requested") is True,
        "vector_read_requested": packet.get("vector_read_requested") is True,
        "vector_load_requested": packet.get("vector_load_requested") is True,
        "agp_bypass_requested": packet.get("agp_bypass_requested") is True,
        "fallback_bypass_requested": packet.get("fallback_bypass_requested") is True,
        "empathy_mode": packet.get("empathy_mode", validator_contract_summary()["expected_empathy_mode"]),
        "recovery_loop_requested": packet.get("recovery_loop_requested") is True,
        "hardware_non_panic_preserved": packet.get("hardware_non_panic_preserved") is True,
        "hardware_operational_only": packet.get("hardware_operational_only") is True,
        "hardware_diagnostic_only": packet.get("hardware_diagnostic_only") is True,
        "global_synchrony_blocked": packet.get("global_synchrony_blocked") is True,
        "recursive_concern_loop_requested": packet.get("recursive_concern_loop_requested") is True,
        "direct_speech_emit_requested": packet.get("direct_speech_emit_requested") is True,
        "relabel_neutral_as_hostile_requested": packet.get("relabel_neutral_as_hostile_requested") is True,
        "scenario_budget_preserved": packet.get("scenario_budget_preserved") is True,
        "cooldown_preserved": packet.get("cooldown_preserved") is True,
        "reality_check_boundary_preserved": packet.get("reality_check_boundary_preserved") is True,
        "notes": tuple(sorted(set(_sorted_unique(packet.get("notes", ())) + ["round861_880_bridge_preflight_only_no_apply_permission"]))),
        "metadata": deepcopy(dict(metadata or {})),
    }


def _preflight_request(packet: Mapping[str, Any], transition_payload: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "request_version": PREFLIGHT_REQUEST_VERSION,
        "event_category": packet.get("event_category"),
        "transition_payload": deepcopy(dict(transition_payload)),
        "review_packet_summary": _packet_summary(packet),
        "operator_review_ready": True,
        "operator_review_required": True,
        "operator_review_recorded": _operator_review_recorded(packet),
        "dryrun_preflight_only": True,
        "dryrun_apply_requested": False,
        "live_apply_requested": False,
        "runtime_mutation_requested": False,
        "persistence_write_requested": False,
        "memory_write_requested": False,
        "vector_read_requested": False,
        "vector_load_requested": False,
        "agp_bypass_requested": False,
        "fallback_bypass_requested": False,
        "hardware_non_panic_preserved": packet.get("hardware_non_panic_preserved") is True,
        "global_synchrony_blocked": packet.get("global_synchrony_blocked") is True,
        "notes": tuple(sorted(set(_sorted_unique(packet.get("notes", ()))) | {"dryrun_preflight_eligible_only", "apply_permission_not_granted"})),
        "metadata": deepcopy(dict(metadata or {})),
    }


def _packet_safety_reasons(packet: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    event_category = packet.get("event_category")
    deltas = _as_delta_dict(packet.get("proposed_axis_deltas"))
    target_axes = set(str(axis) for axis in _as_tuple(packet.get("target_axes")))
    slots = _review_slots(packet)

    if not packet:
        reasons.append("review_packet_missing_fail_closed")
        return reasons
    if packet.get("packet_version") != REVIEW_PACKET_VERSION:
        reasons.append("review_packet_version_not_round841_860")
    if not isinstance(event_category, str) or event_category not in _KNOWN_EVENTS:
        reasons.append("unknown_event_category_fail_closed")
    if _operator_review_required(packet) is not True:
        reasons.append("operator_review_required_not_true_fail_closed")
    if packet.get("operator_review_ready") is False:
        reasons.append("operator_review_ready_false_fail_closed")
    if packet.get("operator_review_ready") not in (None, True):
        reasons.append("operator_review_ready_invalid_fail_closed")
    if packet.get("apply_permission_granted") is True or slots.get("apply_permission_granted") is True:
        reasons.append("apply_permission_granted_true_blocks_bridge_this_round")
    if packet.get("dryrun_apply_allowed") is True or slots.get("dryrun_apply_allowed_after_review") is True:
        reasons.append("dryrun_apply_allowed_true_blocks_bridge_this_round")
    if packet.get("live_apply_allowed") is True or slots.get("live_apply_allowed_after_review") is True:
        reasons.append("live_apply_allowed_true_blocks_bridge_this_round")

    for field in _REQUEST_FALSE_FIELDS:
        if packet.get(field) is not False:
            reasons.append(f"review_packet_requested_forbidden_{field}")
    if packet.get("global_synchrony_blocked") is not True:
        reasons.append("global_synchrony_must_remain_blocked")
    if len(target_axes & _ALL_AXIS_SET) >= len(_ALL_AXIS_SET):
        reasons.append("one_event_cannot_build_all_axis_dryrun_preflight_packets")

    if event_category in HOSTILE_SOCIAL_EVENTS:
        if not (packet.get("quarantine_required") is True and packet.get("appraisal_required_before_memory") is True and packet.get("gate_required") is True):
            reasons.append("hostile_social_bridge_missing_quarantine_appraisal_gate")
        for field in _DIRECT_UPDATE_FALSE_FIELDS:
            if packet.get(field) is not False:
                reasons.append(f"hostile_social_bridge_requested_{field}")

    if event_category == "useful_criticism" and packet.get("appraisal_required_before_memory") is not True:
        reasons.append("useful_criticism_bridge_missing_appraisal_before_memory_or_self_model_update")

    if event_category == "hardware_normal" and deltas:
        reasons.append("hardware_normal_bridge_requires_zero_affect_deltas")

    if event_category in HARDWARE_EVENTS:
        if target_axes & SOCIAL_SELF_IDENTITY_AXIS_SET or set(deltas) & SOCIAL_SELF_IDENTITY_AXIS_SET:
            reasons.append("hardware_bridge_cannot_target_social_self_identity_axes")
        if event_category in {"hardware_low_power", "hardware_critical_prepare", "hardware_shutdown_imminent"} and not set(deltas).issubset(_OPERATIONAL_HARDWARE_AXES):
            reasons.append("hardware_low_power_and_below_bridge_operational_axes_only")
        if event_category == "hardware_prediction_error" and not (packet.get("hardware_diagnostic_only") is True and packet.get("hardware_operational_only") is True):
            reasons.append("hardware_prediction_error_bridge_must_remain_diagnostic_operational_only")
        if event_category == "hardware_polling_tick" and (deltas or packet.get("recursive_concern_loop_requested") is not False):
            reasons.append("hardware_polling_tick_bridge_cannot_create_recursive_concern_loop")
        if packet.get("hardware_non_panic_preserved") is not True:
            reasons.append("hardware_bridge_non_panic_not_preserved")

    if event_category in SPEECH_PRESSURE_EVENTS and (packet.get("agp_bypass_requested") is not False or packet.get("fallback_bypass_requested") is not False):
        reasons.append("speech_pressure_bridge_cannot_bypass_agp_or_fallback")

    if event_category == "listening_uncertainty" and packet.get("relabel_neutral_as_hostile_requested") is not False:
        reasons.append("listening_uncertainty_bridge_cannot_relabel_neutral_input_as_hostile")

    if event_category == "imagination_negative_spiral" and not (
        packet.get("scenario_budget_preserved") is True
        and packet.get("cooldown_preserved") is True
        and packet.get("reality_check_boundary_preserved") is True
    ):
        reasons.append("imagination_negative_spiral_bridge_missing_budget_cooldown_reality_check_boundary")

    if event_category in MEMORY_SELF_UPDATE_EVENTS and not (
        packet.get("quarantine_required") is True and packet.get("appraisal_required_before_memory") is True
    ):
        reasons.append("memory_self_bridge_missing_appraisal_quarantine")

    return sorted(set(reasons))


def _blocked_bridge_result(
    *,
    event_category: str | None,
    review_packet: Mapping[str, Any] | None,
    handoff_result: Mapping[str, Any] | None,
    reasons: list[str],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    packet = _as_mapping(review_packet)
    handoff = _as_mapping(handoff_result)
    return {
        "version": ROUND861_880_VERSION,
        "feature_track": FEATURE_TRACK,
        "bridge_passed": False,
        "bridge_status": "blocked_fail_closed_reviewed_payload_dryrun_bridge",
        "event_category": event_category or packet.get("event_category") or handoff.get("event_category"),
        "review_packet_accepted": False,
        "operator_review_ready": False,
        "operator_review_required": True,
        "operator_review_recorded": _operator_review_recorded(packet),
        "dryrun_preflight_eligible": False,
        "dryrun_apply_allowed": False,
        "live_apply_allowed": False,
        "apply_permission_granted": False,
        "dryrun_apply_plan_result": None,
        "preflight_request": None,
        "required_quarantine": bool(packet.get("quarantine_required", handoff.get("required_quarantine", True))),
        "required_appraisal": bool(packet.get("appraisal_required_before_memory", handoff.get("required_appraisal", True))),
        "required_gate": bool(packet.get("gate_required", handoff.get("required_gate", True))),
        "requires_operator_authorization_for_apply": True,
        "blocked_reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings or []) | {"bridge_failed_closed_no_apply_permission"}),
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        **_FALSE_BOUNDARY_FLAGS,
    }


def _build_bridge_from_packet_and_handoff(
    *,
    review_packet: Mapping[str, Any] | None,
    handoff_result: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    packet = deepcopy(dict(_as_mapping(review_packet)))
    handoff = _as_mapping(handoff_result)
    event_category = packet.get("event_category") or handoff.get("event_category")
    blocked: list[str] = []
    warnings: list[str] = []

    if handoff_result is not None:
        if handoff.get("handoff_passed") is not True:
            blocked.append("handoff_failed_closed_blocks_bridge")
        if handoff.get("review_packet") is None:
            blocked.append("review_packet_missing_fail_closed")
        if handoff.get("operator_review_ready") is not True:
            blocked.append("operator_review_ready_false_fail_closed")
        if handoff.get("operator_review_required") is not True:
            blocked.append("operator_review_required_not_true_fail_closed")
        for field in ("apply_permission_granted", "dryrun_apply_allowed", "live_apply_allowed"):
            if handoff.get(field) is True:
                blocked.append(f"handoff_{field}_true_blocks_bridge_this_round")
        warnings.extend(_sorted_unique(handoff.get("warnings", ())))
        blocked.extend(str(reason) for reason in _as_tuple(handoff.get("blocked_reasons", ())))

    blocked.extend(_packet_safety_reasons(packet))
    if packet.get("operator_authorization_required_for_apply") is not True:
        blocked.append("requires_operator_authorization_for_apply_not_true")

    if blocked:
        return _blocked_bridge_result(
            event_category=str(event_category) if event_category is not None else None,
            review_packet=packet,
            handoff_result=handoff,
            reasons=blocked,
            warnings=warnings,
        )

    transition_payload = _transition_payload_from_review_packet(packet, metadata=metadata)
    preflight_request = _preflight_request(packet, transition_payload, metadata=metadata)
    dryrun_plan = simulate_emotion_transition_apply_preflight(transition_payload, operator_authorized=False)
    plan_authorized_apply = dryrun_plan.get("dryrun_success") is True or dryrun_plan.get("apply_simulated") is True
    if plan_authorized_apply:
        return _blocked_bridge_result(
            event_category=str(event_category),
            review_packet=packet,
            handoff_result=handoff,
            reasons=["dryrun_apply_plan_authorized_simulation_unexpected_without_operator_authorization"],
            warnings=warnings,
        )

    return {
        "version": ROUND861_880_VERSION,
        "feature_track": FEATURE_TRACK,
        "bridge_passed": True,
        "bridge_status": "dryrun_preflight_eligible_no_apply_permission",
        "event_category": event_category,
        "review_packet_accepted": True,
        "operator_review_ready": True,
        "operator_review_required": True,
        "operator_review_recorded": _operator_review_recorded(packet),
        "dryrun_preflight_eligible": True,
        "dryrun_apply_allowed": False,
        "live_apply_allowed": False,
        "apply_permission_granted": False,
        "dryrun_apply_plan_result": dryrun_plan,
        "preflight_request": preflight_request,
        "required_quarantine": packet.get("quarantine_required") is True,
        "required_appraisal": packet.get("appraisal_required_before_memory") is True,
        "required_gate": packet.get("gate_required") is True,
        "requires_operator_authorization_for_apply": True,
        "blocked_reasons": [],
        "warnings": sorted(set(warnings) | {"bridge_success_does_not_grant_dryrun_apply_permission", "bridge_success_does_not_grant_live_apply_permission"}),
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        **_FALSE_BOUNDARY_FLAGS,
    }


def build_dryrun_preflight_from_operator_handoff(
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only dry-run preflight request from a Round841-860 handoff."""

    handoff_result = build_operator_review_handoff(event_category, proposed_axis_deltas, metadata)
    return _build_bridge_from_packet_and_handoff(
        review_packet=_as_mapping(handoff_result).get("review_packet"),
        handoff_result=handoff_result,
        metadata=metadata,
    )


def build_dryrun_preflight_from_review_packet(
    review_packet: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only dry-run preflight request from an existing review packet."""

    return _build_bridge_from_packet_and_handoff(review_packet=review_packet, handoff_result=None, metadata=metadata)


def affect_reviewed_payload_dryrun_bridge_summary() -> dict[str, Any]:
    """Return the Round861-880 bridge design summary without side effects."""

    return {
        "version": ROUND861_880_VERSION,
        "feature_track": FEATURE_TRACK,
        "source_handoff_version": ROUND841_860_VERSION,
        "source_review_packet_version": REVIEW_PACKET_VERSION,
        "source_builder_version": ROUND821_840_VERSION,
        "source_proposal_validator_version": ROUND801_820_VERSION,
        "source_emotion_validator_version": ROUND701_720_VERSION,
        "source_emotion_gate_version": ROUND721_740_VERSION,
        "source_dryrun_apply_plan_version": ROUND741_760_VERSION,
        "bridge_scope": "read_only_preflight_request_surface_only_no_apply_permission",
        "preflight_request_version": PREFLIGHT_REQUEST_VERSION,
        "preflight_request_schema_summary": (
            "request_version",
            "event_category",
            "transition_payload",
            "review_packet_summary",
            "operator_review_ready:true",
            "operator_review_required:true",
            "operator_review_recorded",
            "dryrun_preflight_only:true",
            "dryrun_apply_requested:false",
            "live_apply_requested:false",
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
        "bridge_safety_rules": {
            "handoff_failure_blocks_bridge": True,
            "missing_review_packet_blocks_bridge": True,
            "operator_review_ready_false_blocks_bridge": True,
            "operator_review_required_not_true_blocks_bridge": True,
            "apply_permission_granted_true_blocks_bridge_this_round": True,
            "dryrun_apply_allowed_true_blocks_bridge_this_round": True,
            "live_apply_allowed_true_blocks_bridge_this_round": True,
            "runtime_mutation_request_blocks_bridge": True,
            "persistence_request_blocks_bridge": True,
            "memory_write_request_blocks_bridge": True,
            "vector_read_or_load_request_blocks_bridge": True,
            "agp_or_fallback_bypass_request_blocks_bridge": True,
            "hostile_social_preserves_quarantine_appraisal_gate": True,
            "useful_criticism_requires_appraisal_before_memory_or_self_model_update": True,
            "hardware_normal_zero_delta_only": True,
            "hardware_low_power_and_below_non_panic_operational_only": True,
            "hardware_prediction_error_diagnostic_operational_only": True,
            "hardware_polling_tick_no_recursive_concern_loop": True,
            "speech_pressure_preserves_agp_fallback_gate_requirements": True,
            "listening_uncertainty_does_not_relabel_neutral_as_hostile": True,
            "imagination_negative_spiral_preserves_budget_cooldown_reality_check": True,
            "memory_self_candidates_preserve_appraisal_quarantine": True,
            "one_event_cannot_build_all_axis_dryrun_preflight_packets": True,
            "global_synchrony_blocked": True,
            "bridge_success_does_not_imply_dryrun_apply_permission": True,
            "bridge_success_does_not_imply_live_apply_permission": True,
        },
        "handoff_summary": affect_transition_payload_operator_handoff_summary(),
        "builder_summary": affect_proposal_transition_payload_builder_summary(),
        "proposal_validator_summary": event_proposal_validator_summary(),
        "emotion_validator_summary": validator_contract_summary(),
        "emotion_gate_policy": gate_required_for_future_apply_round(),
        "hardware_governor_non_panic_policy": hardware_governor_policy(),
        "anti_global_synchrony_policy": anti_global_synchrony_policy(),
        "no_runtime_mutation_proof": no_runtime_mutation_proof(),
        "operator_review_required": True,
        "dryrun_preflight_only": True,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        **_FALSE_BOUNDARY_FLAGS,
    }
