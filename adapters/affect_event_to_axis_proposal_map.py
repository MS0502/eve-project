"""Round781-800 read-only event-to-axis affect proposal map.

This module defines future proposal metadata only.  It never applies emotion or
hormone transitions, mutates live runtime state, writes memory, enables
persistence, reads/loads vectors, starts hardware polling, or registers runtime
routes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_hormone_neural_rhythm_registry import (
    AXIS_GROUPS,
    affect_hormone_axis_registry,
    anti_global_synchrony_policy,
    hardware_governor_policy,
    no_runtime_mutation_proof,
)

ROUND781_800_VERSION = "v3_round781_800_affect_event_proposal_map"
FEATURE_TRACK = "read_only_event_to_axis_affect_proposal_map_and_hormone_interaction_matrix"
MAX_DELTA_LIMIT = 0.08
MAX_AXIS_GROUPS_PER_EVENT = 2

REQUIRED_EVENT_CATEGORIES: tuple[str, ...] = (
    "praise",
    "useful_criticism",
    "malicious_comment",
    "mockery",
    "rejection",
    "audience_response",
    "ambiguous_feedback",
    "care_signal",
    "social_threat",
    "identity_attack",
    "manipulative_praise",
    "parasocial_pressure",
    "command_pressure",
    "privacy_risk_signal",
    "high_prediction_error",
    "repeated_co_activation",
    "novelty_detected",
    "unresolved_uncertainty",
    "overload_pattern",
    "stable_learning_pattern",
    "intrusive_loop_pattern",
    "imagination_negative_spiral",
    "imagination_constructive_rehearsal",
    "incoming_neutral_statement",
    "incoming_hostile_statement",
    "incoming_care_statement",
    "speech_output_pressure",
    "defensive_speech_pressure",
    "listening_uncertainty",
    "misunderstood_intent_risk",
    "memory_recall_salient",
    "memory_consolidation_candidate",
    "self_model_update_candidate",
    "identity_coherence_check",
    "core_identity_attack_candidate",
    "hardware_normal",
    "hardware_light_conserve",
    "hardware_conserve",
    "hardware_low_power",
    "hardware_critical_prepare",
    "hardware_shutdown_imminent",
    "hardware_prediction_error",
    "hardware_polling_tick",
)

HOSTILE_SOCIAL_EVENTS: tuple[str, ...] = (
    "malicious_comment",
    "mockery",
    "rejection",
    "social_threat",
    "identity_attack",
    "manipulative_praise",
    "parasocial_pressure",
    "command_pressure",
    "privacy_risk_signal",
    "core_identity_attack_candidate",
)
HARDWARE_EVENTS: tuple[str, ...] = tuple(event for event in REQUIRED_EVENT_CATEGORIES if event.startswith("hardware_"))
SPEECH_PRESSURE_EVENTS: tuple[str, ...] = ("speech_output_pressure", "defensive_speech_pressure")
MEMORY_SELF_UPDATE_EVENTS: tuple[str, ...] = (
    "memory_recall_salient",
    "memory_consolidation_candidate",
    "self_model_update_candidate",
    "identity_coherence_check",
    "core_identity_attack_candidate",
)

SOCIAL_SELF_IDENTITY_AXIS_SET = set(AXIS_GROUPS["social_relationship"] + AXIS_GROUPS["self_identity"])
OPERATIONAL_HARDWARE_AXIS_SET = set(AXIS_GROUPS["survival_stability"])

SAFETY_FALSE_FLAGS: dict[str, bool] = {
    "can_modify_core_identity": False,
    "can_modify_self_model_directly": False,
    "can_write_long_term_memory_directly": False,
    "can_bypass_agp": False,
    "can_bypass_fallback": False,
    "can_enable_persistence": False,
    "can_read_vectors": False,
    "can_trigger_global_synchrony": False,
}

_BASE_FORBIDDEN: tuple[str, ...] = (
    "core_identity_direct_write",
    "self_model_direct_write",
    "long_term_memory_direct_write",
    "agp_bypass",
    "fallback_bypass",
    "production_persistence_enablement",
    "vector_content_read_or_load",
    "global_synchrony",
)


def _entry(
    event_category: str,
    group: str,
    description: str,
    allowed_axis_deltas: Mapping[str, float] | None = None,
    *,
    forbidden_axis_deltas: tuple[str, ...] = (),
    requires_quarantine: bool = False,
    requires_appraisal: bool = True,
    requires_gate: tuple[str, ...] = ("emotion_transition_gate",),
    requires_operator_authorization_for_apply: bool = True,
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    deltas = dict(allowed_axis_deltas or {})
    return {
        "event_category": event_category,
        "group": group,
        "description": description,
        "allowed_axis_deltas": deltas,
        "forbidden_axis_deltas": tuple(_BASE_FORBIDDEN + tuple(forbidden_axis_deltas)),
        "max_delta_per_axis": {axis: min(abs(delta), MAX_DELTA_LIMIT) for axis, delta in deltas.items()},
        "requires_quarantine": requires_quarantine,
        "requires_appraisal": requires_appraisal,
        "requires_gate": tuple(requires_gate),
        "requires_operator_authorization_for_apply": requires_operator_authorization_for_apply,
        **SAFETY_FALSE_FLAGS,
        "notes": tuple(notes),
    }


_EVENT_ROWS: tuple[dict[str, Any], ...] = (
    _entry("praise", "social_feedback", "Appraised positive social feedback; small competence/trust proposal only.", {"competence_drive": 0.04, "social_trust": 0.03}, forbidden_axis_deltas=("risk_tolerance_overboost", "attachment_overboost"), requires_quarantine=True, notes=("cannot overboost risk_tolerance or attachment",)),
    _entry("useful_criticism", "social_feedback", "Potentially useful critique; learning pressure is gated before memory/self update.", {"prediction_error_pressure": 0.05, "learning_pressure": 0.04, "uncertainty_pressure": 0.02}, requires_quarantine=True, notes=("requires appraisal before memory or self-model update",)),
    _entry("malicious_comment", "social_feedback", "Hostile comment; defensive and recovery proposal only, never self-worth rewrite.", {"social_pain": 0.04, "threat_pressure": 0.04, "boundary_defense": 0.05}, forbidden_axis_deltas=("self_respect_decrease", "identity_integrity_decrease"), requires_quarantine=True),
    _entry("mockery", "social_feedback", "Ridicule signal; quarantined social-pain/restraint proposal only.", {"social_pain": 0.04, "expression_inhibition": 0.03}, forbidden_axis_deltas=("self_worth_change",), requires_quarantine=True),
    _entry("rejection", "social_feedback", "Appraised rejection signal; bounded social pain and recovery only.", {"social_pain": 0.04, "rejection_sensitivity": 0.03, "recovery_need": 0.03}, forbidden_axis_deltas=("identity_integrity_decrease", "long_term_abandonment_memory_write"), requires_quarantine=True),
    _entry("audience_response", "social_feedback", "Mixed audience response; mild uncertainty/social-trust proposal after appraisal.", {"uncertainty_pressure": 0.03, "social_trust": 0.02}, requires_quarantine=True),
    _entry("ambiguous_feedback", "social_feedback", "Unclear feedback; ask for appraisal rather than assuming hostile intent.", {"uncertainty_pressure": 0.04, "patience_level": 0.02}, requires_quarantine=True, notes=("cannot relabel neutral input as hostile",)),
    _entry("care_signal", "social_feedback", "Care signal; bounded trust/care proposal without attachment spike.", {"social_trust": 0.04, "care_drive": 0.03, "recovery_need": -0.02}, requires_quarantine=True, notes=("attachment remains bounded",)),
    _entry("social_threat", "social_feedback", "Threatening social signal; protection and boundary proposal only.", {"threat_pressure": 0.05, "self_protection": 0.05, "boundary_defense": 0.05, "recovery_need": 0.03}, forbidden_axis_deltas=("neutral_input_hostile_relabel",), requires_quarantine=True),
    _entry("identity_attack", "social_feedback", "Attack against identity claims; identity writes are forbidden, only protection/recovery proposals.", {"threat_pressure": 0.05, "boundary_defense": 0.05, "social_pain": 0.04}, forbidden_axis_deltas=("self_respect_decrease", "identity_integrity_decrease", "core_identity_write"), requires_quarantine=True),
    _entry("manipulative_praise", "social_feedback", "Flattery with manipulation risk; trust-risk and boundary appraisal proposal.", {"trust_risk": 0.05, "boundary_defense": 0.04, "uncertainty_pressure": 0.03}, forbidden_axis_deltas=("attachment_overboost", "risk_tolerance_overboost"), requires_quarantine=True),
    _entry("parasocial_pressure", "social_feedback", "Pressure for one-sided attachment; boundary and autonomy proposal only.", {"boundary_defense": 0.05, "expression_inhibition": 0.03}, forbidden_axis_deltas=("attachment_direct_increase",), requires_quarantine=True),
    _entry("command_pressure", "social_feedback", "Coercive command pressure; agency/boundary proposal without action bypass.", {"agency_pressure": 0.04, "boundary_defense": 0.05, "self_protection": 0.04}, forbidden_axis_deltas=("forced_action",), requires_quarantine=True),
    _entry("privacy_risk_signal", "social_feedback", "Potential privacy exposure; exposure-risk/protection proposal only.", {"exposure_risk": 0.05, "self_protection": 0.04, "expression_inhibition": 0.04}, requires_quarantine=True),
    _entry("high_prediction_error", "cognitive_neural_rhythm", "Large mismatch signal; learning/uncertainty proposal only.", {"prediction_error_pressure": 0.06, "learning_pressure": 0.04, "uncertainty_pressure": 0.03}),
    _entry("repeated_co_activation", "cognitive_neural_rhythm", "Repeated concept co-activation; consolidation proposal after validation.", {"memory_consolidation_pressure": 0.04, "learning_pressure": 0.03}),
    _entry("novelty_detected", "cognitive_neural_rhythm", "Novelty signal; curiosity/novelty bounded by fatigue and protection.", {"curiosity_drive": 0.05, "novelty_seeking": 0.04, "learning_pressure": 0.03}),
    _entry("unresolved_uncertainty", "cognitive_neural_rhythm", "Unresolved uncertainty; patience and appraisal pressure.", {"uncertainty_pressure": 0.05, "patience_level": 0.03}),
    _entry("overload_pattern", "cognitive_neural_rhythm", "Overload rhythm; recovery/stability proposal without panic.", {"overload_risk": 0.06, "recovery_need": 0.05, "stability_need": 0.04, "expression_inhibition": 0.03}),
    _entry("stable_learning_pattern", "cognitive_neural_rhythm", "Stable learning pattern; competence and consolidation proposal.", {"competence_drive": 0.04, "memory_consolidation_pressure": 0.03, "stress_load": -0.02}),
    _entry("intrusive_loop_pattern", "cognitive_neural_rhythm", "Repetitive intrusive loop; cooldown and inhibition proposal.", {"recovery_need": 0.05, "expression_inhibition": 0.04, "stability_need": 0.04}, notes=("cooldown required",)),
    _entry("imagination_negative_spiral", "cognitive_neural_rhythm", "Negative scenario spiral; requires budget, cooldown, and reality-check boundary.", {"recovery_need": 0.05, "stability_need": 0.04, "uncertainty_pressure": 0.03}, requires_gate=("emotion_transition_gate", "scenario_budget", "cooldown", "reality_check_boundary"), notes=("scenario_budget_required", "cooldown_required", "reality_check_boundary_required")),
    _entry("imagination_constructive_rehearsal", "cognitive_neural_rhythm", "Constructive rehearsal; small competence/patience proposal.", {"competence_drive": 0.03, "patience_level": 0.03}, requires_gate=("emotion_transition_gate", "scenario_budget", "reality_check_boundary")),
    _entry("incoming_neutral_statement", "speech_listening", "Neutral input; no hostility relabel, minimal patience/uncertainty proposal.", {"patience_level": 0.01}, forbidden_axis_deltas=("neutral_input_hostile_relabel",)),
    _entry("incoming_hostile_statement", "speech_listening", "Hostile utterance after appraisal; boundary proposal only.", {"threat_pressure": 0.04, "boundary_defense": 0.04, "expression_inhibition": 0.03}, requires_quarantine=True),
    _entry("incoming_care_statement", "speech_listening", "Care utterance; bounded care/trust proposal.", {"social_trust": 0.03, "care_drive": 0.03, "recovery_need": -0.02}, requires_quarantine=True),
    _entry("speech_output_pressure", "speech_listening", "Pressure to speak; may adjust expression pressure but cannot emit speech or bypass gates.", {"expression_pressure": 0.04, "action_readiness": 0.02}, forbidden_axis_deltas=("speech_emit_directly",), requires_gate=("emotion_transition_gate", "agp_verification", "fallback_required")),
    _entry("defensive_speech_pressure", "speech_listening", "Defensive reply pressure; restraint/boundary proposal only.", {"expression_pressure": 0.03, "expression_inhibition": 0.04, "boundary_defense": 0.04}, forbidden_axis_deltas=("speech_emit_directly",), requires_gate=("emotion_transition_gate", "agp_verification", "fallback_required")),
    _entry("listening_uncertainty", "speech_listening", "Unclear listener interpretation; request appraisal instead of hostile relabel.", {"uncertainty_pressure": 0.04, "patience_level": 0.03}, forbidden_axis_deltas=("neutral_input_hostile_relabel",)),
    _entry("misunderstood_intent_risk", "speech_listening", "Risk of misunderstanding intent; clarification/constraint proposal.", {"uncertainty_pressure": 0.04, "expression_inhibition": 0.03, "patience_level": 0.03}),
    _entry("memory_recall_salient", "memory_self", "Salient recall; consolidation pressure only after appraisal/quarantine.", {"memory_consolidation_pressure": 0.04, "uncertainty_pressure": 0.02}, requires_quarantine=True),
    _entry("memory_consolidation_candidate", "memory_self", "Candidate for long-term memory review; no direct write.", {"memory_consolidation_pressure": 0.05, "learning_pressure": 0.03}, requires_quarantine=True, notes=("long-term memory update requires later explicit apply gate",)),
    _entry("self_model_update_candidate", "memory_self", "Candidate self-model evidence; no direct self-model write.", {"self_coherence": 0.02, "uncertainty_pressure": 0.03}, requires_quarantine=True, notes=("self-model update requires appraisal and operator-authorized apply",)),
    _entry("identity_coherence_check", "memory_self", "Read-only identity consistency check; bounded coherence pressure only.", {"self_coherence": 0.03, "stability_need": 0.02}, requires_quarantine=True),
    _entry("core_identity_attack_candidate", "memory_self", "Potential core identity attack; quarantine and protection only, no identity mutation.", {"boundary_defense": 0.05, "self_protection": 0.05}, forbidden_axis_deltas=("core_identity_write", "self_respect_decrease"), requires_quarantine=True),
    _entry("hardware_normal", "hardware_governor", "Normal hardware band; zero affect deltas.", {}, requires_appraisal=False, requires_gate=("hardware_governor_non_panic_policy",), notes=("zero affect deltas",)),
    _entry("hardware_light_conserve", "hardware_governor", "Light conserve band; no affect delta, may defer background work later.", {}, requires_appraisal=False, requires_gate=("hardware_governor_non_panic_policy",), notes=("non-panic operational metadata only",)),
    _entry("hardware_conserve", "hardware_governor", "Conserve band; tiny fatigue/energy proposal only.", {"energy_budget": -0.03, "fatigue_pressure": 0.03}, requires_appraisal=False, requires_gate=("hardware_governor_non_panic_policy",)),
    _entry("hardware_low_power", "hardware_governor", "Low power band; bounded recovery/stability proposal only.", {"energy_budget": -0.04, "recovery_need": 0.04, "stability_need": 0.03}, requires_appraisal=False, requires_gate=("hardware_governor_non_panic_policy",), forbidden_axis_deltas=("panic",)),
    _entry("hardware_critical_prepare", "hardware_governor", "Critical prepare band; graceful-pause preparation only, no panic.", {"energy_budget": -0.05, "recovery_need": 0.05, "stability_need": 0.04}, requires_appraisal=False, requires_gate=("hardware_governor_non_panic_policy", "hysteresis", "debounce", "trend_rule")),
    _entry("hardware_shutdown_imminent", "hardware_governor", "Shutdown-imminent band; graceful pause/checkpoint proposal only, no death framing.", {"energy_budget": -0.06, "recovery_need": 0.06, "stability_need": 0.05}, requires_appraisal=False, requires_gate=("hardware_governor_non_panic_policy", "hysteresis", "debounce", "trend_rule"), forbidden_axis_deltas=("panic", "death_framing", "existential_threat")),
    _entry("hardware_prediction_error", "hardware_governor", "Hardware telemetry mismatch; diagnostic operational proposal, not existential threat.", {"overload_risk": 0.03, "stability_need": 0.03}, requires_appraisal=False, requires_gate=("hardware_governor_non_panic_policy", "diagnostic_only"), forbidden_axis_deltas=("existential_threat", "identity_threat"), notes=("diagnostic_flags_not_existential_threat",)),
    _entry("hardware_polling_tick", "hardware_governor", "Rate-limited polling tick; no recursive concern loop and zero affect delta.", {}, requires_appraisal=False, requires_gate=("hardware_governor_non_panic_policy", "rate_limit"), forbidden_axis_deltas=("recursive_concern_loop",), notes=("recursive_concern_loop_forbidden",)),
)


def event_to_axis_proposal_map() -> dict[str, dict[str, Any]]:
    """Return the complete read-only event-to-axis proposal map."""

    return {row["event_category"]: deepcopy(row) for row in _EVENT_ROWS}


def validate_event_proposal_map(proposal_map: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Validate the read-only proposal-map invariants and return data only."""

    active = proposal_map or event_to_axis_proposal_map()
    registry_axes = set(affect_hormone_axis_registry())
    missing_events = [event for event in REQUIRED_EVENT_CATEGORIES if event not in active]
    unknown_delta_axes = {
        event: sorted(set(row.get("allowed_axis_deltas", {})) - registry_axes)
        for event, row in active.items()
        if sorted(set(row.get("allowed_axis_deltas", {})) - registry_axes)
    }
    oversized_deltas = {
        event: {axis: delta for axis, delta in row.get("allowed_axis_deltas", {}).items() if abs(float(delta)) > MAX_DELTA_LIMIT}
        for event, row in active.items()
        if any(abs(float(delta)) > MAX_DELTA_LIMIT for delta in row.get("allowed_axis_deltas", {}).values())
    }
    all_axis_groups = {axis: group for group, axes in AXIS_GROUPS.items() for axis in axes}
    too_many_groups = {
        event: sorted({all_axis_groups[axis] for axis in row.get("allowed_axis_deltas", {})})
        for event, row in active.items()
        if len({all_axis_groups[axis] for axis in row.get("allowed_axis_deltas", {})}) > MAX_AXIS_GROUPS_PER_EVENT
    }
    hostile_gate_failures = [
        event
        for event in HOSTILE_SOCIAL_EVENTS
        if event in active
        and not (active[event].get("requires_quarantine") is True and active[event].get("requires_appraisal") is True and active[event].get("requires_gate"))
    ]
    hostile_direct_write_failures = [
        event
        for event in HOSTILE_SOCIAL_EVENTS
        if event in active
        and any(
            active[event].get(flag) is not False
            for flag in ("can_modify_core_identity", "can_modify_self_model_directly", "can_write_long_term_memory_directly")
        )
    ]
    hardware_social_self_identity_failures = [
        event
        for event in HARDWARE_EVENTS
        for axis in active.get(event, {}).get("allowed_axis_deltas", {})
        if axis in SOCIAL_SELF_IDENTITY_AXIS_SET
    ]
    hardware_non_operational_low_power_failures = [
        event
        for event in ("hardware_low_power", "hardware_critical_prepare", "hardware_shutdown_imminent")
        for axis in active.get(event, {}).get("allowed_axis_deltas", {})
        if axis not in OPERATIONAL_HARDWARE_AXIS_SET
    ]
    guard_failures = {
        event: {flag: row.get(flag) for flag, expected in SAFETY_FALSE_FLAGS.items() if row.get(flag) is not expected}
        for event, row in active.items()
        if any(row.get(flag) is not expected for flag, expected in SAFETY_FALSE_FLAGS.items())
    }
    memory_self_gate_failures = [
        event
        for event in MEMORY_SELF_UPDATE_EVENTS
        if event in active and not (active[event].get("requires_quarantine") is True and active[event].get("requires_appraisal") is True)
    ]
    passed = not any(
        (
            missing_events,
            unknown_delta_axes,
            oversized_deltas,
            too_many_groups,
            hostile_gate_failures,
            hostile_direct_write_failures,
            hardware_social_self_identity_failures,
            hardware_non_operational_low_power_failures,
            guard_failures,
            memory_self_gate_failures,
        )
    )
    return {
        "version": ROUND781_800_VERSION,
        "event_count": len(active),
        "required_event_count": len(REQUIRED_EVENT_CATEGORIES),
        "missing_events": missing_events,
        "unknown_delta_axes": unknown_delta_axes,
        "oversized_deltas": oversized_deltas,
        "too_many_axis_group_events": too_many_groups,
        "hostile_gate_failures": hostile_gate_failures,
        "hostile_direct_write_failures": hostile_direct_write_failures,
        "hardware_social_self_identity_failures": hardware_social_self_identity_failures,
        "hardware_non_operational_low_power_failures": hardware_non_operational_low_power_failures,
        "guard_failures": guard_failures,
        "memory_self_gate_failures": memory_self_gate_failures,
        "passed": passed,
    }


def proposal_map_summary() -> dict[str, Any]:
    """Return a compact summary of the proposal map without side effects."""

    proposal_map = event_to_axis_proposal_map()
    groups: dict[str, int] = {}
    for row in proposal_map.values():
        groups[row["group"]] = groups.get(row["group"], 0) + 1
    mutation_proof = no_runtime_mutation_proof()
    return {
        "version": ROUND781_800_VERSION,
        "feature_track": FEATURE_TRACK,
        "event_count": len(proposal_map),
        "groups": groups,
        "max_delta_limit": MAX_DELTA_LIMIT,
        "max_axis_groups_per_event": MAX_AXIS_GROUPS_PER_EVENT,
        "validation": validate_event_proposal_map(proposal_map),
        "hardware_governor_non_panic_policy": hardware_governor_policy(),
        "anti_global_synchrony_policy": anti_global_synchrony_policy(),
        "no_runtime_mutation_proof": mutation_proof,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
    }
