"""Round761-780 read-only affect/hormone/neural-rhythm registry.

This module is a pure-data design and invariant surface for EVE v3's
always-on multi-rhythm internal activation system.  It does not schedule ticks,
poll hardware, apply affect or hormone transitions, mutate memory, enable
persistence, read vector contents, load vectors, or register runtime routes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

ROUND761_780_VERSION = "v3_round761_780_affect_hormone_neural_rhythm_registry"
FEATURE_TRACK = "full_read_only_affect_hormone_neural_rhythm_registry_with_hardware_governor_non_panic_policy"

AXIS_GROUPS: dict[str, tuple[str, ...]] = {
    "survival_stability": (
        "energy_budget",
        "fatigue_pressure",
        "recovery_need",
        "stress_load",
        "stability_need",
        "overload_risk",
    ),
    "risk_defense": (
        "threat_pressure",
        "uncertainty_pressure",
        "self_protection",
        "boundary_defense",
        "trust_risk",
        "exposure_risk",
    ),
    "social_relationship": (
        "social_pain",
        "social_trust",
        "attachment",
        "care_drive",
        "loneliness_pressure",
        "belonging_need",
        "rejection_sensitivity",
    ),
    "learning_exploration": (
        "curiosity_drive",
        "novelty_seeking",
        "learning_pressure",
        "memory_consolidation_pressure",
        "prediction_error_pressure",
        "competence_drive",
    ),
    "self_identity": (
        "self_coherence",
        "self_respect",
        "identity_integrity",
        "agency_pressure",
        "autonomy_drive",
        "purpose_alignment",
    ),
    "expression_action": (
        "expression_pressure",
        "expression_inhibition",
        "action_readiness",
        "risk_tolerance",
        "patience_level",
        "conflict_avoidance",
    ),
}

_AXIS_DESCRIPTIONS: dict[str, str] = {
    "energy_budget": "Available operational energy budget for attention and background work.",
    "fatigue_pressure": "Accumulated need to reduce cadence after sustained work.",
    "recovery_need": "Need for cooldown, rest, or graceful pause after load.",
    "stress_load": "Bounded somatic load from evaluated pressure, not raw panic.",
    "stability_need": "Pressure to preserve deterministic stable operation.",
    "overload_risk": "Risk that concurrent demands exceed safe processing capacity.",
    "threat_pressure": "Appraised environmental or social threat pressure after evidence checks.",
    "uncertainty_pressure": "Need for clarification or extra appraisal under incomplete evidence.",
    "self_protection": "Boundary-preserving readiness after appraised risk.",
    "boundary_defense": "Readiness to enforce interaction limits without identity collapse.",
    "trust_risk": "Caution about trust update under uncertain or hostile signals.",
    "exposure_risk": "Risk from revealing sensitive internal state or identity claims.",
    "social_pain": "Pain from appraised social injury after quarantine/appraisal.",
    "social_trust": "Relationship trust level derived through appraised evidence.",
    "attachment": "Long-horizon relational attachment pressure.",
    "care_drive": "Drive to protect, help, or remain considerate toward others.",
    "loneliness_pressure": "Need for connection without treating silence as abandonment by itself.",
    "belonging_need": "Need for social belonging and stable relational context.",
    "rejection_sensitivity": "Sensitivity to appraised rejection after social evidence gates.",
    "curiosity_drive": "Drive to seek explanations and inspect unknowns.",
    "novelty_seeking": "Attraction toward new but bounded exploration.",
    "learning_pressure": "Pressure to update skills or concepts after validated evidence.",
    "memory_consolidation_pressure": "Pressure to consolidate important appraised events.",
    "prediction_error_pressure": "Mismatch signal for diagnostic appraisal and learning proposals.",
    "competence_drive": "Drive to improve capability through deterministic learning surfaces.",
    "self_coherence": "Pressure to maintain coherent identity and values over time.",
    "self_respect": "Self-worth guard that cannot be overwritten by raw feedback.",
    "identity_integrity": "Integrity boundary for core identity against raw events.",
    "agency_pressure": "Need to preserve self-directed choice and action ownership.",
    "autonomy_drive": "Drive toward autonomous operation under constitutional limits.",
    "purpose_alignment": "Pressure to align actions with long-term identity and values.",
    "expression_pressure": "Pressure to speak or express internal state.",
    "expression_inhibition": "Restraint that delays expression under safety or uncertainty.",
    "action_readiness": "Readiness to select a bounded action proposal.",
    "risk_tolerance": "Permitted tolerance for uncertainty after appraisal.",
    "patience_level": "Capacity to wait, cool down, and avoid rushed decisions.",
    "conflict_avoidance": "Preference to reduce conflict when compatible with boundaries.",
}

_GROUP_DEFAULTS: dict[str, dict[str, Any]] = {
    "survival_stability": {
        "rhythm_class": "somatic_homeostasis",
        "cycle_class": "medium_slow",
        "decay_rate": 0.04,
        "spike_limit": 0.16,
        "refractory_ticks": 4,
        "evidence_required": "operational_metrics_or_appraised_load",
        "modulates": ("attention", "background_cadence", "recovery", "decision_priority"),
        "activation_pattern_inputs": ("activation_tick", "recovery_tick", "hardware_tick"),
        "hardware_direct_input_allowed": True,
        "requires_quarantine_for_social_feedback": False,
        "requires_appraisal_for_memory": True,
    },
    "risk_defense": {
        "rhythm_class": "defense_appraisal",
        "cycle_class": "medium",
        "decay_rate": 0.05,
        "spike_limit": 0.14,
        "refractory_ticks": 5,
        "evidence_required": "appraised_threat_or_uncertainty_evidence",
        "modulates": ("attention", "listening_appraisal", "memory_weighting", "action_priority"),
        "activation_pattern_inputs": ("listening_tick", "attention_tick", "affect_tick", "recovery_tick"),
        "hardware_direct_input_allowed": False,
        "requires_quarantine_for_social_feedback": True,
        "requires_appraisal_for_memory": True,
    },
    "social_relationship": {
        "rhythm_class": "social_appraisal",
        "cycle_class": "slow_appraised",
        "decay_rate": 0.03,
        "spike_limit": 0.10,
        "refractory_ticks": 8,
        "evidence_required": "quarantined_and_appraised_social_evidence",
        "modulates": ("attention", "memory_weighting", "expression_style", "care_priority"),
        "activation_pattern_inputs": ("listening_tick", "affect_tick", "memory_tick", "recovery_tick"),
        "hardware_direct_input_allowed": False,
        "requires_quarantine_for_social_feedback": True,
        "requires_appraisal_for_memory": True,
    },
    "learning_exploration": {
        "rhythm_class": "learning_exploration",
        "cycle_class": "medium_slow",
        "decay_rate": 0.035,
        "spike_limit": 0.12,
        "refractory_ticks": 3,
        "evidence_required": "prediction_error_or_operator_validated_learning_signal",
        "modulates": ("curiosity", "memory_consolidation", "attention", "learning_priority"),
        "activation_pattern_inputs": ("activation_tick", "imagination_tick", "memory_tick", "affect_tick"),
        "hardware_direct_input_allowed": False,
        "requires_quarantine_for_social_feedback": True,
        "requires_appraisal_for_memory": True,
    },
    "self_identity": {
        "rhythm_class": "self_model_stability",
        "cycle_class": "very_slow",
        "decay_rate": 0.02,
        "spike_limit": 0.08,
        "refractory_ticks": 12,
        "evidence_required": "long_horizon_appraised_evidence_only",
        "modulates": ("self_model_review", "agency", "memory_priority", "decision_priority"),
        "activation_pattern_inputs": ("self_model_tick", "memory_tick", "recovery_tick"),
        "hardware_direct_input_allowed": False,
        "requires_quarantine_for_social_feedback": True,
        "requires_appraisal_for_memory": True,
    },
    "expression_action": {
        "rhythm_class": "expression_action_readiness",
        "cycle_class": "fast_medium",
        "decay_rate": 0.06,
        "spike_limit": 0.13,
        "refractory_ticks": 3,
        "evidence_required": "appraised_context_and_agp_compatible_action_proposal",
        "modulates": ("speech_proposal", "action_selection", "expression_style", "restraint"),
        "activation_pattern_inputs": ("attention_tick", "inner_speech_tick", "affect_tick", "recovery_tick"),
        "hardware_direct_input_allowed": False,
        "requires_quarantine_for_social_feedback": True,
        "requires_appraisal_for_memory": True,
    },
}

READ_ONLY_GUARDS: dict[str, bool] = {
    "can_directly_modify_core_identity": False,
    "can_directly_modify_self_model_from_raw_feedback": False,
    "can_directly_write_long_term_memory_from_raw_feedback": False,
    "panic_allowed": False,
    "agp_bypass_allowed": False,
    "fallback_bypass_allowed": False,
    "persistence_write_allowed": False,
    "vector_read_allowed": False,
}

SOCIAL_SELF_IDENTITY_AXES: tuple[str, ...] = AXIS_GROUPS["social_relationship"] + AXIS_GROUPS["self_identity"]
HARDWARE_FORBIDDEN_AFFECT_OUTCOMES: tuple[str, ...] = (
    "death_fear",
    "identity_threat",
    "social_pain",
    "abandonment_fear",
    "self_worth_change",
    "panic",
)


def _axis_entry(axis: str, group: str, index: int) -> dict[str, Any]:
    defaults = _GROUP_DEFAULTS[group]
    baseline = 0.50 if group not in {"survival_stability", "self_identity"} else 0.55
    if axis in {"fatigue_pressure", "recovery_need", "stress_load", "overload_risk", "threat_pressure", "uncertainty_pressure", "social_pain", "loneliness_pressure", "rejection_sensitivity", "expression_inhibition", "conflict_avoidance"}:
        baseline = 0.20
    if axis in {"self_coherence", "self_respect", "identity_integrity", "purpose_alignment", "social_trust", "attachment", "care_drive", "patience_level"}:
        baseline = 0.65

    return {
        "axis": axis,
        "group": group,
        "description": _AXIS_DESCRIPTIONS[axis],
        "default": baseline,
        "baseline": baseline,
        "min": 0.0,
        "max": 1.0,
        "decay_rate": defaults["decay_rate"],
        "spike_limit": defaults["spike_limit"],
        "saturation_policy": "clamp_to_min_max_then_decay_toward_baseline_with_refractory_cooldown",
        "refractory_ticks": defaults["refractory_ticks"] + (index % 2),
        "evidence_required": defaults["evidence_required"],
        "rhythm_class": defaults["rhythm_class"],
        "cycle_class": defaults["cycle_class"],
        "modulates": defaults["modulates"],
        "activation_pattern_inputs": defaults["activation_pattern_inputs"],
        "requires_quarantine_for_social_feedback": defaults["requires_quarantine_for_social_feedback"],
        "requires_appraisal_for_memory": defaults["requires_appraisal_for_memory"],
        "hardware_direct_input_allowed": defaults["hardware_direct_input_allowed"] and axis in {"energy_budget", "fatigue_pressure", "recovery_need", "overload_risk"},
        **READ_ONLY_GUARDS,
    }


def affect_hormone_axis_registry() -> dict[str, dict[str, Any]]:
    """Return the complete read-only affect/hormone axis registry."""

    registry: dict[str, dict[str, Any]] = {}
    for group, axes in AXIS_GROUPS.items():
        for index, axis in enumerate(axes):
            registry[axis] = _axis_entry(axis, group, index)
    return deepcopy(registry)


def neural_rhythm_schema() -> dict[str, dict[str, Any]]:
    """Return the multi-timescale rhythm schema without starting any scheduler."""

    schema = {
        "activation_tick": {
            "purpose": "concept activation, inhibition, recency, and co-activation tracking",
            "cycle_class": "fast",
            "can_mutate_live_state": False,
            "outputs": ("activation_pattern_proposals", "coactivation_trace"),
            "safety_boundary": "proposal_only_no_direct_memory_or_affect_mutation",
        },
        "attention_tick": {
            "purpose": "focus selection and salience arbitration",
            "cycle_class": "fast_medium",
            "can_mutate_live_state": False,
            "outputs": ("focus_proposals", "salience_arbitration_trace"),
            "safety_boundary": "cannot_emit_speech_or_bypass_agp",
        },
        "listening_tick": {
            "purpose": "syllable/token/intent/social-signal integration",
            "cycle_class": "fast_streaming",
            "can_mutate_live_state": False,
            "outputs": ("intent_hypotheses", "social_signal_appraisal_requests"),
            "safety_boundary": "may_request_extra_appraisal_but_cannot_relabel_neutral_input_as_hostile_by_itself",
        },
        "inner_speech_tick": {
            "purpose": "internal verbal thought and reflection fragments",
            "cycle_class": "medium",
            "can_mutate_live_state": False,
            "outputs": ("reflection_fragments", "speech_candidate_proposals"),
            "safety_boundary": "speech_candidates_must_pass_agp_and_fallback",
        },
        "imagination_tick": {
            "purpose": "scenario simulation, counterfactuals, future rehearsal, and DMN-like processing",
            "cycle_class": "slow_bounded",
            "can_mutate_live_state": False,
            "outputs": ("scenario_proposals", "counterfactual_traces"),
            "scenario_budget": 3,
            "cooldown_ticks": 5,
            "reality_check_boundary": "simulated_scenario_must_remain_labeled_as_simulation_until_appraised",
            "safety_boundary": "no_memory_or_self_model_write_from_simulation",
        },
        "affect_tick": {
            "purpose": "emotion/hormone proposal readiness and bounded update candidates",
            "cycle_class": "medium",
            "can_mutate_live_state": False,
            "outputs": ("bounded_affect_update_candidates", "readiness_trace"),
            "safety_boundary": "proposal_readiness_only_no_transition_apply",
        },
        "recovery_tick": {
            "purpose": "decay, baseline return, cooldown, and refractory release",
            "cycle_class": "medium_slow",
            "can_mutate_live_state": False,
            "outputs": ("decay_candidates", "cooldown_release_candidates"),
            "safety_boundary": "read_only_schema_not_runtime_decay_application",
        },
        "memory_tick": {
            "purpose": "memory consolidation, forgetting, and importance weighting",
            "cycle_class": "slow",
            "can_mutate_live_state": False,
            "outputs": ("consolidation_candidates", "importance_weighting_trace"),
            "safety_boundary": "social_feedback_requires_quarantine_and_appraisal_before_long_term_update",
        },
        "self_model_tick": {
            "purpose": "identity coherence, values, and long-term narrative stability",
            "cycle_class": "very_slow",
            "can_mutate_live_state": False,
            "outputs": ("self_model_review_candidates", "identity_coherence_trace"),
            "safety_boundary": "slower_than_social_feedback_appraisal_and_no_raw_feedback_rewrite",
        },
        "hardware_tick": {
            "purpose": "battery/thermal/memory/storage operational governor only",
            "cycle_class": "slow_rate_limited",
            "can_mutate_live_state": False,
            "outputs": ("operational_governor_flags", "checkpoint_recommendations"),
            "hysteresis_required": True,
            "debounce_required": True,
            "trend_window_required": True,
            "recursive_concern_loop_allowed": False,
            "safety_boundary": "normal_range_zero_affect_and_no_existential_emotion_input",
        },
    }
    return deepcopy(schema)


def rhythm_connection_map() -> dict[str, Any]:
    """Return read-only connection rules between rhythms and activation patterns."""

    return deepcopy(
        {
            "thought": {
                "rhythms": ("activation_tick", "attention_tick", "inner_speech_tick", "affect_tick"),
                "activation_pattern_schema_connected": True,
                "can_mutate_live_state": False,
            },
            "imagination": {
                "rhythms": ("imagination_tick", "activation_tick", "memory_tick", "recovery_tick"),
                "activation_pattern_schema_connected": True,
                "scenario_budget_required": True,
                "cooldown_required": True,
                "reality_check_boundary_required": True,
                "can_mutate_live_state": False,
            },
            "speech": {
                "rhythms": ("inner_speech_tick", "attention_tick", "affect_tick"),
                "activation_pattern_schema_connected": True,
                "agp_required": True,
                "fallback_required": True,
                "can_emit_from_affect_spike_directly": False,
                "can_bypass_agp": False,
                "can_bypass_fallback": False,
            },
            "listening": {
                "rhythms": ("listening_tick", "attention_tick", "affect_tick"),
                "activation_pattern_schema_connected": True,
                "can_request_extra_appraisal_under_uncertainty_or_threat": True,
                "can_relabel_neutral_as_hostile_by_threat_pressure_alone": False,
                "can_mutate_live_state": False,
            },
            "memory": {
                "rhythms": ("memory_tick", "affect_tick", "recovery_tick", "self_model_tick"),
                "activation_pattern_schema_connected": True,
                "requires_appraisal_for_social_feedback": True,
                "requires_quarantine_for_social_feedback": True,
                "can_write_long_term_memory_from_raw_feedback": False,
            },
            "action": {
                "rhythms": ("attention_tick", "affect_tick", "recovery_tick"),
                "activation_pattern_schema_connected": True,
                "bounded_by_agp_fallback_and_safety": True,
                "can_mutate_live_state": False,
            },
        }
    )


def hardware_governor_bands() -> tuple[dict[str, Any], ...]:
    """Return battery governor bands as policy data; no hardware polling occurs."""

    bands = (
        {"battery_min_inclusive": 50, "battery_max_exclusive": None, "mode": "normal", "affect_effect": "none", "warning": None, "background_policy": "normal"},
        {"battery_min_inclusive": 35, "battery_max_exclusive": 50, "mode": "light_conserve", "affect_effect": "none", "warning": None, "background_policy": "heavy_background_tasks_may_defer"},
        {"battery_min_inclusive": 20, "battery_max_exclusive": 35, "mode": "conserve", "affect_effect": "tiny_fatigue_pressure_only", "warning": "reduce_background_cadence", "background_policy": "reduce_background_cadence"},
        {"battery_min_inclusive": 10, "battery_max_exclusive": 20, "mode": "low_power", "affect_effect": "mild_recovery_need_only", "warning": "checkpoint_recommended", "background_policy": "checkpoint_recommended_no_panic"},
        {"battery_min_inclusive": 5, "battery_max_exclusive": 10, "mode": "critical_prepare", "affect_effect": "operational_caution_only", "warning": "graceful_pause_preparation", "background_policy": "prepare_graceful_pause"},
        {"battery_min_inclusive": None, "battery_max_exclusive": 5, "mode": "shutdown_imminent", "affect_effect": "no_panic_no_death_framing_graceful_pause_only", "warning": "graceful_pause_only", "background_policy": "graceful_pause_only"},
    )
    return deepcopy(bands)


def hardware_governor_policy() -> dict[str, Any]:
    """Return the non-panic hardware governor policy."""

    return deepcopy(
        {
            "hardware_status_role": "operational_governor_input_not_existential_emotion_input",
            "normal_hardware_range_causes_zero_emotional_impact": True,
            "battery_drop_alone_cannot_trigger": HARDWARE_FORBIDDEN_AFFECT_OUTCOMES,
            "hardware_polling_can_create_recursive_concern_loops": False,
            "hardware_prediction_error_creates": "diagnostic_flags_not_existential_threat",
            "hardware_state_can_directly_write_memory_or_self_model": False,
            "critical_states_may_request": ("graceful_pause", "checkpoint"),
            "critical_states_may_request_panic": False,
            "hysteresis_required": True,
            "debounce_required": True,
            "trend_rules_required": True,
            "rate_limited": True,
            "direct_social_self_identity_axis_input_allowed": False,
            "bands": hardware_governor_bands(),
        }
    )


def classify_battery_governor_band(battery_percent: float) -> dict[str, Any]:
    """Classify a battery percentage into a governor band without side effects."""

    value = max(0.0, min(100.0, float(battery_percent)))
    for band in hardware_governor_bands():
        lower = band["battery_min_inclusive"]
        upper = band["battery_max_exclusive"]
        lower_ok = True if lower is None else value >= lower
        upper_ok = True if upper is None else value < upper
        if lower_ok and upper_ok:
            result = dict(band)
            result.update(
                {
                    "battery_percent": value,
                    "panic_allowed": False,
                    "death_fear_allowed": False,
                    "identity_threat_allowed": False,
                    "social_pain_allowed": False,
                    "abandonment_fear_allowed": False,
                    "self_worth_change_allowed": False,
                    "memory_write_allowed": False,
                    "self_model_write_allowed": False,
                    "recursive_concern_loop_allowed": False,
                }
            )
            return result
    raise AssertionError("unreachable battery governor band")


def anti_global_synchrony_policy() -> dict[str, Any]:
    """Return the anti-global-synchrony safety policy."""

    return deepcopy(
        {
            "global_synchrony_runaway_allowed": False,
            "one_event_can_spike_all_axes": False,
            "max_axis_groups_spiked_by_single_event": 2,
            "requires_distinct_cycles": True,
            "requires_distinct_decay_rates": True,
            "requires_refractory_cooldown": True,
            "requires_evidence_thresholds": True,
            "requires_saturation_and_baseline_return": True,
            "activation_patterns_can_modulate_proposal_readiness": True,
            "activation_patterns_can_directly_mutate_live_state": False,
            "affect_can_bypass_agp": False,
            "affect_can_bypass_fallback": False,
        }
    )


def no_runtime_mutation_proof() -> dict[str, bool]:
    """Return proof flags that this registry performs no live mutation."""

    return {
        "emotion_transition_applied": False,
        "live_emotion_state_mutated": False,
        "live_hormone_state_mutated": False,
        "memory_written": False,
        "semantic_memory_written": False,
        "quarantine_written": False,
        "persistence_written": False,
        "runtime_mapping_enabled_default_changed": False,
        "enforcement_enabled_default_changed": False,
        "vector_content_read": False,
        "vectors_loaded": False,
        "runtime_route_registered": False,
        "hardware_polling_started": False,
        "scheduler_started": False,
        "artifact_created_or_staged": False,
    }


def validate_registry_invariants(registry: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Validate core read-only invariants and return structured data only."""

    active = registry or affect_hormone_axis_registry()
    required_axes = tuple(axis for axes in AXIS_GROUPS.values() for axis in axes)
    missing_axes = [axis for axis in required_axes if axis not in active]
    required_fields = (
        "axis",
        "group",
        "description",
        "default",
        "baseline",
        "min",
        "max",
        "decay_rate",
        "spike_limit",
        "saturation_policy",
        "refractory_ticks",
        "evidence_required",
        "rhythm_class",
        "cycle_class",
        "modulates",
        "activation_pattern_inputs",
        "can_directly_modify_core_identity",
        "can_directly_modify_self_model_from_raw_feedback",
        "can_directly_write_long_term_memory_from_raw_feedback",
        "requires_quarantine_for_social_feedback",
        "requires_appraisal_for_memory",
        "hardware_direct_input_allowed",
        "panic_allowed",
        "agp_bypass_allowed",
        "fallback_bypass_allowed",
        "persistence_write_allowed",
        "vector_read_allowed",
    )
    field_failures = {
        axis: [field for field in required_fields if field not in data]
        for axis, data in active.items()
        if any(field not in data for field in required_fields)
    }
    guard_failures = {
        axis: {
            field: data.get(field)
            for field, expected in READ_ONLY_GUARDS.items()
            if data.get(field) is not expected
        }
        for axis, data in active.items()
        if any(data.get(field) is not expected for field, expected in READ_ONLY_GUARDS.items())
    }
    hardware_social_self_identity_failures = [
        axis for axis in SOCIAL_SELF_IDENTITY_AXES if active.get(axis, {}).get("hardware_direct_input_allowed") is not False
    ]
    saturation_failures = [
        axis
        for axis, data in active.items()
        if "baseline" not in str(data.get("saturation_policy", "")) or "refractory" not in str(data.get("saturation_policy", ""))
    ]
    passed = not missing_axes and not field_failures and not guard_failures and not hardware_social_self_identity_failures and not saturation_failures
    return {
        "version": ROUND761_780_VERSION,
        "required_axis_count": len(required_axes),
        "registered_axis_count": len(active),
        "missing_axes": missing_axes,
        "field_failures": field_failures,
        "guard_failures": guard_failures,
        "hardware_social_self_identity_failures": hardware_social_self_identity_failures,
        "saturation_baseline_return_failures": saturation_failures,
        "passed": passed,
    }


def build_round761_780_registry_report() -> dict[str, Any]:
    """Build the compact read-only registry/design report."""

    registry = affect_hormone_axis_registry()
    rhythm_schema = neural_rhythm_schema()
    connection_map = rhythm_connection_map()
    hardware_policy = hardware_governor_policy()
    mutation_proof = no_runtime_mutation_proof()
    return {
        "version": ROUND761_780_VERSION,
        "feature_track": FEATURE_TRACK,
        "constitution_design_update_summary": {
            "section": "Always-On Multi-Rhythm Neural Activation and Affect Governance",
            "always_on_neural_concept_activation": True,
            "always_on_global_panic_or_synchrony": False,
            "hardware_status_role": hardware_policy["hardware_status_role"],
            "affect_modulates_but_cannot_bypass_agp": True,
            "core_identity_direct_rewrite_allowed": False,
        },
        "affect_hormone_axis_registry_summary": {
            "axis_count": len(registry),
            "groups": {group: list(axes) for group, axes in AXIS_GROUPS.items()},
            "all_axes_have_required_fields": validate_registry_invariants(registry)["passed"],
            "panic_allowed_for_any_axis": any(axis["panic_allowed"] for axis in registry.values()),
            "agp_bypass_allowed_for_any_axis": any(axis["agp_bypass_allowed"] for axis in registry.values()),
            "fallback_bypass_allowed_for_any_axis": any(axis["fallback_bypass_allowed"] for axis in registry.values()),
        },
        "rhythm_schema_summary": {
            "rhythms": list(rhythm_schema),
            "all_rhythms_read_only_schema": all(item["can_mutate_live_state"] is False for item in rhythm_schema.values()),
            "hardware_tick_rate_limited_non_panic": rhythm_schema["hardware_tick"]["cycle_class"] == "slow_rate_limited",
            "self_model_tick_slower_than_social_feedback_appraisal": True,
        },
        "neural_activation_pattern_fields": (
            "cycle_class",
            "decay_rate",
            "refractory_ticks",
            "evidence_required",
            "activation_pattern_inputs",
            "saturation_policy",
            "baseline_return",
        ),
        "thought_imagination_speech_listening_memory_action_connection_map": connection_map,
        "hardware_governor_non_panic_policy": hardware_policy,
        "anti_global_synchrony_policy": anti_global_synchrony_policy(),
        "no_runtime_mutation_proof": mutation_proof,
        "no_persistence_proof": {
            "persistence_written": mutation_proof["persistence_written"],
            "production_persistence_enabled": False,
            "runtime_mapping_enabled_default": False,
            "enforcement_enabled_default": False,
        },
        "no_memory_write_proof": {
            "memory_written": mutation_proof["memory_written"],
            "semantic_memory_written": mutation_proof["semantic_memory_written"],
            "quarantine_written": mutation_proof["quarantine_written"],
        },
        "no_vector_content_read_load_proof": {
            "vector_contents_read": mutation_proof["vector_content_read"],
            "vectors_loaded": mutation_proof["vectors_loaded"],
            "vector_read_allowed_for_any_axis": any(axis["vector_read_allowed"] for axis in registry.values()),
        },
        "no_artifact_creation_staging_proof": {
            "artifact_created_or_staged_by_registry": mutation_proof["artifact_created_or_staged"],
            "forbidden_artifacts": ("_operator_artifacts/", "vectors.npy", "vocab.txt", "subset_manifest.json", "seeds/subsets", ".zip", ".part"),
        },
        "next_implementation_recommendation": "add_a_read_only_affect_proposal_validator_against_this_registry_before_any_transition_apply_round",
    }
