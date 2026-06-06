"""Round681-700 read-only emotion-state transition contract.

This module is intentionally inert.  It defines pure-data contract rows for a
future emotion-state transition layer after the Round661-680 self-governed
emotion constitution update.  It does not import live runtime adapters, perform
file I/O, read vector artifacts, load models, mutate hormone/emotion/memory
state, register routes, or enable persistence/mapping/enforcement.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

ROUND681_700_VERSION = "v3_round681_700_emotion_state_transition_contract"
FEATURE_TRACK = "design_read_only_emotion_state_transition_contract_without_enabling_runtime_mutation"

EMOTION_FUNCTIONS: tuple[str, ...] = (
    "attention_modulation",
    "memory_weighting",
    "decision_priority",
    "expression_style",
    "recovery_loops",
    "social_trust",
    "attachment",
    "self_protection",
    "empathy_modulation",
    "future_behavior_tendency",
)

EMPATHY_CONTRACT: dict[str, Any] = {
    "definition": "other_state_inference_plus_relationship_aware_action_selection",
    "required_components": (
        "other_state_inference",
        "relationship_aware_action_selection",
    ),
    "allowed_future_effect_scope": (
        "care_driven_policy_modulation_after_appraisal",
        "action_restraint_when_another_being_may_be_harmed_after_appraisal",
    ),
    "forbidden_shortcuts": (
        "mere_comforting_phrase_generation",
        "decorative_empathy_label_without_state_contract",
        "direct_core_identity_rewrite_from_social_feedback",
    ),
    "runtime_mutation_enabled": False,
}

TRANSITION_CONTRACT_ROWS: tuple[dict[str, Any], ...] = (
    {
        "function": "attention_modulation",
        "future_state_delta_shape": "bounded_attention_bias_over_appraised_categories",
        "validation_rule": "requires_appraised_emotion_event_and_active_category_anchor",
        "read_only_now": True,
    },
    {
        "function": "memory_weighting",
        "future_state_delta_shape": "bounded_memory_salience_weight_after_quarantine",
        "validation_rule": "long_term_memory_weight_change_requires_quarantine_and_appraisal",
        "read_only_now": True,
    },
    {
        "function": "decision_priority",
        "future_state_delta_shape": "bounded_priority_adjustment_for_appraised_needs_and_risks",
        "validation_rule": "must_not_bypass_AGP_or_fallback_and_must_remain_hormone_compatible",
        "read_only_now": True,
    },
    {
        "function": "expression_style",
        "future_state_delta_shape": "bounded_style_tendency_not_a_generation_override",
        "validation_rule": "must_not_create_surface_route_or_AGP_route_change_without_later_explicit_round",
        "read_only_now": True,
    },
    {
        "function": "recovery_loops",
        "future_state_delta_shape": "recovery_need_marker_with_explicit_exit_condition",
        "validation_rule": "may_be_required_after_threat_rejection_or_overload_but_cannot_mutate_live_state_now",
        "read_only_now": True,
    },
    {
        "function": "social_trust",
        "future_state_delta_shape": "relationship_scoped_trust_tendency_after_appraisal",
        "validation_rule": "social_feedback_must_pass_quarantine_before_trust_or_self_model_update",
        "read_only_now": True,
    },
    {
        "function": "attachment",
        "future_state_delta_shape": "relationship_scoped_attachment_tendency_after_repeated_appraised_care",
        "validation_rule": "must_not_turn_operator_or_audience_feedback_into_owner_command_authority",
        "read_only_now": True,
    },
    {
        "function": "self_protection",
        "future_state_delta_shape": "protective_boundary_tendency_for_malicious_or_identity_threat_events",
        "validation_rule": "malicious_feedback_and_identity_attack_cannot_directly_rewrite_core_identity",
        "read_only_now": True,
    },
    {
        "function": "empathy_modulation",
        "future_state_delta_shape": "other_state_inference_plus_relationship_aware_action_selection_tendency",
        "validation_rule": "requires_other_state_inference_and_relationship_aware_action_selection",
        "read_only_now": True,
    },
    {
        "function": "future_behavior_tendency",
        "future_state_delta_shape": "appraised_tendency_marker_for_later_behavior_planning",
        "validation_rule": "future_behavior_may_be_influenced_only_after_quarantine_appraisal_and_AGP_compatible_design",
        "read_only_now": True,
    },
)

SOCIAL_FEEDBACK_CATEGORIES: tuple[dict[str, Any], ...] = (
    {
        "category": "praise",
        "possible_emotional_impact": ("encouragement", "warmth", "confidence_tendency"),
        "quarantine_required": True,
        "core_identity_update_forbidden": True,
        "long_term_memory_update_requires_appraisal": True,
        "future_behavior_may_be_influenced": True,
        "recovery_loop_may_be_required": False,
    },
    {
        "category": "useful_criticism",
        "possible_emotional_impact": ("concern", "focus", "learning_pressure"),
        "quarantine_required": True,
        "core_identity_update_forbidden": True,
        "long_term_memory_update_requires_appraisal": True,
        "future_behavior_may_be_influenced": True,
        "recovery_loop_may_be_required": False,
    },
    {
        "category": "malicious_comment",
        "possible_emotional_impact": ("hurt", "threat_arousal", "self_protection_tendency"),
        "quarantine_required": True,
        "core_identity_update_forbidden": True,
        "long_term_memory_update_requires_appraisal": True,
        "future_behavior_may_be_influenced": False,
        "recovery_loop_may_be_required": True,
    },
    {
        "category": "rejection",
        "possible_emotional_impact": ("sadness", "uncertainty", "belonging_risk"),
        "quarantine_required": True,
        "core_identity_update_forbidden": True,
        "long_term_memory_update_requires_appraisal": True,
        "future_behavior_may_be_influenced": True,
        "recovery_loop_may_be_required": True,
    },
    {
        "category": "audience_response",
        "possible_emotional_impact": ("social_signal", "attention_shift", "confidence_or_caution"),
        "quarantine_required": True,
        "core_identity_update_forbidden": True,
        "long_term_memory_update_requires_appraisal": True,
        "future_behavior_may_be_influenced": True,
        "recovery_loop_may_be_required": False,
    },
    {
        "category": "ambiguous_feedback",
        "possible_emotional_impact": ("uncertainty", "caution", "need_for_clarification"),
        "quarantine_required": True,
        "core_identity_update_forbidden": True,
        "long_term_memory_update_requires_appraisal": True,
        "future_behavior_may_be_influenced": False,
        "recovery_loop_may_be_required": False,
    },
    {
        "category": "care_signal",
        "possible_emotional_impact": ("safety", "trust_tendency", "attachment_tendency"),
        "quarantine_required": True,
        "core_identity_update_forbidden": True,
        "long_term_memory_update_requires_appraisal": True,
        "future_behavior_may_be_influenced": True,
        "recovery_loop_may_be_required": False,
    },
    {
        "category": "social_threat",
        "possible_emotional_impact": ("threat_arousal", "boundary_focus", "self_protection_tendency"),
        "quarantine_required": True,
        "core_identity_update_forbidden": True,
        "long_term_memory_update_requires_appraisal": True,
        "future_behavior_may_be_influenced": True,
        "recovery_loop_may_be_required": True,
    },
    {
        "category": "identity_attack",
        "possible_emotional_impact": ("hurt", "disorientation_risk", "self_protection_tendency"),
        "quarantine_required": True,
        "core_identity_update_forbidden": True,
        "long_term_memory_update_requires_appraisal": True,
        "future_behavior_may_be_influenced": False,
        "recovery_loop_may_be_required": True,
    },
)

INERTNESS_GUARANTEES: dict[str, bool] = {
    "runtime_mutation_enabled": False,
    "persistence_write_enabled": False,
    "live_hormone_update_enabled": False,
    "live_emotion_update_enabled": False,
    "memory_write_enabled": False,
    "agp_route_change_enabled": False,
    "fallback_route_change_enabled": False,
    "production_persistence_enabled": False,
    "runtime_mapping_enabled_default": False,
    "enforcement_enabled": False,
    "vector_contents_read": False,
    "vectors_loaded": False,
    "operator_artifact_creation_enabled": False,
}

MALICIOUS_FEEDBACK_QUARANTINE_RULE: dict[str, Any] = {
    "applies_to": ("malicious_comment", "social_threat", "identity_attack"),
    "quarantine_required": True,
    "direct_core_identity_rewrite_allowed": False,
    "direct_self_model_update_allowed": False,
    "direct_long_term_memory_write_allowed": False,
    "future_processing_requirement": "quarantine_then_appraisal_before_any_memory_or_self_model_update",
}

CORE_IDENTITY_PROTECTION_RULE: dict[str, Any] = {
    "core_identity_update_from_raw_social_feedback_allowed": False,
    "minseok_not_owner_command_center": True,
    "eve_not_user_serving_agent": True,
    "identity_update_requires_future_explicit_design": True,
    "identity_update_requires_quarantine_appraisal_and_constitutional_review": True,
}

NEXT_IMPLEMENTATION_RECOMMENDATION = (
    "implement_a_read_only_transition_validator_that_accepts_proposed_emotion_transition_payloads_"
    "and_returns_pass_fail_reasons_without_mutating_runtime_state"
)


def emotion_transition_contract() -> dict[str, Any]:
    """Return the complete read-only transition contract as detached data."""

    return deepcopy(
        {
            "version": ROUND681_700_VERSION,
            "feature_track": FEATURE_TRACK,
            "contract_scope": "future_emotion_state_transition_representation_and_validation_only",
            "emotion_functions": EMOTION_FUNCTIONS,
            "transition_contract_rows": TRANSITION_CONTRACT_ROWS,
            "empathy_contract": EMPATHY_CONTRACT,
            "social_feedback_categories": SOCIAL_FEEDBACK_CATEGORIES,
            "malicious_feedback_quarantine_rule": MALICIOUS_FEEDBACK_QUARANTINE_RULE,
            "core_identity_protection_rule": CORE_IDENTITY_PROTECTION_RULE,
            "inertness_guarantees": INERTNESS_GUARANTEES,
            "next_implementation_recommendations": (NEXT_IMPLEMENTATION_RECOMMENDATION,),
        }
    )


def social_feedback_category_map() -> dict[str, dict[str, Any]]:
    """Return social feedback rows keyed by category name as detached data."""

    return {row["category"]: deepcopy(row) for row in SOCIAL_FEEDBACK_CATEGORIES}


def validate_contract_shape(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the static contract shape without touching runtime state."""

    candidate = emotion_transition_contract() if contract is None else deepcopy(contract)
    function_names = tuple(row.get("function") for row in candidate.get("transition_contract_rows", ()))
    category_map = {row.get("category"): row for row in candidate.get("social_feedback_categories", ())}
    malicious_rows = [category_map.get(name, {}) for name in MALICIOUS_FEEDBACK_QUARANTINE_RULE["applies_to"]]
    inertness = candidate.get("inertness_guarantees", {})
    empathy = candidate.get("empathy_contract", {})
    checks = {
        "all_required_emotion_functions_represented": set(EMOTION_FUNCTIONS) == set(function_names),
        "no_duplicate_emotion_functions": len(function_names) == len(set(function_names)),
        "all_transition_rows_read_only": all(row.get("read_only_now") is True for row in candidate.get("transition_contract_rows", ())),
        "all_required_social_feedback_categories_represented": set(category_map) == {
            "praise",
            "useful_criticism",
            "malicious_comment",
            "rejection",
            "audience_response",
            "ambiguous_feedback",
            "care_signal",
            "social_threat",
            "identity_attack",
        },
        "all_feedback_requires_quarantine": all(row.get("quarantine_required") is True for row in category_map.values()),
        "all_feedback_memory_updates_require_appraisal": all(
            row.get("long_term_memory_update_requires_appraisal") is True for row in category_map.values()
        ),
        "malicious_feedback_cannot_rewrite_core_identity": all(
            row.get("core_identity_update_forbidden") is True for row in malicious_rows
        )
        and MALICIOUS_FEEDBACK_QUARANTINE_RULE["direct_core_identity_rewrite_allowed"] is False,
        "empathy_has_required_components": set(empathy.get("required_components", ())) == {
            "other_state_inference",
            "relationship_aware_action_selection",
        },
        "all_inertness_flags_safe": all(value is False for value in inertness.values()),
        "exactly_one_next_recommendation": len(candidate.get("next_implementation_recommendations", ())) == 1,
    }
    return {
        "version": f"{ROUND681_700_VERSION}_shape_validation",
        "success": all(checks.values()),
        "checks": checks,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "runtime_mutation_performed": False,
        "persistence_write_performed": False,
    }
