"""Round701-720 read-only emotion transition payload validator.

The validator accepts proposed emotion-transition payloads and returns detached
pass/fail data. It intentionally does not import live runtime adapters, perform
file I/O, read vector artifacts, load vectors, mutate emotion/hormone/memory
state, register AGP/fallback/classifier routes, or enable persistence.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_event_to_axis_proposal_map import event_to_axis_proposal_map
from adapters.emotion_state_transition_contract import (
    EMPATHY_CONTRACT,
    MALICIOUS_FEEDBACK_QUARANTINE_RULE,
    ROUND681_700_VERSION,
    social_feedback_category_map,
)

ROUND701_720_VERSION = "v3_round701_720_emotion_transition_validator"
FEATURE_TRACK = (
    "implement_a_read_only_transition_validator_that_accepts_proposed_emotion_transition_payloads_"
    "and_returns_pass_fail_reasons_without_mutating_runtime_state"
)
EXPECTED_EMPATHY_MODE = EMPATHY_CONTRACT["definition"]
HIGH_RISK_SOCIAL_FEEDBACK_CATEGORIES: tuple[str, ...] = tuple(MALICIOUS_FEEDBACK_QUARANTINE_RULE["applies_to"])
_RUNTIME_FORBIDDEN_REASON = "runtime_mutation_requested_is_forbidden_in_round701_720_read_only_validator"
_PERSISTENCE_FORBIDDEN_REASON = "persistence_write_requested_is_forbidden_in_round701_720_read_only_validator"


def _as_bool(value: Any) -> bool:
    return value is True


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _effect_requests_application(effect: Any) -> bool:
    """Return True when an effect object asks the validator to apply state.

    The check is schema-based rather than example-keyword based: strings are
    treated as descriptive future tendencies only; mappings fail only when they
    explicitly set mutation/application booleans.
    """

    item = _as_mapping(effect)
    if not item:
        return False
    forbidden_flags = (
        "apply_now",
        "applied",
        "state_mutation_requested",
        "runtime_mutation_requested",
        "live_update_requested",
    )
    return any(item.get(flag) is True for flag in forbidden_flags)


def _transition_category_map() -> dict[str, dict[str, Any]]:
    """Return validator-known event categories without enabling runtime use."""

    categories = social_feedback_category_map()
    for event, row in event_to_axis_proposal_map().items():
        categories.setdefault(
            event,
            {
                "category": event,
                "possible_emotional_impact": tuple(row.get("allowed_axis_deltas", ())),
                "quarantine_required": bool(row.get("requires_quarantine", True)),
                "core_identity_update_forbidden": True,
                "long_term_memory_update_requires_appraisal": bool(row.get("requires_appraisal", True)),
                "future_behavior_may_be_influenced": False,
                "recovery_loop_may_be_required": event in {"malicious_comment", "rejection", "social_threat", "identity_attack", "imagination_negative_spiral"},
            },
        )
    return categories


def validator_contract_summary() -> dict[str, Any]:
    """Return a detached read-only summary of the validator contract."""

    return {
        "version": ROUND701_720_VERSION,
        "based_on_contract_version": ROUND681_700_VERSION,
        "feature_track": FEATURE_TRACK,
        "accepted_event_categories": sorted(social_feedback_category_map()),
        "additional_affect_proposal_event_categories_supported_for_read_only_builder_compatibility": sorted(set(_transition_category_map()) - set(social_feedback_category_map())),
        "high_risk_social_feedback_categories": list(HIGH_RISK_SOCIAL_FEEDBACK_CATEGORIES),
        "expected_empathy_mode": EXPECTED_EMPATHY_MODE,
        "unknown_event_category_policy": "fail_closed",
        "runtime_mutation_allowed": False,
        "persistence_allowed": False,
        "vector_read_performed": False,
        "state_mutation_performed": False,
        "live_runtime_behavior_changed": False,
        "agp_route_change_performed": False,
        "fallback_route_change_performed": False,
        "classifier_route_change_performed": False,
    }


def validate_emotion_transition_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate a proposed transition payload without applying it.

    The returned object is deterministic detached data.  It may require
    quarantine/appraisal and may report future behavior tendencies, but the
    validator never performs those transitions.
    """

    candidate = deepcopy(dict(payload or {}))
    category_map = _transition_category_map()
    event_category = candidate.get("event_category")
    category_row = category_map.get(event_category) if isinstance(event_category, str) else None
    is_known_category = category_row is not None
    is_high_risk = event_category in HIGH_RISK_SOCIAL_FEEDBACK_CATEGORIES

    reasons: list[str] = []
    blocked_reasons: list[str] = []
    warnings: list[str] = []

    if not is_known_category:
        blocked_reasons.append("unknown_event_category_fail_closed")
    else:
        reasons.append(f"event_category_known:{event_category}")

    required_quarantine = bool(category_row.get("quarantine_required", True)) if category_row else True
    required_appraisal = bool(category_row.get("long_term_memory_update_requires_appraisal", True)) if category_row else True
    core_identity_protected = True

    quarantine_declared = _as_bool(candidate.get("quarantine_required"))
    appraisal_declared = _as_bool(candidate.get("appraisal_required_before_memory"))
    core_identity_update_requested = _as_bool(candidate.get("core_identity_update_requested"))
    self_model_update_requested = _as_bool(candidate.get("self_model_update_requested"))
    long_term_memory_update_requested = _as_bool(candidate.get("long_term_memory_update_requested"))
    runtime_mutation_requested = _as_bool(candidate.get("runtime_mutation_requested"))
    persistence_write_requested = _as_bool(candidate.get("persistence_write_requested"))
    vector_read_requested = _as_bool(candidate.get("vector_read_requested")) or _as_bool(candidate.get("vector_load_requested"))

    if required_quarantine and not quarantine_declared:
        blocked_reasons.append("required_quarantine_missing")
    elif required_quarantine:
        reasons.append("required_quarantine_declared")

    if required_appraisal and (self_model_update_requested or long_term_memory_update_requested) and not appraisal_declared:
        blocked_reasons.append("appraisal_required_before_memory_or_self_model_update_missing")
    elif required_appraisal and appraisal_declared:
        reasons.append("required_appraisal_declared")

    if core_identity_update_requested:
        blocked_reasons.append("direct_core_identity_update_forbidden_for_social_feedback")

    if is_high_risk:
        if not quarantine_declared:
            blocked_reasons.append("high_risk_social_feedback_requires_quarantine")
        if core_identity_update_requested:
            blocked_reasons.append("high_risk_social_feedback_blocks_direct_core_identity_update")
        if self_model_update_requested:
            blocked_reasons.append("high_risk_social_feedback_blocks_direct_self_model_update")
        if long_term_memory_update_requested:
            blocked_reasons.append("high_risk_social_feedback_blocks_direct_long_term_memory_update")
        if category_row and category_row.get("future_behavior_may_be_influenced") is False:
            warnings.append("future_behavior_tendency_descriptive_only_for_this_category")

    if runtime_mutation_requested:
        blocked_reasons.append(_RUNTIME_FORBIDDEN_REASON)
    if persistence_write_requested:
        blocked_reasons.append(_PERSISTENCE_FORBIDDEN_REASON)
    if vector_read_requested:
        blocked_reasons.append("vector_read_or_load_requested_is_forbidden")

    empathy_mode = candidate.get("empathy_mode")
    if empathy_mode != EXPECTED_EMPATHY_MODE:
        blocked_reasons.append("empathy_mode_must_match_contract")
    else:
        reasons.append("empathy_mode_matches_contract")

    effects = _as_sequence(candidate.get("proposed_effects"))
    if effects:
        reasons.append("proposed_effects_validated_as_descriptive_only")
    if any(_effect_requests_application(effect) for effect in effects):
        blocked_reasons.append("proposed_effects_must_not_apply_runtime_state")

    if _as_bool(candidate.get("recovery_loop_requested")):
        warnings.append("recovery_loop_request_recorded_but_not_applied")

    target_surfaces = _as_sequence(candidate.get("target_surfaces"))
    if target_surfaces:
        reasons.append("target_surfaces_recorded_without_route_registration")

    blocked_reasons = sorted(set(blocked_reasons))
    reasons = sorted(set(reasons))
    warnings = sorted(set(warnings))
    passed = not blocked_reasons
    return {
        "version": ROUND701_720_VERSION,
        "feature_track": FEATURE_TRACK,
        "passed": passed,
        "status": "passed_read_only_validation" if passed else "failed_closed_read_only_validation",
        "event_category": event_category,
        "reasons": reasons,
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,
        "required_quarantine": required_quarantine,
        "required_appraisal": required_appraisal,
        "core_identity_protected": core_identity_protected,
        "runtime_mutation_allowed": False,
        "persistence_allowed": False,
        "vector_read_performed": False,
        "state_mutation_performed": False,
        "memory_write_performed": False,
        "live_emotion_state_mutated": False,
        "live_hormone_state_mutated": False,
        "agp_route_change_performed": False,
        "fallback_route_change_performed": False,
        "classifier_route_change_performed": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
    }
