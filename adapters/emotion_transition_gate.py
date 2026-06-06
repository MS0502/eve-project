"""Round721-740 read-only emotion transition gate.

This module wraps the Round701-720 emotion transition validator as an explicit
operator/test gate for future apply rounds.  It is intentionally pure and
read-only: it does not import live runtime adapters, perform file I/O, read or
load vector artifacts, mutate emotion/hormone/memory state, register routes, or
enable persistence/runtime mapping/enforcement.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.emotion_state_transition_contract import (
    EMPATHY_CONTRACT,
    MALICIOUS_FEEDBACK_QUARANTINE_RULE,
    ROUND681_700_VERSION,
)
from adapters.emotion_transition_validator import (
    ROUND701_720_VERSION,
    validate_emotion_transition_payload,
    validator_contract_summary,
)

ROUND721_740_VERSION = "v3_round721_740_emotion_transition_gate"
FEATURE_TRACK = "integrate_read_only_emotion_transition_validator_as_explicit_operator_test_gate_before_future_apply_round"
EXPECTED_EMPATHY_MODE = EMPATHY_CONTRACT["definition"]

_GATE_BLOCKED_STATUS = "blocked_read_only_gate_failed_closed"
_GATE_PASSED_STATUS = "passed_read_only_gate_future_apply_still_blocked"

READ_ONLY_GATE_INERTNESS: dict[str, bool] = {
    "apply_allowed": False,
    "runtime_mutation_allowed": False,
    "persistence_allowed": False,
    "vector_read_performed": False,
    "vector_load_performed": False,
    "state_mutation_performed": False,
    "memory_write_performed": False,
    "live_emotion_state_mutated": False,
    "live_hormone_state_mutated": False,
    "agp_route_changed": False,
    "fallback_route_changed": False,
    "classifier_route_changed": False,
    "runtime_mapping_enabled": False,
    "enforcement_enabled": False,
    "production_persistence_enabled": False,
    "future_apply_requires_explicit_operator_authorization": True,
}

FUTURE_APPLY_POLICY: dict[str, Any] = {
    "version": ROUND721_740_VERSION,
    "policy_name": "future_emotion_transition_apply_round_gate_policy",
    "current_round_applies_transitions": False,
    "future_emotion_apply_rounds_must_first_pass_gate": True,
    "gate_passing_automatically_allows_persistence": False,
    "gate_passing_automatically_allows_runtime_mapping": False,
    "gate_passing_automatically_allows_enforcement": False,
    "live_mutation_requires_separate_explicit_operator_authorized_round": True,
    "malicious_social_threat_identity_attack_direct_rewrites_blocked": True,
    "blocked_direct_rewrite_surfaces": (
        "core_identity",
        "self_model",
        "long_term_memory",
    ),
    "high_risk_social_feedback_categories": tuple(MALICIOUS_FEEDBACK_QUARANTINE_RULE["applies_to"]),
    "production_persistence_remains_no_go": True,
    "runtime_mapping_enabled_default": False,
    "enforcement_enabled_default": False,
    "default_runtime_remains_no_load": True,
    "vector_contents_read": False,
    "vectors_loaded": False,
}


def gate_required_for_future_apply_round() -> dict[str, Any]:
    """Return the read-only policy that blocks future apply rounds without the gate."""

    return deepcopy(FUTURE_APPLY_POLICY)


def _merge_blocked_reasons(validator_result: Mapping[str, Any], gate_reasons: list[str]) -> list[str]:
    validator_reasons = validator_result.get("blocked_reasons", [])
    merged = [str(reason) for reason in validator_reasons]
    merged.extend(gate_reasons)
    return sorted(set(merged))


def build_emotion_transition_gate_report(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build a deterministic read-only gate report for a proposed payload.

    The gate delegates payload safety checks to the Round701-720 validator and
    then adds an explicit apply-round boundary: passing the gate is necessary
    for a future apply round, but it is not sufficient to mutate runtime state
    or enable persistence/runtime mapping/enforcement.
    """

    candidate = deepcopy(dict(payload or {}))
    validator_result = validate_emotion_transition_payload(candidate)
    validator_passed = validator_result.get("passed") is True

    gate_blocked_reasons: list[str] = []
    if validator_result.get("vector_read_performed") is not False:
        gate_blocked_reasons.append("validator_reported_vector_read_performed")
    if validator_result.get("state_mutation_performed") is not False:
        gate_blocked_reasons.append("validator_reported_state_mutation_performed")
    if validator_result.get("memory_write_performed") is not False:
        gate_blocked_reasons.append("validator_reported_memory_write_performed")
    if validator_result.get("agp_route_change_performed") is not False:
        gate_blocked_reasons.append("validator_reported_agp_route_change_performed")
    if validator_result.get("fallback_route_change_performed") is not False:
        gate_blocked_reasons.append("validator_reported_fallback_route_change_performed")
    if validator_result.get("classifier_route_change_performed") is not False:
        gate_blocked_reasons.append("validator_reported_classifier_route_change_performed")
    if validator_result.get("runtime_mutation_allowed") is not False:
        gate_blocked_reasons.append("validator_reported_runtime_mutation_allowed")
    if validator_result.get("persistence_allowed") is not False:
        gate_blocked_reasons.append("validator_reported_persistence_allowed")

    blocked_reasons = _merge_blocked_reasons(validator_result, gate_blocked_reasons)
    gate_passed = validator_passed and not gate_blocked_reasons
    warnings = sorted(
        set(str(warning) for warning in validator_result.get("warnings", []))
        | {"gate_pass_does_not_authorize_apply_persistence_runtime_mapping_or_enforcement"}
    )

    return {
        "version": ROUND721_740_VERSION,
        "feature_track": FEATURE_TRACK,
        "based_on_contract_version": ROUND681_700_VERSION,
        "wrapped_validator_version": ROUND701_720_VERSION,
        "gate_passed": gate_passed,
        "gate_status": _GATE_PASSED_STATUS if gate_passed else _GATE_BLOCKED_STATUS,
        "validator_passed": validator_passed,
        "fail_closed": not gate_passed,
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,
        "validator_result": validator_result,
        "validator_contract_summary": validator_contract_summary(),
        "future_apply_policy": gate_required_for_future_apply_round(),
        **READ_ONLY_GATE_INERTNESS,
    }


def validate_emotion_transition_gate(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate a proposed payload through the explicit read-only gate."""

    return build_emotion_transition_gate_report(payload)
