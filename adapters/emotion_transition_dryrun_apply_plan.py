"""Round741-760 operator-authorized dry-run emotion transition apply plan.

This module consumes the Round721-740 read-only emotion transition gate and
simulates the preflight checks a future live apply round would need.  It is a
pure-data planning surface only: it never applies transitions, mutates live
emotion/hormone/memory state, writes persistence, changes runtime mapping or
enforcement, reads/loads vectors, creates artifacts, or registers runtime
routes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.emotion_transition_gate import (
    ROUND721_740_VERSION,
    build_emotion_transition_gate_report,
    gate_required_for_future_apply_round,
)

ROUND741_760_VERSION = "v3_round741_760_emotion_transition_dryrun_apply_plan"
FEATURE_TRACK = "design_operator_authorized_dry_run_emotion_transition_apply_plan_without_mutating_live_state"

_DRYRUN_SUCCESS_STATUS = "dryrun_apply_preflight_simulated_success_live_apply_still_blocked"
_DRYRUN_UNAUTHORIZED_STATUS = "blocked_dryrun_apply_requires_explicit_operator_authorization"
_DRYRUN_GATE_FAILED_STATUS = "blocked_dryrun_apply_requires_gate_pass"

_REQUIRED_PRECONDITIONS: tuple[str, ...] = (
    "round721_740_gate_must_pass",
    "explicit_operator_authorization_must_be_true_for_dryrun",
    "future_live_apply_requires_separate_operator_authorized_round",
    "dryrun_success_does_not_authorize_live_mutation",
    "dryrun_success_does_not_authorize_persistence",
    "dryrun_success_does_not_authorize_runtime_mapping",
    "dryrun_success_does_not_authorize_enforcement",
    "dryrun_success_does_not_authorize_memory_writes",
    "dryrun_success_does_not_authorize_vector_read_or_load",
)

_DRYRUN_INERTNESS: dict[str, bool] = {
    "apply_performed": False,
    "live_emotion_mutated": False,
    "live_hormone_mutated": False,
    "memory_written": False,
    "persistence_written": False,
    "runtime_mapping_changed": False,
    "enforcement_changed": False,
    "vector_read_performed": False,
    "vector_load_performed": False,
    "artifact_created_or_staged": False,
    "agp_route_changed": False,
    "fallback_route_changed": False,
    "classifier_route_changed": False,
}


def dryrun_apply_requires_gate_pass() -> bool:
    """Return True because dry-run preflight must fail closed unless the gate passes."""

    return True


def dryrun_apply_requires_explicit_operator_authorization() -> bool:
    """Return True because even dry-run apply simulation requires operator authorization."""

    return True


def future_live_apply_policy() -> dict[str, Any]:
    """Return the Round741-760 policy boundary for any future live apply round."""

    gate_policy = gate_required_for_future_apply_round()
    return {
        "version": ROUND741_760_VERSION,
        "based_on_gate_version": ROUND721_740_VERSION,
        "policy_name": "operator_authorized_dryrun_emotion_transition_apply_plan_policy",
        "this_round_applies_transitions": False,
        "future_live_apply_requires_separate_explicit_operator_authorized_round": True,
        "dryrun_success_authorizes_live_mutation": False,
        "dryrun_success_authorizes_persistence": False,
        "dryrun_success_authorizes_runtime_mapping": False,
        "dryrun_success_authorizes_enforcement": False,
        "dryrun_success_authorizes_memory_writes": False,
        "dryrun_success_authorizes_vector_read_or_load": False,
        "malicious_social_threat_identity_attack_direct_rewrites_blocked": True,
        "blocked_direct_rewrite_surfaces": tuple(gate_policy["blocked_direct_rewrite_surfaces"]),
        "production_persistence_remains_no_go": True,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "default_runtime_remains_no_load": True,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "gate_policy": gate_policy,
    }


def _status(operator_authorized: bool, gate_passed: bool) -> str:
    if not operator_authorized:
        return _DRYRUN_UNAUTHORIZED_STATUS
    if not gate_passed:
        return _DRYRUN_GATE_FAILED_STATUS
    return _DRYRUN_SUCCESS_STATUS


def _blocked_reasons(operator_authorized: bool, gate_result: Mapping[str, Any]) -> list[str]:
    reasons = [str(reason) for reason in gate_result.get("blocked_reasons", [])]
    if not operator_authorized:
        reasons.append("explicit_operator_authorization_missing")
    if gate_result.get("gate_passed") is not True:
        reasons.append("round721_740_gate_did_not_pass")
    return sorted(set(reasons))


def _warnings(gate_result: Mapping[str, Any]) -> list[str]:
    warnings = {str(warning) for warning in gate_result.get("warnings", [])}
    warnings.update(
        {
            "dryrun_apply_plan_does_not_apply_transitions",
            "dryrun_success_does_not_authorize_live_mutation",
            "dryrun_success_does_not_authorize_persistence_runtime_mapping_enforcement_memory_or_vectors",
        }
    )
    return sorted(warnings)


def build_emotion_transition_dryrun_apply_plan(
    payload: Mapping[str, Any] | None,
    operator_authorized: bool = False,
) -> dict[str, Any]:
    """Build a detached dry-run apply plan without mutating any live state.

    The plan consumes the read-only Round721-740 gate.  Missing operator
    authorization or a failed gate both fail closed.  When both are present, the
    function reports only simulated preflight success; ``apply_performed`` and
    all mutation/write/load/route flags remain ``False``.
    """

    candidate = deepcopy(dict(payload or {}))
    authorized = operator_authorized is True
    gate_result = build_emotion_transition_gate_report(candidate)
    gate_passed = gate_result.get("gate_passed") is True
    dryrun_success = authorized and gate_passed
    blocked_reasons = _blocked_reasons(authorized, gate_result)

    return {
        "version": ROUND741_760_VERSION,
        "feature_track": FEATURE_TRACK,
        "based_on_gate_version": ROUND721_740_VERSION,
        "dryrun_success": dryrun_success,
        "dryrun_status": _status(authorized, gate_passed),
        "operator_authorized": authorized,
        "gate_passed": gate_passed,
        "gate_status": str(gate_result.get("gate_status", "unknown_gate_status")),
        "apply_simulated": dryrun_success,
        **_DRYRUN_INERTNESS,
        "required_preconditions": list(_REQUIRED_PRECONDITIONS),
        "blocked_reasons": blocked_reasons,
        "warnings": _warnings(gate_result),
        "gate_result": gate_result,
        "future_live_apply_policy": future_live_apply_policy(),
    }


def simulate_emotion_transition_apply_preflight(
    payload: Mapping[str, Any] | None,
    operator_authorized: bool = False,
) -> dict[str, Any]:
    """Alias for the dry-run apply plan preflight simulation."""

    return build_emotion_transition_dryrun_apply_plan(payload, operator_authorized=operator_authorized)
