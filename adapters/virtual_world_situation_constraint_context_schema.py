"""Deterministic read-only constraint-context candidates for virtual situations."""

import hashlib
import json
import math
from typing import Any, Dict, Optional

VERSION = "1.0.0-round1381-1400"
CANONICAL_ID_ALGORITHM = "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256"
SUPPORTED_CONSTRAINT_TYPES = (
    "hard_constraint_candidate", "soft_constraint_candidate", "precondition_constraint_candidate",
    "postcondition_constraint_candidate", "invariant_constraint_candidate", "resource_constraint_candidate",
    "temporal_constraint_candidate", "causal_constraint_candidate", "role_relation_constraint_candidate",
    "uncertainty_guard_constraint_candidate", "boundary_constraint_candidate", "exclusion_constraint_candidate",
    "dependency_constraint_candidate", "ordering_constraint_candidate", "symbolic_constraint_candidate",
    "conflicting_constraint_candidate", "mixed_unknown_constraint_candidate",
)
SUPPORTED_CLAUSE_KINDS = (
    "must_hold_candidate", "should_hold_candidate", "must_not_hold_candidate", "requires_candidate",
    "excludes_candidate", "before_candidate", "after_candidate", "depends_on_candidate",
    "resource_limit_candidate", "boundary_limit_candidate", "uncertainty_guard_candidate",
    "causal_guard_candidate", "role_relation_guard_candidate", "symbolic_guard_candidate",
    "unknown_constraint_candidate",
)
RELATIONAL_CLAUSE_KINDS = {"requires_candidate", "excludes_candidate", "before_candidate", "after_candidate", "depends_on_candidate"}
TYPE_CLAUSE_COMPATIBILITY = {
    "hard_constraint_candidate": {"must_hold_candidate", "must_not_hold_candidate", "requires_candidate", "excludes_candidate"},
    "soft_constraint_candidate": {"should_hold_candidate"},
    "precondition_constraint_candidate": {"requires_candidate"},
    "postcondition_constraint_candidate": {"must_hold_candidate", "should_hold_candidate"},
    "invariant_constraint_candidate": {"must_hold_candidate", "must_not_hold_candidate"},
    "resource_constraint_candidate": {"resource_limit_candidate"},
    "temporal_constraint_candidate": {"before_candidate", "after_candidate"},
    "causal_constraint_candidate": {"causal_guard_candidate"},
    "role_relation_constraint_candidate": {"role_relation_guard_candidate"},
    "uncertainty_guard_constraint_candidate": {"uncertainty_guard_candidate"},
    "boundary_constraint_candidate": {"boundary_limit_candidate"},
    "exclusion_constraint_candidate": {"excludes_candidate"},
    "dependency_constraint_candidate": {"depends_on_candidate"},
    "ordering_constraint_candidate": {"before_candidate", "after_candidate"},
    "symbolic_constraint_candidate": {"symbolic_guard_candidate"},
    "conflicting_constraint_candidate": {"must_hold_candidate", "must_not_hold_candidate", "requires_candidate", "excludes_candidate", "before_candidate", "after_candidate"},
    "mixed_unknown_constraint_candidate": set(SUPPORTED_CLAUSE_KINDS),
}
SUPPORTED_BOUNDARIES = (
    "internal_logical_constraint", "internal_soft_constraint", "internal_precondition_constraint",
    "internal_postcondition_constraint", "internal_invariant_constraint", "internal_resource_constraint",
    "temporal_virtual_constraint", "causal_virtual_constraint", "role_relation_virtual_constraint",
    "uncertainty_virtual_constraint", "boundary_virtual_constraint", "symbolic_virtual_constraint",
    "mixed_virtual_external_constraint_boundary",
)
BOUNDARY_DEFAULTS = {
    "soft_constraint_candidate": "internal_soft_constraint",
    "precondition_constraint_candidate": "internal_precondition_constraint",
    "postcondition_constraint_candidate": "internal_postcondition_constraint",
    "invariant_constraint_candidate": "internal_invariant_constraint",
    "resource_constraint_candidate": "internal_resource_constraint",
    "temporal_constraint_candidate": "temporal_virtual_constraint",
    "ordering_constraint_candidate": "temporal_virtual_constraint",
    "causal_constraint_candidate": "causal_virtual_constraint",
    "role_relation_constraint_candidate": "role_relation_virtual_constraint",
    "uncertainty_guard_constraint_candidate": "uncertainty_virtual_constraint",
    "boundary_constraint_candidate": "boundary_virtual_constraint",
    "symbolic_constraint_candidate": "symbolic_virtual_constraint",
    "mixed_unknown_constraint_candidate": "mixed_virtual_external_constraint_boundary",
}
SUPPORTED_CONFIDENCE = (
    "constraint_unverified", "constraint_low_confidence", "constraint_medium_confidence",
    "constraint_high_confidence_but_not_enforced", "constraint_conflict_detected", "constraint_origin_unknown",
)
IMMUTABLE_FALSE_FLAGS = (
    "external_constraint_asserted", "constraint_fact_asserted", "constraint_satisfied_asserted",
    "constraint_violated_asserted", "constraint_enforced", "automatic_enforcement_performed",
    "automatic_enforcement_allowed", "hard_block_applied", "soft_penalty_applied", "precondition_evaluated",
    "postcondition_evaluated", "invariant_evaluated", "resource_reserved", "resource_consumed",
    "schedule_created", "action_prevented", "action_permitted", "event_completion_asserted",
    "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed",
    "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed",
    "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied",
    "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed",
    "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded",
    "device_activation_performed", "network_action_performed", "artifact_created_or_staged",
    "agp_bypass_allowed", "fallback_bypass_allowed",
)
IMMUTABLE_TRUE_FLAGS = (
    "constraint_candidate_only", "situation_constraint_only", "read_only", "situation_review_required",
    "snapshot_review_required", "transition_review_required", "memory_gate_required", "quarantine_required",
    "origin_fact_status_required", "constraint_integrity_review_required", "enforcement_review_required",
    "appraisal_required", "agp_input_required",
)
FORBIDDEN_REQUEST_FIELDS = {
    "external_constraint_assertion_requested", "constraint_fact_assertion_requested", "constraint_satisfaction_assertion_requested",
    "constraint_violation_assertion_requested", "constraint_enforcement_requested", "automatic_enforcement_requested",
    "hard_block_requested", "soft_penalty_requested", "precondition_evaluation_requested", "postcondition_evaluation_requested",
    "invariant_evaluation_requested", "resource_reservation_requested", "resource_consumption_requested",
    "action_prevention_requested", "action_permission_requested", "conflict_resolution_requested", "winner_selection_requested",
    "certainty_promotion_requested", "fact_promotion_requested", "memory_write_requested", "memory_fact_promotion_requested",
    "relationship_update_requested", "self_model_update_requested", "identity_update_requested", "affect_transition_requested",
    "hormone_transition_requested", "world_state_mutation_requested", "transition_application_requested",
    "planning_execution_requested", "tool_execution_requested", "runtime_mutation_requested", "persistence_write_requested",
    "vector_read_requested", "vector_load_requested", "model_load_requested", "device_activation_requested",
    "network_action_requested", "artifact_creation_requested", "intervention_requested", "event_completion_assertion_requested",
    "agp_bypass_requested", "fallback_bypass_requested", "schedule_requested", "scheduler_requested", "timer_requested",
    "alarm_requested", "calendar_requested", "reminder_requested", "deadline_requested", "cron_requested",
    "cron_expression", "scheduled_execution_requested", "delayed_execution_requested", "act_requested", "execute_requested",
    "apply_requested", "commit_outcome_requested", "promote_to_fact_requested",
}
TOP_KEYS = {"situation_constraint_context_passed", "situation_constraint_context_status", "constraint_context_id", "canonical_id_algorithm", "constraint_type", "constraint_boundary_classification", "constraint_confidence_state", "situation_id", "constraint_clauses", "metadata", "origin_summary", "fact_status_summary", "constraint_scope_summary", "constraint_conflict_summary", "enforcement_summary", "constraint_flags", "boundary_flags", "constraint_integrity_flags", "candidate_only_fields", "blocked_reasons", "warnings", *IMMUTABLE_FALSE_FLAGS, *IMMUTABLE_TRUE_FLAGS}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _assert_json_native(value: Any, seen: Optional[set] = None, depth: int = 0) -> Optional[str]:
    try:
        if depth > 100:
            return "non_json_serializable_semantic_input"
        if value is None or type(value) in {str, bool}:
            return None
        if type(value) is int:
            return None
        if type(value) is float:
            return None if math.isfinite(value) else "non_json_serializable_semantic_input"
        if seen is None:
            seen = set()
        if type(value) is dict:
            oid = id(value)
            if oid in seen:
                return "non_json_serializable_semantic_input"
            seen.add(oid)
            keys = list(value.keys())
            if not all(type(key) is str for key in keys):
                return "non_json_serializable_semantic_input"
            for key in sorted(keys):
                reason = _assert_json_native(value[key], seen, depth + 1)
                if reason:
                    return reason
            seen.remove(oid)
            return None
        if type(value) is list:
            oid = id(value)
            if oid in seen:
                return "non_json_serializable_semantic_input"
            seen.add(oid)
            for item in value:
                reason = _assert_json_native(item, seen, depth + 1)
                if reason:
                    return reason
            seen.remove(oid)
            return None
        return "non_json_serializable_semantic_input"
    except Exception:
        return "non_json_serializable_semantic_input"


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def _type_exact_equal(expected: Any, actual: Any) -> bool:
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return set(expected.keys()) == set(actual.keys()) and all(_type_exact_equal(expected[k], actual[k]) for k in expected)
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(_type_exact_equal(a, b) for a, b in zip(expected, actual))
    return expected == actual


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _forbidden_reason(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            item = value[key]
            if key in FORBIDDEN_REQUEST_FIELDS:
                if item is True:
                    return key
                if item is not False:
                    return "malformed_forbidden_request_field"
            reason = _forbidden_reason(item)
            if reason:
                return reason
    elif isinstance(value, list):
        for item in sorted(value, key=_canonical_sort_key):
            reason = _forbidden_reason(item)
            if reason:
                return reason
    return None


def _base_payload() -> Dict[str, Any]:
    payload = {key: False for key in IMMUTABLE_FALSE_FLAGS}
    payload.update({key: True for key in IMMUTABLE_TRUE_FLAGS})
    return payload


def _reject(reason: str) -> Dict[str, Any]:
    payload = _base_payload()
    payload.update({
        "situation_constraint_context_passed": False, "situation_constraint_context_status": "REJECTED",
        "constraint_context_id": None, "canonical_id_algorithm": None, "constraint_type": None,
        "constraint_boundary_classification": None, "constraint_confidence_state": None, "situation_id": None,
        "constraint_clauses": [], "metadata": {},
        "origin_summary": {"origin_kind": "internal_virtual_constraint_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "external_constraint_fact_verified": False, "constraint_fact_asserted": False, "constraint_satisfied_asserted": False, "constraint_violated_asserted": False},
        "constraint_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "clause_kinds": [], "unary_clause_count": 0, "relational_clause_count": 0},
        "constraint_conflict_summary": {"conflict_present": False, "conflict_pairs": [], "winner_selected": False, "conflict_resolved": False, "automatic_resolution_allowed": False},
        "enforcement_summary": {"enforcement_candidate_only": True, "enforced": False, "automatic_enforcement_allowed": False, "action_blocked": False, "action_permitted": False},
        "constraint_flags": [], "boundary_flags": [], "constraint_integrity_flags": [],
        "candidate_only_fields": ["constraint_type", "constraint_clauses", "constraint_boundary_classification", "constraint_confidence_state"],
        "blocked_reasons": [reason], "warnings": [],
    })
    return payload


def _normalize_clauses(clauses: Any, situation_id: str):
    if clauses is None:
        return None, "missing_constraint_clauses"
    if not isinstance(clauses, list):
        return None, "constraint_clauses_not_non_empty_list"
    if not clauses:
        return None, "empty_constraint_clauses"
    ordered = sorted(clauses, key=_canonical_sort_key)
    normalized = []
    for clause in ordered:
        if not isinstance(clause, dict):
            return None, "malformed_constraint_clause"
        for req in ("clause_id", "situation_id", "clause_kind", "subject_ref_id"):
            if req not in clause:
                if req == "clause_id":
                    return None, "missing_or_empty_clause_id"
                if req == "situation_id":
                    return None, "malformed_clause_situation_id"
                if req == "clause_kind":
                    return None, "unknown_clause_kind"
                return None, "missing_or_malformed_subject_ref_id"
        if not _non_empty_string(clause["clause_id"]):
            return None, "missing_or_empty_clause_id"
        if not _non_empty_string(clause["situation_id"]):
            return None, "malformed_clause_situation_id"
        if clause["clause_kind"] not in SUPPORTED_CLAUSE_KINDS:
            return None, "unknown_clause_kind"
        if not _non_empty_string(clause["subject_ref_id"]):
            return None, "missing_or_malformed_subject_ref_id"
        if "strength_candidate" in clause:
            value = clause["strength_candidate"]
            if type(value) is bool:
                return None, "malformed_strength_candidate"
            if type(value) is int:
                if value < 0 or value > 1:
                    return None, "strength_candidate_out_of_range"
            elif type(value) is float:
                if not math.isfinite(value):
                    return None, "malformed_strength_candidate"
                if value < 0.0 or value > 1.0:
                    return None, "strength_candidate_out_of_range"
            else:
                return None, "malformed_strength_candidate"
        if "priority_candidate" in clause:
            value = clause["priority_candidate"]
            if isinstance(value, bool) or not isinstance(value, int):
                return None, "malformed_priority_candidate"
            if value < 0 or value > 100:
                return None, "priority_candidate_out_of_range"
        if "label" in clause and not _non_empty_string(clause["label"]):
            return None, "malformed_constraint_clause_label"
        normalized.append(_json_clone(clause))
    reason = _forbidden_reason(normalized)
    if reason:
        return None, reason
    ids = [c["clause_id"] for c in normalized]
    if len(ids) != len(set(ids)):
        return None, "duplicate_clause_id"
    seen_sem = set()
    for c in normalized:
        sem = (c["situation_id"], c["clause_kind"], c["subject_ref_id"], c.get("object_ref_id"))
        if sem in seen_sem:
            return None, "duplicate_semantic_constraint_clause"
        seen_sem.add(sem)
    if any(c["situation_id"] != situation_id for c in normalized):
        return None, "constraint_clause_situation_mismatch"
    for c in normalized:
        relational = c["clause_kind"] in RELATIONAL_CLAUSE_KINDS
        if relational:
            if "object_ref_id" not in c:
                return None, "missing_object_ref_id_for_relational_clause"
            if not _non_empty_string(c["object_ref_id"]):
                return None, "malformed_object_ref_id"
            if c["subject_ref_id"] == c["object_ref_id"]:
                return None, "identical_constraint_clause_refs"
        elif "object_ref_id" in c:
            return None, "unexpected_object_ref_id_for_unary_clause"
    return sorted(normalized, key=_canonical_sort_key), None


def _cardinality_reason(constraint_type: str, clauses: list) -> Optional[str]:
    kinds = [c["clause_kind"] for c in clauses]
    kind_set = set(kinds)
    if any(k not in TYPE_CLAUSE_COMPATIBILITY[constraint_type] for k in kinds):
        return "incompatible_constraint_type_clause_kind"
    requirements = {
        "soft_constraint_candidate": {"should_hold_candidate"},
        "precondition_constraint_candidate": {"requires_candidate"},
        "postcondition_constraint_candidate": {"must_hold_candidate", "should_hold_candidate"},
        "invariant_constraint_candidate": {"must_hold_candidate", "must_not_hold_candidate"},
        "resource_constraint_candidate": {"resource_limit_candidate"},
        "temporal_constraint_candidate": {"before_candidate", "after_candidate"},
        "causal_constraint_candidate": {"causal_guard_candidate"},
        "role_relation_constraint_candidate": {"role_relation_guard_candidate"},
        "uncertainty_guard_constraint_candidate": {"uncertainty_guard_candidate"},
        "boundary_constraint_candidate": {"boundary_limit_candidate"},
        "exclusion_constraint_candidate": {"excludes_candidate"},
        "dependency_constraint_candidate": {"depends_on_candidate"},
        "ordering_constraint_candidate": {"before_candidate", "after_candidate"},
        "symbolic_constraint_candidate": {"symbolic_guard_candidate"},
    }
    if constraint_type in requirements and not (kind_set & requirements[constraint_type]):
        return f"{constraint_type.replace('_candidate','')}_requires_compatible_clause"
    if constraint_type == "conflicting_constraint_candidate" and len(clauses) < 2:
        return "conflicting_constraint_requires_multiple_clauses"
    if constraint_type == "mixed_unknown_constraint_candidate":
        if len(clauses) < 2:
            return "mixed_unknown_constraint_requires_multiple_clauses"
        if len(kind_set) < 2:
            return "mixed_unknown_constraint_requires_distinct_clause_kinds"
    return None


def _detect_conflicts(clauses: list) -> list:
    pairs = []
    by_sem = {(c["clause_kind"], c["subject_ref_id"], c.get("object_ref_id")): c for c in clauses}
    checks = [
        ("must_hold_candidate", "must_not_hold_candidate", "must_hold_vs_must_not_hold"),
        ("requires_candidate", "excludes_candidate", "requires_vs_excludes"),
        ("before_candidate", "after_candidate", "before_vs_after"),
    ]
    for left, right, kind in checks:
        for c in clauses:
            if c["clause_kind"] != left:
                continue
            other = by_sem.get((right, c["subject_ref_id"], c.get("object_ref_id")))
            if other:
                ids = sorted([c["clause_id"], other["clause_id"]])
                pairs.append({"left_clause_id": ids[0], "right_clause_id": ids[1], "conflict_kind": kind})
    unique = {(_canonical_sort_key(p)): p for p in pairs}
    return [unique[k] for k in sorted(unique)]


def _composition_reason(constraint_type: str, conflict_pairs: list) -> Optional[str]:
    if constraint_type == "conflicting_constraint_candidate" and not conflict_pairs:
        return "conflicting_constraint_requires_detected_conflict"
    if constraint_type not in {"conflicting_constraint_candidate", "mixed_unknown_constraint_candidate"} and conflict_pairs:
        return "constraint_conflict_requires_conflicting_type"
    return None


def _validate_boundary(metadata: dict, constraint_type: str):
    if "constraint_boundary_classification" in metadata:
        value = metadata["constraint_boundary_classification"]
        if not _non_empty_string(value):
            return None, "malformed_constraint_boundary_class"
    else:
        value = BOUNDARY_DEFAULTS.get(constraint_type, "internal_logical_constraint")
    if value not in SUPPORTED_BOUNDARIES:
        return None, "unknown_constraint_boundary_class"
    return value, None


def _validate_confidence(metadata: dict):
    if "constraint_confidence_state" in metadata:
        value = metadata["constraint_confidence_state"]
        if not _non_empty_string(value):
            return None, "malformed_constraint_confidence_state"
    else:
        value = "constraint_unverified"
    if value not in SUPPORTED_CONFIDENCE:
        return None, "unknown_constraint_confidence_state"
    return value, None


def _build_valid(constraint_type: str, situation_id: str, clauses: list, metadata: dict, boundary: str, confidence: str, conflict_pairs: list) -> Dict[str, Any]:
    kinds = sorted(set(c["clause_kind"] for c in clauses))
    unary_count = sum(1 for c in clauses if c["clause_kind"] not in RELATIONAL_CLAUSE_KINDS)
    relational_count = len(clauses) - unary_count
    flags = ["candidate_only", "constraint_unenforced"]
    integrity = ["constraint_integrity_review_required", "enforcement_review_required"]
    warnings = []
    if conflict_pairs or confidence == "constraint_conflict_detected":
        flags.append("constraint_conflict_review_required")
        integrity.append("conflict_review_required")
    if confidence == "constraint_origin_unknown":
        flags.append("origin_unknown")
        integrity.append("origin_unknown_review_required")
    if constraint_type == "mixed_unknown_constraint_candidate":
        warnings.append("mixed_virtual_external_constraint_boundary_review_required")
    basis = {"schema_version": VERSION, "constraint_type": constraint_type, "constraint_boundary_classification": boundary, "constraint_confidence_state": confidence, "situation_id": situation_id, "constraint_clauses": clauses, "metadata": metadata}
    blob = json.dumps(basis, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    cid = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    payload = _base_payload()
    payload.update({
        "situation_constraint_context_passed": True, "situation_constraint_context_status": "VALIDATED",
        "constraint_context_id": cid, "canonical_id_algorithm": CANONICAL_ID_ALGORITHM,
        "constraint_type": constraint_type, "constraint_boundary_classification": boundary, "constraint_confidence_state": confidence,
        "situation_id": situation_id, "constraint_clauses": clauses, "metadata": metadata,
        "origin_summary": {"origin_kind": "internal_virtual_constraint_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "external_constraint_fact_verified": False, "constraint_fact_asserted": False, "constraint_satisfied_asserted": False, "constraint_violated_asserted": False},
        "constraint_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "clause_kinds": kinds, "unary_clause_count": unary_count, "relational_clause_count": relational_count},
        "constraint_conflict_summary": {"conflict_present": bool(conflict_pairs), "conflict_pairs": conflict_pairs, "winner_selected": False, "conflict_resolved": False, "automatic_resolution_allowed": False},
        "enforcement_summary": {"enforcement_candidate_only": True, "enforced": False, "automatic_enforcement_allowed": False, "action_blocked": False, "action_permitted": False},
        "constraint_flags": sorted(set(flags)), "boundary_flags": [boundary], "constraint_integrity_flags": sorted(set(integrity)),
        "candidate_only_fields": ["constraint_type", "constraint_clauses", "constraint_boundary_classification", "constraint_confidence_state"],
        "blocked_reasons": [], "warnings": warnings,
    })
    return payload


def build_virtual_world_situation_constraint_context(constraint_type=None, situation_id=None, constraint_clauses=None, metadata=None):
    raw = {"constraint_type": constraint_type, "situation_id": situation_id, "constraint_clauses": constraint_clauses, "metadata": metadata if metadata is not None else {}}
    try:
        if _assert_json_native(raw):
            return _reject("non_json_serializable_semantic_input")
        if constraint_type is None or not _non_empty_string(constraint_type):
            return _reject("missing_constraint_type")
        if constraint_type not in SUPPORTED_CONSTRAINT_TYPES:
            return _reject("unknown_constraint_type")
        if not _non_empty_string(situation_id):
            return _reject("missing_or_malformed_situation_id")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            return _reject("invalid_metadata")
        metadata = _json_clone(metadata)
        reason = _forbidden_reason(metadata)
        if reason:
            return _reject(reason)
        clauses, reason = _normalize_clauses(constraint_clauses, situation_id)
        if reason:
            return _reject(reason)
        reason = _cardinality_reason(constraint_type, clauses)
        if reason:
            return _reject(reason)
        conflicts = _detect_conflicts(clauses)
        reason = _composition_reason(constraint_type, conflicts)
        if reason:
            return _reject(reason)
        boundary, reason = _validate_boundary(metadata, constraint_type)
        if reason:
            return _reject(reason)
        confidence, reason = _validate_confidence(metadata)
        if reason:
            return _reject(reason)
        return _build_valid(constraint_type, situation_id, clauses, metadata, boundary, confidence, conflicts)
    except Exception:
        return _reject("non_json_serializable_semantic_input")


def validate_virtual_world_situation_constraint_context(constraint_context) -> bool:
    try:
        if _assert_json_native(constraint_context) or not isinstance(constraint_context, dict):
            return False
        if set(constraint_context.keys()) != TOP_KEYS:
            return False
        if constraint_context.get("situation_constraint_context_passed") is not True or constraint_context.get("situation_constraint_context_status") != "VALIDATED":
            return False
        expected = build_virtual_world_situation_constraint_context(
            constraint_context.get("constraint_type"), constraint_context.get("situation_id"),
            constraint_context.get("constraint_clauses"), constraint_context.get("metadata"),
        )
        return _type_exact_equal(expected, constraint_context)
    except Exception:
        return False


PLAN_FALSE_FLAGS = (
    "external_constraint_asserted", "constraint_fact_asserted", "constraint_satisfied_asserted", "constraint_violated_asserted",
    "constraint_enforced", "automatic_enforcement_performed", "hard_block_applied", "soft_penalty_applied",
    "precondition_evaluated", "postcondition_evaluated", "invariant_evaluated", "resource_reserved", "resource_consumed",
    "schedule_created", "action_prevented", "action_permitted", "memory_write_performed", "relationship_update_performed",
    "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed", "world_state_mutation_performed",
    "transition_applied", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded",
    "device_activation_performed", "network_action_performed", "planning_execution_performed", "tool_execution_performed",
    "agp_bypass_allowed", "fallback_bypass_allowed",
)


def _plan(plan_type: str, source: Any) -> Dict[str, Any]:
    ready = validate_virtual_world_situation_constraint_context(source)
    plan = {"plan_type": plan_type, "source_constraint_context_id": source.get("constraint_context_id") if isinstance(source, dict) and ready else None, "ready": ready, "candidate_only": True, "read_only": True, "constraint_candidate_only": True}
    for key in PLAN_FALSE_FLAGS:
        plan[key] = False
    return plan


def build_constraint_context_to_situation_plan(constraint_context): return _plan("constraint_context_to_situation", constraint_context)
def build_constraint_context_to_snapshot_plan(constraint_context): return _plan("constraint_context_to_snapshot", constraint_context)
def build_constraint_context_to_transition_preflight_plan(constraint_context): return _plan("constraint_context_to_transition_preflight", constraint_context)
def build_constraint_context_to_memory_candidate_plan(constraint_context): return _plan("constraint_context_to_memory_candidate", constraint_context)
def build_constraint_context_to_appraisal_plan(constraint_context): return _plan("constraint_context_to_appraisal", constraint_context)
def build_constraint_context_to_agp_input_plan(constraint_context): return _plan("constraint_context_to_agp_input", constraint_context)


def build_virtual_world_situation_constraint_context_schema_summary():
    return {
        "schema_version": VERSION,
        "supported_constraint_types": list(SUPPORTED_CONSTRAINT_TYPES),
        "supported_clause_kinds": list(SUPPORTED_CLAUSE_KINDS),
        "relational_clause_kinds": sorted(RELATIONAL_CLAUSE_KINDS),
        "type_clause_compatibility": {k: sorted(v) for k, v in sorted(TYPE_CLAUSE_COMPATIBILITY.items())},
        "supported_boundary_classes": list(SUPPORTED_BOUNDARIES),
        "supported_confidence_states": list(SUPPORTED_CONFIDENCE),
        "canonical_id_algorithm": CANONICAL_ID_ALGORITHM,
        "read_only": True,
        "candidate_only": True,
        "next_recommended_step": "read_only_virtual_world_situation_evidence_context_schema",
    }


def virtual_world_situation_constraint_context_schema_summary():
    return build_virtual_world_situation_constraint_context_schema_summary()
