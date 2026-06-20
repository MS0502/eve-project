"""Deterministic read-only hypothesis-context candidates for virtual situations."""

import hashlib
import json
import math
from typing import Any, Dict, Optional

VERSION = "1.0.0-round1421-1440"
CANONICAL_ID_ALGORITHM = "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256"
SUPPORTED_HYPOTHESIS_TYPES = (
    "explanatory_hypothesis_candidate", "causal_hypothesis_candidate", "temporal_hypothesis_candidate",
    "state_hypothesis_candidate", "identity_hypothesis_candidate", "boundary_hypothesis_candidate",
    "simulation_hypothesis_candidate", "symbolic_hypothesis_candidate", "dmn_hypothesis_candidate",
    "dream_hypothesis_candidate", "predictive_hypothesis_candidate", "counterfactual_hypothesis_candidate",
    "competing_hypothesis_candidate", "unknown_origin_hypothesis_candidate", "mixed_unknown_hypothesis_candidate",
)
SUPPORTED_HYPOTHESIS_KINDS = (
    "explanatory_proposition_candidate", "causal_proposition_candidate", "temporal_proposition_candidate",
    "state_proposition_candidate", "identity_proposition_candidate", "boundary_proposition_candidate",
    "simulation_proposition_candidate", "symbolic_proposition_candidate", "dmn_proposition_candidate",
    "dream_proposition_candidate", "predictive_proposition_candidate", "counterfactual_proposition_candidate",
    "alternative_proposition_candidate", "unknown_origin_proposition_candidate",
)
SUPPORTED_CANDIDATE_ROLES = ("focal_candidate", "alternative_candidate", "counter_candidate", "neutral_unknown")
TYPE_KIND_COMPATIBILITY = {
    "explanatory_hypothesis_candidate": {"explanatory_proposition_candidate"},
    "causal_hypothesis_candidate": {"causal_proposition_candidate"},
    "temporal_hypothesis_candidate": {"temporal_proposition_candidate"},
    "state_hypothesis_candidate": {"state_proposition_candidate"},
    "identity_hypothesis_candidate": {"identity_proposition_candidate"},
    "boundary_hypothesis_candidate": {"boundary_proposition_candidate"},
    "simulation_hypothesis_candidate": {"simulation_proposition_candidate"},
    "symbolic_hypothesis_candidate": {"symbolic_proposition_candidate"},
    "dmn_hypothesis_candidate": {"dmn_proposition_candidate"},
    "dream_hypothesis_candidate": {"dream_proposition_candidate"},
    "predictive_hypothesis_candidate": {"predictive_proposition_candidate"},
    "counterfactual_hypothesis_candidate": {"counterfactual_proposition_candidate"},
    "competing_hypothesis_candidate": {"explanatory_proposition_candidate", "causal_proposition_candidate", "temporal_proposition_candidate", "state_proposition_candidate", "identity_proposition_candidate", "boundary_proposition_candidate", "simulation_proposition_candidate", "symbolic_proposition_candidate", "predictive_proposition_candidate", "counterfactual_proposition_candidate", "alternative_proposition_candidate"},
    "unknown_origin_hypothesis_candidate": {"unknown_origin_proposition_candidate"},
    "mixed_unknown_hypothesis_candidate": set(SUPPORTED_HYPOTHESIS_KINDS),
}
KIND_ROLE_COMPATIBILITY = {
    "explanatory_proposition_candidate": {"focal_candidate", "alternative_candidate", "neutral_unknown"},
    "causal_proposition_candidate": {"focal_candidate", "alternative_candidate", "neutral_unknown"},
    "temporal_proposition_candidate": {"focal_candidate", "alternative_candidate", "neutral_unknown"},
    "state_proposition_candidate": {"focal_candidate", "alternative_candidate", "neutral_unknown"},
    "identity_proposition_candidate": {"focal_candidate", "alternative_candidate", "neutral_unknown"},
    "boundary_proposition_candidate": {"focal_candidate", "alternative_candidate", "neutral_unknown"},
    "simulation_proposition_candidate": {"focal_candidate", "alternative_candidate", "neutral_unknown"},
    "symbolic_proposition_candidate": {"focal_candidate", "alternative_candidate", "neutral_unknown"},
    "dmn_proposition_candidate": {"focal_candidate", "alternative_candidate", "neutral_unknown"},
    "dream_proposition_candidate": {"focal_candidate", "alternative_candidate", "neutral_unknown"},
    "predictive_proposition_candidate": {"focal_candidate", "alternative_candidate"},
    "counterfactual_proposition_candidate": {"counter_candidate"},
    "alternative_proposition_candidate": {"alternative_candidate"},
    "unknown_origin_proposition_candidate": {"neutral_unknown"},
}
RELATIONAL_KINDS = {"causal_proposition_candidate", "temporal_proposition_candidate", "boundary_proposition_candidate", "predictive_proposition_candidate", "counterfactual_proposition_candidate", "alternative_proposition_candidate"}
SUPPORTED_BOUNDARIES = ("internal_explanatory_hypothesis", "causal_virtual_hypothesis", "temporal_virtual_hypothesis", "state_virtual_hypothesis", "identity_virtual_hypothesis", "boundary_virtual_hypothesis", "simulation_hypothesis", "symbolic_hypothesis", "dmn_hypothesis", "dream_hypothesis", "predictive_virtual_hypothesis", "counterfactual_virtual_hypothesis", "unknown_origin_hypothesis", "mixed_virtual_external_hypothesis_boundary")
BOUNDARY_DEFAULTS = {"causal_hypothesis_candidate": "causal_virtual_hypothesis", "temporal_hypothesis_candidate": "temporal_virtual_hypothesis", "state_hypothesis_candidate": "state_virtual_hypothesis", "identity_hypothesis_candidate": "identity_virtual_hypothesis", "boundary_hypothesis_candidate": "boundary_virtual_hypothesis", "simulation_hypothesis_candidate": "simulation_hypothesis", "symbolic_hypothesis_candidate": "symbolic_hypothesis", "dmn_hypothesis_candidate": "dmn_hypothesis", "dream_hypothesis_candidate": "dream_hypothesis", "predictive_hypothesis_candidate": "predictive_virtual_hypothesis", "counterfactual_hypothesis_candidate": "counterfactual_virtual_hypothesis", "unknown_origin_hypothesis_candidate": "unknown_origin_hypothesis", "mixed_unknown_hypothesis_candidate": "mixed_virtual_external_hypothesis_boundary"}
SUPPORTED_CONFIDENCE = ("hypothesis_unverified", "hypothesis_low_confidence", "hypothesis_medium_confidence", "hypothesis_high_confidence_but_not_fact", "hypothesis_competing", "hypothesis_origin_unknown")
IMMUTABLE_FALSE_FLAGS = ("external_hypothesis_asserted", "hypothesis_verified", "hypothesis_fact_asserted", "hypothesis_accepted", "hypothesis_rejected", "conclusion_asserted", "truth_resolved", "winner_selected", "competition_resolved", "automatic_resolution_performed", "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted", "counterfactual_applied", "evidence_verified", "evidence_sufficient_asserted", "random_sampling_performed", "probability_draw_performed", "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "artifact_created_or_staged", "schedule_created", "action_prevented", "action_permitted", "agp_bypass_allowed", "fallback_bypass_allowed")
IMMUTABLE_TRUE_FLAGS = ("hypothesis_candidate_only", "situation_hypothesis_only", "read_only", "situation_review_required", "snapshot_review_required", "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required", "hypothesis_integrity_review_required", "evidence_review_required", "competition_review_required", "resolution_review_required", "appraisal_required", "agp_input_required")
FORBIDDEN_REQUEST_FIELDS = {"external_hypothesis_assertion_requested", "hypothesis_verification_requested", "hypothesis_fact_assertion_requested", "hypothesis_fact_promotion_requested", "hypothesis_acceptance_requested", "hypothesis_rejection_requested", "conclusion_assertion_requested", "conclusion_selection_requested", "winner_selection_requested", "competition_resolution_requested", "truth_resolution_requested", "automatic_resolution_requested", "prediction_guarantee_requested", "future_outcome_assertion_requested", "counterfactual_application_requested", "evidence_verification_requested", "evidence_sufficiency_assertion_requested", "certainty_promotion_requested", "fact_promotion_requested", "random_sampling_requested", "probability_draw_requested", "memory_write_requested", "memory_fact_promotion_requested", "relationship_update_requested", "self_model_update_requested", "identity_update_requested", "affect_transition_requested", "hormone_transition_requested", "world_state_mutation_requested", "transition_application_requested", "planning_execution_requested", "tool_execution_requested", "runtime_mutation_requested", "persistence_write_requested", "vector_read_requested", "vector_load_requested", "model_load_requested", "device_activation_requested", "network_action_requested", "artifact_creation_requested", "intervention_requested", "event_completion_assertion_requested", "action_prevention_requested", "action_permission_requested", "agp_bypass_requested", "fallback_bypass_requested", "schedule_requested", "scheduler_requested", "timer_requested", "alarm_requested", "calendar_requested", "reminder_requested", "deadline_requested", "cron_requested", "cron_expression", "scheduled_execution_requested", "delayed_execution_requested", "act_requested", "execute_requested", "apply_requested", "commit_outcome_requested", "promote_to_fact_requested"}
TOP_KEYS = {"situation_hypothesis_context_passed", "situation_hypothesis_context_status", "hypothesis_context_id", "canonical_id_algorithm", "hypothesis_type", "hypothesis_boundary_classification", "hypothesis_confidence_state", "situation_id", "hypothesis_items", "metadata", "origin_summary", "fact_status_summary", "hypothesis_scope_summary", "hypothesis_competition_summary", "hypothesis_support_summary", "resolution_summary", "hypothesis_flags", "boundary_flags", "hypothesis_integrity_flags", "candidate_only_fields", "blocked_reasons", "warnings", *IMMUTABLE_FALSE_FLAGS, *IMMUTABLE_TRUE_FLAGS}


def _non_empty_string(value: Any) -> bool:
    return type(value) is str and bool(value.strip())


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
            for item in sorted(value, key=_canonical_sort_key):
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


def _semantic_item_key(item: Dict[str, Any]) -> str:
    return _canonical_sort_key({"situation_id": item["situation_id"], "hypothesis_kind": item["hypothesis_kind"], "candidate_role": item["candidate_role"], "subject_ref_id": item["subject_ref_id"], "object_ref_id": item.get("object_ref_id"), "claim_ref_id": item["claim_ref_id"], "basis_evidence_ref_ids": item["basis_evidence_ref_ids"], "related_context_id": item.get("related_context_id")})


def _collect_forbidden_reasons(value: Any, reasons: list) -> None:
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            item = value[key]
            if key in FORBIDDEN_REQUEST_FIELDS:
                if item is True:
                    reasons.append(key)
                elif item is not False:
                    reasons.append("malformed_forbidden_request_field")
            _collect_forbidden_reasons(item, reasons)
    elif isinstance(value, list):
        for item in sorted(value, key=_canonical_sort_key):
            _collect_forbidden_reasons(item, reasons)


def _forbidden_reason(value: Any) -> Optional[str]:
    reasons = []
    _collect_forbidden_reasons(value, reasons)
    if "malformed_forbidden_request_field" in reasons:
        return "malformed_forbidden_request_field"
    return sorted(reasons)[0] if reasons else None


def _base_payload() -> Dict[str, Any]:
    payload = {key: False for key in IMMUTABLE_FALSE_FLAGS}
    payload.update({key: True for key in IMMUTABLE_TRUE_FLAGS})
    return payload


def _reject(reason: str) -> Dict[str, Any]:
    payload = _base_payload()
    payload.update({
        "situation_hypothesis_context_passed": False, "situation_hypothesis_context_status": "REJECTED", "hypothesis_context_id": None,
        "canonical_id_algorithm": None, "hypothesis_type": None, "hypothesis_boundary_classification": None,
        "hypothesis_confidence_state": None, "situation_id": None, "hypothesis_items": [], "metadata": {},
        "origin_summary": {"origin_kind": "internal_virtual_hypothesis_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "external_hypothesis_fact_verified": False, "hypothesis_fact_asserted": False, "hypothesis_accepted": False, "hypothesis_rejected": False, "conclusion_asserted": False, "truth_resolved": False},
        "hypothesis_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "hypothesis_kinds": [], "candidate_roles": [], "claim_ref_ids": [], "item_count": 0},
        "hypothesis_competition_summary": {"focal_item_count": 0, "alternative_item_count": 0, "counter_item_count": 0, "neutral_item_count": 0, "competition_present": False, "winner_selected": False},
        "hypothesis_support_summary": {"basis_evidence_ref_ids": [], "basis_evidence_ref_count": 0, "external_evidence_verified": False, "evidence_sufficient_asserted": False},
        "resolution_summary": {"resolution_candidate_only": True, "resolved": False, "accepted": False, "rejected": False, "winner_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False},
        "hypothesis_flags": [], "boundary_flags": [], "hypothesis_integrity_flags": [],
        "candidate_only_fields": ["hypothesis_type", "hypothesis_items", "hypothesis_boundary_classification", "hypothesis_confidence_state"],
        "blocked_reasons": [reason], "warnings": [],
    })
    return payload


def _normalize_items(items: Any, situation_id: str):
    if items is None:
        return None, "missing_hypothesis_items"
    if not isinstance(items, list):
        return None, "hypothesis_items_not_non_empty_list"
    if not items:
        return None, "empty_hypothesis_items"
    normalized = []
    for item in sorted(items, key=_canonical_sort_key):
        if not isinstance(item, dict):
            return None, "malformed_hypothesis_item"
        for req, reason in (("hypothesis_item_id", "missing_or_empty_hypothesis_item_id"), ("situation_id", "malformed_hypothesis_item_situation_id"), ("hypothesis_kind", "unknown_hypothesis_kind"), ("candidate_role", "unknown_candidate_role"), ("subject_ref_id", "missing_or_malformed_subject_ref_id"), ("claim_ref_id", "missing_or_malformed_claim_ref_id"), ("basis_evidence_ref_ids", "missing_or_malformed_basis_evidence_ref_ids")):
            if req not in item:
                return None, reason
        if not _non_empty_string(item["hypothesis_item_id"]): return None, "missing_or_empty_hypothesis_item_id"
        if not _non_empty_string(item["situation_id"]): return None, "malformed_hypothesis_item_situation_id"
        if item["hypothesis_kind"] not in SUPPORTED_HYPOTHESIS_KINDS: return None, "unknown_hypothesis_kind"
        if item["candidate_role"] not in SUPPORTED_CANDIDATE_ROLES: return None, "unknown_candidate_role"
        if not _non_empty_string(item["subject_ref_id"]): return None, "missing_or_malformed_subject_ref_id"
        if not _non_empty_string(item["claim_ref_id"]): return None, "missing_or_malformed_claim_ref_id"
        refs = item["basis_evidence_ref_ids"]
        if type(refs) is not list or not refs or any(not _non_empty_string(ref) for ref in refs):
            return None, "missing_or_malformed_basis_evidence_ref_ids"
        if len(refs) != len(set(refs)):
            return None, "duplicate_basis_evidence_ref_id"
        if "related_context_id" in item and not _non_empty_string(item["related_context_id"]): return None, "malformed_related_context_id"
        for field, malformed, out_range in (("confidence_candidate", "malformed_confidence_candidate", "confidence_candidate_out_of_range"), ("plausibility_candidate", "malformed_plausibility_candidate", "plausibility_candidate_out_of_range")):
            if field in item:
                value = item[field]
                if type(value) is bool or type(value) not in {int, float}:
                    return None, malformed
                if type(value) is float and not math.isfinite(value):
                    return None, malformed
                if value < 0 or value > 1:
                    return None, out_range
        if "label" in item and not _non_empty_string(item["label"]): return None, "malformed_hypothesis_item_label"
        clone = _json_clone(item)
        clone["basis_evidence_ref_ids"] = sorted(clone["basis_evidence_ref_ids"])
        normalized.append(clone)
    reason = _forbidden_reason(normalized)
    if reason:
        return None, reason
    ids = [i["hypothesis_item_id"] for i in normalized]
    if len(ids) != len(set(ids)):
        return None, "duplicate_hypothesis_item_id"
    seen = set()
    for item in normalized:
        key = _semantic_item_key(item)
        if key in seen:
            return None, "duplicate_semantic_hypothesis_item"
        seen.add(key)
    if any(i["situation_id"] != situation_id for i in normalized):
        return None, "hypothesis_item_situation_mismatch"
    for item in normalized:
        if item["hypothesis_kind"] in RELATIONAL_KINDS:
            if "object_ref_id" not in item:
                return None, "missing_object_ref_id_for_relational_hypothesis"
            if not _non_empty_string(item["object_ref_id"]):
                return None, "malformed_object_ref_id"
            if item["object_ref_id"] == item["subject_ref_id"]:
                return None, "identical_hypothesis_refs"
        elif "object_ref_id" in item:
            return None, "unexpected_object_ref_id_for_unary_hypothesis"
    for item in normalized:
        if item["candidate_role"] not in KIND_ROLE_COMPATIBILITY[item["hypothesis_kind"]]:
            return None, "incompatible_hypothesis_kind_candidate_role"
    return sorted(normalized, key=_canonical_sort_key), None


def _type_reason(hypothesis_type: str, items: list) -> Optional[str]:
    if any(i["hypothesis_kind"] not in TYPE_KIND_COMPATIBILITY[hypothesis_type] for i in items):
        return "incompatible_hypothesis_type_kind"
    roles = [i["candidate_role"] for i in items]
    kinds = {i["hypothesis_kind"] for i in items}
    required_kind = {"explanatory_hypothesis_candidate":"explanatory_proposition_candidate", "causal_hypothesis_candidate":"causal_proposition_candidate", "temporal_hypothesis_candidate":"temporal_proposition_candidate", "state_hypothesis_candidate":"state_proposition_candidate", "identity_hypothesis_candidate":"identity_proposition_candidate", "boundary_hypothesis_candidate":"boundary_proposition_candidate", "simulation_hypothesis_candidate":"simulation_proposition_candidate", "symbolic_hypothesis_candidate":"symbolic_proposition_candidate", "dmn_hypothesis_candidate":"dmn_proposition_candidate", "dream_hypothesis_candidate":"dream_proposition_candidate", "predictive_hypothesis_candidate":"predictive_proposition_candidate", "counterfactual_hypothesis_candidate":"counterfactual_proposition_candidate", "unknown_origin_hypothesis_candidate":"unknown_origin_proposition_candidate"}
    if hypothesis_type in required_kind and required_kind[hypothesis_type] not in kinds:
        return f"{hypothesis_type.replace('_candidate','')}_requires_{required_kind[hypothesis_type].replace('_candidate','')}"
    if hypothesis_type == "competing_hypothesis_candidate":
        if len(items) < 2: return "competing_hypothesis_requires_multiple_items"
        if "focal_candidate" not in roles: return "competing_hypothesis_missing_focal"
        if "alternative_candidate" not in roles and "counter_candidate" not in roles: return "competing_hypothesis_missing_alternative_or_counter"
    if hypothesis_type == "mixed_unknown_hypothesis_candidate":
        if len(items) < 2: return "mixed_unknown_hypothesis_requires_multiple_items"
        if len(kinds) < 2 and len(set(roles)) < 2: return "mixed_unknown_hypothesis_requires_distinct_dimensions"
    return None


def _validate_boundary(metadata: dict, hypothesis_type: str):
    if "hypothesis_boundary_classification" in metadata:
        value = metadata["hypothesis_boundary_classification"]
        if not _non_empty_string(value):
            return None, "malformed_hypothesis_boundary_class"
    else:
        value = BOUNDARY_DEFAULTS.get(hypothesis_type, "internal_explanatory_hypothesis")
    if value not in SUPPORTED_BOUNDARIES:
        return None, "unknown_hypothesis_boundary_class"
    return value, None


def _validate_confidence(metadata: dict):
    if "hypothesis_confidence_state" in metadata:
        value = metadata["hypothesis_confidence_state"]
        if not _non_empty_string(value):
            return None, "malformed_hypothesis_confidence_state"
    else:
        value = "hypothesis_unverified"
    if value not in SUPPORTED_CONFIDENCE:
        return None, "unknown_hypothesis_confidence_state"
    return value, None


def _build_valid(hypothesis_type: str, situation_id: str, items: list, metadata: dict, boundary: str, confidence: str) -> Dict[str, Any]:
    focal = sum(1 for i in items if i["candidate_role"] == "focal_candidate")
    alternative = sum(1 for i in items if i["candidate_role"] == "alternative_candidate")
    counter = sum(1 for i in items if i["candidate_role"] == "counter_candidate")
    neutral = sum(1 for i in items if i["candidate_role"] == "neutral_unknown")
    basis_refs = sorted({ref for i in items for ref in i["basis_evidence_ref_ids"]})
    flags = ["candidate_only", "hypothesis_unverified"]
    integrity = ["hypothesis_integrity_review_required", "evidence_review_required", "competition_review_required", "resolution_review_required"]
    if confidence == "hypothesis_competing" or (focal and (alternative or counter)):
        flags.append("competition_review_required")
    if confidence == "hypothesis_origin_unknown" or hypothesis_type == "unknown_origin_hypothesis_candidate":
        flags.append("origin_unknown"); integrity.append("origin_unknown_review_required")
    basis = {"schema_version": VERSION, "hypothesis_type": hypothesis_type, "hypothesis_boundary_classification": boundary, "hypothesis_confidence_state": confidence, "situation_id": situation_id, "hypothesis_items": items, "metadata": metadata}
    blob = json.dumps(basis, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = _base_payload()
    payload.update({
        "situation_hypothesis_context_passed": True, "situation_hypothesis_context_status": "VALIDATED",
        "hypothesis_context_id": hashlib.sha256(blob.encode("utf-8")).hexdigest(), "canonical_id_algorithm": CANONICAL_ID_ALGORITHM,
        "hypothesis_type": hypothesis_type, "hypothesis_boundary_classification": boundary, "hypothesis_confidence_state": confidence,
        "situation_id": situation_id, "hypothesis_items": items, "metadata": metadata,
        "origin_summary": {"origin_kind": "internal_virtual_hypothesis_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "external_hypothesis_fact_verified": False, "hypothesis_fact_asserted": False, "hypothesis_accepted": False, "hypothesis_rejected": False, "conclusion_asserted": False, "truth_resolved": False},
        "hypothesis_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "hypothesis_kinds": sorted({i["hypothesis_kind"] for i in items}), "candidate_roles": sorted({i["candidate_role"] for i in items}), "claim_ref_ids": sorted({i["claim_ref_id"] for i in items}), "item_count": len(items)},
        "hypothesis_competition_summary": {"focal_item_count": focal, "alternative_item_count": alternative, "counter_item_count": counter, "neutral_item_count": neutral, "competition_present": bool(focal and (alternative or counter)), "winner_selected": False},
        "hypothesis_support_summary": {"basis_evidence_ref_ids": basis_refs, "basis_evidence_ref_count": len(basis_refs), "external_evidence_verified": False, "evidence_sufficient_asserted": False},
        "resolution_summary": {"resolution_candidate_only": True, "resolved": False, "accepted": False, "rejected": False, "winner_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False},
        "hypothesis_flags": sorted(set(flags)), "boundary_flags": [boundary], "hypothesis_integrity_flags": sorted(set(integrity)),
        "candidate_only_fields": ["hypothesis_type", "hypothesis_items", "hypothesis_boundary_classification", "hypothesis_confidence_state"],
        "blocked_reasons": [], "warnings": ["mixed_virtual_external_hypothesis_boundary_review_required"] if hypothesis_type == "mixed_unknown_hypothesis_candidate" else [],
    })
    return payload


def build_virtual_world_situation_hypothesis_context(hypothesis_type=None, situation_id=None, hypothesis_items=None, metadata=None):
    raw = {"hypothesis_type": hypothesis_type, "situation_id": situation_id, "hypothesis_items": hypothesis_items, "metadata": metadata if metadata is not None else {}}
    try:
        if _assert_json_native(raw): return _reject("non_json_serializable_semantic_input")
        if hypothesis_type is None or not _non_empty_string(hypothesis_type): return _reject("missing_hypothesis_type")
        if hypothesis_type not in SUPPORTED_HYPOTHESIS_TYPES: return _reject("unknown_hypothesis_type")
        if not _non_empty_string(situation_id): return _reject("missing_or_malformed_situation_id")
        if metadata is None: metadata = {}
        if not isinstance(metadata, dict): return _reject("invalid_metadata")
        metadata = _json_clone(metadata)
        reason = _forbidden_reason(metadata)
        if reason: return _reject(reason)
        items, reason = _normalize_items(hypothesis_items, situation_id)
        if reason: return _reject(reason)
        reason = _type_reason(hypothesis_type, items)
        if reason: return _reject(reason)
        boundary, reason = _validate_boundary(metadata, hypothesis_type)
        if reason: return _reject(reason)
        confidence, reason = _validate_confidence(metadata)
        if reason: return _reject(reason)
        return _build_valid(hypothesis_type, situation_id, items, metadata, boundary, confidence)
    except Exception:
        return _reject("non_json_serializable_semantic_input")


def validate_virtual_world_situation_hypothesis_context(hypothesis_context) -> bool:
    try:
        if _assert_json_native(hypothesis_context) or not isinstance(hypothesis_context, dict): return False
        if set(hypothesis_context.keys()) != TOP_KEYS: return False
        if hypothesis_context.get("situation_hypothesis_context_passed") is not True or hypothesis_context.get("situation_hypothesis_context_status") != "VALIDATED": return False
        expected = build_virtual_world_situation_hypothesis_context(hypothesis_context.get("hypothesis_type"), hypothesis_context.get("situation_id"), hypothesis_context.get("hypothesis_items"), hypothesis_context.get("metadata"))
        return _type_exact_equal(expected, hypothesis_context)
    except Exception:
        return False

PLAN_FALSE_FLAGS = tuple(k for k in IMMUTABLE_FALSE_FLAGS if k != "artifact_created_or_staged") + ("relationship_update_performed", "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed")


def _plan(plan_type: str, source: Any) -> Dict[str, Any]:
    ready = validate_virtual_world_situation_hypothesis_context(source)
    plan = {"plan_type": plan_type, "source_hypothesis_context_id": source.get("hypothesis_context_id") if isinstance(source, dict) and ready else None, "ready": ready, "candidate_only": True, "read_only": True, "hypothesis_candidate_only": True}
    for key in PLAN_FALSE_FLAGS:
        plan[key] = False
    return plan


def build_hypothesis_context_to_situation_plan(hypothesis_context): return _plan("hypothesis_context_to_situation", hypothesis_context)
def build_hypothesis_context_to_snapshot_plan(hypothesis_context): return _plan("hypothesis_context_to_snapshot", hypothesis_context)
def build_hypothesis_context_to_transition_preflight_plan(hypothesis_context): return _plan("hypothesis_context_to_transition_preflight", hypothesis_context)
def build_hypothesis_context_to_memory_candidate_plan(hypothesis_context): return _plan("hypothesis_context_to_memory_candidate", hypothesis_context)
def build_hypothesis_context_to_appraisal_plan(hypothesis_context): return _plan("hypothesis_context_to_appraisal", hypothesis_context)
def build_hypothesis_context_to_agp_input_plan(hypothesis_context): return _plan("hypothesis_context_to_agp_input", hypothesis_context)


def build_virtual_world_situation_hypothesis_context_schema_summary():
    return {"schema_version": VERSION, "schema_name": "read_only_virtual_world_situation_hypothesis_context_schema", "supported_hypothesis_types": list(SUPPORTED_HYPOTHESIS_TYPES), "supported_hypothesis_kinds": list(SUPPORTED_HYPOTHESIS_KINDS), "supported_candidate_roles": list(SUPPORTED_CANDIDATE_ROLES), "type_kind_compatibility": {k: sorted(v) for k, v in sorted(TYPE_KIND_COMPATIBILITY.items())}, "kind_role_compatibility": {k: sorted(v) for k, v in sorted(KIND_ROLE_COMPATIBILITY.items())}, "relational_hypothesis_kinds": sorted(RELATIONAL_KINDS), "supported_boundary_classes": list(SUPPORTED_BOUNDARIES), "supported_confidence_states": list(SUPPORTED_CONFIDENCE), "canonical_id_algorithm": CANONICAL_ID_ALGORITHM, "read_only": True, "candidate_only": True, "next_recommended_step": "read_only_virtual_world_situation_inference_context_schema", "permanent_validation_contract": ["input types", "required and optional fields", "enums", "compatibility matrices", "situation and reference coherence", "relational and unary reference shape", "cardinality", "composition", "ordering", "duplicates", "validation precedence", "strict JSON", "recursive forbidden fields", "deterministic IDs", "exact top-level schema", "nested payload integrity", "immutable safety fields", "downstream plan safety"]}


def virtual_world_situation_hypothesis_context_schema_summary():
    return build_virtual_world_situation_hypothesis_context_schema_summary()
