"""Deterministic read-only inference-context candidates for virtual situations."""

import hashlib
import json
import math
from typing import Any, Dict, Optional

VERSION = "1.0.0-round1441-1460"
CANONICAL_ID_ALGORITHM = "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256"
SUPPORTED_INFERENCE_TYPES = (
    "deductive_inference_candidate", "abductive_inference_candidate", "inductive_inference_candidate",
    "causal_inference_candidate", "temporal_inference_candidate", "state_inference_candidate",
    "identity_inference_candidate", "boundary_inference_candidate", "simulation_inference_candidate",
    "symbolic_inference_candidate", "dmn_inference_candidate", "dream_inference_candidate",
    "predictive_inference_candidate", "counterfactual_inference_candidate", "competing_inference_candidate",
    "unknown_origin_inference_candidate", "mixed_unknown_inference_candidate",
)
SUPPORTED_INFERENCE_KINDS = (
    "deductive_step_candidate", "abductive_step_candidate", "inductive_step_candidate", "causal_step_candidate",
    "temporal_step_candidate", "state_step_candidate", "identity_step_candidate", "boundary_step_candidate",
    "simulation_step_candidate", "symbolic_step_candidate", "dmn_step_candidate", "dream_step_candidate",
    "predictive_step_candidate", "counterfactual_step_candidate", "alternative_step_candidate", "unknown_origin_step_candidate",
)
SUPPORTED_CANDIDATE_ROLES = ("focal_conclusion_candidate", "alternative_conclusion_candidate", "counter_conclusion_candidate", "neutral_unknown")
SUPPORTED_DERIVATION_SOURCE_CLASSES = (
    "internal_hypothesis_context", "internal_evidence_context", "internal_constraint_context", "internal_uncertainty_context",
    "internal_temporal_context", "internal_causal_context", "internal_simulation", "internal_symbolic",
    "internal_dmn", "internal_dream", "external_unverified_candidate", "unknown_origin_candidate",
)
TYPE_KIND_COMPATIBILITY = {
    "deductive_inference_candidate": {"deductive_step_candidate"},
    "abductive_inference_candidate": {"abductive_step_candidate"},
    "inductive_inference_candidate": {"inductive_step_candidate"},
    "causal_inference_candidate": {"causal_step_candidate"},
    "temporal_inference_candidate": {"temporal_step_candidate"},
    "state_inference_candidate": {"state_step_candidate"},
    "identity_inference_candidate": {"identity_step_candidate"},
    "boundary_inference_candidate": {"boundary_step_candidate"},
    "simulation_inference_candidate": {"simulation_step_candidate"},
    "symbolic_inference_candidate": {"symbolic_step_candidate"},
    "dmn_inference_candidate": {"dmn_step_candidate"},
    "dream_inference_candidate": {"dream_step_candidate"},
    "predictive_inference_candidate": {"predictive_step_candidate"},
    "counterfactual_inference_candidate": {"counterfactual_step_candidate"},
    "competing_inference_candidate": {"deductive_step_candidate", "abductive_step_candidate", "inductive_step_candidate", "causal_step_candidate", "temporal_step_candidate", "state_step_candidate", "identity_step_candidate", "boundary_step_candidate", "simulation_step_candidate", "symbolic_step_candidate", "predictive_step_candidate", "counterfactual_step_candidate", "alternative_step_candidate"},
    "unknown_origin_inference_candidate": {"unknown_origin_step_candidate"},
    "mixed_unknown_inference_candidate": set(SUPPORTED_INFERENCE_KINDS),
}
_BASE_ROLES = {"focal_conclusion_candidate", "alternative_conclusion_candidate", "neutral_unknown"}
KIND_ROLE_COMPATIBILITY = {k: set(_BASE_ROLES) for k in SUPPORTED_INFERENCE_KINDS[:12]}
KIND_ROLE_COMPATIBILITY.update({"predictive_step_candidate": {"focal_conclusion_candidate", "alternative_conclusion_candidate"}, "counterfactual_step_candidate": {"counter_conclusion_candidate"}, "alternative_step_candidate": {"alternative_conclusion_candidate"}, "unknown_origin_step_candidate": {"neutral_unknown"}})
KIND_SOURCE_COMPATIBILITY = {
    "deductive_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_constraint_context"},
    "abductive_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_uncertainty_context", "unknown_origin_candidate"},
    "inductive_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "external_unverified_candidate", "unknown_origin_candidate"},
    "causal_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_causal_context"},
    "temporal_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_temporal_context"},
    "state_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_constraint_context"},
    "identity_step_candidate": {"internal_hypothesis_context", "internal_evidence_context"},
    "boundary_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_uncertainty_context"},
    "simulation_step_candidate": {"internal_simulation"}, "symbolic_step_candidate": {"internal_symbolic"},
    "dmn_step_candidate": {"internal_dmn"}, "dream_step_candidate": {"internal_dream"},
    "predictive_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_temporal_context", "internal_causal_context"},
    "counterfactual_step_candidate": {"internal_hypothesis_context", "internal_simulation"},
    "alternative_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_simulation"},
    "unknown_origin_step_candidate": {"unknown_origin_candidate"},
}
RELATIONAL_KINDS = {"causal_step_candidate", "temporal_step_candidate", "boundary_step_candidate", "predictive_step_candidate", "counterfactual_step_candidate", "alternative_step_candidate"}
SUPPORTED_BOUNDARIES = ("internal_deductive_inference", "internal_abductive_inference", "internal_inductive_inference", "causal_virtual_inference", "temporal_virtual_inference", "state_virtual_inference", "identity_virtual_inference", "boundary_virtual_inference", "simulation_inference", "symbolic_inference", "dmn_inference", "dream_inference", "predictive_virtual_inference", "counterfactual_virtual_inference", "unknown_origin_inference", "mixed_virtual_external_inference_boundary")
BOUNDARY_DEFAULTS = dict(zip(SUPPORTED_INFERENCE_TYPES, ("internal_deductive_inference", "internal_abductive_inference", "internal_inductive_inference", "causal_virtual_inference", "temporal_virtual_inference", "state_virtual_inference", "identity_virtual_inference", "boundary_virtual_inference", "simulation_inference", "symbolic_inference", "dmn_inference", "dream_inference", "predictive_virtual_inference", "counterfactual_virtual_inference", "internal_abductive_inference", "unknown_origin_inference", "mixed_virtual_external_inference_boundary")))
SUPPORTED_CONFIDENCE = ("inference_unverified", "inference_low_confidence", "inference_medium_confidence", "inference_high_confidence_but_not_fact", "inference_competing", "inference_origin_unknown")
IMMUTABLE_FALSE_FLAGS = ("external_inference_asserted", "inference_verified", "inference_fact_asserted", "inference_accepted", "inference_rejected", "conclusion_asserted", "truth_resolved", "winner_selected", "competition_resolved", "automatic_resolution_performed", "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted", "counterfactual_applied", "basis_verified", "basis_sufficient_asserted", "random_sampling_performed", "probability_draw_performed", "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "artifact_created_or_staged", "schedule_created", "action_prevented", "action_permitted", "agp_bypass_allowed", "fallback_bypass_allowed")
IMMUTABLE_TRUE_FLAGS = ("inference_candidate_only", "situation_inference_only", "read_only", "situation_review_required", "snapshot_review_required", "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required", "inference_integrity_review_required", "basis_review_required", "competition_review_required", "resolution_review_required", "appraisal_required", "agp_input_required")
FORBIDDEN_REQUEST_FIELDS = {"external_inference_assertion_requested", "inference_verification_requested", "inference_fact_assertion_requested", "inference_fact_promotion_requested", "inference_acceptance_requested", "inference_rejection_requested", "conclusion_assertion_requested", "conclusion_selection_requested", "winner_selection_requested", "competition_resolution_requested", "truth_resolution_requested", "automatic_resolution_requested", "prediction_guarantee_requested", "future_outcome_assertion_requested", "counterfactual_application_requested", "basis_verification_requested", "basis_sufficiency_assertion_requested", "certainty_promotion_requested", "fact_promotion_requested", "random_sampling_requested", "probability_draw_requested", "memory_write_requested", "memory_fact_promotion_requested", "relationship_update_requested", "self_model_update_requested", "identity_update_requested", "affect_transition_requested", "hormone_transition_requested", "world_state_mutation_requested", "transition_application_requested", "planning_execution_requested", "tool_execution_requested", "runtime_mutation_requested", "persistence_write_requested", "vector_read_requested", "vector_load_requested", "model_load_requested", "device_activation_requested", "network_action_requested", "artifact_creation_requested", "intervention_requested", "event_completion_assertion_requested", "action_prevention_requested", "action_permission_requested", "agp_bypass_requested", "fallback_bypass_requested", "schedule_requested", "scheduler_requested", "timer_requested", "alarm_requested", "calendar_requested", "reminder_requested", "deadline_requested", "cron_requested", "cron_expression", "scheduled_execution_requested", "delayed_execution_requested", "act_requested", "execute_requested", "apply_requested", "commit_outcome_requested", "promote_to_fact_requested"}
TOP_KEYS = {"situation_inference_context_passed", "situation_inference_context_status", "inference_context_id", "canonical_id_algorithm", "inference_type", "inference_boundary_classification", "inference_confidence_state", "situation_id", "inference_items", "metadata", "origin_summary", "fact_status_summary", "inference_scope_summary", "inference_competition_summary", "inference_support_summary", "resolution_summary", "inference_flags", "boundary_flags", "inference_integrity_flags", "candidate_only_fields", "blocked_reasons", "warnings", *IMMUTABLE_FALSE_FLAGS, *IMMUTABLE_TRUE_FLAGS}
PLAN_FALSE_FLAGS = tuple(k for k in IMMUTABLE_FALSE_FLAGS if k != "artifact_created_or_staged") + ("relationship_update_performed", "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed")

def _non_empty_string(value: Any) -> bool:
    return type(value) is str and bool(value.strip())

def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def _assert_json_native(value: Any, seen: Optional[set] = None, depth: int = 0) -> Optional[str]:
    try:
        if depth > 100: return "non_json_serializable_semantic_input"
        if value is None or type(value) in {str, bool}: return None
        if type(value) is int: return None
        if type(value) is float: return None if math.isfinite(value) else "non_json_serializable_semantic_input"
        if seen is None: seen = set()
        if type(value) is dict:
            oid = id(value)
            if oid in seen: return "non_json_serializable_semantic_input"
            seen.add(oid); keys = list(value.keys())
            if not all(type(k) is str for k in keys): return "non_json_serializable_semantic_input"
            for key in sorted(keys):
                reason = _assert_json_native(value[key], seen, depth + 1)
                if reason: return reason
            seen.remove(oid); return None
        if type(value) is list:
            oid = id(value)
            if oid in seen: return "non_json_serializable_semantic_input"
            seen.add(oid)
            for item in sorted(value, key=_canonical_sort_key):
                reason = _assert_json_native(item, seen, depth + 1)
                if reason: return reason
            seen.remove(oid); return None
        return "non_json_serializable_semantic_input"
    except Exception:
        return "non_json_serializable_semantic_input"

def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))

def _type_exact_equal(expected: Any, actual: Any) -> bool:
    if type(expected) is not type(actual): return False
    if isinstance(expected, dict): return set(expected.keys()) == set(actual.keys()) and all(_type_exact_equal(expected[k], actual[k]) for k in expected)
    if isinstance(expected, list): return len(expected) == len(actual) and all(_type_exact_equal(a, b) for a, b in zip(expected, actual))
    return expected == actual

def _collect_forbidden_reasons(value: Any, reasons: list) -> None:
    if type(value) is dict:
        for key in sorted(value.keys()):
            item = value[key]
            if key in FORBIDDEN_REQUEST_FIELDS:
                if item is True: reasons.append(key)
                elif item is not False: reasons.append("malformed_forbidden_request_field")
            _collect_forbidden_reasons(item, reasons)
    elif type(value) is list:
        for item in sorted(value, key=_canonical_sort_key): _collect_forbidden_reasons(item, reasons)

def _forbidden_reason(value: Any) -> Optional[str]:
    reasons = []; _collect_forbidden_reasons(value, reasons)
    if "malformed_forbidden_request_field" in reasons: return "malformed_forbidden_request_field"
    return sorted(reasons)[0] if reasons else None

def _base_payload() -> Dict[str, Any]:
    payload = {k: False for k in IMMUTABLE_FALSE_FLAGS}; payload.update({k: True for k in IMMUTABLE_TRUE_FLAGS}); return payload

def _reject(reason: str) -> Dict[str, Any]:
    payload = _base_payload(); payload.update({
        "situation_inference_context_passed": False, "situation_inference_context_status": "REJECTED", "inference_context_id": None,
        "canonical_id_algorithm": None, "inference_type": None, "inference_boundary_classification": None, "inference_confidence_state": None,
        "situation_id": None, "inference_items": [], "metadata": {},
        "origin_summary": {"origin_kind": "internal_virtual_inference_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "external_inference_fact_verified": False, "inference_fact_asserted": False, "inference_accepted": False, "inference_rejected": False, "conclusion_asserted": False, "truth_resolved": False},
        "inference_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "inference_kinds": [], "candidate_roles": [], "derivation_source_classes": [], "conclusion_ref_ids": [], "item_count": 0},
        "inference_competition_summary": {"focal_item_count": 0, "alternative_item_count": 0, "counter_item_count": 0, "neutral_item_count": 0, "competition_present": False, "winner_selected": False},
        "inference_support_summary": {"premise_ref_ids": [], "premise_ref_count": 0, "hypothesis_context_ref_ids": [], "hypothesis_context_ref_count": 0, "evidence_ref_ids": [], "evidence_ref_count": 0, "external_basis_verified": False, "basis_sufficient_asserted": False},
        "resolution_summary": {"resolution_candidate_only": True, "resolved": False, "accepted": False, "rejected": False, "winner_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False},
        "inference_flags": [], "boundary_flags": [], "inference_integrity_flags": [],
        "candidate_only_fields": ["inference_type", "inference_items", "inference_boundary_classification", "inference_confidence_state"],
        "blocked_reasons": [reason], "warnings": [],
    }); return payload

def _ref_list(item, field, malformed, duplicate, required=True):
    if field not in item: return None, malformed
    refs = item[field]
    if type(refs) is not list or (required and not refs) or any(not _non_empty_string(r) for r in refs): return None, malformed
    if len(refs) != len(set(refs)): return None, duplicate
    return sorted(refs), None

def _semantic_item_key(item: Dict[str, Any]) -> str:
    return _canonical_sort_key({"situation_id": item["situation_id"], "inference_kind": item["inference_kind"], "candidate_role": item["candidate_role"], "derivation_source_class": item["derivation_source_class"], "subject_ref_id": item["subject_ref_id"], "object_ref_id": item.get("object_ref_id"), "conclusion_ref_id": item["conclusion_ref_id"], "premise_ref_ids": item["premise_ref_ids"], "hypothesis_context_ref_ids": item["hypothesis_context_ref_ids"], "evidence_ref_ids": item.get("evidence_ref_ids", []), "related_context_id": item.get("related_context_id")})

def _normalize_items(items: Any, situation_id: str):
    if items is None: return None, "missing_inference_items"
    if type(items) is not list: return None, "inference_items_not_non_empty_list"
    if not items: return None, "empty_inference_items"
    staged = []
    for item in sorted(items, key=_canonical_sort_key):
        if type(item) is not dict: return None, "malformed_inference_item"
        for req, reason in (("inference_item_id", "missing_or_empty_inference_item_id"), ("situation_id", "malformed_inference_item_situation_id"), ("inference_kind", "unknown_inference_kind"), ("candidate_role", "unknown_candidate_role"), ("derivation_source_class", "unknown_derivation_source_class"), ("subject_ref_id", "missing_or_malformed_subject_ref_id"), ("conclusion_ref_id", "missing_or_malformed_conclusion_ref_id"), ("premise_ref_ids", "missing_or_malformed_premise_ref_ids"), ("hypothesis_context_ref_ids", "missing_or_malformed_hypothesis_context_ref_ids")):
            if req not in item: return None, reason
        if not _non_empty_string(item["inference_item_id"]): return None, "missing_or_empty_inference_item_id"
        if not _non_empty_string(item["situation_id"]): return None, "malformed_inference_item_situation_id"
        if item["inference_kind"] not in SUPPORTED_INFERENCE_KINDS: return None, "unknown_inference_kind"
        if item["candidate_role"] not in SUPPORTED_CANDIDATE_ROLES: return None, "unknown_candidate_role"
        if item["derivation_source_class"] not in SUPPORTED_DERIVATION_SOURCE_CLASSES: return None, "unknown_derivation_source_class"
        if not _non_empty_string(item["subject_ref_id"]): return None, "missing_or_malformed_subject_ref_id"
        if not _non_empty_string(item["conclusion_ref_id"]): return None, "missing_or_malformed_conclusion_ref_id"
        if "related_context_id" in item and not _non_empty_string(item["related_context_id"]): return None, "malformed_related_context_id"
        for field, malformed, out_range in (("confidence_candidate", "malformed_confidence_candidate", "confidence_candidate_out_of_range"), ("coherence_candidate", "malformed_coherence_candidate", "coherence_candidate_out_of_range")):
            if field in item:
                value = item[field]
                if type(value) is bool or type(value) not in {int, float}: return None, malformed
                if type(value) is float and not math.isfinite(value): return None, malformed
                if value < 0 or value > 1: return None, out_range
        if "label" in item and not _non_empty_string(item["label"]): return None, "malformed_inference_item_label"
        staged.append(_json_clone(item))
    reason = _forbidden_reason(staged)
    if reason: return None, reason
    ids = [i["inference_item_id"] for i in staged]
    if len(ids) != len(set(ids)): return None, "duplicate_inference_item_id"
    normalized = []
    for item in staged:
        for field, malformed, duplicate, required in (("premise_ref_ids", "missing_or_malformed_premise_ref_ids", "duplicate_premise_ref_id", True), ("hypothesis_context_ref_ids", "missing_or_malformed_hypothesis_context_ref_ids", "duplicate_hypothesis_context_ref_id", True), ("evidence_ref_ids", "malformed_evidence_ref_ids", "duplicate_evidence_ref_id", False)):
            if field == "evidence_ref_ids" and field not in item: continue
            refs, reason = _ref_list(item, field, malformed, duplicate, required)
            if reason: return None, reason
            item[field] = refs
        normalized.append(item)
    seen = set()
    for item in normalized:
        key = _semantic_item_key(item)
        if key in seen: return None, "duplicate_semantic_inference_item"
        seen.add(key)
    if any(i["situation_id"] != situation_id for i in normalized): return None, "inference_item_situation_mismatch"
    for item in normalized:
        if item["inference_kind"] in RELATIONAL_KINDS:
            if "object_ref_id" not in item: return None, "missing_object_ref_id_for_relational_inference"
            if not _non_empty_string(item["object_ref_id"]): return None, "malformed_object_ref_id"
            if item["object_ref_id"] == item["subject_ref_id"]: return None, "identical_inference_refs"
        elif "object_ref_id" in item: return None, "unexpected_object_ref_id_for_unary_inference"
    for item in normalized:
        if item["candidate_role"] not in KIND_ROLE_COMPATIBILITY[item["inference_kind"]]: return None, "incompatible_inference_kind_candidate_role"
    for item in normalized:
        if item["derivation_source_class"] not in KIND_SOURCE_COMPATIBILITY[item["inference_kind"]]: return None, "incompatible_inference_kind_derivation_source"
    return sorted(normalized, key=_canonical_sort_key), None

def _type_reason(inference_type: str, items: list) -> Optional[str]:
    if any(i["inference_kind"] not in TYPE_KIND_COMPATIBILITY[inference_type] for i in items): return "incompatible_inference_type_kind"
    roles = [i["candidate_role"] for i in items]
    dims = {(i["inference_kind"], i["candidate_role"], i["derivation_source_class"]) for i in items}
    if inference_type == "competing_inference_candidate":
        if len(items) < 2: return "competing_inference_requires_multiple_items"
        if "focal_conclusion_candidate" not in roles: return "competing_inference_missing_focal"
        if "alternative_conclusion_candidate" not in roles and "counter_conclusion_candidate" not in roles: return "competing_inference_missing_alternative_or_counter"
        if len(dims) < 2: return "competing_inference_requires_multiple_items"
    if inference_type == "mixed_unknown_inference_candidate":
        if len(items) < 2: return "mixed_unknown_inference_requires_multiple_items"
        if len({i["inference_kind"] for i in items}) < 2 and len({i["candidate_role"] for i in items}) < 2 and len({i["derivation_source_class"] for i in items}) < 2: return "mixed_unknown_inference_requires_distinct_dimensions"
    return None

def _validate_boundary(metadata: dict, inference_type: str):
    if "inference_boundary_classification" in metadata:
        value = metadata["inference_boundary_classification"]
        if not _non_empty_string(value): return None, "malformed_inference_boundary_class"
    else: value = BOUNDARY_DEFAULTS[inference_type]
    if value not in SUPPORTED_BOUNDARIES: return None, "unknown_inference_boundary_class"
    return value, None

def _validate_confidence(metadata: dict):
    if "inference_confidence_state" in metadata:
        value = metadata["inference_confidence_state"]
        if not _non_empty_string(value): return None, "malformed_inference_confidence_state"
    else: value = "inference_unverified"
    if value not in SUPPORTED_CONFIDENCE: return None, "unknown_inference_confidence_state"
    return value, None

def _build_valid(inference_type: str, situation_id: str, items: list, metadata: dict, boundary: str, confidence: str) -> Dict[str, Any]:
    focal = sum(1 for i in items if i["candidate_role"] == "focal_conclusion_candidate"); alternative = sum(1 for i in items if i["candidate_role"] == "alternative_conclusion_candidate"); counter = sum(1 for i in items if i["candidate_role"] == "counter_conclusion_candidate"); neutral = sum(1 for i in items if i["candidate_role"] == "neutral_unknown")
    premise_refs = sorted({r for i in items for r in i["premise_ref_ids"]}); hypo_refs = sorted({r for i in items for r in i["hypothesis_context_ref_ids"]}); evidence_refs = sorted({r for i in items for r in i.get("evidence_ref_ids", [])})
    flags = ["candidate_only", "inference_unverified"]
    integrity = ["inference_integrity_review_required", "basis_review_required", "competition_review_required", "resolution_review_required"]
    if confidence == "inference_competing" or (focal and (alternative or counter)): flags.append("competition_review_required")
    if confidence == "inference_origin_unknown" or inference_type == "unknown_origin_inference_candidate": flags.append("origin_unknown"); integrity.append("origin_unknown_review_required")
    basis = {"schema_version": VERSION, "inference_type": inference_type, "inference_boundary_classification": boundary, "inference_confidence_state": confidence, "situation_id": situation_id, "inference_items": items, "metadata": metadata}
    blob = json.dumps(basis, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = _base_payload(); payload.update({
        "situation_inference_context_passed": True, "situation_inference_context_status": "VALIDATED", "inference_context_id": hashlib.sha256(blob.encode("utf-8")).hexdigest(), "canonical_id_algorithm": CANONICAL_ID_ALGORITHM,
        "inference_type": inference_type, "inference_boundary_classification": boundary, "inference_confidence_state": confidence, "situation_id": situation_id, "inference_items": items, "metadata": metadata,
        "origin_summary": {"origin_kind": "internal_virtual_inference_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "external_inference_fact_verified": False, "inference_fact_asserted": False, "inference_accepted": False, "inference_rejected": False, "conclusion_asserted": False, "truth_resolved": False},
        "inference_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "inference_kinds": sorted({i["inference_kind"] for i in items}), "candidate_roles": sorted({i["candidate_role"] for i in items}), "derivation_source_classes": sorted({i["derivation_source_class"] for i in items}), "conclusion_ref_ids": sorted({i["conclusion_ref_id"] for i in items}), "item_count": len(items)},
        "inference_competition_summary": {"focal_item_count": focal, "alternative_item_count": alternative, "counter_item_count": counter, "neutral_item_count": neutral, "competition_present": bool(focal and (alternative or counter)), "winner_selected": False},
        "inference_support_summary": {"premise_ref_ids": premise_refs, "premise_ref_count": len(premise_refs), "hypothesis_context_ref_ids": hypo_refs, "hypothesis_context_ref_count": len(hypo_refs), "evidence_ref_ids": evidence_refs, "evidence_ref_count": len(evidence_refs), "external_basis_verified": False, "basis_sufficient_asserted": False},
        "resolution_summary": {"resolution_candidate_only": True, "resolved": False, "accepted": False, "rejected": False, "winner_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False},
        "inference_flags": sorted(set(flags)), "boundary_flags": [boundary], "inference_integrity_flags": sorted(set(integrity)), "candidate_only_fields": ["inference_type", "inference_items", "inference_boundary_classification", "inference_confidence_state"],
        "blocked_reasons": [], "warnings": ["mixed_virtual_external_inference_boundary_review_required"] if inference_type == "mixed_unknown_inference_candidate" else [],
    }); return payload

def build_virtual_world_situation_inference_context(inference_type=None, situation_id=None, inference_items=None, metadata=None):
    raw = {"inference_type": inference_type, "situation_id": situation_id, "inference_items": inference_items, "metadata": metadata if metadata is not None else {}}
    try:
        if _assert_json_native(raw): return _reject("non_json_serializable_semantic_input")
        if inference_type is None or not _non_empty_string(inference_type): return _reject("missing_inference_type")
        if inference_type not in SUPPORTED_INFERENCE_TYPES: return _reject("unknown_inference_type")
        if not _non_empty_string(situation_id): return _reject("missing_or_malformed_situation_id")
        if metadata is None: metadata = {}
        if type(metadata) is not dict: return _reject("invalid_metadata")
        metadata = _json_clone(metadata)
        reason = _forbidden_reason(metadata)
        if reason: return _reject(reason)
        items, reason = _normalize_items(inference_items, situation_id)
        if reason: return _reject(reason)
        reason = _type_reason(inference_type, items)
        if reason: return _reject(reason)
        boundary, reason = _validate_boundary(metadata, inference_type)
        if reason: return _reject(reason)
        confidence, reason = _validate_confidence(metadata)
        if reason: return _reject(reason)
        return _build_valid(inference_type, situation_id, items, metadata, boundary, confidence)
    except Exception:
        return _reject("non_json_serializable_semantic_input")

def validate_virtual_world_situation_inference_context(inference_context) -> bool:
    try:
        if _assert_json_native(inference_context) or type(inference_context) is not dict: return False
        if set(inference_context.keys()) != TOP_KEYS: return False
        if inference_context.get("situation_inference_context_passed") is not True or inference_context.get("situation_inference_context_status") != "VALIDATED": return False
        expected = build_virtual_world_situation_inference_context(inference_context.get("inference_type"), inference_context.get("situation_id"), inference_context.get("inference_items"), inference_context.get("metadata"))
        return _type_exact_equal(expected, inference_context)
    except Exception:
        return False

def _plan(plan_type: str, source: Any) -> Dict[str, Any]:
    ready = validate_virtual_world_situation_inference_context(source)
    plan = {"plan_type": plan_type, "source_inference_context_id": source.get("inference_context_id") if type(source) is dict and ready else None, "ready": ready, "candidate_only": True, "read_only": True, "inference_candidate_only": True}
    for key in PLAN_FALSE_FLAGS: plan[key] = False
    return plan

def build_inference_context_to_situation_plan(inference_context): return _plan("inference_context_to_situation", inference_context)
def build_inference_context_to_snapshot_plan(inference_context): return _plan("inference_context_to_snapshot", inference_context)
def build_inference_context_to_transition_preflight_plan(inference_context): return _plan("inference_context_to_transition_preflight", inference_context)
def build_inference_context_to_memory_candidate_plan(inference_context): return _plan("inference_context_to_memory_candidate", inference_context)
def build_inference_context_to_appraisal_plan(inference_context): return _plan("inference_context_to_appraisal", inference_context)
def build_inference_context_to_agp_input_plan(inference_context): return _plan("inference_context_to_agp_input", inference_context)

def build_virtual_world_situation_inference_context_schema_summary():
    return {"schema_version": VERSION, "schema_name": "read_only_virtual_world_situation_inference_context_schema", "supported_inference_types": list(SUPPORTED_INFERENCE_TYPES), "supported_inference_kinds": list(SUPPORTED_INFERENCE_KINDS), "supported_candidate_roles": list(SUPPORTED_CANDIDATE_ROLES), "supported_derivation_source_classes": list(SUPPORTED_DERIVATION_SOURCE_CLASSES), "type_kind_compatibility": {k: sorted(v) for k, v in sorted(TYPE_KIND_COMPATIBILITY.items())}, "kind_role_compatibility": {k: sorted(v) for k, v in sorted(KIND_ROLE_COMPATIBILITY.items())}, "kind_source_compatibility": {k: sorted(v) for k, v in sorted(KIND_SOURCE_COMPATIBILITY.items())}, "relational_inference_kinds": sorted(RELATIONAL_KINDS), "supported_boundary_classes": list(SUPPORTED_BOUNDARIES), "supported_confidence_states": list(SUPPORTED_CONFIDENCE), "canonical_id_algorithm": CANONICAL_ID_ALGORITHM, "read_only": True, "candidate_only": True, "next_recommended_step": "read_only_virtual_world_situation_conclusion_candidate_schema", "permanent_validation_contract": ["candidate inference is not fact or conclusion", "literal enums and compatibility matrices", "item and reference list schemas", "relational and unary reference shape", "semantic duplicate identity", "situation coherence", "competing and mixed composition", "boundary and confidence states", "nested summaries", "strict JSON", "recursive forbidden fields", "deterministic invalid ordering", "validation precedence", "deterministic canonical ID", "recursive type-exact integrity", "status coherence", "downstream read-only plans", "prohibited side effects", "fail-closed behavior"]}

def virtual_world_situation_inference_context_schema_summary():
    return build_virtual_world_situation_inference_context_schema_summary()
