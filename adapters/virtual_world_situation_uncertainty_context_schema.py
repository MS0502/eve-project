"""Deterministic read-only uncertainty-context candidates for virtual situations."""

import hashlib
import json
import math
from typing import Any, Dict, Optional

VERSION = "1.0.0-round1361-1380"
CANONICAL_ID_ALGORITHM = "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256"
SUPPORTED_UNCERTAINTY_TYPES = (
    "missing_information_candidate", "conflicting_evidence_candidate", "source_reliability_uncertainty_candidate",
    "temporal_uncertainty_candidate", "causal_uncertainty_candidate", "state_uncertainty_candidate",
    "identity_uncertainty_candidate", "boundary_uncertainty_candidate", "stochastic_uncertainty_candidate",
    "ambiguity_candidate", "simulation_uncertainty_candidate", "symbolic_uncertainty_candidate",
    "dmn_uncertainty_candidate", "dream_uncertainty_candidate", "unknown_origin_uncertainty_candidate",
    "mixed_unknown_uncertainty_candidate",
)
SUPPORTED_FACTOR_KINDS = (
    "missing_information", "supporting_evidence", "challenging_evidence", "neutral_unknown", "source_unverified",
    "temporal_ambiguity", "causal_ambiguity", "state_ambiguity", "identity_ambiguity", "boundary_ambiguity",
    "stochastic_variation", "simulation_divergence", "symbolic_ambiguity", "dmn_ambiguity", "dream_ambiguity", "unknown_origin",
)
SUPPORTED_POLARITIES = ("support_candidate", "challenge_candidate", "neutral_unknown")
TYPE_FACTOR_COMPATIBILITY = {
    "missing_information_candidate": {"missing_information"},
    "conflicting_evidence_candidate": {"supporting_evidence", "challenging_evidence"},
    "source_reliability_uncertainty_candidate": {"source_unverified"},
    "temporal_uncertainty_candidate": {"temporal_ambiguity"},
    "causal_uncertainty_candidate": {"causal_ambiguity"},
    "state_uncertainty_candidate": {"state_ambiguity"},
    "identity_uncertainty_candidate": {"identity_ambiguity"},
    "boundary_uncertainty_candidate": {"boundary_ambiguity"},
    "stochastic_uncertainty_candidate": {"stochastic_variation"},
    "ambiguity_candidate": {"neutral_unknown", "temporal_ambiguity", "causal_ambiguity", "state_ambiguity", "identity_ambiguity", "boundary_ambiguity", "symbolic_ambiguity"},
    "simulation_uncertainty_candidate": {"simulation_divergence"},
    "symbolic_uncertainty_candidate": {"symbolic_ambiguity"},
    "dmn_uncertainty_candidate": {"dmn_ambiguity"},
    "dream_uncertainty_candidate": {"dream_ambiguity"},
    "unknown_origin_uncertainty_candidate": {"unknown_origin"},
    "mixed_unknown_uncertainty_candidate": {"missing_information", "neutral_unknown", "source_unverified", "temporal_ambiguity", "causal_ambiguity", "state_ambiguity", "identity_ambiguity", "boundary_ambiguity", "stochastic_variation", "simulation_divergence", "symbolic_ambiguity", "dmn_ambiguity", "dream_ambiguity", "unknown_origin"},
}
KIND_POLARITY = {"supporting_evidence": "support_candidate", "challenging_evidence": "challenge_candidate"}
SUPPORTED_BOUNDARIES = (
    "internal_epistemic_uncertainty", "internal_stochastic_uncertainty", "source_provenance_uncertainty",
    "temporal_virtual_uncertainty", "causal_virtual_uncertainty", "state_virtual_uncertainty", "identity_virtual_uncertainty",
    "symbolic_virtual_uncertainty", "dmn_virtual_uncertainty", "simulated_virtual_uncertainty", "dream_virtual_uncertainty",
    "mixed_virtual_external_uncertainty_boundary",
)
BOUNDARY_DEFAULTS = {
    "source_reliability_uncertainty_candidate": "source_provenance_uncertainty",
    "temporal_uncertainty_candidate": "temporal_virtual_uncertainty",
    "causal_uncertainty_candidate": "causal_virtual_uncertainty",
    "state_uncertainty_candidate": "state_virtual_uncertainty",
    "identity_uncertainty_candidate": "identity_virtual_uncertainty",
    "stochastic_uncertainty_candidate": "internal_stochastic_uncertainty",
    "simulation_uncertainty_candidate": "simulated_virtual_uncertainty",
    "symbolic_uncertainty_candidate": "symbolic_virtual_uncertainty",
    "dmn_uncertainty_candidate": "dmn_virtual_uncertainty",
    "dream_uncertainty_candidate": "dream_virtual_uncertainty",
    "mixed_unknown_uncertainty_candidate": "mixed_virtual_external_uncertainty_boundary",
}
SUPPORTED_CONFIDENCE = (
    "uncertainty_unverified", "uncertainty_low_confidence", "uncertainty_medium_confidence",
    "uncertainty_high_confidence_but_not_fact", "uncertainty_conflict_detected", "uncertainty_origin_unknown",
)
IMMUTABLE_FALSE_FLAGS = (
    "external_uncertainty_asserted", "uncertainty_fact_asserted", "external_evidence_verified", "resolved_truth_asserted",
    "uncertainty_resolved", "external_resolution_verified", "automatic_resolution_performed", "automatic_resolution_allowed",
    "prediction_guaranteed", "future_outcome_asserted", "random_sampling_performed", "probability_draw_performed",
    "intervention_performed", "event_completion_asserted", "identity_asserted", "memory_fact_asserted", "memory_write_performed",
    "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed",
    "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed",
    "tool_execution_performed", "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed",
    "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "artifact_created_or_staged",
    "agp_bypass_allowed", "fallback_bypass_allowed",
)
IMMUTABLE_TRUE_FLAGS = (
    "uncertainty_candidate_only", "situation_uncertainty_only", "read_only", "situation_review_required", "snapshot_review_required",
    "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required",
    "uncertainty_integrity_review_required", "appraisal_required", "agp_input_required",
)
FORBIDDEN_REQUEST_FIELDS = {
    "external_uncertainty_assertion_requested", "uncertainty_fact_assertion_requested", "resolved_truth_assertion_requested",
    "uncertainty_resolution_requested", "automatic_resolution_requested", "conflict_auto_resolution_requested",
    "certainty_promotion_requested", "fact_promotion_requested", "external_evidence_verification_requested",
    "source_trust_promotion_requested", "identity_resolution_requested", "boundary_resolution_requested",
    "prediction_guarantee_requested", "future_outcome_assertion_requested", "random_sampling_requested", "probability_draw_requested",
    "memory_write_requested", "memory_fact_promotion_requested", "relationship_update_requested", "self_model_update_requested",
    "affect_transition_requested", "hormone_transition_requested", "world_state_mutation_requested", "transition_application_requested",
    "planning_execution_requested", "tool_execution_requested", "runtime_mutation_requested", "persistence_write_requested",
    "vector_read_requested", "vector_load_requested", "model_load_requested", "device_activation_requested", "network_action_requested",
    "artifact_creation_requested", "intervention_requested", "event_completion_assertion_requested", "agp_bypass_requested",
    "fallback_bypass_requested", "schedule_requested", "scheduler_requested", "timer_requested", "alarm_requested", "calendar_requested",
    "reminder_requested", "deadline_requested", "cron_requested", "cron_expression", "scheduled_execution_requested",
    "delayed_execution_requested", "act_requested", "execute_requested", "apply_requested", "commit_outcome_requested", "promote_to_fact_requested",
}
TOP_KEYS = {"situation_uncertainty_context_passed", "situation_uncertainty_context_status", "uncertainty_context_id", "canonical_id_algorithm", "uncertainty_type", "uncertainty_boundary_classification", "uncertainty_confidence_state", "situation_id", "uncertainty_factors", "metadata", "origin_summary", "fact_status_summary", "uncertainty_scope_summary", "evidence_balance_summary", "resolution_summary", "uncertainty_flags", "boundary_flags", "uncertainty_integrity_flags", "candidate_only_fields", "blocked_reasons", "warnings", *IMMUTABLE_FALSE_FLAGS, *IMMUTABLE_TRUE_FLAGS}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _assert_json_native(value: Any, seen: Optional[set] = None, depth: int = 0) -> Optional[str]:
    if depth > 100:
        return "non_json_serializable_semantic_input"
    if value is None or isinstance(value, (str, bool)):
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else "non_json_serializable_semantic_input"
    if seen is None:
        seen = set()
    if isinstance(value, dict):
        oid = id(value)
        if oid in seen:
            return "non_json_serializable_semantic_input"
        seen.add(oid)
        for key, item in value.items():
            if not isinstance(key, str):
                return "non_json_serializable_semantic_input"
            reason = _assert_json_native(item, seen, depth + 1)
            if reason:
                return reason
        seen.remove(oid)
        return None
    if isinstance(value, list):
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


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def _forbidden_reason(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_REQUEST_FIELDS:
                if item is True:
                    return key
                if item is not False:
                    return "malformed_forbidden_request_field"
            reason = _forbidden_reason(item)
            if reason:
                return reason
    elif isinstance(value, list):
        for item in value:
            reason = _forbidden_reason(item)
            if reason:
                return reason
    return None


def _reject(reason: str) -> Dict[str, Any]:
    payload = {key: False for key in IMMUTABLE_FALSE_FLAGS}
    payload.update({key: True for key in IMMUTABLE_TRUE_FLAGS})
    payload.update({
        "situation_uncertainty_context_passed": False, "situation_uncertainty_context_status": "REJECTED",
        "uncertainty_context_id": None, "canonical_id_algorithm": None, "uncertainty_type": None,
        "uncertainty_boundary_classification": None, "uncertainty_confidence_state": None, "situation_id": None,
        "uncertainty_factors": [], "metadata": {},
        "origin_summary": {"origin_kind": "internal_virtual_uncertainty_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "external_uncertainty_fact_verified": False, "uncertainty_fact_asserted": False, "resolved_truth_asserted": False, "prediction_guaranteed": False},
        "uncertainty_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "factor_kinds": []},
        "evidence_balance_summary": {"supporting_factor_count": 0, "challenging_factor_count": 0, "neutral_factor_count": 0, "conflict_present": False, "external_evidence_verified": False},
        "resolution_summary": {"resolution_candidate_only": True, "resolved": False, "external_resolution_verified": False, "automatic_resolution_allowed": False},
        "uncertainty_flags": [], "boundary_flags": [], "uncertainty_integrity_flags": [],
        "candidate_only_fields": ["uncertainty_type", "uncertainty_factors", "uncertainty_boundary_classification", "uncertainty_confidence_state"],
        "blocked_reasons": [reason], "warnings": [],
    })
    return payload


def _normalize_factors(factors: Any, situation_id: str):
    if factors is None:
        return None, "missing_uncertainty_factors"
    if not isinstance(factors, list):
        return None, "uncertainty_factors_not_non_empty_list"
    if not factors:
        return None, "empty_uncertainty_factors"
    normalized = []
    for factor in factors:
        if not isinstance(factor, dict):
            return None, "malformed_uncertainty_factor"
        for req in ("factor_id", "situation_id", "factor_kind", "polarity"):
            if req not in factor:
                return None, "missing_or_empty_factor_id" if req == "factor_id" else "malformed_uncertainty_factor"
        if not _non_empty_string(factor["factor_id"]):
            return None, "missing_or_empty_factor_id"
        if not _non_empty_string(factor["situation_id"]):
            return None, "malformed_factor_situation_id"
        if factor["factor_kind"] not in SUPPORTED_FACTOR_KINDS:
            return None, "unknown_factor_kind"
        if factor["polarity"] not in SUPPORTED_POLARITIES:
            return None, "unknown_factor_polarity"
        if "related_context_id" in factor and not _non_empty_string(factor["related_context_id"]):
            return None, "malformed_related_context_id"
        if "weight_candidate" in factor:
            w = factor["weight_candidate"]
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w):
                return None, "malformed_weight_candidate"
            if w < 0.0 or w > 1.0:
                return None, "weight_candidate_out_of_range"
        if "label" in factor and not _non_empty_string(factor["label"]):
            return None, "malformed_uncertainty_factor_label"
        normalized.append(_json_clone(factor))
    reason = _forbidden_reason(normalized)
    if reason:
        return None, reason
    ids = [f["factor_id"] for f in normalized]
    if len(ids) != len(set(ids)):
        return None, "duplicate_factor_id"
    seen_sem = set()
    for factor in normalized:
        sem = (factor["situation_id"], factor["factor_kind"], factor["polarity"], factor.get("related_context_id"))
        if sem in seen_sem:
            return None, "duplicate_semantic_uncertainty_factor"
        seen_sem.add(sem)
    if any(f["situation_id"] != situation_id for f in normalized):
        return None, "uncertainty_factor_situation_mismatch"
    for factor in normalized:
        expected = KIND_POLARITY.get(factor["factor_kind"], "neutral_unknown")
        if factor["polarity"] != expected:
            return None, "incompatible_factor_kind_polarity"
    return sorted(normalized, key=lambda f: json.dumps(f, sort_keys=True, ensure_ascii=False, separators=(",", ":"))), None


def _composition_reason(uncertainty_type: str, factors: list) -> Optional[str]:
    kinds = [f["factor_kind"] for f in factors]
    kind_set = set(kinds)
    if any(k not in TYPE_FACTOR_COMPATIBILITY[uncertainty_type] for k in kinds):
        return "incompatible_uncertainty_type_factor_kind"
    if uncertainty_type == "conflicting_evidence_candidate":
        if len(factors) < 2: return "conflicting_evidence_requires_multiple_factors"
        if "supporting_evidence" not in kind_set: return "conflicting_evidence_missing_support"
        if "challenging_evidence" not in kind_set: return "conflicting_evidence_missing_challenge"
    if uncertainty_type == "ambiguity_candidate":
        if len(factors) < 2: return "ambiguity_requires_multiple_factors"
        if len(kind_set) < 2: return "ambiguity_requires_distinct_factor_kinds"
    if uncertainty_type == "mixed_unknown_uncertainty_candidate":
        if len(factors) < 2: return "mixed_unknown_requires_multiple_factors"
        if len(kind_set) < 2: return "mixed_unknown_requires_distinct_factor_kinds"
    return None


def _validate_boundary(metadata: dict, uncertainty_type: str):
    value = metadata.get("uncertainty_boundary_classification", BOUNDARY_DEFAULTS.get(uncertainty_type, "internal_epistemic_uncertainty"))
    if not _non_empty_string(value) or value in {"None", "False", "0", "[]", "{}"}:
        return None, "malformed_uncertainty_boundary_class"
    if value not in SUPPORTED_BOUNDARIES:
        return None, "unknown_uncertainty_boundary_class"
    return value, None


def _validate_confidence(metadata: dict):
    value = metadata.get("uncertainty_confidence_state", "uncertainty_unverified")
    if not _non_empty_string(value) or value in {"None", "False", "0", "[]", "{}"}:
        return None, "malformed_uncertainty_confidence_state"
    if value not in SUPPORTED_CONFIDENCE:
        return None, "unknown_uncertainty_confidence_state"
    return value, None


def _build_valid(uncertainty_type: str, situation_id: str, factors: list, metadata: dict, boundary: str, confidence: str) -> Dict[str, Any]:
    support = sum(1 for f in factors if f["polarity"] == "support_candidate")
    challenge = sum(1 for f in factors if f["polarity"] == "challenge_candidate")
    neutral = sum(1 for f in factors if f["polarity"] == "neutral_unknown")
    flags = ["uncertainty_unresolved", "candidate_only"]
    integrity = ["uncertainty_integrity_review_required"]
    boundary_flags = [boundary]
    warnings = []
    if support and challenge or confidence == "uncertainty_conflict_detected":
        flags.append("conflict_uncertainty_present"); integrity.append("conflict_review_required")
    if confidence == "uncertainty_origin_unknown" or uncertainty_type == "unknown_origin_uncertainty_candidate":
        flags.append("origin_unknown"); integrity.append("origin_unknown_review_required")
    if uncertainty_type == "mixed_unknown_uncertainty_candidate":
        warnings.append("mixed_virtual_external_uncertainty_boundary_review_required")
    basis = {"schema_version": VERSION, "uncertainty_type": uncertainty_type, "uncertainty_boundary_classification": boundary, "uncertainty_confidence_state": confidence, "situation_id": situation_id, "uncertainty_factors": factors, "metadata": metadata}
    blob = json.dumps(basis, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    uid = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    payload = {key: False for key in IMMUTABLE_FALSE_FLAGS}
    payload.update({key: True for key in IMMUTABLE_TRUE_FLAGS})
    payload.update({
        "situation_uncertainty_context_passed": True, "situation_uncertainty_context_status": "VALIDATED",
        "uncertainty_context_id": uid, "canonical_id_algorithm": CANONICAL_ID_ALGORITHM,
        "uncertainty_type": uncertainty_type, "uncertainty_boundary_classification": boundary,
        "uncertainty_confidence_state": confidence, "situation_id": situation_id,
        "uncertainty_factors": factors, "metadata": metadata,
        "origin_summary": {"origin_kind": "internal_virtual_uncertainty_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "external_uncertainty_fact_verified": False, "uncertainty_fact_asserted": False, "resolved_truth_asserted": False, "prediction_guaranteed": False},
        "uncertainty_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "factor_kinds": sorted(set(f["factor_kind"] for f in factors))},
        "evidence_balance_summary": {"supporting_factor_count": support, "challenging_factor_count": challenge, "neutral_factor_count": neutral, "conflict_present": bool(support and challenge), "external_evidence_verified": False},
        "resolution_summary": {"resolution_candidate_only": True, "resolved": False, "external_resolution_verified": False, "automatic_resolution_allowed": False},
        "uncertainty_flags": sorted(set(flags)), "boundary_flags": sorted(set(boundary_flags)),
        "uncertainty_integrity_flags": sorted(set(integrity)),
        "candidate_only_fields": ["uncertainty_type", "uncertainty_factors", "uncertainty_boundary_classification", "uncertainty_confidence_state"],
        "blocked_reasons": [], "warnings": warnings,
    })
    return payload


def build_virtual_world_situation_uncertainty_context(uncertainty_type=None, situation_id=None, uncertainty_factors=None, metadata=None):
    raw = {"uncertainty_type": uncertainty_type, "situation_id": situation_id, "uncertainty_factors": uncertainty_factors, "metadata": metadata if metadata is not None else {}}
    if _assert_json_native(raw):
        return _reject("non_json_serializable_semantic_input")
    if uncertainty_type is None or not _non_empty_string(uncertainty_type):
        return _reject("missing_uncertainty_type")
    if uncertainty_type not in SUPPORTED_UNCERTAINTY_TYPES:
        return _reject("unknown_uncertainty_type")
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
    factors, reason = _normalize_factors(uncertainty_factors, situation_id)
    if reason:
        return _reject(reason)
    reason = _composition_reason(uncertainty_type, factors)
    if reason:
        return _reject(reason)
    boundary, reason = _validate_boundary(metadata, uncertainty_type)
    if reason:
        return _reject(reason)
    confidence, reason = _validate_confidence(metadata)
    if reason:
        return _reject(reason)
    return _build_valid(uncertainty_type, situation_id, factors, metadata, boundary, confidence)


def validate_virtual_world_situation_uncertainty_context(uncertainty_context) -> bool:
    try:
        if _assert_json_native(uncertainty_context) or not isinstance(uncertainty_context, dict):
            return False
        if set(uncertainty_context.keys()) != TOP_KEYS:
            return False
        if not uncertainty_context.get("situation_uncertainty_context_passed") or uncertainty_context.get("situation_uncertainty_context_status") != "VALIDATED":
            return False
        expected = build_virtual_world_situation_uncertainty_context(
            uncertainty_context.get("uncertainty_type"), uncertainty_context.get("situation_id"),
            uncertainty_context.get("uncertainty_factors"), uncertainty_context.get("metadata"),
        )
        return expected == uncertainty_context
    except Exception:
        return False


def _plan(plan_type: str, source: Any) -> Dict[str, Any]:
    ready = validate_virtual_world_situation_uncertainty_context(source)
    plan = {"plan_type": plan_type, "source_uncertainty_context_id": source.get("uncertainty_context_id") if isinstance(source, dict) and ready else None, "ready": ready, "candidate_only": True, "read_only": True, "uncertainty_candidate_only": True}
    for key in ("external_uncertainty_asserted", "uncertainty_fact_asserted", "uncertainty_resolved", "automatic_resolution_performed", "random_sampling_performed", "probability_draw_performed", "prediction_guaranteed", "memory_write_performed", "relationship_update_performed", "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed", "world_state_mutation_performed", "transition_applied", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "planning_execution_performed", "tool_execution_performed", "agp_bypass_allowed", "fallback_bypass_allowed"):
        plan[key] = False
    return plan


def build_uncertainty_context_to_situation_plan(uncertainty_context): return _plan("uncertainty_context_to_situation", uncertainty_context)
def build_uncertainty_context_to_snapshot_plan(uncertainty_context): return _plan("uncertainty_context_to_snapshot", uncertainty_context)
def build_uncertainty_context_to_transition_preflight_plan(uncertainty_context): return _plan("uncertainty_context_to_transition_preflight", uncertainty_context)
def build_uncertainty_context_to_memory_candidate_plan(uncertainty_context): return _plan("uncertainty_context_to_memory_candidate", uncertainty_context)
def build_uncertainty_context_to_appraisal_plan(uncertainty_context): return _plan("uncertainty_context_to_appraisal", uncertainty_context)
def build_uncertainty_context_to_agp_input_plan(uncertainty_context): return _plan("uncertainty_context_to_agp_input", uncertainty_context)


def build_virtual_world_situation_uncertainty_context_schema_summary():
    return {"schema_version": VERSION, "supported_uncertainty_types": list(SUPPORTED_UNCERTAINTY_TYPES), "supported_factor_kinds": list(SUPPORTED_FACTOR_KINDS), "supported_polarities": list(SUPPORTED_POLARITIES), "supported_boundary_classes": list(SUPPORTED_BOUNDARIES), "supported_confidence_states": list(SUPPORTED_CONFIDENCE), "canonical_id_algorithm": CANONICAL_ID_ALGORITHM, "read_only": True, "candidate_only": True, "next_recommended_step": "read_only_virtual_world_situation_constraint_context_schema"}


def virtual_world_situation_uncertainty_context_schema_summary():
    return build_virtual_world_situation_uncertainty_context_schema_summary()
