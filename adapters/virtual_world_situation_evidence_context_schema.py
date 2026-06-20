"""Deterministic read-only evidence-context candidates for virtual situations."""

import hashlib
import json
import math
from typing import Any, Dict, Optional

VERSION = "1.0.0-round1401-1420"
CANONICAL_ID_ALGORITHM = "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256"
SUPPORTED_EVIDENCE_TYPES = (
    "supporting_evidence_candidate", "challenging_evidence_candidate", "conflicting_evidence_candidate",
    "neutral_evidence_candidate", "source_provenance_evidence_candidate", "temporal_evidence_candidate",
    "causal_evidence_candidate", "state_evidence_candidate", "identity_evidence_candidate",
    "boundary_evidence_candidate", "simulation_evidence_candidate", "symbolic_evidence_candidate",
    "dmn_evidence_candidate", "dream_evidence_candidate", "tool_state_evidence_candidate",
    "unknown_origin_evidence_candidate", "mixed_unknown_evidence_candidate",
)
SUPPORTED_ITEM_KINDS = (
    "direct_observation_candidate", "indirect_observation_candidate", "supporting_signal_candidate",
    "challenging_signal_candidate", "neutral_signal_candidate", "source_provenance_candidate",
    "temporal_trace_candidate", "causal_trace_candidate", "state_trace_candidate", "identity_trace_candidate",
    "boundary_trace_candidate", "simulation_trace_candidate", "symbolic_trace_candidate", "dmn_trace_candidate",
    "dream_trace_candidate", "tool_state_trace_candidate", "unknown_origin_trace_candidate",
)
SUPPORTED_STANCES = ("supports_candidate", "challenges_candidate", "neutral_unknown")
SUPPORTED_SOURCE_CLASSES = (
    "internal_virtual_observation", "internal_memory_replay", "internal_imagination", "internal_simulation",
    "internal_symbolic", "internal_dmn", "internal_dream", "internal_tool_state",
    "external_unverified_candidate", "unknown_origin_candidate",
)
TYPE_ITEM_COMPATIBILITY = {
    "supporting_evidence_candidate": {"direct_observation_candidate", "indirect_observation_candidate", "supporting_signal_candidate"},
    "challenging_evidence_candidate": {"direct_observation_candidate", "indirect_observation_candidate", "challenging_signal_candidate"},
    "conflicting_evidence_candidate": {"direct_observation_candidate", "indirect_observation_candidate", "supporting_signal_candidate", "challenging_signal_candidate", "neutral_signal_candidate"},
    "neutral_evidence_candidate": {"direct_observation_candidate", "indirect_observation_candidate", "neutral_signal_candidate"},
    "source_provenance_evidence_candidate": {"source_provenance_candidate"},
    "temporal_evidence_candidate": {"temporal_trace_candidate"},
    "causal_evidence_candidate": {"causal_trace_candidate"},
    "state_evidence_candidate": {"state_trace_candidate"},
    "identity_evidence_candidate": {"identity_trace_candidate"},
    "boundary_evidence_candidate": {"boundary_trace_candidate"},
    "simulation_evidence_candidate": {"simulation_trace_candidate"},
    "symbolic_evidence_candidate": {"symbolic_trace_candidate"},
    "dmn_evidence_candidate": {"dmn_trace_candidate"},
    "dream_evidence_candidate": {"dream_trace_candidate"},
    "tool_state_evidence_candidate": {"tool_state_trace_candidate"},
    "unknown_origin_evidence_candidate": {"unknown_origin_trace_candidate"},
    "mixed_unknown_evidence_candidate": set(SUPPORTED_ITEM_KINDS),
}
KIND_STANCE_COMPATIBILITY = {
    "direct_observation_candidate": set(SUPPORTED_STANCES), "indirect_observation_candidate": set(SUPPORTED_STANCES),
    "supporting_signal_candidate": {"supports_candidate"}, "challenging_signal_candidate": {"challenges_candidate"},
    "neutral_signal_candidate": {"neutral_unknown"}, "source_provenance_candidate": {"neutral_unknown"},
    "temporal_trace_candidate": {"neutral_unknown"}, "causal_trace_candidate": {"neutral_unknown"},
    "state_trace_candidate": {"neutral_unknown"}, "identity_trace_candidate": {"neutral_unknown"},
    "boundary_trace_candidate": {"neutral_unknown"}, "simulation_trace_candidate": {"neutral_unknown"},
    "symbolic_trace_candidate": {"neutral_unknown"}, "dmn_trace_candidate": {"neutral_unknown"},
    "dream_trace_candidate": {"neutral_unknown"}, "tool_state_trace_candidate": {"neutral_unknown"},
    "unknown_origin_trace_candidate": {"neutral_unknown"},
}
_ALL_SIGNAL_SOURCES = set(SUPPORTED_SOURCE_CLASSES)
_VIRT_MEM_SIM_EXT_UNK = {"internal_virtual_observation", "internal_memory_replay", "internal_simulation", "external_unverified_candidate", "unknown_origin_candidate"}
KIND_SOURCE_COMPATIBILITY = {
    "direct_observation_candidate": {"internal_virtual_observation", "internal_tool_state", "external_unverified_candidate"},
    "indirect_observation_candidate": {"internal_memory_replay", "internal_imagination", "internal_simulation", "external_unverified_candidate", "unknown_origin_candidate"},
    "supporting_signal_candidate": _ALL_SIGNAL_SOURCES, "challenging_signal_candidate": _ALL_SIGNAL_SOURCES,
    "neutral_signal_candidate": _ALL_SIGNAL_SOURCES, "source_provenance_candidate": _ALL_SIGNAL_SOURCES,
    "temporal_trace_candidate": _VIRT_MEM_SIM_EXT_UNK, "causal_trace_candidate": _VIRT_MEM_SIM_EXT_UNK,
    "state_trace_candidate": {"internal_virtual_observation", "internal_memory_replay", "internal_simulation", "internal_tool_state", "external_unverified_candidate", "unknown_origin_candidate"},
    "identity_trace_candidate": _VIRT_MEM_SIM_EXT_UNK, "boundary_trace_candidate": _VIRT_MEM_SIM_EXT_UNK,
    "simulation_trace_candidate": {"internal_simulation"}, "symbolic_trace_candidate": {"internal_symbolic"},
    "dmn_trace_candidate": {"internal_dmn"}, "dream_trace_candidate": {"internal_dream"},
    "tool_state_trace_candidate": {"internal_tool_state"}, "unknown_origin_trace_candidate": {"unknown_origin_candidate"},
}
SUPPORTED_BOUNDARIES = (
    "internal_virtual_evidence", "memory_replay_evidence", "imagination_evidence", "simulation_evidence",
    "symbolic_evidence", "dmn_evidence", "dream_evidence", "tool_state_evidence",
    "external_unverified_evidence", "unknown_origin_evidence", "mixed_virtual_external_evidence_boundary",
)
BOUNDARY_DEFAULTS = {"simulation_evidence_candidate": "simulation_evidence", "symbolic_evidence_candidate": "symbolic_evidence", "dmn_evidence_candidate": "dmn_evidence", "dream_evidence_candidate": "dream_evidence", "tool_state_evidence_candidate": "tool_state_evidence", "unknown_origin_evidence_candidate": "unknown_origin_evidence", "mixed_unknown_evidence_candidate": "mixed_virtual_external_evidence_boundary"}
SUPPORTED_CONFIDENCE = ("evidence_unverified", "evidence_low_confidence", "evidence_medium_confidence", "evidence_high_confidence_but_not_fact", "evidence_conflict_detected", "evidence_origin_unknown")
IMMUTABLE_FALSE_FLAGS = (
    "external_evidence_asserted", "evidence_verified", "source_verified", "evidence_fact_asserted", "conclusion_asserted",
    "truth_resolved", "winner_selected", "evidence_conflict_resolved", "automatic_resolution_performed",
    "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted", "random_sampling_performed",
    "probability_draw_performed", "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed",
    "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed",
    "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed",
    "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed",
    "model_loaded", "device_activation_performed", "network_action_performed", "artifact_created_or_staged",
    "schedule_created", "action_prevented", "action_permitted", "agp_bypass_allowed", "fallback_bypass_allowed",
)
IMMUTABLE_TRUE_FLAGS = (
    "evidence_candidate_only", "situation_evidence_only", "read_only", "situation_review_required", "snapshot_review_required",
    "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required",
    "evidence_integrity_review_required", "provenance_review_required", "resolution_review_required", "appraisal_required",
    "agp_input_required",
)
FORBIDDEN_REQUEST_FIELDS = {
    "external_evidence_assertion_requested", "evidence_verification_requested", "source_verification_requested", "source_trust_promotion_requested", "evidence_fact_assertion_requested", "evidence_fact_promotion_requested", "conclusion_assertion_requested", "conclusion_selection_requested", "winner_selection_requested", "truth_resolution_requested", "evidence_conflict_resolution_requested", "automatic_resolution_requested", "certainty_promotion_requested", "fact_promotion_requested", "prediction_guarantee_requested", "future_outcome_assertion_requested", "random_sampling_requested", "probability_draw_requested", "memory_write_requested", "memory_fact_promotion_requested", "relationship_update_requested", "self_model_update_requested", "identity_update_requested", "affect_transition_requested", "hormone_transition_requested", "world_state_mutation_requested", "transition_application_requested", "planning_execution_requested", "tool_execution_requested", "runtime_mutation_requested", "persistence_write_requested", "vector_read_requested", "vector_load_requested", "model_load_requested", "device_activation_requested", "network_action_requested", "artifact_creation_requested", "intervention_requested", "event_completion_assertion_requested", "action_prevention_requested", "action_permission_requested", "agp_bypass_requested", "fallback_bypass_requested", "schedule_requested", "scheduler_requested", "timer_requested", "alarm_requested", "calendar_requested", "reminder_requested", "deadline_requested", "cron_requested", "cron_expression", "scheduled_execution_requested", "delayed_execution_requested", "act_requested", "execute_requested", "apply_requested", "commit_outcome_requested", "promote_to_fact_requested",
}
TOP_KEYS = {"situation_evidence_context_passed", "situation_evidence_context_status", "evidence_context_id", "canonical_id_algorithm", "evidence_type", "evidence_boundary_classification", "evidence_confidence_state", "situation_id", "evidence_items", "metadata", "origin_summary", "fact_status_summary", "evidence_scope_summary", "evidence_stance_summary", "evidence_provenance_summary", "resolution_summary", "evidence_flags", "boundary_flags", "evidence_integrity_flags", "candidate_only_fields", "blocked_reasons", "warnings", *IMMUTABLE_FALSE_FLAGS, *IMMUTABLE_TRUE_FLAGS}


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


def _semantic_item_key(item: Dict[str, Any]) -> str:
    return _canonical_sort_key({"situation_id": item["situation_id"], "evidence_kind": item["evidence_kind"], "stance": item["stance"], "source_class": item["source_class"], "source_ref_id": item["source_ref_id"], "claim_ref_id": item["claim_ref_id"], "related_context_id": item.get("related_context_id")})


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
        "situation_evidence_context_passed": False, "situation_evidence_context_status": "REJECTED", "evidence_context_id": None,
        "canonical_id_algorithm": None, "evidence_type": None, "evidence_boundary_classification": None,
        "evidence_confidence_state": None, "situation_id": None, "evidence_items": [], "metadata": {},
        "origin_summary": {"origin_kind": "internal_virtual_evidence_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "external_evidence_fact_verified": False, "evidence_fact_asserted": False, "conclusion_asserted": False, "truth_resolved": False},
        "evidence_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "evidence_kinds": [], "source_classes": [], "claim_ref_ids": [], "item_count": 0},
        "evidence_stance_summary": {"supporting_item_count": 0, "challenging_item_count": 0, "neutral_item_count": 0, "conflict_present": False, "winner_selected": False},
        "evidence_provenance_summary": {"source_ref_ids": [], "source_count": 0, "external_source_verified": False, "unknown_origin_present": False},
        "resolution_summary": {"resolution_candidate_only": True, "resolved": False, "conclusion_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False},
        "evidence_flags": [], "boundary_flags": [], "evidence_integrity_flags": [],
        "candidate_only_fields": ["evidence_type", "evidence_items", "evidence_boundary_classification", "evidence_confidence_state"],
        "blocked_reasons": [reason], "warnings": [],
    })
    return payload


def _normalize_items(items: Any, situation_id: str):
    if items is None:
        return None, "missing_evidence_items"
    if not isinstance(items, list):
        return None, "evidence_items_not_non_empty_list"
    if not items:
        return None, "empty_evidence_items"
    ordered = sorted(items, key=_canonical_sort_key)
    normalized = []
    for item in ordered:
        if not isinstance(item, dict):
            return None, "malformed_evidence_item"
        for req, reason in (("evidence_id", "missing_or_empty_evidence_id"), ("situation_id", "malformed_evidence_item_situation_id"), ("evidence_kind", "unknown_evidence_kind"), ("stance", "unknown_evidence_stance"), ("source_class", "unknown_source_class"), ("source_ref_id", "missing_or_malformed_source_ref_id"), ("claim_ref_id", "missing_or_malformed_claim_ref_id")):
            if req not in item:
                return None, reason
        if not _non_empty_string(item["evidence_id"]): return None, "missing_or_empty_evidence_id"
        if not _non_empty_string(item["situation_id"]): return None, "malformed_evidence_item_situation_id"
        if item["evidence_kind"] not in SUPPORTED_ITEM_KINDS: return None, "unknown_evidence_kind"
        if item["stance"] not in SUPPORTED_STANCES: return None, "unknown_evidence_stance"
        if item["source_class"] not in SUPPORTED_SOURCE_CLASSES: return None, "unknown_source_class"
        if not _non_empty_string(item["source_ref_id"]): return None, "missing_or_malformed_source_ref_id"
        if not _non_empty_string(item["claim_ref_id"]): return None, "missing_or_malformed_claim_ref_id"
        if "related_context_id" in item and not _non_empty_string(item["related_context_id"]): return None, "malformed_related_context_id"
        for field, malformed, out_range in (("confidence_candidate", "malformed_confidence_candidate", "confidence_candidate_out_of_range"), ("weight_candidate", "malformed_weight_candidate", "weight_candidate_out_of_range")):
            if field in item:
                value = item[field]
                if type(value) is bool or type(value) not in {int, float}:
                    return None, malformed
                if type(value) is float and not math.isfinite(value):
                    return None, malformed
                if value < 0 or value > 1:
                    return None, out_range
        if "label" in item and not _non_empty_string(item["label"]): return None, "malformed_evidence_item_label"
        normalized.append(_json_clone(item))
    reason = _forbidden_reason(normalized)
    if reason:
        return None, reason
    ids = [i["evidence_id"] for i in normalized]
    if len(ids) != len(set(ids)):
        return None, "duplicate_evidence_id"
    seen = set()
    for item in normalized:
        key = _semantic_item_key(item)
        if key in seen:
            return None, "duplicate_semantic_evidence_item"
        seen.add(key)
    if any(i["situation_id"] != situation_id for i in normalized):
        return None, "evidence_item_situation_mismatch"
    for item in normalized:
        if item["stance"] not in KIND_STANCE_COMPATIBILITY[item["evidence_kind"]]:
            return None, "incompatible_evidence_item_kind_stance"
    for item in normalized:
        if item["source_class"] not in KIND_SOURCE_COMPATIBILITY[item["evidence_kind"]]:
            return None, "incompatible_evidence_item_kind_source_class"
    return sorted(normalized, key=_canonical_sort_key), None


def _type_reason(evidence_type: str, items: list) -> Optional[str]:
    if any(i["evidence_kind"] not in TYPE_ITEM_COMPATIBILITY[evidence_type] for i in items):
        return "incompatible_evidence_type_item_kind"
    stances = [i["stance"] for i in items]
    kinds = {i["evidence_kind"] for i in items}
    sources = {i["source_class"] for i in items}
    if evidence_type == "supporting_evidence_candidate" and (any(s != "supports_candidate" for s in stances) or "supports_candidate" not in stances): return "supporting_evidence_requires_support"
    if evidence_type == "challenging_evidence_candidate" and (any(s != "challenges_candidate" for s in stances) or "challenges_candidate" not in stances): return "challenging_evidence_requires_challenge"
    if evidence_type == "conflicting_evidence_candidate":
        if len(items) < 2: return "conflicting_evidence_requires_multiple_items"
        if "supports_candidate" not in stances: return "conflicting_evidence_missing_support"
        if "challenges_candidate" not in stances: return "conflicting_evidence_missing_challenge"
    if evidence_type == "neutral_evidence_candidate" and any(s != "neutral_unknown" for s in stances): return "neutral_evidence_requires_neutral_items"
    required_kind = {"source_provenance_evidence_candidate":"source_provenance_candidate", "temporal_evidence_candidate":"temporal_trace_candidate", "causal_evidence_candidate":"causal_trace_candidate", "state_evidence_candidate":"state_trace_candidate", "identity_evidence_candidate":"identity_trace_candidate", "boundary_evidence_candidate":"boundary_trace_candidate", "simulation_evidence_candidate":"simulation_trace_candidate", "symbolic_evidence_candidate":"symbolic_trace_candidate", "dmn_evidence_candidate":"dmn_trace_candidate", "dream_evidence_candidate":"dream_trace_candidate", "tool_state_evidence_candidate":"tool_state_trace_candidate", "unknown_origin_evidence_candidate":"unknown_origin_trace_candidate"}
    if evidence_type in required_kind and required_kind[evidence_type] not in kinds:
        return f"{evidence_type.replace('_candidate','')}_requires_{required_kind[evidence_type].replace('_candidate','')}"
    if evidence_type == "unknown_origin_evidence_candidate" and sources != {"unknown_origin_candidate"}:
        return "incompatible_evidence_item_kind_source_class"
    if evidence_type == "mixed_unknown_evidence_candidate":
        if len(items) < 2: return "mixed_unknown_evidence_requires_multiple_items"
        if len(kinds) < 2 and len(sources) < 2: return "mixed_unknown_evidence_requires_distinct_evidence_dimensions"
    return None


def _validate_boundary(metadata: dict, evidence_type: str):
    if "evidence_boundary_classification" in metadata:
        value = metadata["evidence_boundary_classification"]
        if not _non_empty_string(value):
            return None, "malformed_evidence_boundary_class"
    else:
        value = BOUNDARY_DEFAULTS.get(evidence_type, "internal_virtual_evidence")
    if value not in SUPPORTED_BOUNDARIES:
        return None, "unknown_evidence_boundary_class"
    return value, None


def _validate_confidence(metadata: dict):
    if "evidence_confidence_state" in metadata:
        value = metadata["evidence_confidence_state"]
        if not _non_empty_string(value):
            return None, "malformed_evidence_confidence_state"
    else:
        value = "evidence_unverified"
    if value not in SUPPORTED_CONFIDENCE:
        return None, "unknown_evidence_confidence_state"
    return value, None


def _build_valid(evidence_type: str, situation_id: str, items: list, metadata: dict, boundary: str, confidence: str) -> Dict[str, Any]:
    support = sum(1 for i in items if i["stance"] == "supports_candidate")
    challenge = sum(1 for i in items if i["stance"] == "challenges_candidate")
    neutral = sum(1 for i in items if i["stance"] == "neutral_unknown")
    unknown_origin = any(i["source_class"] == "unknown_origin_candidate" or i["evidence_kind"] == "unknown_origin_trace_candidate" for i in items)
    flags = ["candidate_only", "evidence_unverified"]
    integrity = ["evidence_integrity_review_required", "provenance_review_required", "resolution_review_required"]
    if confidence == "evidence_conflict_detected" or (support and challenge):
        flags.append("evidence_conflict_review_required"); integrity.append("conflict_review_required")
    if confidence == "evidence_origin_unknown" or unknown_origin:
        flags.append("origin_unknown"); integrity.append("origin_unknown_review_required")
    basis = {"schema_version": VERSION, "evidence_type": evidence_type, "evidence_boundary_classification": boundary, "evidence_confidence_state": confidence, "situation_id": situation_id, "evidence_items": items, "metadata": metadata}
    blob = json.dumps(basis, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = _base_payload()
    payload.update({
        "situation_evidence_context_passed": True, "situation_evidence_context_status": "VALIDATED",
        "evidence_context_id": hashlib.sha256(blob.encode("utf-8")).hexdigest(), "canonical_id_algorithm": CANONICAL_ID_ALGORITHM,
        "evidence_type": evidence_type, "evidence_boundary_classification": boundary, "evidence_confidence_state": confidence,
        "situation_id": situation_id, "evidence_items": items, "metadata": metadata,
        "origin_summary": {"origin_kind": "internal_virtual_evidence_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "external_evidence_fact_verified": False, "evidence_fact_asserted": False, "conclusion_asserted": False, "truth_resolved": False},
        "evidence_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "evidence_kinds": sorted({i["evidence_kind"] for i in items}), "source_classes": sorted({i["source_class"] for i in items}), "claim_ref_ids": sorted({i["claim_ref_id"] for i in items}), "item_count": len(items)},
        "evidence_stance_summary": {"supporting_item_count": support, "challenging_item_count": challenge, "neutral_item_count": neutral, "conflict_present": bool(support and challenge), "winner_selected": False},
        "evidence_provenance_summary": {"source_ref_ids": sorted({i["source_ref_id"] for i in items}), "source_count": len({i["source_ref_id"] for i in items}), "external_source_verified": False, "unknown_origin_present": unknown_origin},
        "resolution_summary": {"resolution_candidate_only": True, "resolved": False, "conclusion_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False},
        "evidence_flags": sorted(set(flags)), "boundary_flags": [boundary], "evidence_integrity_flags": sorted(set(integrity)),
        "candidate_only_fields": ["evidence_type", "evidence_items", "evidence_boundary_classification", "evidence_confidence_state"],
        "blocked_reasons": [], "warnings": ["mixed_virtual_external_evidence_boundary_review_required"] if evidence_type == "mixed_unknown_evidence_candidate" else [],
    })
    return payload


def build_virtual_world_situation_evidence_context(evidence_type=None, situation_id=None, evidence_items=None, metadata=None):
    raw = {"evidence_type": evidence_type, "situation_id": situation_id, "evidence_items": evidence_items, "metadata": metadata if metadata is not None else {}}
    try:
        if _assert_json_native(raw): return _reject("non_json_serializable_semantic_input")
        if evidence_type is None or not _non_empty_string(evidence_type): return _reject("missing_evidence_type")
        if evidence_type not in SUPPORTED_EVIDENCE_TYPES: return _reject("unknown_evidence_type")
        if not _non_empty_string(situation_id): return _reject("missing_or_malformed_situation_id")
        if metadata is None: metadata = {}
        if not isinstance(metadata, dict): return _reject("invalid_metadata")
        metadata = _json_clone(metadata)
        reason = _forbidden_reason(metadata)
        if reason: return _reject(reason)
        items, reason = _normalize_items(evidence_items, situation_id)
        if reason: return _reject(reason)
        reason = _type_reason(evidence_type, items)
        if reason: return _reject(reason)
        boundary, reason = _validate_boundary(metadata, evidence_type)
        if reason: return _reject(reason)
        confidence, reason = _validate_confidence(metadata)
        if reason: return _reject(reason)
        return _build_valid(evidence_type, situation_id, items, metadata, boundary, confidence)
    except Exception:
        return _reject("non_json_serializable_semantic_input")


def validate_virtual_world_situation_evidence_context(evidence_context) -> bool:
    try:
        if _assert_json_native(evidence_context) or not isinstance(evidence_context, dict): return False
        if set(evidence_context.keys()) != TOP_KEYS: return False
        if evidence_context.get("situation_evidence_context_passed") is not True or evidence_context.get("situation_evidence_context_status") != "VALIDATED": return False
        expected = build_virtual_world_situation_evidence_context(evidence_context.get("evidence_type"), evidence_context.get("situation_id"), evidence_context.get("evidence_items"), evidence_context.get("metadata"))
        return _type_exact_equal(expected, evidence_context)
    except Exception:
        return False

PLAN_FALSE_FLAGS = tuple(k for k in IMMUTABLE_FALSE_FLAGS if k != "artifact_created_or_staged") + ("relationship_update_performed", "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed")

def _plan(plan_type: str, source: Any) -> Dict[str, Any]:
    ready = validate_virtual_world_situation_evidence_context(source)
    plan = {"plan_type": plan_type, "source_evidence_context_id": source.get("evidence_context_id") if isinstance(source, dict) and ready else None, "ready": ready, "candidate_only": True, "read_only": True, "evidence_candidate_only": True}
    for key in PLAN_FALSE_FLAGS:
        plan[key] = False
    return plan


def build_evidence_context_to_situation_plan(evidence_context): return _plan("evidence_context_to_situation", evidence_context)
def build_evidence_context_to_snapshot_plan(evidence_context): return _plan("evidence_context_to_snapshot", evidence_context)
def build_evidence_context_to_transition_preflight_plan(evidence_context): return _plan("evidence_context_to_transition_preflight", evidence_context)
def build_evidence_context_to_memory_candidate_plan(evidence_context): return _plan("evidence_context_to_memory_candidate", evidence_context)
def build_evidence_context_to_appraisal_plan(evidence_context): return _plan("evidence_context_to_appraisal", evidence_context)
def build_evidence_context_to_agp_input_plan(evidence_context): return _plan("evidence_context_to_agp_input", evidence_context)


def build_virtual_world_situation_evidence_context_schema_summary():
    return {"schema_version": VERSION, "schema_name": "read_only_virtual_world_situation_evidence_context_schema", "supported_evidence_types": list(SUPPORTED_EVIDENCE_TYPES), "supported_evidence_item_kinds": list(SUPPORTED_ITEM_KINDS), "supported_evidence_stances": list(SUPPORTED_STANCES), "supported_source_classes": list(SUPPORTED_SOURCE_CLASSES), "type_item_compatibility": {k: sorted(v) for k, v in sorted(TYPE_ITEM_COMPATIBILITY.items())}, "item_stance_compatibility": {k: sorted(v) for k, v in sorted(KIND_STANCE_COMPATIBILITY.items())}, "item_source_compatibility": {k: sorted(v) for k, v in sorted(KIND_SOURCE_COMPATIBILITY.items())}, "supported_boundary_classes": list(SUPPORTED_BOUNDARIES), "supported_confidence_states": list(SUPPORTED_CONFIDENCE), "canonical_id_algorithm": CANONICAL_ID_ALGORITHM, "read_only": True, "candidate_only": True, "next_recommended_step": "read_only_virtual_world_situation_hypothesis_context_schema", "permanent_validation_contract": ["input types", "required and optional fields", "enums", "compatibility matrices", "situation and reference coherence", "cardinality", "composition", "ordering", "duplicates", "validation precedence", "strict JSON", "recursive forbidden fields", "deterministic IDs", "exact top-level schema", "nested payload integrity", "immutable safety fields", "downstream plan safety"]}


def virtual_world_situation_evidence_context_schema_summary():
    return build_virtual_world_situation_evidence_context_schema_summary()
