"""Deterministic read-only causal-context candidates for virtual situations."""

import hashlib
import json
import math
from typing import Any, Dict, Optional, Tuple

VERSION = "1.0.0-round1341-1360"
CANONICAL_ID_ALGORITHM = "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256"

SUPPORTED_CAUSAL_TYPES = (
    "direct_cause_candidate", "indirect_cause_candidate", "contributing_factor_candidate",
    "enabling_condition_candidate", "preventing_condition_candidate", "candidate_consequence",
    "common_cause_candidate", "causal_chain_candidate", "causal_direction_unknown_candidate",
    "correlation_only_candidate", "counterfactual_cause_candidate", "simulation_cause_candidate",
    "symbolic_cause_candidate", "dmn_cause_candidate", "dream_cause_candidate",
    "mixed_unknown_causal_candidate",
)
SUPPORTED_CAUSAL_BOUNDARY_CLASSES = (
    "internal_logical_causality", "symbolic_virtual_causality", "dmn_virtual_causality",
    "simulated_virtual_causality", "dream_virtual_causality", "mixed_virtual_external_causal_boundary",
)
SUPPORTED_CAUSAL_CONFIDENCE_STATES = (
    "causal_unverified", "causal_low_confidence", "causal_medium_confidence",
    "causal_high_confidence_but_not_fact", "causal_conflict_detected",
    "causal_direction_unknown", "causal_origin_unknown",
)
SUPPORTED_LINK_KINDS = (
    "cause_candidate", "consequence_candidate", "contributing_candidate", "enabling_candidate",
    "preventing_candidate", "common_cause_candidate", "correlation_candidate",
    "counterfactual_candidate", "unknown_direction_candidate",
)
DIRECTIONAL_CAUSAL_TYPES = {
    "direct_cause_candidate", "indirect_cause_candidate", "contributing_factor_candidate",
    "enabling_condition_candidate", "preventing_condition_candidate", "candidate_consequence",
    "counterfactual_cause_candidate", "simulation_cause_candidate",
}
BOUNDARY_DEFAULTS = {
    "symbolic_cause_candidate": "symbolic_virtual_causality",
    "dmn_cause_candidate": "dmn_virtual_causality",
    "simulation_cause_candidate": "simulated_virtual_causality",
    "dream_cause_candidate": "dream_virtual_causality",
    "mixed_unknown_causal_candidate": "mixed_virtual_external_causal_boundary",
}
IMMUTABLE_FALSE_FLAGS = (
    "external_cause_asserted", "external_consequence_asserted", "external_causal_direction_asserted",
    "external_causal_chain_asserted", "causal_fact_asserted", "external_fact_asserted",
    "prediction_guaranteed", "future_outcome_asserted", "intervention_performed",
    "counterfactual_execution_performed", "event_completion_asserted", "identity_asserted",
    "memory_fact_asserted", "memory_write_performed", "memory_write_allowed",
    "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed",
    "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied",
    "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed",
    "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded",
    "device_activation_performed", "network_action_performed", "artifact_created_or_staged",
    "agp_bypass_allowed", "fallback_bypass_allowed",
)
IMMUTABLE_TRUE_FLAGS = (
    "causal_candidate_only", "situation_causal_only", "read_only", "situation_review_required",
    "snapshot_review_required", "transition_review_required", "memory_gate_required", "quarantine_required",
    "origin_fact_status_required", "causal_integrity_review_required", "appraisal_required", "agp_input_required",
)
FORBIDDEN_REQUEST_FIELDS = {
    "external_cause_assertion_requested", "external_consequence_assertion_requested",
    "causal_fact_assertion_requested", "causal_direction_verification_requested",
    "causal_chain_verification_requested", "prediction_guarantee_requested",
    "future_outcome_assertion_requested", "intervention_requested", "counterfactual_execution_requested",
    "event_completion_assertion_requested", "memory_write_requested", "memory_fact_promotion_requested",
    "relationship_update_requested", "self_model_update_requested", "affect_transition_requested",
    "hormone_transition_requested", "world_state_mutation_requested", "transition_application_requested",
    "planning_execution_requested", "tool_execution_requested", "runtime_mutation_requested",
    "persistence_write_requested", "vector_read_requested", "vector_load_requested", "model_load_requested",
    "device_activation_requested", "network_action_requested", "artifact_creation_requested",
    "agp_bypass_requested", "fallback_bypass_requested", "act_requested", "execute_requested",
    "apply_requested", "intervene_requested", "simulate_and_apply_requested", "commit_outcome_requested",
    "promote_to_fact_requested",
}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _assert_json_native(value: Any, seen: Optional[set] = None, depth: int = 0) -> Optional[str]:
    if depth > 1000:
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
                seen.remove(oid); return "non_json_serializable_semantic_input"
            err = _assert_json_native(item, seen, depth + 1)
            if err:
                seen.remove(oid); return err
        seen.remove(oid); return None
    if isinstance(value, list):
        oid = id(value)
        if oid in seen:
            return "non_json_serializable_semantic_input"
        seen.add(oid)
        for item in value:
            err = _assert_json_native(item, seen, depth + 1)
            if err:
                seen.remove(oid); return err
        seen.remove(oid); return None
    return "non_json_serializable_semantic_input"


def _canonical_json_dumps(value: Any) -> Tuple[Optional[str], Optional[str]]:
    try:
        err = _assert_json_native(value)
        if err:
            return None, err
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False), None
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None, "non_json_serializable_semantic_input"


def _forbidden_request_reason(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        for key in sorted(value):
            item = value[key]
            if key in FORBIDDEN_REQUEST_FIELDS:
                if item is True:
                    return key
                if item is not False:
                    return "malformed_forbidden_request_field"
            nested = _forbidden_request_reason(item)
            if nested:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _forbidden_request_reason(item)
            if nested:
                return nested
    return None


def _boundary_for_type(causal_type: str, metadata: Dict[str, Any]):
    if "causal_boundary_classification" not in metadata:
        return BOUNDARY_DEFAULTS.get(causal_type, "internal_logical_causality"), None
    value = metadata["causal_boundary_classification"]
    if not _non_empty_string(value):
        return None, "malformed_causal_boundary_class"
    if value not in SUPPORTED_CAUSAL_BOUNDARY_CLASSES:
        return value, "unknown_causal_boundary_class"
    return value, None


def _confidence_for_metadata(metadata: Dict[str, Any]):
    if "causal_confidence_state" not in metadata:
        return "causal_unverified", None
    value = metadata["causal_confidence_state"]
    if not _non_empty_string(value):
        return None, "malformed_causal_confidence_state"
    if value not in SUPPORTED_CAUSAL_CONFIDENCE_STATES:
        return value, "unknown_causal_confidence_state"
    return value, None


def _normalize_links(links: Any, causal_type: Optional[str]) -> Tuple[Optional[list], Optional[str]]:
    if links is None:
        return None, "missing_causal_links"
    if not isinstance(links, list):
        return None, "causal_links_not_non_empty_list"
    if not links:
        return None, "empty_causal_links"
    normalized = []
    seen_ids = set(); seen_sem = set(); seqs = []
    for link in links:
        if not isinstance(link, dict):
            return None, "malformed_causal_link"
        reason = _forbidden_request_reason(link)
        if reason:
            return None, reason
        for field in ("link_id", "source_situation_id", "target_situation_id", "link_kind"):
            if field not in link:
                return None, "malformed_causal_link"
        if not _non_empty_string(link["link_id"]):
            return None, "missing_or_empty_link_id"
        if link["link_id"] in seen_ids:
            return None, "duplicate_link_id"
        seen_ids.add(link["link_id"])
        if not _non_empty_string(link["source_situation_id"]) or not _non_empty_string(link["target_situation_id"]):
            return None, "malformed_link_situation_id"
        if link["source_situation_id"] == link["target_situation_id"]:
            return None, "identical_link_source_target"
        if link["link_kind"] not in SUPPORTED_LINK_KINDS:
            return None, "unknown_link_kind"
        item = dict(link)
        if "sequence_index" in item:
            seq = item["sequence_index"]
            if not isinstance(seq, int) or isinstance(seq, bool):
                return None, "malformed_sequence_index"
            if seq < 0:
                return None, "negative_sequence_index"
            seqs.append(seq)
        if "weight_candidate" in item:
            weight = item["weight_candidate"]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
                return None, "malformed_weight_candidate"
            if weight < 0.0 or weight > 1.0:
                return None, "weight_candidate_out_of_range"
        if "label" in item and not _non_empty_string(item["label"]):
            return None, "malformed_causal_link"
        sem = (item["source_situation_id"], item["target_situation_id"], item["link_kind"], item.get("sequence_index"))
        if sem in seen_sem:
            return None, "duplicate_semantic_link"
        seen_sem.add(sem)
        normalized.append(item)
    if causal_type == "causal_chain_candidate":
        if len(seqs) != len(normalized):
            return None, "causal_chain_sequence_missing"
        if len(set(seqs)) != len(seqs):
            return None, "causal_chain_sequence_duplicated"
        if sorted(seqs) != list(range(len(seqs))):
            return None, "causal_chain_sequence_non_contiguous"
        normalized.sort(key=lambda item: item["sequence_index"])
    else:
        normalized.sort(key=lambda item: _canonical_json_dumps(item)[0] or "")
    return normalized, None


def _base_payload(causal_type, subject_situation_id, object_situation_id, causal_links, metadata):
    payload = {
        "situation_causal_context_passed": False,
        "situation_causal_context_status": "REJECTED",
        "causal_context_id": None,
        "canonical_id_algorithm": None,
        "causal_type": causal_type if isinstance(causal_type, str) else None,
        "causal_boundary_classification": None,
        "causal_confidence_state": None,
        "subject_situation_id": subject_situation_id if isinstance(subject_situation_id, str) else None,
        "object_situation_id": object_situation_id if isinstance(object_situation_id, str) else None,
        "causal_links": causal_links if isinstance(causal_links, list) else [],
        "metadata": metadata if isinstance(metadata, dict) else {},
        "origin_summary": {"origin_kind": "internal_virtual_causal_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "verified_external_cause": False, "verified_external_consequence": False, "causal_fact_asserted": False, "prediction_guaranteed": False},
        "causal_direction_summary": {"direction_candidate_only": True, "external_direction_verified": False, "direction_known": False},
        "causal_chain_summary": {"chain_candidate_only": True, "external_chain_verified": False, "link_count": 0, "sequence_validated": False},
        "counterfactual_summary": {"counterfactual_candidate_only": True, "intervention_performed": False, "external_outcome_verified": False},
        "uncertainty_flags": [], "boundary_flags": [], "causal_integrity_flags": [],
        "candidate_only_fields": ["causal_type", "subject_situation_id", "object_situation_id", "causal_links", "metadata"],
        "blocked_reasons": [], "warnings": [],
    }
    for flag in IMMUTABLE_FALSE_FLAGS:
        payload[flag] = False
    for flag in IMMUTABLE_TRUE_FLAGS:
        payload[flag] = True
    return payload


def _reject(payload, reason):
    payload["situation_causal_context_status"] = "REJECTED"
    payload["situation_causal_context_passed"] = False
    payload["blocked_reasons"].append(reason)
    return payload


def _id_basis(causal_type, boundary, confidence, subject, obj, links, metadata):
    return {"schema_version": VERSION, "causal_type": causal_type, "causal_boundary_classification": boundary,
            "causal_confidence_state": confidence, "subject_situation_id": subject, "object_situation_id": obj,
            "causal_links": links, "metadata": metadata}


def _stable_id(basis):
    canonical, error = _canonical_json_dumps(basis)
    if error:
        return None, error
    return "vw-situation-causal-context-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest(), None


def build_virtual_world_situation_causal_context(causal_type=None, subject_situation_id=None, object_situation_id=None, causal_links=None, metadata=None):
    if metadata is None:
        metadata = {}
    semantic = {"causal_type": causal_type, "subject_situation_id": subject_situation_id, "object_situation_id": object_situation_id, "causal_links": causal_links, "metadata": metadata}
    try:
        err = _assert_json_native(semantic)
    except (TypeError, ValueError, OverflowError, RecursionError):
        err = "non_json_serializable_semantic_input"
    if err:
        return _reject(_base_payload(causal_type, subject_situation_id, object_situation_id, [], {}), "non_json_serializable_semantic_input")
    payload = _base_payload(causal_type, subject_situation_id, object_situation_id, causal_links, metadata)
    if causal_type is None:
        return _reject(payload, "missing_causal_type")
    if causal_type not in SUPPORTED_CAUSAL_TYPES:
        return _reject(payload, "unknown_causal_type")
    if not _non_empty_string(subject_situation_id):
        return _reject(payload, "missing_or_malformed_subject_situation_id")
    if not _non_empty_string(object_situation_id):
        return _reject(payload, "missing_or_malformed_object_situation_id")
    if subject_situation_id == object_situation_id and causal_type in DIRECTIONAL_CAUSAL_TYPES:
        return _reject(payload, "identical_directional_situation_ids")
    if subject_situation_id == object_situation_id and causal_type == "common_cause_candidate":
        return _reject(payload, "identical_common_cause_outcome_ids")
    reason = _forbidden_request_reason(metadata)
    if reason:
        return _reject(payload, reason)
    links, reason = _normalize_links(causal_links, causal_type)
    if reason:
        return _reject(payload, reason)
    payload["causal_links"] = links
    boundary, reason = _boundary_for_type(causal_type, metadata)
    payload["causal_boundary_classification"] = boundary
    if reason:
        return _reject(payload, reason)
    confidence, reason = _confidence_for_metadata(metadata)
    payload["causal_confidence_state"] = confidence
    if reason:
        return _reject(payload, reason)
    direction_known = causal_type not in {"correlation_only_candidate", "causal_direction_unknown_candidate", "common_cause_candidate", "mixed_unknown_causal_candidate"}
    payload["causal_direction_summary"]["direction_known"] = bool(direction_known)
    payload["causal_chain_summary"]["link_count"] = len(links)
    payload["causal_chain_summary"]["sequence_validated"] = causal_type == "causal_chain_candidate"
    if causal_type == "correlation_only_candidate":
        payload["causal_integrity_flags"].append("correlation_is_not_causation")
        payload["warnings"].append("correlation_only_candidate_does_not_assert_causation")
    if causal_type == "causal_direction_unknown_candidate":
        payload["uncertainty_flags"].append("causal_direction_unknown")
    if causal_type == "contributing_factor_candidate":
        payload["causal_integrity_flags"].append("not_promoted_to_sole_cause")
    if causal_type == "enabling_condition_candidate":
        payload["causal_integrity_flags"].append("not_asserted_as_sufficient_cause")
    if causal_type == "preventing_condition_candidate":
        payload["causal_integrity_flags"].append("external_prevention_not_asserted")
    if causal_type == "candidate_consequence":
        payload["uncertainty_flags"].append("external_outcome_not_verified")
    if causal_type == "common_cause_candidate":
        payload["causal_integrity_flags"].append("shared_origin_not_verified")
    if causal_type == "counterfactual_cause_candidate":
        payload["boundary_flags"].append("counterfactual_no_intervention")
    if causal_type == "simulation_cause_candidate":
        payload["boundary_flags"].append("simulation_only")
    if causal_type in {"symbolic_cause_candidate", "dmn_cause_candidate", "dream_cause_candidate"}:
        payload["boundary_flags"].append("internal_symbolic_causality_only")
    if boundary == "mixed_virtual_external_causal_boundary":
        payload["boundary_flags"].append("mixed_external_causal_assertion_blocked")
        payload["warnings"].append("mixed_boundary_requires_operator_review_without_blocking_valid_candidate")
    if confidence == "causal_high_confidence_but_not_fact":
        payload["warnings"].append("high_confidence_remains_non_factual")
    if confidence == "causal_conflict_detected":
        payload["uncertainty_flags"].append("causal_conflict_detected")
        payload["causal_integrity_flags"].append("conflicting_causal_candidates_require_review")
    if confidence == "causal_direction_unknown":
        payload["uncertainty_flags"].append("causal_direction_unknown")
        payload["causal_direction_summary"]["direction_known"] = False
    if confidence == "causal_origin_unknown":
        payload["uncertainty_flags"].append("causal_origin_unknown")
    cid, err = _stable_id(_id_basis(causal_type, boundary, confidence, subject_situation_id, object_situation_id, links, metadata))
    if err:
        return _reject(payload, err)
    payload["causal_context_id"] = cid
    payload["canonical_id_algorithm"] = CANONICAL_ID_ALGORITHM
    payload["situation_causal_context_passed"] = True
    payload["situation_causal_context_status"] = "VALIDATED"
    return payload


def validate_virtual_world_situation_causal_context(causal_context):
    try:
        if _assert_json_native(causal_context) is not None or not isinstance(causal_context, dict):
            return False
        if causal_context.get("situation_causal_context_passed") is not True:
            return False
        if causal_context.get("situation_causal_context_status") != "VALIDATED" or causal_context.get("blocked_reasons") != []:
            return False
        if causal_context.get("canonical_id_algorithm") != CANONICAL_ID_ALGORITHM:
            return False
        for flag in IMMUTABLE_FALSE_FLAGS:
            if causal_context.get(flag) is not False:
                return False
        for flag in IMMUTABLE_TRUE_FLAGS:
            if causal_context.get(flag) is not True:
                return False
        links, reason = _normalize_links(causal_context.get("causal_links"), causal_context.get("causal_type"))
        if reason:
            return False
        expected = build_virtual_world_situation_causal_context(
            causal_type=causal_context.get("causal_type"),
            subject_situation_id=causal_context.get("subject_situation_id"),
            object_situation_id=causal_context.get("object_situation_id"),
            causal_links=links,
            metadata=causal_context.get("metadata"),
        )
        if expected.get("situation_causal_context_passed") is not True:
            return False
        if set(causal_context.keys()) != set(expected.keys()):
            return False
        return causal_context == expected
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False


def _plan(causal_context, plan_type):
    return {
        "plan_type": plan_type,
        "source_causal_context_id": causal_context.get("causal_context_id") if isinstance(causal_context, dict) else None,
        "ready": validate_virtual_world_situation_causal_context(causal_context),
        "candidate_only": True, "read_only": True, "causal_candidate_only": True,
        "external_cause_asserted": False, "external_consequence_asserted": False, "causal_fact_asserted": False,
        "intervention_performed": False, "memory_write_performed": False, "relationship_update_performed": False,
        "self_model_update_performed": False, "affect_transition_performed": False, "hormone_transition_performed": False,
        "world_state_mutation_performed": False, "transition_applied": False, "persistence_write_performed": False,
        "vector_read_performed": False, "vector_load_performed": False, "model_loaded": False,
        "device_activation_performed": False, "network_action_performed": False, "planning_execution_performed": False,
        "tool_execution_performed": False, "agp_bypass_allowed": False, "fallback_bypass_allowed": False,
    }


def build_causal_context_to_situation_plan(causal_context): return _plan(causal_context, "causal_context_to_situation_candidate")
def build_causal_context_to_snapshot_plan(causal_context): return _plan(causal_context, "causal_context_to_snapshot_candidate")
def build_causal_context_to_transition_preflight_plan(causal_context): return _plan(causal_context, "causal_context_to_transition_preflight_candidate")
def build_causal_context_to_memory_candidate_plan(causal_context): return _plan(causal_context, "causal_context_to_memory_candidate")
def build_causal_context_to_appraisal_plan(causal_context): return _plan(causal_context, "causal_context_to_appraisal_candidate")
def build_causal_context_to_agp_input_plan(causal_context): return _plan(causal_context, "causal_context_to_agp_input_candidate")


def build_virtual_world_situation_causal_context_schema_summary():
    return {
        "schema_name": "read_only_virtual_world_situation_causal_context_schema",
        "version": VERSION,
        "supported_causal_types": list(SUPPORTED_CAUSAL_TYPES),
        "supported_causal_boundary_classes": list(SUPPORTED_CAUSAL_BOUNDARY_CLASSES),
        "supported_causal_confidence_states": list(SUPPORTED_CAUSAL_CONFIDENCE_STATES),
        "supported_link_kinds": list(SUPPORTED_LINK_KINDS),
        "deterministic_id_algorithm": "canonical_json_sorted_keys_sha256",
        "candidate_only": True, "read_only": True,
        "prohibited_side_effects": list(IMMUTABLE_FALSE_FLAGS),
        "required_review_gates": list(IMMUTABLE_TRUE_FLAGS),
        "recommended_next_implementation_step": "read_only_virtual_world_situation_uncertainty_context_schema",
    }


def virtual_world_situation_causal_context_schema_summary():
    return build_virtual_world_situation_causal_context_schema_summary()
