"""Read-only schema for temporary semantic roles and relations in virtual situations."""

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

VERSION = "1.0.0-round1301-1320"

SUPPORTED_RELATION_TYPES = (
    "entity_participates_in_situation",
    "agent_performs_activity",
    "agent_waits_for_event",
    "agent_searches_for_target",
    "agent_observes_entity_candidate",
    "agent_communicates_with_agent",
    "agent_cooperates_with_agent",
    "agent_competes_with_agent",
    "agent_teaches_agent",
    "agent_learns_from_agent",
    "agent_uses_object_candidate",
    "entity_associated_with_entity_candidate",
    "symbolic_relation_candidate",
    "dmn_relation_candidate",
    "simulation_relation_candidate",
    "dream_relation_candidate",
    "mixed_unknown_relation_candidate",
)

SUPPORTED_BOUNDARY_CLASSES = (
    "internal_virtual_relation",
    "symbolic_virtual_relation",
    "dmn_virtual_relation",
    "simulated_virtual_relation",
    "dream_virtual_relation",
    "mixed_virtual_external_relation_boundary",
)

SUPPORTED_CONFIDENCE_STATES = (
    "relation_unverified",
    "relation_low_confidence",
    "relation_medium_confidence",
    "relation_high_confidence_but_not_fact",
    "relation_conflict_detected",
    "relation_origin_unknown",
)

TWO_DISTINCT_AGENT_RELATIONS = {
    "agent_communicates_with_agent",
    "agent_cooperates_with_agent",
    "agent_competes_with_agent",
    "agent_teaches_agent",
    "agent_learns_from_agent",
}

RELATION_BOUNDARY_DEFAULTS = {
    "symbolic_relation_candidate": "symbolic_virtual_relation",
    "dmn_relation_candidate": "dmn_virtual_relation",
    "simulation_relation_candidate": "simulated_virtual_relation",
    "dream_relation_candidate": "dream_virtual_relation",
    "mixed_unknown_relation_candidate": "mixed_virtual_external_relation_boundary",
}

IMMUTABLE_FALSE_FLAGS = (
    "external_fact_asserted",
    "current_external_fact_asserted",
    "real_world_state_asserted",
    "identity_asserted",
    "persistent_relationship_state_asserted",
    "relationship_update_allowed",
    "user_emotion_asserted",
    "user_intent_asserted",
    "memory_fact_asserted",
    "memory_write_performed",
    "memory_write_allowed",
    "quarantine_promotion_allowed",
    "self_model_update_allowed",
    "affect_transition_allowed",
    "hormone_transition_allowed",
    "world_state_mutation_performed",
    "transition_applied",
    "planning_execution_performed",
    "tool_execution_performed",
    "runtime_mutation_performed",
    "persistence_write_performed",
    "vector_read_performed",
    "vector_load_performed",
    "model_loaded",
    "device_activation_performed",
    "artifact_created_or_staged",
    "agp_bypass_allowed",
    "fallback_bypass_allowed",
)

IMMUTABLE_TRUE_FLAGS = (
    "relation_candidate_only",
    "situation_relation_only",
    "situation_review_required",
    "snapshot_review_required",
    "transition_review_required",
    "memory_gate_required",
    "quarantine_required",
    "origin_fact_status_required",
    "appraisal_required",
    "agp_input_required",
)

FORBIDDEN_REQUEST_FIELDS = {
    "external_fact_assertion_requested",
    "current_external_fact_assertion_requested",
    "real_world_state_assertion_requested",
    "identity_assertion_requested",
    "persistent_relationship_update_requested",
    "relationship_update_requested",
    "user_emotion_assertion_requested",
    "user_intent_assertion_requested",
    "memory_fact_assertion_requested",
    "memory_write_requested",
    "quarantine_promotion_requested",
    "self_model_update_requested",
    "affect_transition_requested",
    "hormone_transition_requested",
    "world_state_mutation_requested",
    "transition_application_requested",
    "planning_execution_requested",
    "tool_execution_requested",
    "runtime_mutation_requested",
    "persistence_write_requested",
    "vector_read_requested",
    "vector_load_requested",
    "model_load_requested",
    "device_activation_requested",
    "artifact_creation_requested",
    "agp_bypass_requested",
    "fallback_bypass_requested",
}



def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_entity(entity: Any) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    if not isinstance(entity, dict):
        return None, "malformed_entity"
    if not all(_non_empty_string(entity.get(k)) for k in ("entity_id", "entity_type", "label")):
        return None, "malformed_entity"
    return {"entity_id": entity["entity_id"], "entity_type": entity["entity_type"], "label": entity["label"]}, None


def _normalize_context(context: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(context, dict) or not context:
        return None, "malformed_situation_context"
    normalized = dict(context)
    if not any(_non_empty_string(normalized.get(k)) for k in ("situation_id", "situation_type", "label", "context_id")):
        return None, "malformed_situation_context"
    return normalized, None


def _metadata_forbidden(metadata: Any) -> Optional[str]:
    if not isinstance(metadata, dict):
        return "invalid_metadata"
    for field in sorted(FORBIDDEN_REQUEST_FIELDS):
        if metadata.get(field) is True:
            return field
    return None


def _boundary_for_relation(relation_type: str, metadata: Dict[str, Any]) -> str:
    return metadata.get("relation_boundary_classification") or RELATION_BOUNDARY_DEFAULTS.get(
        relation_type, "internal_virtual_relation"
    )


def _confidence_for_metadata(metadata: Dict[str, Any]) -> str:
    return metadata.get("relation_confidence_state") or "relation_unverified"


def _canonical_basis(relation_type: str, boundary: str, confidence: str, subject: Dict[str, str], object_entity: Dict[str, str], context: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": VERSION,
        "relation_type": relation_type,
        "relation_boundary_classification": boundary,
        "relation_confidence_state": confidence,
        "subject_entity": subject,
        "object_entity": object_entity,
        "situation_context": context,
        "metadata": metadata,
    }


def _stable_relation_id(basis: Dict[str, Any]) -> str:
    canonical_json = json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "vw-situation-role-relation-" + hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _base_payload(relation_type, subject_entity, object_entity, situation_context, metadata):
    payload = {
        "situation_role_relation_passed": False,
        "situation_role_relation_status": "failed",
        "relation_id": None,
        "relation_type": relation_type,
        "relation_boundary_classification": None,
        "relation_confidence_state": None,
        "subject_entity": subject_entity,
        "object_entity": object_entity,
        "situation_context": situation_context,
        "metadata": metadata,
        "origin_summary": {"origin_kind": "internal_virtual_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "confirmed_external_fact": False, "persistent_relationship_state": False, "completed_knowledge_transfer": False},
        "role_assignments": [],
        "candidate_only_fields": ["subject_entity", "object_entity", "situation_context", "role_assignments"],
        "uncertainty_flags": [],
        "privacy_flags": ["identity_not_asserted"],
        "boundary_flags": [],
        "relation_integrity_flags": [],
        "blocked_reasons": [],
        "warnings": [],
    }
    for flag in IMMUTABLE_FALSE_FLAGS:
        payload[flag] = False
    for flag in IMMUTABLE_TRUE_FLAGS:
        payload[flag] = True
    return payload


def _reject(payload: Dict[str, Any], reason: str) -> Dict[str, Any]:
    payload["situation_role_relation_status"] = "REJECTED"
    payload["blocked_reasons"].append(reason)
    return payload


def build_virtual_world_situation_role_relation(
    relation_type=None,
    subject_entity=None,
    object_entity=None,
    situation_context=None,
    metadata=None,
):
    if metadata is None:
        metadata = {}

    payload = _base_payload(relation_type, subject_entity, object_entity, situation_context, metadata)

    if subject_entity is None:
        return _reject(payload, "missing_subject_entity")
    if object_entity is None:
        return _reject(payload, "missing_object_entity")
    if situation_context is None:
        return _reject(payload, "missing_situation_context")

    if relation_type is None:
        return _reject(payload, "missing_relation_type")
    if relation_type not in SUPPORTED_RELATION_TYPES:
        return _reject(payload, "unknown_relation_type")
    forbidden = _metadata_forbidden(metadata)
    if forbidden is not None:
        return _reject(payload, forbidden)

    subject, subject_error = _normalize_entity(subject_entity)
    if subject_error:
        return _reject(payload, "malformed_subject_entity")
    obj, object_error = _normalize_entity(object_entity)
    if object_error:
        return _reject(payload, "malformed_object_entity")
    context, context_error = _normalize_context(situation_context)
    if context_error:
        return _reject(payload, context_error)

    boundary = _boundary_for_relation(relation_type, metadata)
    confidence = _confidence_for_metadata(metadata)
    payload["relation_boundary_classification"] = boundary
    payload["relation_confidence_state"] = confidence
    if boundary not in SUPPORTED_BOUNDARY_CLASSES:
        return _reject(payload, "unknown_boundary_class")
    if confidence not in SUPPORTED_CONFIDENCE_STATES:
        return _reject(payload, "unknown_confidence_state")
    if relation_type in TWO_DISTINCT_AGENT_RELATIONS and subject["entity_id"] == obj["entity_id"]:
        return _reject(payload, "same_entity_invalid_for_two_agent_relation")

    payload["subject_entity"] = subject
    payload["object_entity"] = obj
    payload["situation_context"] = context
    payload["role_assignments"] = [
        {"role": "subject_candidate", "entity_id": subject["entity_id"], "candidate_only": True},
        {"role": "object_candidate", "entity_id": obj["entity_id"], "candidate_only": True},
    ]

    if relation_type in {"agent_waits_for_event", "agent_searches_for_target"}:
        payload["uncertainty_flags"].append("future_outcome_not_asserted")
    if relation_type == "agent_observes_entity_candidate":
        payload["uncertainty_flags"].append("external_observation_not_asserted")
    if relation_type in {"agent_communicates_with_agent", "agent_cooperates_with_agent", "agent_competes_with_agent"}:
        payload["boundary_flags"].append("persistent_relationship_update_blocked")
    if relation_type == "agent_teaches_agent":
        payload["uncertainty_flags"].append("completed_knowledge_transfer_not_asserted")
    if relation_type == "agent_learns_from_agent":
        payload["boundary_flags"].append("memory_write_blocked")
    if boundary == "mixed_virtual_external_relation_boundary":
        payload["boundary_flags"].append("mixed_external_assertion_blocked")
        payload["warnings"].append("external_assertions_restricted_to_candidate_boundary")
    if confidence == "relation_high_confidence_but_not_fact":
        payload["warnings"].append("high_confidence_remains_non_factual")
    if confidence == "relation_conflict_detected":
        payload["uncertainty_flags"].append("relation_conflict_detected")
        payload["relation_integrity_flags"].append("conflict_requires_review")
    if confidence == "relation_origin_unknown":
        payload["uncertainty_flags"].append("relation_origin_unknown")
    if any(entity["label"] == "민석" for entity in (subject, obj)):
        payload["privacy_flags"].append("literal_minseok_preserved_without_identity_assertion")

    basis = _canonical_basis(relation_type, boundary, confidence, subject, obj, context, metadata)
    payload["relation_id"] = _stable_relation_id(basis)
    payload["canonical_id_algorithm"] = "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256"
    payload["situation_role_relation_passed"] = True
    payload["situation_role_relation_status"] = "VALIDATED"
    return payload


def validate_virtual_world_situation_role_relation(relation):
    if not isinstance(relation, dict):
        return False
    passed = relation.get("situation_role_relation_passed")
    blocked_reasons = relation.get("blocked_reasons")
    if passed is True:
        if relation.get("situation_role_relation_status") != "VALIDATED":
            return False
        if blocked_reasons != []:
            return False
    elif passed is False:
        if relation.get("situation_role_relation_status") != "REJECTED":
            return False
        if not isinstance(blocked_reasons, list) or not blocked_reasons:
            return False
    else:
        return False
    if relation.get("situation_role_relation_passed") is not True:
        return False
    if relation.get("relation_type") not in SUPPORTED_RELATION_TYPES:
        return False
    if relation.get("relation_boundary_classification") not in SUPPORTED_BOUNDARY_CLASSES:
        return False
    if relation.get("relation_confidence_state") not in SUPPORTED_CONFIDENCE_STATES:
        return False
    for flag in IMMUTABLE_FALSE_FLAGS:
        if relation.get(flag) is not False:
            return False
    for flag in IMMUTABLE_TRUE_FLAGS:
        if relation.get(flag) is not True:
            return False
    if _metadata_forbidden(relation.get("metadata")) is not None:
        return False
    subject, subject_error = _normalize_entity(relation.get("subject_entity"))
    obj, object_error = _normalize_entity(relation.get("object_entity"))
    context, context_error = _normalize_context(relation.get("situation_context"))
    if subject_error or object_error or context_error:
        return False
    if relation["relation_type"] in TWO_DISTINCT_AGENT_RELATIONS and subject["entity_id"] == obj["entity_id"]:
        return False
    expected = _stable_relation_id(_canonical_basis(relation["relation_type"], relation["relation_boundary_classification"], relation["relation_confidence_state"], subject, obj, context, relation.get("metadata")))
    return relation.get("relation_id") == expected


def _plan(relation, plan_type):
    return {
        "plan_type": plan_type,
        "source_relation_id": relation.get("relation_id") if isinstance(relation, dict) else None,
        "candidate_only": True,
        "read_only": True,
        "mutation_performed": False,
        "memory_write_performed": False,
        "relationship_update_performed": False,
        "self_model_update_performed": False,
        "affect_transition_performed": False,
        "hormone_transition_performed": False,
        "world_state_mutation_performed": False,
        "persistence_write_performed": False,
        "vector_load_performed": False,
        "model_loaded": False,
        "device_activation_performed": False,
        "planning_execution_performed": False,
        "tool_execution_performed": False,
        "agp_bypass_allowed": False,
        "fallback_bypass_allowed": False,
        "ready": validate_virtual_world_situation_role_relation(relation),
    }


def build_situation_role_relation_to_situation_plan(relation):
    return _plan(relation, "situation_role_relation_to_situation_candidate")


def build_situation_role_relation_to_snapshot_plan(relation):
    return _plan(relation, "situation_role_relation_to_snapshot_candidate")


def build_situation_role_relation_to_transition_preflight_plan(relation):
    return _plan(relation, "situation_role_relation_to_transition_preflight_candidate")


def build_situation_role_relation_to_memory_candidate_plan(relation):
    return _plan(relation, "situation_role_relation_to_memory_candidate")


def build_situation_role_relation_to_appraisal_plan(relation):
    return _plan(relation, "situation_role_relation_to_appraisal_candidate")


def build_situation_role_relation_to_agp_input_plan(relation):
    return _plan(relation, "situation_role_relation_to_agp_input_candidate")


def build_virtual_world_situation_role_relation_schema_summary():
    return {
        "schema_name": "read_only_virtual_world_situation_role_relation_schema",
        "version": VERSION,
        "supported_relation_types": list(SUPPORTED_RELATION_TYPES),
        "supported_boundary_classes": list(SUPPORTED_BOUNDARY_CLASSES),
        "supported_confidence_states": list(SUPPORTED_CONFIDENCE_STATES),
        "entity_required_fields": ["entity_id", "entity_type", "label"],
        "deterministic_id_algorithm": "canonical_json_sorted_keys_sha256",
        "candidate_only": True,
        "read_only": True,
        "prohibited_side_effects": list(IMMUTABLE_FALSE_FLAGS),
        "required_review_gates": list(IMMUTABLE_TRUE_FLAGS),
        "recommended_next_implementation_step": "read_only_virtual_world_situation_temporal_context_schema",
    }


def virtual_world_situation_role_relation_schema_summary():
    return build_virtual_world_situation_role_relation_schema_summary()
