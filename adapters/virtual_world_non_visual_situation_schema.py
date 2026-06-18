import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

VERSION = "1.0.1-round1281-deterministic"

SUPPORTED_SITUATION_TYPES = {
    "waiting",
    "searching",
    "negotiating",
    "traveling",
    "observing",
    "planning",
    "cooperating",
    "competing",
    "communicating",
    "learning",
    "teaching",
    "resting",
    "working",
    "symbolic",
    "mixed_unknown",
}

READ_ONLY_FALSE_INVARIANTS = (
    "identity_asserted",
    "external_fact_asserted",
    "memory_fact_asserted",
    "memory_write_performed",
    "self_model_update_allowed",
    "relationship_update_allowed",
    "affect_transition_allowed",
    "hormone_transition_allowed",
    "world_state_mutation_performed",
    "transition_applied",
    "planning_execution_performed",
    "runtime_mutation_performed",
    "persistence_write_performed",
    "tool_execution_performed",
    "model_activation_performed",
    "device_activation_performed",
    "artifact_created_or_staged",
    "agp_bypass_allowed",
    "fallback_bypass_allowed",
)

FORBIDDEN_REQUEST_FIELDS = {
    "memory_write_requested",
    "self_model_update_requested",
    "relationship_update_requested",
    "affect_transition_requested",
    "hormone_transition_requested",
    "world_mutation_requested",
    "transition_application_requested",
    "planning_execution_requested",
    "persistence_requested",
    "tool_execution_requested",
    "model_activation_requested",
    "device_activation_requested",
    "artifact_creation_requested",
}


def _reject(payload: Dict[str, Any], status: str, reason: str) -> Dict[str, Any]:
    payload["virtual_world_situation_status"] = status
    payload["blocked_reasons"].append(reason)
    return payload


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_entities(entities: Any) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    if not isinstance(entities, list):
        return None, "invalid_entities"

    normalized: List[Dict[str, str]] = []
    seen_ids = set()
    for entity in entities:
        if not isinstance(entity, dict):
            return None, "malformed_entity"
        entity_id = entity.get("entity_id")
        entity_type = entity.get("entity_type")
        label = entity.get("label")
        if not all(_is_non_empty_string(value) for value in (entity_id, entity_type, label)):
            return None, "malformed_entity"
        if entity_id in seen_ids:
            return None, "duplicate_entity_id"
        seen_ids.add(entity_id)
        normalized.append(
            {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "label": label,
            }
        )

    return sorted(normalized, key=lambda item: item["entity_id"]), None


def _normalize_relationships(
    relationships: Any, entity_ids: set
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    if not isinstance(relationships, list):
        return None, "invalid_relationships"

    normalized: List[Dict[str, str]] = []
    for relationship in relationships:
        if not isinstance(relationship, dict):
            return None, "malformed_relationship"
        source_id = relationship.get("source_entity_id")
        target_id = relationship.get("target_entity_id")
        relationship_type = relationship.get("relationship_type")
        if not all(_is_non_empty_string(value) for value in (source_id, target_id, relationship_type)):
            return None, "malformed_relationship"
        if source_id not in entity_ids or target_id not in entity_ids:
            return None, "relationship_unknown_entity_reference"
        normalized.append(
            {
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "relationship_type": relationship_type,
            }
        )

    return sorted(
        normalized,
        key=lambda item: (
            item["source_entity_id"], item["target_entity_id"], item["relationship_type"]
        ),
    ), None


def _metadata_requests_forbidden(metadata: Any) -> Optional[str]:
    if not isinstance(metadata, dict):
        return "invalid_metadata"
    for field in FORBIDDEN_REQUEST_FIELDS:
        if metadata.get(field) is True:
            return field
    return None


def _canonical_situation_basis(
    situation_type: str,
    entities: List[Dict[str, str]],
    relationships: List[Dict[str, str]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": VERSION,
        "situation_type": situation_type,
        "entities": entities,
        "relationships": relationships,
        "metadata": metadata,
    }


def _stable_situation_id(canonical_basis: Dict[str, Any]) -> str:
    canonical_json = json.dumps(
        canonical_basis, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return f"vw-non-visual-{digest}"


def build_virtual_world_non_visual_situation(
    situation_type: Optional[str] = None,
    entities: Optional[List[Dict[str, str]]] = None,
    relationships: Optional[List[Dict[str, str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a candidate-only, read-only virtual non-visual situation schema.
    """
    if entities is None:
        entities = []
    if relationships is None:
        relationships = []
    if metadata is None:
        metadata = {}

    payload = {
        "virtual_world_situation_passed": False,
        "virtual_world_situation_status": "failed",
        "situation_id": None,
        "situation_type": situation_type,
        "entities": entities,
        "relationships": relationships,
        "metadata": metadata,
        "snapshot_plan_candidate_only": True,
        "transition_plan_candidate_only": True,
        "memory_handoff_plan_candidate_only": True,
        "read_only": True,
        "blocked_reasons": [],
    }
    for invariant in READ_ONLY_FALSE_INVARIANTS:
        payload[invariant] = False

    if situation_type is None:
        return _reject(payload, "REJECTED: missing situation type", "missing_situation_type")
    if situation_type not in SUPPORTED_SITUATION_TYPES:
        return _reject(
            payload,
            f"REJECTED: unknown situation type: {situation_type}",
            "unknown_situation_type",
        )

    forbidden_reason = _metadata_requests_forbidden(metadata)
    if forbidden_reason is not None:
        return _reject(payload, f"REJECTED: {forbidden_reason}", forbidden_reason)

    normalized_entities, entity_error = _normalize_entities(entities)
    if entity_error is not None:
        return _reject(payload, f"REJECTED: {entity_error}", entity_error)

    entity_ids = {entity["entity_id"] for entity in normalized_entities or []}
    normalized_relationships, relationship_error = _normalize_relationships(
        relationships, entity_ids
    )
    if relationship_error is not None:
        return _reject(payload, f"REJECTED: {relationship_error}", relationship_error)

    canonical_basis = _canonical_situation_basis(
        situation_type, normalized_entities or [], normalized_relationships or [], metadata
    )
    payload["situation_id"] = _stable_situation_id(canonical_basis)
    payload["entities"] = normalized_entities or []
    payload["relationships"] = normalized_relationships or []
    payload["canonical_id_algorithm"] = "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256"
    payload["virtual_world_situation_passed"] = True
    payload["virtual_world_situation_status"] = "VALIDATED"
    return payload


def validate_virtual_world_non_visual_situation(situation: Dict[str, Any]) -> bool:
    """
    Validate schema shape, deterministic ID, structural references, and read-only invariants.
    """
    if not isinstance(situation, dict):
        return False
    if situation.get("virtual_world_situation_passed") is not True:
        return False
    if situation.get("read_only") is not True:
        return False
    if situation.get("situation_type") not in SUPPORTED_SITUATION_TYPES:
        return False
    for invariant in READ_ONLY_FALSE_INVARIANTS:
        if situation.get(invariant) is not False:
            return False
    for candidate_field in (
        "snapshot_plan_candidate_only",
        "transition_plan_candidate_only",
        "memory_handoff_plan_candidate_only",
    ):
        if situation.get(candidate_field) is not True:
            return False

    normalized_entities, entity_error = _normalize_entities(situation.get("entities"))
    if entity_error is not None:
        return False
    entity_ids = {entity["entity_id"] for entity in normalized_entities or []}
    normalized_relationships, relationship_error = _normalize_relationships(
        situation.get("relationships"), entity_ids
    )
    if relationship_error is not None:
        return False
    metadata = situation.get("metadata")
    if _metadata_requests_forbidden(metadata) is not None:
        return False

    expected_id = _stable_situation_id(
        _canonical_situation_basis(
            situation["situation_type"], normalized_entities or [], normalized_relationships or [], metadata
        )
    )
    return situation.get("situation_id") == expected_id


def build_virtual_world_situation_to_snapshot_plan(situation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_situation_to_snapshot",
        "source_situation_id": situation.get("situation_id"),
        "candidate_only": True,
        "read_only": True,
        "mutation_performed": False,
        "ready": validate_virtual_world_non_visual_situation(situation),
    }


def build_virtual_world_situation_to_transition_plan(situation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_situation_to_transition",
        "source_situation_id": situation.get("situation_id"),
        "candidate_only": True,
        "read_only": True,
        "transition_applied": False,
        "mutation_performed": False,
        "ready": validate_virtual_world_non_visual_situation(situation),
    }


def build_virtual_world_situation_to_memory_plan(situation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_situation_to_memory",
        "source_situation_id": situation.get("situation_id"),
        "candidate_only": True,
        "read_only": True,
        "memory_write_performed": False,
        "mutation_performed": False,
        "ready": validate_virtual_world_non_visual_situation(situation),
    }
