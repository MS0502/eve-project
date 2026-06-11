"""Read-only virtual world non-visual situation schema.

Round1281-1300 introduces a deterministic schema for describing virtual
situations such as waiting, searching, negotiating, or learning.  The module is
pure data construction and validation: it does not write memory, mutate virtual
world state, update self/affect/hormone state, load models, or persist files.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

SUPPORTED_SITUATION_TYPES: List[str] = [
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
]

READ_ONLY_FALSE_FLAGS: List[str] = [
    "identity_asserted",
    "external_fact_asserted",
    "memory_fact_asserted",
    "memory_write_performed",
    "self_model_update_allowed",
    "affect_transition_allowed",
    "hormone_transition_allowed",
    "world_state_mutation_performed",
    "transition_applied",
    "planning_execution_performed",
    "runtime_mutation_performed",
    "persistence_write_performed",
    "artifact_created_or_staged",
]

FORBIDDEN_REQUESTS = {
    "memory_write": "memory write request is forbidden for situation schema",
    "self_model_update": "self-model update request is forbidden for situation schema",
    "affect_transition": "affect transition request is forbidden for situation schema",
    "hormone_transition": "hormone transition request is forbidden for situation schema",
    "world_state_mutation": "world-state mutation request is forbidden for situation schema",
    "transition_apply": "transition application request is forbidden for situation schema",
    "planning_execution": "planning execution request is forbidden for situation schema",
    "persistence_write": "persistence write request is forbidden for situation schema",
    "artifact_create": "artifact creation request is forbidden for situation schema",
}

FORBIDDEN_ASSERTIONS = {
    "identity_assertion": "identity assertion is forbidden for situation schema",
    "external_fact_assertion": "external fact assertion is forbidden for situation schema",
    "memory_fact_assertion": "memory fact assertion is forbidden for situation schema",
}


def _stable_id(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_entities(entities: Any) -> List[str]:
    reasons: List[str] = []
    if not isinstance(entities, list):
        return ["entities must be a list"]

    seen = set()
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            reasons.append(f"entity[{index}] must be a dict")
            continue
        entity_id = entity.get("entity_id")
        if not _is_non_empty_string(entity_id):
            reasons.append(f"entity[{index}].entity_id must be a non-empty string")
        elif entity_id in seen:
            reasons.append(f"entity[{index}].entity_id duplicates an earlier entity")
        else:
            seen.add(entity_id)
        if "entity_role" in entity and not _is_non_empty_string(entity.get("entity_role")):
            reasons.append(f"entity[{index}].entity_role must be a non-empty string when present")
        if "is_identity_fact" in entity and entity.get("is_identity_fact") is not False:
            reasons.append(f"entity[{index}].is_identity_fact must remain False")
    return reasons


def _validate_relationships(relationships: Any, entities: List[Dict[str, Any]]) -> List[str]:
    reasons: List[str] = []
    if not isinstance(relationships, list):
        return ["relationships must be a list"]

    entity_ids = {
        entity.get("entity_id")
        for entity in entities
        if isinstance(entity, dict) and _is_non_empty_string(entity.get("entity_id"))
    }
    for index, relationship in enumerate(relationships):
        if not isinstance(relationship, dict):
            reasons.append(f"relationship[{index}] must be a dict")
            continue
        source = relationship.get("source_entity_id")
        target = relationship.get("target_entity_id")
        relationship_type = relationship.get("relationship_type")
        if not _is_non_empty_string(source):
            reasons.append(f"relationship[{index}].source_entity_id must be a non-empty string")
        elif entity_ids and source not in entity_ids:
            reasons.append(f"relationship[{index}].source_entity_id must reference an entity")
        if not _is_non_empty_string(target):
            reasons.append(f"relationship[{index}].target_entity_id must be a non-empty string")
        elif entity_ids and target not in entity_ids:
            reasons.append(f"relationship[{index}].target_entity_id must reference an entity")
        if not _is_non_empty_string(relationship_type):
            reasons.append(f"relationship[{index}].relationship_type must be a non-empty string")
        if "relationship_asserted" in relationship and relationship.get("relationship_asserted") is not False:
            reasons.append(f"relationship[{index}].relationship_asserted must remain False")
    return reasons


def build_virtual_world_non_visual_situation(
    situation_type: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None,
    situation_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a read-only virtual situation candidate and fail closed on invalid input."""

    if entities is None:
        entities = []
    if relationships is None:
        relationships = []
    if situation_context is None:
        situation_context = {}
    if metadata is None:
        metadata = {}

    blocked_reasons: List[str] = []
    warnings: List[str] = []

    if not _is_non_empty_string(situation_type):
        blocked_reasons.append("missing situation type fails closed")
        normalized_situation_type = "unknown"
    else:
        normalized_situation_type = situation_type.strip()
        if normalized_situation_type not in SUPPORTED_SITUATION_TYPES:
            blocked_reasons.append("unknown situation type fails closed")

    entity_reasons = _validate_entities(entities)
    relationship_reasons = _validate_relationships(relationships, entities if isinstance(entities, list) else [])
    blocked_reasons.extend(entity_reasons)
    blocked_reasons.extend(relationship_reasons)

    if not isinstance(situation_context, dict):
        blocked_reasons.append("situation_context must be a dict")
        situation_context = {}
    if not isinstance(metadata, dict):
        blocked_reasons.append("metadata must be a dict")
        metadata = {}

    for request in situation_context.get("requests", []):
        if request in FORBIDDEN_REQUESTS:
            blocked_reasons.append(FORBIDDEN_REQUESTS[request])
    for assertion in situation_context.get("assertions", []):
        if assertion in FORBIDDEN_ASSERTIONS:
            blocked_reasons.append(FORBIDDEN_ASSERTIONS[assertion])

    stable_payload = {
        "situation_type": normalized_situation_type,
        "entities": entities,
        "relationships": relationships,
        "situation_context": situation_context,
        "metadata": metadata,
    }
    situation = {
        "schema_type": "virtual_world_non_visual_situation",
        "schema_version": "round1281_1300",
        "situation_id": f"vw-non-visual-situation-{_stable_id(stable_payload)}",
        "virtual_world_non_visual_situation_valid": not blocked_reasons,
        "virtual_world_non_visual_situation_status": "read_only_validated" if not blocked_reasons else "failed_closed",
        "situation_type": normalized_situation_type,
        "supported_situation_types": list(SUPPORTED_SITUATION_TYPES),
        "entities": entities,
        "relationships": relationships,
        "situation_context": situation_context,
        "metadata": metadata,
        "situation_only": True,
        "schema_only": True,
        "validation_only": True,
        "read_only_only": True,
        "memory_candidate_only": False,
        "fact_candidate_only": False,
        "identity_candidate_only": False,
        "affect_state_candidate_only": False,
        "world_mutation_candidate_only": False,
        "snapshot_plan_available": not blocked_reasons,
        "transition_plan_available": not blocked_reasons,
        "memory_plan_available": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,
    }
    for flag in READ_ONLY_FALSE_FLAGS:
        situation[flag] = False
    return situation


def validate_virtual_world_non_visual_situation(situation: Dict[str, Any]) -> bool:
    """Validate schema shape and immutable read-only invariants."""

    if not isinstance(situation, dict):
        return False
    if situation.get("schema_type") != "virtual_world_non_visual_situation":
        return False
    if situation.get("situation_only") is not True:
        return False
    if situation.get("schema_only") is not True:
        return False
    if situation.get("validation_only") is not True:
        return False
    if situation.get("read_only_only") is not True:
        return False
    for flag in READ_ONLY_FALSE_FLAGS:
        if situation.get(flag) is not False:
            return False

    blocked_reasons = situation.get("blocked_reasons")
    if not isinstance(blocked_reasons, list):
        return False
    is_valid = situation.get("virtual_world_non_visual_situation_valid") is True
    if is_valid:
        if situation.get("virtual_world_non_visual_situation_status") != "read_only_validated":
            return False
        if blocked_reasons:
            return False
        if situation.get("situation_type") not in SUPPORTED_SITUATION_TYPES:
            return False
    else:
        if situation.get("virtual_world_non_visual_situation_status") != "failed_closed":
            return False
        if not blocked_reasons:
            return False

    if _validate_entities(situation.get("entities")):
        return False
    if _validate_relationships(situation.get("relationships"), situation.get("entities", [])):
        return False
    return True


def _build_plan(situation: Dict[str, Any], plan_type: str, target_schema: str) -> Dict[str, Any]:
    source_valid = (
        validate_virtual_world_non_visual_situation(situation)
        and isinstance(situation, dict)
        and situation.get("virtual_world_non_visual_situation_valid") is True
    )
    plan_payload = {
        "plan_type": plan_type,
        "target_schema": target_schema,
        "situation_id": situation.get("situation_id") if isinstance(situation, dict) else None,
    }
    plan = {
        "schema_type": f"virtual_world_situation_to_{target_schema}_plan",
        "schema_version": "round1281_1300",
        "plan_id": f"{plan_type}-{_stable_id(plan_payload)}",
        "plan_type": plan_type,
        "source_schema_type": "virtual_world_non_visual_situation",
        "target_schema_type": target_schema,
        "source_situation_id": situation.get("situation_id") if isinstance(situation, dict) else None,
        "source_situation_type": situation.get("situation_type") if isinstance(situation, dict) else "unknown",
        "plan_valid": source_valid,
        "plan_status": "read_only_plan_available" if source_valid else "failed_closed",
        "read_only_only": True,
        "schema_only": True,
        "validation_only": True,
        "plan_execution_allowed": False,
        "plan_execution_performed": False,
        "snapshot_write_performed": False,
        "snapshot_applied": False,
        "memory_candidate_created": False,
        "memory_write_allowed": False,
        "blocked_reasons": [] if source_valid else ["source situation is invalid"],
        "warnings": [],
    }
    for flag in READ_ONLY_FALSE_FLAGS:
        plan[flag] = False
    return plan


def build_virtual_world_situation_to_snapshot_plan(situation: Dict[str, Any]) -> Dict[str, Any]:
    """Build a read-only handoff plan toward snapshot review, without creating a snapshot."""

    return _build_plan(situation, "virtual_world_situation_to_snapshot_plan", "snapshot")


def build_virtual_world_situation_to_transition_plan(situation: Dict[str, Any]) -> Dict[str, Any]:
    """Build a read-only handoff plan toward transition preflight, without applying it."""

    return _build_plan(situation, "virtual_world_situation_to_transition_plan", "transition")


def build_virtual_world_situation_to_memory_plan(situation: Dict[str, Any]) -> Dict[str, Any]:
    """Build a read-only handoff plan toward memory review, without writing memory."""

    return _build_plan(situation, "virtual_world_situation_to_memory_plan", "memory")
