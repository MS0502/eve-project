import uuid
from typing import Dict, Any, List, Optional

VERSION = "1.0.0-round1281"

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
    "mixed_unknown"
}

def build_virtual_world_non_visual_situation(
    situation_type: Optional[str] = None,
    entities: Optional[List[str]] = None,
    relationships: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Builds a pure read-only schema representing virtual situations that are not images.
    """
    situation_id = str(uuid.uuid4())

    # Defaults
    if entities is None:
        entities = []
    if relationships is None:
        relationships = {}
    if metadata is None:
        metadata = {}

    payload = {
        "virtual_world_situation_passed": False,
        "virtual_world_situation_status": "failed",
        "situation_id": situation_id,
        "situation_type": situation_type,
        "entities": entities,
        "relationships": relationships,
        "metadata": metadata,
        "identity_asserted": False,
        "external_fact_asserted": False,
        "memory_fact_asserted": False,
        "memory_write_performed": False,
        "self_model_update_allowed": False,
        "affect_transition_allowed": False,
        "hormone_transition_allowed": False,
        "world_state_mutation_performed": False,
        "transition_applied": False,
        "planning_execution_performed": False,
        "runtime_mutation_performed": False,
        "persistence_write_performed": False,
        "artifact_created_or_staged": False,
        "read_only": True,
        "blocked_reasons": []
    }

    # Fail closed conditions
    if situation_type is None:
        payload["virtual_world_situation_status"] = "REJECTED: missing situation type"
        payload["blocked_reasons"].append("missing_situation_type")
        return payload

    if situation_type not in SUPPORTED_SITUATION_TYPES:
        payload["virtual_world_situation_status"] = f"REJECTED: unknown situation type: {situation_type}"
        payload["blocked_reasons"].append("unknown_situation_type")
        return payload

    if not isinstance(entities, list):
        payload["virtual_world_situation_status"] = "REJECTED: invalid entities"
        payload["blocked_reasons"].append("invalid_entities")
        return payload

    if not isinstance(relationships, dict):
        payload["virtual_world_situation_status"] = "REJECTED: invalid relationships"
        payload["blocked_reasons"].append("invalid_relationships")
        return payload

    # If it reached here, validation passed
    payload["virtual_world_situation_passed"] = True
    payload["virtual_world_situation_status"] = "VALIDATED"

    return payload

def validate_virtual_world_non_visual_situation(situation: Dict[str, Any]) -> bool:
    """
    Validates that a situation payload is valid and adheres to read-only invariants.
    """
    if not isinstance(situation, dict):
        return False

    if situation.get("virtual_world_situation_passed") is not True:
        return False

    # Invariants
    invariants = [
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
        "artifact_created_or_staged"
    ]

    for inv in invariants:
        if situation.get(inv) is not False:
            return False

    if situation.get("read_only") is not True:
        return False

    if situation.get("situation_type") not in SUPPORTED_SITUATION_TYPES:
        return False

    return True

def build_virtual_world_situation_to_snapshot_plan(situation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_situation_to_snapshot",
        "source_situation_id": situation.get("situation_id"),
        "ready": validate_virtual_world_non_visual_situation(situation)
    }

def build_virtual_world_situation_to_transition_plan(situation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_situation_to_transition",
        "source_situation_id": situation.get("situation_id"),
        "ready": validate_virtual_world_non_visual_situation(situation)
    }

def build_virtual_world_situation_to_memory_plan(situation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_situation_to_memory",
        "source_situation_id": situation.get("situation_id"),
        "ready": validate_virtual_world_non_visual_situation(situation)
    }
