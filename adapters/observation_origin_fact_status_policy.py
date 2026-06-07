"""Read-only observation origin and fact status policy for EVE v3.1."""

from __future__ import annotations

from typing import Any

ROUND1021_1040_VERSION = "v3_round1021_1040_observation_origin_fact_status_policy"
FEATURE_TRACK = "read_only_observation_origin_and_fact_status_policy"

NEXT_IMPLEMENTATION_RECOMMENDATION = "read_only_virtual_visual_observation_schema"

SUPPORTED_ORIGINS = (
    "external_visual",
    "external_audio",
    "screen_visual",
    "tool_state",
    "internal_state",
    "virtual_world_visual",
    "memory_replay",
    "imagination",
    "simulation",
    "dream_dmn",
)

SUPPORTED_FACT_STATUSES = (
    "observed_external",
    "observed_internal_virtual",
    "reconstructed_memory",
    "imagined_candidate",
    "simulated_future",
    "symbolic_visualization",
)

FALSE_FLAGS = {
    "memory_write_performed": False,
    "self_model_update_allowed": False,
    "affect_transition_allowed": False,
    "hormone_transition_allowed": False,
    "runtime_mutation_performed": False,
    "persistence_write_performed": False,
    "vector_read_performed": False,
    "vector_load_performed": False,
    "artifact_created_or_staged": False,
    "agp_bypass_allowed": False,
    "fallback_bypass_allowed": False,
}

def build_observation_origin_fact_status_policy_summary() -> dict[str, Any]:
    return {
        "version": ROUND1021_1040_VERSION,
        "feature_track": FEATURE_TRACK,
        "supported_origins": SUPPORTED_ORIGINS,
        "supported_fact_statuses": SUPPORTED_FACT_STATUSES,
        "false_flags_enforced": list(FALSE_FLAGS.keys()),
        "next_implementation_recommendations": [NEXT_IMPLEMENTATION_RECOMMENDATION],
    }

def observation_origin_fact_status_policy_summary() -> dict[str, Any]:
    return build_observation_origin_fact_status_policy_summary()

def build_origin_fact_status_record(origin: str, fact_status: str, source: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    record: dict[str, Any] = {
        "origin": origin,
        "fact_status": fact_status,
        "source": source,
        "metadata": metadata,
        "origin_fact_status_passed": True,
        "blocked_reasons": [],
        "warnings": [],
    }

    if origin not in SUPPORTED_ORIGINS:
        record["origin_fact_status_passed"] = False
        record["blocked_reasons"].append(f"unsupported_origin_{origin}")

    if fact_status not in SUPPORTED_FACT_STATUSES:
        record["origin_fact_status_passed"] = False
        record["blocked_reasons"].append(f"unsupported_fact_status_{fact_status}")

    record.update(FALSE_FLAGS)

    # external_fact_asserted = false unless origin/fact_status permits it
    if origin in ("external_visual", "external_audio", "screen_visual") and fact_status == "observed_external":
        record["external_fact_asserted"] = True
    else:
        record["external_fact_asserted"] = False

    guard_results = [
        external_fact_assertion_guard(record),
        virtual_world_reality_boundary_guard(record),
        memory_replay_provenance_guard(record),
        imagination_memory_contamination_guard(record),
        simulation_result_boundary_guard(record),
        dream_dmn_boundary_guard(record),
    ]

    for guard_result in guard_results:
        if not guard_result["guard_passed"]:
            record["origin_fact_status_passed"] = False
            if guard_result["blocked_reason"] not in record["blocked_reasons"]:
                record["blocked_reasons"].append(guard_result["blocked_reason"])
            record["warnings"].extend(guard_result.get("warnings", []))

    return record

def validate_origin_fact_status_record(record: dict[str, Any]) -> dict[str, Any]:
    passed = record.get("origin_fact_status_passed", False)
    return {
        "validation_passed": passed,
        "blocked_reasons": record.get("blocked_reasons", []),
        "record_origin": record.get("origin"),
        "record_fact_status": record.get("fact_status"),
    }

def external_fact_assertion_guard(record: dict[str, Any]) -> dict[str, Any]:
    origin = record.get("origin")
    fact_status = record.get("fact_status")

    if origin in ("virtual_world_visual", "memory_replay", "imagination", "simulation", "dream_dmn"):
        if fact_status == "observed_external":
            return {"guard_passed": False, "blocked_reason": "external_fact_assertion_guard_failed_internal_origin_cannot_be_external_fact"}

    if fact_status == "observed_external" and origin not in ("external_visual", "external_audio", "screen_visual", "tool_state"):
        return {"guard_passed": False, "blocked_reason": "external_fact_assertion_guard_failed_fact_status_mismatch"}

    return {"guard_passed": True, "blocked_reason": None}

def virtual_world_reality_boundary_guard(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("origin") == "virtual_world_visual" and record.get("fact_status") != "observed_internal_virtual":
        return {"guard_passed": False, "blocked_reason": "virtual_world_reality_boundary_guard_failed"}
    return {"guard_passed": True, "blocked_reason": None}

def memory_replay_provenance_guard(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("origin") == "memory_replay" and record.get("fact_status") != "reconstructed_memory":
        return {"guard_passed": False, "blocked_reason": "memory_replay_provenance_guard_failed"}
    return {"guard_passed": True, "blocked_reason": None}

def imagination_memory_contamination_guard(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("origin") == "imagination" and record.get("fact_status") != "imagined_candidate":
        return {"guard_passed": False, "blocked_reason": "imagination_memory_contamination_guard_failed"}
    return {"guard_passed": True, "blocked_reason": None}

def simulation_result_boundary_guard(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("origin") == "simulation" and record.get("fact_status") != "simulated_future":
        return {"guard_passed": False, "blocked_reason": "simulation_result_boundary_guard_failed"}
    return {"guard_passed": True, "blocked_reason": None}

def dream_dmn_boundary_guard(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("origin") == "dream_dmn" and record.get("fact_status") != "symbolic_visualization":
        return {"guard_passed": False, "blocked_reason": "dream_dmn_boundary_guard_failed"}
    return {"guard_passed": True, "blocked_reason": None}

def build_external_fact_assertion_guard(record: dict[str, Any]) -> dict[str, Any]:
    return external_fact_assertion_guard(record)

def build_memory_contamination_guard(record: dict[str, Any]) -> dict[str, Any]:
    return imagination_memory_contamination_guard(record)

def build_origin_fact_status_compatibility_plan(observation: dict[str, Any]) -> dict[str, Any]:
    origin = observation.get("origin", "unknown")
    fact_status = observation.get("fact_status", "unknown")
    record = build_origin_fact_status_record(origin, fact_status)
    return {
        "compatibility_plan_passed": record["origin_fact_status_passed"],
        "origin_fact_status_record": record,
        "external_fact_asserted": record["external_fact_asserted"],
        "memory_write_allowed": False,
        "affect_transition_allowed": False,
    }
