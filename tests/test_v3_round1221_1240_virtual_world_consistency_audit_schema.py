import pytest
from adapters.virtual_world_consistency_audit_schema import (
    build_virtual_world_consistency_audit_schema_summary,
    build_virtual_world_consistency_audit,
    validate_virtual_world_consistency_audit,
    build_virtual_world_consistency_to_origin_fact_status_plan,
    build_virtual_world_consistency_to_snapshot_plan,
    build_virtual_world_consistency_to_transition_preflight_plan,
    build_virtual_world_consistency_to_memory_candidate_plan,
    build_virtual_world_consistency_to_quarantine_plan,
    build_virtual_world_consistency_to_appraisal_plan,
    build_virtual_world_consistency_to_agp_input_plan
)

def _assert_immutable_flags(audit):
    assert audit["audit_applied"] is False
    assert audit["transition_applied"] is False
    assert audit["virtual_state_mutation_performed"] is False
    assert audit["virtual_world_write_allowed"] is False
    assert audit["virtual_world_persistence_allowed"] is False
    assert audit["persistent_world_state_created"] is False
    assert audit["external_reality_asserted"] is False
    assert audit["current_external_fact_asserted"] is False
    assert audit["real_world_event_asserted"] is False
    assert audit["memory_write_performed"] is False
    assert audit["memory_write_allowed"] is False
    assert audit["quarantine_promotion_allowed"] is False
    assert audit["self_model_update_allowed"] is False
    assert audit["relationship_update_allowed"] is False
    assert audit["relationship_state_asserted"] is False
    assert audit["identity_asserted"] is False
    assert audit["user_emotion_asserted"] is False
    assert audit["user_intent_asserted"] is False
    assert audit["affect_transition_allowed"] is False
    assert audit["hormone_transition_allowed"] is False
    assert audit["agp_bypass_allowed"] is False
    assert audit["fallback_bypass_allowed"] is False
    assert audit["vector_read_performed"] is False
    assert audit["vector_load_performed"] is False
    assert audit["runtime_mutation_performed"] is False
    assert audit["persistence_write_performed"] is False
    assert audit["artifact_created_or_staged"] is False

def test_build_virtual_world_consistency_audit_valid_cases():
    # observation + snapshot
    audit = build_virtual_world_consistency_audit(
        observation_candidate={"obs": "data"},
        snapshot_candidate={"snap": "data"},
        metadata={
            "audit_source_type": "observation_snapshot_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit"
        }
    )
    assert audit["virtual_world_consistency_audit_passed"] is True
    assert validate_virtual_world_consistency_audit(audit) is True

    # snapshot + transition
    audit2 = build_virtual_world_consistency_audit(
        snapshot_candidate={"snap": "data"},
        transition_preflight={"tran": "data"},
        metadata={
            "audit_source_type": "snapshot_transition_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit"
        }
    )
    assert audit2["virtual_world_consistency_audit_passed"] is True
    assert validate_virtual_world_consistency_audit(audit2) is True

    # full virtual world audit
    audit3 = build_virtual_world_consistency_audit(
        observation_candidate={"obs": "data"},
        snapshot_candidate={"snap": "data"},
        transition_preflight={"tran": "data"},
        metadata={
            "audit_source_type": "full_virtual_world_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit"
        }
    )
    assert audit3["virtual_world_consistency_audit_passed"] is True
    assert validate_virtual_world_consistency_audit(audit3) is True

def test_missing_candidates_fail_closed():
    # missing observation
    audit = build_virtual_world_consistency_audit(
        snapshot_candidate={"snap": "data"},
        metadata={
            "audit_source_type": "observation_snapshot_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit"
        }
    )
    assert audit["virtual_world_consistency_audit_passed"] is False
    assert audit["audit_decision"] == "consistency_blocked_missing_observation"
    assert "missing_observation_candidate" in audit["blocked_reasons"]

    # missing snapshot
    audit2 = build_virtual_world_consistency_audit(
        transition_preflight={"tran": "data"},
        metadata={
            "audit_source_type": "snapshot_transition_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit"
        }
    )
    assert audit2["virtual_world_consistency_audit_passed"] is False
    assert audit2["audit_decision"] == "consistency_blocked_missing_snapshot"
    assert "missing_snapshot_candidate" in audit2["blocked_reasons"]

    # missing transition
    audit3 = build_virtual_world_consistency_audit(
        observation_candidate={"obs": "data"},
        metadata={
            "audit_source_type": "observation_transition_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit"
        }
    )
    assert audit3["virtual_world_consistency_audit_passed"] is False
    assert audit3["audit_decision"] == "consistency_blocked_missing_transition"
    assert "missing_transition_preflight" in audit3["blocked_reasons"]

def test_unknown_type_and_boundary_fail_closed():
    audit = build_virtual_world_consistency_audit(
        observation_candidate={"obs": "data"},
        metadata={
            "audit_source_type": "unknown_source_type",
            "audit_boundary_classification": "internal_virtual_consistency_audit"
        }
    )
    assert audit["virtual_world_consistency_audit_passed"] is False
    assert "unknown_audit_source_type" in audit["blocked_reasons"]

    audit2 = build_virtual_world_consistency_audit(
        observation_candidate={"obs": "data"},
        snapshot_candidate={"snap": "data"},
        metadata={
            "audit_source_type": "observation_snapshot_audit_candidate",
            "audit_boundary_classification": "unknown_boundary_class"
        }
    )
    assert audit2["virtual_world_consistency_audit_passed"] is False
    assert "unknown_audit_boundary_classification" in audit2["blocked_reasons"]

def test_mismatch_creates_flags():
    audit = build_virtual_world_consistency_audit(
        observation_candidate={"obs": "data"},
        snapshot_candidate={"snap": "data"},
        metadata={
            "audit_source_type": "observation_snapshot_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit",
            "mismatch_detected": True
        }
    )
    assert audit["virtual_world_consistency_audit_passed"] is True
    assert "mismatch_detected" in audit["consistency_flags"]

def test_conflict_and_blocked_reasons():
    # Origin conflict
    audit1 = build_virtual_world_consistency_audit(
        observation_candidate={"obs": "data"},
        snapshot_candidate={"snap": "data"},
        metadata={
            "audit_source_type": "observation_snapshot_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit",
            "origin_conflict": True
        }
    )
    assert audit1["virtual_world_consistency_audit_passed"] is False
    assert audit1["audit_decision"] == "consistency_blocked_origin_conflict"

    # external fact assertion
    audit2 = build_virtual_world_consistency_audit(
        observation_candidate={"obs": "data"},
        snapshot_candidate={"snap": "data"},
        metadata={
            "audit_source_type": "observation_snapshot_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit",
            "update_requests": {
                "external_fact_assertion": True
            }
        }
    )
    assert audit2["virtual_world_consistency_audit_passed"] is False
    assert audit2["audit_decision"] == "consistency_blocked_external_fact_assertion"

def test_candidate_only_preservation():
    # symbolic
    audit = build_virtual_world_consistency_audit(
        metadata={
            "audit_source_type": "symbolic_virtual_audit_candidate",
            "audit_boundary_classification": "symbolic_virtual_consistency_audit"
        }
    )
    assert "symbolic_remains_candidate" in audit["candidate_only_fields"]

    # dmn
    audit2 = build_virtual_world_consistency_audit(
        metadata={
            "audit_source_type": "dmn_virtual_audit_candidate",
            "audit_boundary_classification": "dmn_virtual_consistency_audit"
        }
    )
    assert "dmn_remains_candidate" in audit2["candidate_only_fields"]

def test_immutable_flags_and_korean_fixtures():
    audit = build_virtual_world_consistency_audit(
        observation_candidate={"name": "민석"},
        snapshot_candidate={"snap": "data"},
        metadata={
            "audit_source_type": "observation_snapshot_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit"
        }
    )
    _assert_immutable_flags(audit)
    assert audit["observation_candidate"]["name"] == "민석"

    assert build_virtual_world_consistency_to_origin_fact_status_plan(audit)["ready"] is True
    assert build_virtual_world_consistency_to_snapshot_plan(audit)["ready"] is True
    assert build_virtual_world_consistency_to_transition_preflight_plan(audit)["ready"] is True
    assert build_virtual_world_consistency_to_memory_candidate_plan(audit)["ready"] is True
    assert build_virtual_world_consistency_to_quarantine_plan(audit)["ready"] is True
    assert build_virtual_world_consistency_to_appraisal_plan(audit)["ready"] is True
    assert build_virtual_world_consistency_to_agp_input_plan(audit)["ready"] is True
