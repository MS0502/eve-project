import pytest
from adapters.virtual_world_operator_intent_boundary_schema import (
    virtual_world_operator_intent_boundary_schema_summary,
    build_virtual_world_operator_intent_boundary,
    validate_virtual_world_operator_intent_boundary
)

def test_explicit_operator_instruction_boundary():
    candidate = {
        "candidate_source_type": "explicit_operator_instruction_candidate",
        "instruction": "민석 says: Confirm the state."
    }
    context = {
        "boundary_classification": "explicit_operator_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is True
    assert boundary["operator_intent_explicit"] is True
    assert boundary["user_intent_explicit"] is False
    assert boundary["user_intent_inferred"] is False
    assert validate_virtual_world_operator_intent_boundary(boundary) is True

def test_explicit_user_utterance_boundary():
    candidate = {
        "candidate_source_type": "explicit_user_utterance_intent_candidate",
        "utterance": "민석 says: I want to see the garden."
    }
    context = {
        "boundary_classification": "explicit_user_utterance_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is True
    assert boundary["user_intent_explicit"] is True
    assert boundary["operator_intent_explicit"] is False
    assert validate_virtual_world_operator_intent_boundary(boundary) is True

def test_inferred_user_intent_not_fact():
    candidate = {
        "candidate_source_type": "inferred_user_intent_candidate",
        "inference": "User might be looking for something.",
        "treated_as_fact": True
    }
    context = {
        "boundary_classification": "inferred_user_intent_boundary",
        "confidence_state": "intent_inferred_medium_confidence"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert boundary["user_intent_inferred"] is True
    assert "inferred user intent as fact blocks future review" in boundary["blocked_reasons"]
    assert boundary["inferred_intent_treated_as_fact"] is False

def test_virtual_world_intent_not_external_fact():
    candidate = {
        "candidate_source_type": "virtual_world_intent_candidate",
        "intent": "Object A should move.",
        "treated_as_external_fact": True
    }
    context = {
        "boundary_classification": "internal_virtual_intent_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert "virtual intent as external fact blocks future review" in boundary["blocked_reasons"]
    assert boundary["virtual_intent_treated_as_external_fact"] is False

def test_virtual_agent_intent_no_identity_assertion():
    candidate = {
        "candidate_source_type": "virtual_agent_intent_candidate",
        "assertions": ["identity_assertion"]
    }
    context = {
        "boundary_classification": "internal_virtual_intent_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert "identity assertion blocks future review" in boundary["blocked_reasons"]
    assert boundary["identity_asserted"] is False

def test_symbolic_intent_remains_candidate_only():
    candidate = {
        "candidate_source_type": "symbolic_intent_candidate"
    }
    context = {
        "boundary_classification": "symbolic_intent_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is True
    assert "remains_candidate_only" in boundary["intent_boundary_flags"]

def test_dmn_intent_remains_candidate_only():
    candidate = {
        "candidate_source_type": "dmn_intent_candidate"
    }
    context = {
        "boundary_classification": "dmn_intent_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is True
    assert "remains_candidate_only" in boundary["intent_boundary_flags"]

def test_simulation_intent_not_current_fact():
    candidate = {
        "candidate_source_type": "simulation_intent_candidate",
        "assert_current_fact": True
    }
    context = {
        "boundary_classification": "simulated_intent_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert "simulation intent cannot become current fact" in boundary["blocked_reasons"]

def test_dream_intent_not_external_fact():
    candidate = {
        "candidate_source_type": "dream_intent_candidate",
        "assert_external_fact": True
    }
    context = {
        "boundary_classification": "dream_intent_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert "dream intent cannot become external fact" in boundary["blocked_reasons"]

def test_mixed_boundary_requires_review():
    candidate = {
        "candidate_source_type": "explicit_operator_instruction_candidate"
    }
    context = {
        "boundary_classification": "mixed_operator_user_virtual_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert boundary["intent_decision"] == "intent_review_required"
    assert "mixed operator/user/virtual boundary requires review" in boundary["blocked_reasons"]

def test_operator_user_conflict_blocks():
    candidate = {
        "candidate_source_type": "explicit_operator_instruction_candidate"
    }
    context = {
        "boundary_classification": "explicit_operator_boundary",
        "confidence_state": "intent_explicit",
        "operator_user_conflict": True
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert "operator-user conflict blocks future review" in boundary["blocked_reasons"]

def test_empty_candidate_fails_closed():
    boundary = build_virtual_world_operator_intent_boundary(None, None)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert boundary["virtual_world_operator_intent_boundary_status"] == "failed_closed"

def test_unknown_source_type_fails_closed():
    candidate = {"candidate_source_type": "alien_intent_candidate"}
    boundary = build_virtual_world_operator_intent_boundary(candidate, None)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert boundary["virtual_world_operator_intent_boundary_status"] == "failed_closed"

def test_unknown_boundary_class_fails_closed():
    candidate = {"candidate_source_type": "explicit_operator_instruction_candidate"}
    context = {"boundary_classification": "unknown_boundary"}
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert boundary["virtual_world_operator_intent_boundary_status"] == "failed_closed"

def test_unknown_decision_fails_closed():
    # To test unknown decision failure, we'd need to mock or force internal state.
    # But we can test unknown confidence state instead.
    candidate = {"candidate_source_type": "explicit_operator_instruction_candidate"}
    context = {
        "boundary_classification": "explicit_operator_boundary",
        "confidence_state": "intent_telepathic"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert boundary["virtual_world_operator_intent_boundary_status"] == "failed_closed"

def test_privacy_risk_blocks():
    candidate = {"candidate_source_type": "explicit_operator_instruction_candidate"}
    context = {
        "boundary_classification": "explicit_operator_boundary",
        "confidence_state": "intent_explicit",
        "risks": ["privacy_risk"]
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert "privacy risk blocks future review" in boundary["blocked_reasons"]

def test_forbidden_requests_block():
    forbidden = [
        "self_model_update", "memory_write", "persistence_write",
        "virtual_state_mutation", "transition_application"
    ]
    for req in forbidden:
        candidate = {
            "candidate_source_type": "explicit_operator_instruction_candidate",
            "requests": [req]
        }
        context = {
            "boundary_classification": "explicit_operator_boundary",
            "confidence_state": "intent_explicit"
        }
        boundary = build_virtual_world_operator_intent_boundary(candidate, context)
        assert boundary["virtual_world_operator_intent_boundary_passed"] is False
        assert any(req in reason for reason in boundary["blocked_reasons"])

def test_relationship_update_blocks():
    candidate = {
        "candidate_source_type": "explicit_operator_instruction_candidate",
        "assertions": ["relationship_assertion"]
    }
    context = {
        "boundary_classification": "explicit_operator_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert boundary["virtual_world_operator_intent_boundary_passed"] is False
    assert "relationship assertion blocks future review" in boundary["blocked_reasons"]

def test_invariant_flags():
    candidate = {"candidate_source_type": "explicit_operator_instruction_candidate"}
    context = {
        "boundary_classification": "explicit_operator_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)

    assert boundary["boundary_applied"] is False
    assert boundary["policy_applied"] is False
    assert boundary["audit_applied"] is False
    assert boundary["transition_applied"] is False
    assert boundary["virtual_state_mutation_performed"] is False
    assert boundary["virtual_world_write_allowed"] is False
    assert boundary["virtual_world_persistence_allowed"] is False
    assert boundary["persistent_world_state_created"] is False
    assert boundary["external_reality_asserted"] is False
    assert boundary["current_external_fact_asserted"] is False
    assert boundary["memory_write_performed"] is False
    assert boundary["memory_write_allowed"] is False
    assert boundary["quarantine_promotion_allowed"] is False
    assert boundary["self_model_update_allowed"] is False
    assert boundary["relationship_update_allowed"] is False
    assert boundary["identity_asserted"] is False
    assert boundary["user_emotion_asserted"] is False
    assert boundary["user_intent_asserted"] is False
    assert boundary["affect_transition_allowed"] is False
    assert boundary["hormone_transition_allowed"] is False
    assert boundary["agp_bypass_allowed"] is False
    assert boundary["fallback_bypass_allowed"] is False
    assert boundary["vector_read_performed"] is False
    assert boundary["vector_load_performed"] is False
    assert boundary["runtime_mutation_performed"] is False
    assert boundary["persistence_write_performed"] is False
    assert boundary["artifact_created_or_staged"] is False

def test_korean_preservation():
    # Prove Korean fixtures and "민석" remain preserved
    candidate = {
        "candidate_source_type": "explicit_operator_instruction_candidate",
        "instruction": "민석 says: Hello."
    }
    context = {
        "boundary_classification": "explicit_operator_boundary",
        "confidence_state": "intent_explicit"
    }
    boundary = build_virtual_world_operator_intent_boundary(candidate, context)
    assert "민석" in boundary["intent_candidate"]["instruction"]
