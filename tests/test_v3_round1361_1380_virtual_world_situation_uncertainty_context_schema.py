import copy
import json
import math

from adapters.virtual_world_situation_uncertainty_context_schema import (
    FORBIDDEN_REQUEST_FIELDS,
    IMMUTABLE_FALSE_FLAGS,
    IMMUTABLE_TRUE_FLAGS,
    SUPPORTED_BOUNDARIES,
    SUPPORTED_CONFIDENCE,
    SUPPORTED_FACTOR_KINDS,
    SUPPORTED_POLARITIES,
    SUPPORTED_UNCERTAINTY_TYPES,
    build_uncertainty_context_to_agp_input_plan,
    build_uncertainty_context_to_appraisal_plan,
    build_uncertainty_context_to_memory_candidate_plan,
    build_uncertainty_context_to_situation_plan,
    build_uncertainty_context_to_snapshot_plan,
    build_uncertainty_context_to_transition_preflight_plan,
    build_virtual_world_situation_uncertainty_context,
    validate_virtual_world_situation_uncertainty_context,
    virtual_world_situation_uncertainty_context_schema_summary,
)


def factor(fid="f1", kind="missing_information", pol="neutral_unknown", sid="sit-민석", **extra):
    data = {"factor_id": fid, "situation_id": sid, "factor_kind": kind, "polarity": pol}
    data.update(extra)
    return data


def valid_case(**kw):
    args = {"uncertainty_type": "missing_information_candidate", "situation_id": "sit-민석", "uncertainty_factors": [factor()], "metadata": {}}
    args.update(kw)
    return build_virtual_world_situation_uncertainty_context(**args)


def assert_reject(payload, reason):
    assert payload["situation_uncertainty_context_passed"] is False
    assert payload["situation_uncertainty_context_status"] == "REJECTED"
    assert payload["blocked_reasons"] == [reason]
    json.dumps(payload, allow_nan=False)


def test_literal_top_level_fields_and_nested_safety_contract():
    p = valid_case()
    assert set(p) == {
        "situation_uncertainty_context_passed", "situation_uncertainty_context_status", "uncertainty_context_id",
        "canonical_id_algorithm", "uncertainty_type", "uncertainty_boundary_classification", "uncertainty_confidence_state",
        "situation_id", "uncertainty_factors", "metadata", "origin_summary", "fact_status_summary", "uncertainty_scope_summary",
        "evidence_balance_summary", "resolution_summary", "uncertainty_flags", "boundary_flags", "uncertainty_integrity_flags",
        "candidate_only_fields", "blocked_reasons", "warnings", "external_uncertainty_asserted", "uncertainty_fact_asserted",
        "external_evidence_verified", "resolved_truth_asserted", "uncertainty_resolved", "external_resolution_verified",
        "automatic_resolution_performed", "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted",
        "random_sampling_performed", "probability_draw_performed", "intervention_performed", "event_completion_asserted",
        "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed", "relationship_update_allowed",
        "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed", "world_state_mutation_performed",
        "transition_applied", "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed",
        "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed",
        "network_action_performed", "artifact_created_or_staged", "agp_bypass_allowed", "fallback_bypass_allowed",
        "uncertainty_candidate_only", "situation_uncertainty_only", "read_only", "situation_review_required", "snapshot_review_required",
        "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required",
        "uncertainty_integrity_review_required", "appraisal_required", "agp_input_required",
    }
    assert p["origin_summary"] == {"origin_kind": "internal_virtual_uncertainty_candidate", "external_origin_verified": False}
    assert p["fact_status_summary"] == {"candidate_only": True, "external_uncertainty_fact_verified": False, "uncertainty_fact_asserted": False, "resolved_truth_asserted": False, "prediction_guaranteed": False}
    assert p["resolution_summary"] == {"resolution_candidate_only": True, "resolved": False, "external_resolution_verified": False, "automatic_resolution_allowed": False}
    for k in IMMUTABLE_FALSE_FLAGS: assert p[k] is False
    for k in IMMUTABLE_TRUE_FLAGS: assert p[k] is True
    assert validate_virtual_world_situation_uncertainty_context(p)


def test_supported_types_boundaries_confidence_and_behavior_flags():
    examples = {
        "missing_information_candidate": [factor("f1", "missing_information")],
        "conflicting_evidence_candidate": [factor("f1", "supporting_evidence", "support_candidate"), factor("f2", "challenging_evidence", "challenge_candidate")],
        "source_reliability_uncertainty_candidate": [factor("f1", "source_unverified")],
        "temporal_uncertainty_candidate": [factor("f1", "temporal_ambiguity")],
        "causal_uncertainty_candidate": [factor("f1", "causal_ambiguity")],
        "state_uncertainty_candidate": [factor("f1", "state_ambiguity")],
        "identity_uncertainty_candidate": [factor("f1", "identity_ambiguity")],
        "boundary_uncertainty_candidate": [factor("f1", "boundary_ambiguity")],
        "stochastic_uncertainty_candidate": [factor("f1", "stochastic_variation")],
        "ambiguity_candidate": [factor("f1", "neutral_unknown"), factor("f2", "temporal_ambiguity")],
        "simulation_uncertainty_candidate": [factor("f1", "simulation_divergence")],
        "symbolic_uncertainty_candidate": [factor("f1", "symbolic_ambiguity")],
        "dmn_uncertainty_candidate": [factor("f1", "dmn_ambiguity")],
        "dream_uncertainty_candidate": [factor("f1", "dream_ambiguity")],
        "unknown_origin_uncertainty_candidate": [factor("f1", "unknown_origin")],
        "mixed_unknown_uncertainty_candidate": [factor("f1", "missing_information"), factor("f2", "unknown_origin")],
    }
    assert set(examples) == set(SUPPORTED_UNCERTAINTY_TYPES)
    for typ, factors in examples.items():
        p = valid_case(uncertainty_type=typ, uncertainty_factors=factors, metadata={"uncertainty_confidence_state":"uncertainty_high_confidence_but_not_fact"})
        assert p["situation_uncertainty_context_passed"] is True, typ
        assert p["fact_status_summary"]["uncertainty_fact_asserted"] is False
        assert p["resolution_summary"]["resolved"] is False
        assert p["identity_asserted"] is False and p["self_model_update_allowed"] is False
        assert p["random_sampling_performed"] is False
    for b in SUPPORTED_BOUNDARIES:
        assert valid_case(metadata={"uncertainty_boundary_classification": b})["situation_uncertainty_context_passed"]
    for c in SUPPORTED_CONFIDENCE:
        p = valid_case(metadata={"uncertainty_confidence_state": c})
        assert p["situation_uncertainty_context_passed"]
    conflict = valid_case(uncertainty_type="conflicting_evidence_candidate", uncertainty_factors=examples["conflicting_evidence_candidate"], metadata={"uncertainty_confidence_state":"uncertainty_conflict_detected"})
    assert conflict["evidence_balance_summary"] == {"supporting_factor_count":1,"challenging_factor_count":1,"neutral_factor_count":0,"conflict_present":True,"external_evidence_verified":False}
    assert "conflict_review_required" in conflict["uncertainty_integrity_flags"]
    unknown = valid_case(uncertainty_type="unknown_origin_uncertainty_candidate", uncertainty_factors=examples["unknown_origin_uncertainty_candidate"], metadata={"uncertainty_confidence_state":"uncertainty_origin_unknown"})
    assert "origin_unknown" in unknown["uncertainty_flags"]
    mixed = valid_case(uncertainty_type="mixed_unknown_uncertainty_candidate", uncertainty_factors=examples["mixed_unknown_uncertainty_candidate"])
    assert mixed["blocked_reasons"] == [] and mixed["warnings"]


def test_input_failures_and_validation_precedence():
    assert_reject(valid_case(uncertainty_type=None), "missing_uncertainty_type")
    assert_reject(valid_case(uncertainty_type="bad"), "unknown_uncertainty_type")
    assert_reject(valid_case(situation_id=""), "missing_or_malformed_situation_id")
    assert_reject(valid_case(metadata=[]), "invalid_metadata")
    assert_reject(valid_case(uncertainty_factors=None), "missing_uncertainty_factors")
    assert_reject(valid_case(uncertainty_factors="x"), "uncertainty_factors_not_non_empty_list")
    assert_reject(valid_case(uncertainty_factors=[]), "empty_uncertainty_factors")
    assert_reject(valid_case(uncertainty_factors=["x"]), "malformed_uncertainty_factor")
    assert_reject(valid_case(uncertainty_factors=[{"situation_id":"sit-민석","factor_kind":"missing_information","polarity":"neutral_unknown"}]), "missing_or_empty_factor_id")
    assert_reject(valid_case(uncertainty_factors=[factor("f1"), factor("f1", related_context_id="r")]), "duplicate_factor_id")
    assert_reject(valid_case(uncertainty_factors=[factor("f1"), factor("f2", label="x", weight_candidate=0.2, extra="y")]), "duplicate_semantic_uncertainty_factor")
    assert_reject(valid_case(uncertainty_factors=[factor("f1", sid="other")]), "uncertainty_factor_situation_mismatch")
    assert_reject(valid_case(uncertainty_factors=[factor("f1", kind="bad")]), "unknown_factor_kind")
    assert_reject(valid_case(uncertainty_factors=[factor("f1", pol="bad")]), "unknown_factor_polarity")
    assert_reject(valid_case(uncertainty_factors=[factor("f1", related_context_id="")]), "malformed_related_context_id")
    assert_reject(valid_case(uncertainty_factors=[factor("f1", weight_candidate=True)]), "malformed_weight_candidate")
    assert_reject(valid_case(uncertainty_factors=[factor("f1", weight_candidate=math.nan)]), "non_json_serializable_semantic_input")
    assert_reject(valid_case(uncertainty_factors=[factor("f1", weight_candidate=2)]), "weight_candidate_out_of_range")
    assert_reject(valid_case(uncertainty_factors=[factor("f1", label="")]), "malformed_uncertainty_factor_label")
    assert_reject(valid_case(uncertainty_factors=[factor("f1", "temporal_ambiguity", "support_candidate")]), "incompatible_factor_kind_polarity")
    assert_reject(valid_case(uncertainty_type="temporal_uncertainty_candidate", uncertainty_factors=[factor("f1", "causal_ambiguity")]), "incompatible_uncertainty_type_factor_kind")
    assert_reject(valid_case(uncertainty_type="conflicting_evidence_candidate", uncertainty_factors=[factor("f1", "supporting_evidence", "support_candidate")]), "conflicting_evidence_requires_multiple_factors")
    assert_reject(valid_case(uncertainty_type="conflicting_evidence_candidate", uncertainty_factors=[factor("f1", "supporting_evidence", "support_candidate"), factor("f2", "supporting_evidence", "support_candidate", related_context_id="r")]), "conflicting_evidence_missing_challenge")
    assert_reject(valid_case(uncertainty_type="conflicting_evidence_candidate", uncertainty_factors=[factor("f1", "challenging_evidence", "challenge_candidate"), factor("f2", "challenging_evidence", "challenge_candidate", related_context_id="r")]), "conflicting_evidence_missing_support")
    assert_reject(valid_case(uncertainty_type="ambiguity_candidate", uncertainty_factors=[factor("f1", "neutral_unknown")]), "ambiguity_requires_multiple_factors")
    assert_reject(valid_case(uncertainty_type="ambiguity_candidate", uncertainty_factors=[factor("f1", "neutral_unknown"), factor("f2", "neutral_unknown", related_context_id="r")]), "ambiguity_requires_distinct_factor_kinds")
    assert_reject(valid_case(uncertainty_type="mixed_unknown_uncertainty_candidate", uncertainty_factors=[factor("f1", "missing_information")]), "mixed_unknown_requires_multiple_factors")
    assert_reject(build_virtual_world_situation_uncertainty_context("bad", 1, {1: object()}, []), "non_json_serializable_semantic_input")
    assert_reject(valid_case(metadata={"uncertainty_boundary_classification":""}), "malformed_uncertainty_boundary_class")
    assert_reject(valid_case(metadata={"uncertainty_boundary_classification":"bad"}), "unknown_uncertainty_boundary_class")
    assert_reject(valid_case(metadata={"uncertainty_confidence_state":""}), "malformed_uncertainty_confidence_state")
    assert_reject(valid_case(metadata={"uncertainty_confidence_state":"bad"}), "unknown_uncertainty_confidence_state")


def test_determinism_and_tamper_detection():
    p1 = valid_case(metadata={"b":2,"a":1}, uncertainty_factors=[factor("f2", label="b"), factor("f1", related_context_id="r", x={"z":1})])
    p2 = valid_case(metadata={"a":1,"b":2}, uncertainty_factors=[{"x":{"z":1},"related_context_id":"r","polarity":"neutral_unknown","factor_kind":"missing_information","situation_id":"sit-민석","factor_id":"f1"}, {"label":"b","polarity":"neutral_unknown","factor_kind":"missing_information","situation_id":"sit-민석","factor_id":"f2"}])
    assert p1["uncertainty_context_id"] == p2["uncertainty_context_id"]
    for changed in [valid_case(metadata={"a":2}), valid_case(uncertainty_factors=[factor(label="x")]), valid_case(uncertainty_factors=[factor(weight_candidate=0.5)]), valid_case(uncertainty_factors=[factor(x="y")])]:
        assert changed["uncertainty_context_id"] != valid_case()["uncertainty_context_id"]
    for key, value in [("situation_uncertainty_context_passed", False),("situation_uncertainty_context_status","X"),("blocked_reasons",["x"]),("uncertainty_context_id","bad"),("canonical_id_algorithm",None),("uncertainty_type","temporal_uncertainty_candidate"),("uncertainty_boundary_classification","temporal_virtual_uncertainty"),("uncertainty_confidence_state","uncertainty_low_confidence"),("situation_id","other"),("metadata",{"x":1}),("uncertainty_factors",[factor("f9")]),("origin_summary",{}),("fact_status_summary",{}),("uncertainty_scope_summary",{}),("evidence_balance_summary",{}),("resolution_summary",{}),("uncertainty_flags",[]),("boundary_flags",[]),("uncertainty_integrity_flags",[]),("candidate_only_fields",[]),("warnings",["x"]),("memory_write_performed",True),("read_only",False)]:
        t = copy.deepcopy(valid_case()); t[key] = value
        assert not validate_virtual_world_situation_uncertainty_context(t), key
    t = copy.deepcopy(valid_case()); del t["metadata"]; assert not validate_virtual_world_situation_uncertainty_context(t)
    t = copy.deepcopy(valid_case()); t["extra"] = 1; assert not validate_virtual_world_situation_uncertainty_context(t)
    t = copy.deepcopy(valid_case()); t["memory_write_requested"] = True; assert not validate_virtual_world_situation_uncertainty_context(t)


def test_strict_json_and_forbidden_requests_and_plans():
    class BadStr:
        def __str__(self): raise RuntimeError("bad")
        def __repr__(self): raise RuntimeError("bad")
    cyc = {}; cyc["x"] = cyc
    cyc_list = []; cyc_list.append(cyc_list)
    bad_values = [{1:"x"}, (1,), {1}, frozenset([1]), b"x", bytearray(b"x"), math.nan, math.inf, -math.inf, BadStr(), cyc, cyc_list]
    for v in bad_values:
        assert_reject(valid_case(metadata={"x": v}), "non_json_serializable_semantic_input")
    deep = cur = {}
    for i in range(1002): cur["x"] = {}; cur = cur["x"]
    assert_reject(valid_case(metadata=deep), "non_json_serializable_semantic_input")
    for bad in ["", "None", "False", "0", "[]", "{}"]:
        assert_reject(valid_case(metadata={"uncertainty_boundary_classification": bad}), "malformed_uncertainty_boundary_class")
        assert_reject(valid_case(metadata={"uncertainty_confidence_state": bad}), "malformed_uncertainty_confidence_state")
    for field in FORBIDDEN_REQUEST_FIELDS:
        assert_reject(valid_case(metadata={field: True}), field)
        assert valid_case(metadata={field: False})["situation_uncertainty_context_passed"]
        assert_reject(valid_case(metadata={field: 1}), "malformed_forbidden_request_field")
        assert_reject(valid_case(metadata={"n":[{field: True}]}), field)
        assert_reject(valid_case(uncertainty_factors=[factor(**{field: True})]), field)
        assert_reject(valid_case(uncertainty_factors=[factor(n={"d": [{field: True}]})]), field)
    builders = [build_uncertainty_context_to_situation_plan, build_uncertainty_context_to_snapshot_plan, build_uncertainty_context_to_transition_preflight_plan, build_uncertainty_context_to_memory_candidate_plan, build_uncertainty_context_to_appraisal_plan, build_uncertainty_context_to_agp_input_plan]
    invalids = [valid_case(uncertainty_type="bad"), None, [], {1:"x"}, copy.deepcopy(valid_case()), valid_case(metadata={"uncertainty_boundary_classification":""}), valid_case(metadata={"uncertainty_confidence_state":""}), valid_case(metadata={"memory_write_requested":True}), valid_case(uncertainty_factors=[factor(memory_write_requested=True)]), valid_case(uncertainty_factors=[factor("f1"), factor("f2")]), valid_case(uncertainty_factors=[factor(sid="other")]), valid_case(uncertainty_factors=[factor("f1","temporal_ambiguity","support_candidate")]), valid_case(uncertainty_type="temporal_uncertainty_candidate"), valid_case(uncertainty_type="ambiguity_candidate", uncertainty_factors=[factor("f1","neutral_unknown")]), valid_case(metadata=deep)]
    invalids[4]["uncertainty_context_id"] = "tamper"
    t = copy.deepcopy(valid_case()); t["resolution_summary"] = {}; invalids.append(t)
    t = copy.deepcopy(valid_case()); t["extra"] = 1; invalids.append(t)
    t = copy.deepcopy(valid_case()); t["read_only"] = False; invalids.append(t)
    for b in builders:
        assert b(valid_case())["ready"] is True
        for src in invalids:
            plan = b(src)
            assert plan["ready"] is False
            for key, value in plan.items():
                if key.endswith("_performed") or key.endswith("_allowed") or key.endswith("_loaded") or key.endswith("_asserted") or key.endswith("_guaranteed"):
                    assert value is False


def test_summary_contains_next_step():
    assert virtual_world_situation_uncertainty_context_schema_summary()["next_recommended_step"] == "read_only_virtual_world_situation_constraint_context_schema"
    assert set(SUPPORTED_FACTOR_KINDS) >= {"missing_information", "supporting_evidence", "challenging_evidence"}
    assert set(SUPPORTED_POLARITIES) == {"support_candidate", "challenge_candidate", "neutral_unknown"}
