import copy
import json
import math

from adapters.virtual_world_situation_constraint_context_schema import (
    FORBIDDEN_REQUEST_FIELDS,
    IMMUTABLE_FALSE_FLAGS,
    IMMUTABLE_TRUE_FLAGS,
    SUPPORTED_BOUNDARIES,
    SUPPORTED_CLAUSE_KINDS,
    SUPPORTED_CONFIDENCE,
    SUPPORTED_CONSTRAINT_TYPES,
    build_constraint_context_to_agp_input_plan,
    build_constraint_context_to_appraisal_plan,
    build_constraint_context_to_memory_candidate_plan,
    build_constraint_context_to_situation_plan,
    build_constraint_context_to_snapshot_plan,
    build_constraint_context_to_transition_preflight_plan,
    build_virtual_world_situation_constraint_context,
    validate_virtual_world_situation_constraint_context,
    virtual_world_situation_constraint_context_schema_summary,
)


def clause(cid="c1", kind="must_hold_candidate", sid="sit-민석", subj="subj", obj=None, **extra):
    data = {"clause_id": cid, "situation_id": sid, "clause_kind": kind, "subject_ref_id": subj}
    if obj is not None:
        data["object_ref_id"] = obj
    data.update(extra)
    return data


def build(**kw):
    args = {"constraint_type": "hard_constraint_candidate", "situation_id": "sit-민석", "constraint_clauses": [clause()], "metadata": {}}
    args.update(kw)
    return build_virtual_world_situation_constraint_context(**args)


def assert_reject(payload, reason):
    assert payload["situation_constraint_context_passed"] is False
    assert payload["situation_constraint_context_status"] == "REJECTED"
    assert payload["blocked_reasons"] == [reason]
    json.dumps(payload, allow_nan=False)


def test_literal_top_level_fields_and_safety_values():
    p = build()
    assert set(p) == {
        "situation_constraint_context_passed", "situation_constraint_context_status", "constraint_context_id", "canonical_id_algorithm",
        "constraint_type", "constraint_boundary_classification", "constraint_confidence_state", "situation_id", "constraint_clauses",
        "metadata", "origin_summary", "fact_status_summary", "constraint_scope_summary", "constraint_conflict_summary", "enforcement_summary",
        "constraint_flags", "boundary_flags", "constraint_integrity_flags", "candidate_only_fields", "blocked_reasons", "warnings",
        "external_constraint_asserted", "constraint_fact_asserted", "constraint_satisfied_asserted", "constraint_violated_asserted",
        "constraint_enforced", "automatic_enforcement_performed", "automatic_enforcement_allowed", "hard_block_applied", "soft_penalty_applied",
        "precondition_evaluated", "postcondition_evaluated", "invariant_evaluated", "resource_reserved", "resource_consumed",
        "schedule_created", "action_prevented", "action_permitted", "event_completion_asserted", "identity_asserted", "memory_fact_asserted",
        "memory_write_performed", "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed",
        "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed",
        "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded",
        "device_activation_performed", "network_action_performed", "artifact_created_or_staged", "agp_bypass_allowed", "fallback_bypass_allowed",
        "constraint_candidate_only", "situation_constraint_only", "read_only", "situation_review_required", "snapshot_review_required",
        "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required", "constraint_integrity_review_required",
        "enforcement_review_required", "appraisal_required", "agp_input_required",
    }
    assert p["origin_summary"] == {"origin_kind": "internal_virtual_constraint_candidate", "external_origin_verified": False}
    assert p["fact_status_summary"] == {"candidate_only": True, "external_constraint_fact_verified": False, "constraint_fact_asserted": False, "constraint_satisfied_asserted": False, "constraint_violated_asserted": False}
    assert p["enforcement_summary"] == {"enforcement_candidate_only": True, "enforced": False, "automatic_enforcement_allowed": False, "action_blocked": False, "action_permitted": False}
    for k in IMMUTABLE_FALSE_FLAGS:
        assert p[k] is False
    for k in IMMUTABLE_TRUE_FLAGS:
        assert p[k] is True
    assert validate_virtual_world_situation_constraint_context(p)


def test_supported_types_clause_kinds_boundaries_confidence_and_scope():
    examples = {
        "hard_constraint_candidate": [clause("c1", "must_hold_candidate")],
        "soft_constraint_candidate": [clause("c1", "should_hold_candidate")],
        "precondition_constraint_candidate": [clause("c1", "requires_candidate", obj="obj")],
        "postcondition_constraint_candidate": [clause("c1", "should_hold_candidate")],
        "invariant_constraint_candidate": [clause("c1", "must_not_hold_candidate")],
        "resource_constraint_candidate": [clause("c1", "resource_limit_candidate")],
        "temporal_constraint_candidate": [clause("c1", "before_candidate", obj="obj")],
        "causal_constraint_candidate": [clause("c1", "causal_guard_candidate")],
        "role_relation_constraint_candidate": [clause("c1", "role_relation_guard_candidate")],
        "uncertainty_guard_constraint_candidate": [clause("c1", "uncertainty_guard_candidate")],
        "boundary_constraint_candidate": [clause("c1", "boundary_limit_candidate")],
        "exclusion_constraint_candidate": [clause("c1", "excludes_candidate", obj="obj")],
        "dependency_constraint_candidate": [clause("c1", "depends_on_candidate", obj="obj")],
        "ordering_constraint_candidate": [clause("c1", "after_candidate", obj="obj")],
        "symbolic_constraint_candidate": [clause("c1", "symbolic_guard_candidate")],
        "conflicting_constraint_candidate": [clause("c1", "requires_candidate", obj="obj"), clause("c2", "excludes_candidate", obj="obj")],
        "mixed_unknown_constraint_candidate": [clause("c1", "must_hold_candidate"), clause("c2", "unknown_constraint_candidate")],
    }
    assert set(examples) == set(SUPPORTED_CONSTRAINT_TYPES)
    all_kinds = {c["clause_kind"] for clauses in examples.values() for c in clauses}
    assert set(SUPPORTED_CLAUSE_KINDS) == all_kinds | {"before_candidate", "must_not_hold_candidate"}
    for typ, clauses in examples.items():
        p = build(constraint_type=typ, constraint_clauses=clauses, metadata={"constraint_confidence_state": "constraint_high_confidence_but_not_enforced"})
        assert p["situation_constraint_context_passed"], typ
        assert p["constraint_fact_asserted"] is False and p["constraint_enforced"] is False and p["action_prevented"] is False
    for boundary in SUPPORTED_BOUNDARIES:
        assert build(metadata={"constraint_boundary_classification": boundary})["situation_constraint_context_passed"]
    for confidence in SUPPORTED_CONFIDENCE:
        p = build(metadata={"constraint_confidence_state": confidence})
        assert p["situation_constraint_context_passed"]
    unknown = build(metadata={"constraint_confidence_state": "constraint_origin_unknown"})
    assert "origin_unknown" in unknown["constraint_flags"]
    conflict_conf = build(metadata={"constraint_confidence_state": "constraint_conflict_detected"})
    assert conflict_conf["constraint_conflict_summary"]["conflict_pairs"] == []
    scope = build(constraint_type="precondition_constraint_candidate", constraint_clauses=[clause("r", "requires_candidate", obj="obj")])
    assert scope["constraint_scope_summary"]["relational_clause_count"] == 1


def test_input_failures_and_precedence():
    assert_reject(build(constraint_type=None), "missing_constraint_type")
    assert_reject(build(constraint_type="bad"), "unknown_constraint_type")
    assert_reject(build(situation_id=""), "missing_or_malformed_situation_id")
    assert_reject(build(metadata=[]), "invalid_metadata")
    assert_reject(build(metadata={"memory_write_requested": True}, constraint_clauses=None), "memory_write_requested")
    assert_reject(build(constraint_clauses=None), "missing_constraint_clauses")
    assert_reject(build(constraint_clauses="x"), "constraint_clauses_not_non_empty_list")
    assert_reject(build(constraint_clauses=[]), "empty_constraint_clauses")
    assert_reject(build(constraint_clauses=["x"]), "malformed_constraint_clause")
    assert_reject(build(constraint_clauses=[{"situation_id":"sit-민석","clause_kind":"must_hold_candidate","subject_ref_id":"s"}]), "missing_or_empty_clause_id")
    assert_reject(build(constraint_clauses=[clause("", "must_hold_candidate")]), "missing_or_empty_clause_id")
    assert_reject(build(constraint_clauses=[clause("x", "must_hold_candidate", sid="")]), "malformed_clause_situation_id")
    assert_reject(build(constraint_clauses=[clause("x", "bad")]), "unknown_clause_kind")
    assert_reject(build(constraint_clauses=[clause("x", "must_hold_candidate", subj="")]), "missing_or_malformed_subject_ref_id")
    assert_reject(build(constraint_clauses=[clause("x", "must_hold_candidate", extra={"tool_execution_requested": True})]), "tool_execution_requested")
    assert_reject(build(constraint_clauses=[clause("x"), clause("x", "must_not_hold_candidate")]), "duplicate_clause_id")
    assert_reject(build(constraint_clauses=[clause("x"), clause("y")]), "duplicate_semantic_constraint_clause")
    assert_reject(build(constraint_clauses=[clause("x", sid="other")]), "constraint_clause_situation_mismatch")
    assert_reject(build(constraint_type="precondition_constraint_candidate", constraint_clauses=[clause("r", "requires_candidate")]), "missing_object_ref_id_for_relational_clause")
    assert_reject(build(constraint_type="precondition_constraint_candidate", constraint_clauses=[clause("r", "requires_candidate", obj="")]), "malformed_object_ref_id")
    assert_reject(build(constraint_type="precondition_constraint_candidate", constraint_clauses=[clause("r", "requires_candidate", subj="a", obj="a")]), "identical_constraint_clause_refs")
    assert_reject(build(constraint_clauses=[clause("u", "must_hold_candidate", obj="o")]), "unexpected_object_ref_id_for_unary_clause")
    assert_reject(build(constraint_clauses=[clause("x", strength_candidate=True)]), "malformed_strength_candidate")
    assert_reject(build(constraint_clauses=[clause("x", strength_candidate=math.inf)]), "non_json_serializable_semantic_input")
    assert_reject(build(constraint_clauses=[clause("x", strength_candidate=2)]), "strength_candidate_out_of_range")
    assert_reject(build(constraint_clauses=[clause("x", priority_candidate=True)]), "malformed_priority_candidate")
    assert_reject(build(constraint_clauses=[clause("x", priority_candidate=1.2)]), "malformed_priority_candidate")
    assert_reject(build(constraint_clauses=[clause("x", priority_candidate=101)]), "priority_candidate_out_of_range")
    assert_reject(build(constraint_clauses=[clause("x", label="")]), "malformed_constraint_clause_label")
    assert_reject(build(constraint_type="soft_constraint_candidate"), "incompatible_constraint_type_clause_kind")
    assert_reject(build(constraint_type="conflicting_constraint_candidate", constraint_clauses=[clause("x")]), "conflicting_constraint_requires_multiple_clauses")
    assert_reject(build(constraint_type="conflicting_constraint_candidate", constraint_clauses=[clause("x"), clause("y", "must_not_hold_candidate", subj="other")]), "conflicting_constraint_requires_detected_conflict")
    assert_reject(build(constraint_type="mixed_unknown_constraint_candidate", constraint_clauses=[clause("x")]), "mixed_unknown_constraint_requires_multiple_clauses")
    assert_reject(build(constraint_type="mixed_unknown_constraint_candidate", constraint_clauses=[clause("x"), clause("y", subj="other")]), "mixed_unknown_constraint_requires_distinct_clause_kinds")
    assert_reject(build(metadata={"constraint_boundary_classification": None, "constraint_confidence_state": None}), "malformed_constraint_boundary_class")
    assert_reject(build(metadata={"constraint_confidence_state": None}), "malformed_constraint_confidence_state")


def test_conflict_semantics_and_determinism():
    cases = [
        ([clause("a", "must_hold_candidate"), clause("b", "must_not_hold_candidate")], "must_hold_vs_must_not_hold"),
        ([clause("a", "requires_candidate", obj="o"), clause("b", "excludes_candidate", obj="o")], "requires_vs_excludes"),
        ([clause("a", "before_candidate", obj="o"), clause("b", "after_candidate", obj="o")], "before_vs_after"),
    ]
    for clauses, kind in cases:
        p1 = build(constraint_type="conflicting_constraint_candidate", constraint_clauses=clauses)
        p2 = build(constraint_type="conflicting_constraint_candidate", constraint_clauses=list(reversed(clauses)))
        assert p1["constraint_conflict_summary"] == p2["constraint_conflict_summary"]
        assert p1["constraint_conflict_summary"]["conflict_pairs"][0]["conflict_kind"] == kind
    assert_reject(build(constraint_clauses=cases[0][0]), "constraint_conflict_requires_conflicting_type")
    mixed = build(constraint_type="mixed_unknown_constraint_candidate", constraint_clauses=cases[0][0])
    assert mixed["situation_constraint_context_passed"] and mixed["constraint_conflict_summary"]["conflict_present"]
    unrelated = build(constraint_type="mixed_unknown_constraint_candidate", constraint_clauses=[clause("a", "must_hold_candidate"), clause("b", "must_not_hold_candidate", subj="other")])
    assert unrelated["constraint_conflict_summary"]["conflict_pairs"] == []


def test_deterministic_ids_and_invalid_ordering():
    a = build(metadata={"b": 2, "a": 1}, constraint_clauses=[clause("b", "must_not_hold_candidate", subj="x"), clause("a")])
    b = build(metadata={"a": 1, "b": 2}, constraint_clauses=[{"subject_ref_id":"subj", "clause_kind":"must_hold_candidate", "situation_id":"sit-민석", "clause_id":"a"}, clause("b", "must_not_hold_candidate", subj="x")])
    assert a["constraint_context_id"] == b["constraint_context_id"]
    assert build()["constraint_context_id"] != build(constraint_clauses=[clause("id2")])["constraint_context_id"]
    assert build()["constraint_context_id"] != build(constraint_clauses=[clause("c1", label="x")])["constraint_context_id"]
    assert build()["constraint_context_id"] != build(constraint_clauses=[clause("c1", strength_candidate=0.5)])["constraint_context_id"]
    assert build()["constraint_context_id"] != build(constraint_clauses=[clause("c1", priority_candidate=5)])["constraint_context_id"]
    assert build()["constraint_context_id"] != build(constraint_clauses=[clause("c1", unknown={"x": 1})])["constraint_context_id"]
    pairs = [
        ([clause("a", a_request=True), clause("b", z_request=True)], [clause("b", z_request=True), clause("a", a_request=True)]),
        ([clause("a", cron_expression="yes"), clause("b", "must_not_hold_candidate", memory_write_requested=True)], [clause("b", "must_not_hold_candidate", memory_write_requested=True), clause("a", cron_expression="yes")]),
        ([{"clause_id":"a"}, clause("b", memory_write_requested=True)], [clause("b", memory_write_requested=True), {"clause_id":"a"}]),
        (["x", {"clause_id":"a"}], [{"clause_id":"a"}, "x"]),
        ([clause("x"), clause("x", "must_not_hold_candidate"), clause("y", "must_hold_candidate", sid="other")], [clause("y", "must_hold_candidate", sid="other"), clause("x", "must_not_hold_candidate"), clause("x")]),
    ]
    for left, right in pairs:
        assert build(constraint_clauses=left) == build(constraint_clauses=right)


def test_strict_json_and_boundary_confidence_distinction():
    bads = [{1: "x"}, (1,), {"x"}, frozenset({"x"}), b"x", bytearray(b"x"), float("nan"), float("inf")]
    for bad in bads:
        assert_reject(build(metadata={"bad": bad}), "non_json_serializable_semantic_input")
    d = {}; d["self"] = d
    assert_reject(build(metadata=d), "non_json_serializable_semantic_input")
    l = []; l.append(l)
    assert_reject(build(metadata={"l": l}), "non_json_serializable_semantic_input")
    assert_reject(build(metadata={"deep": [[[{"x": 1}]]]}), None) if False else None
    for key, malformed, unknown in [("constraint_boundary_classification", "malformed_constraint_boundary_class", "unknown_constraint_boundary_class"), ("constraint_confidence_state", "malformed_constraint_confidence_state", "unknown_constraint_confidence_state")]:
        for value in ["", None, False, 0, [], {}]:
            assert_reject(build(metadata={key: value}), malformed)
        for value in ["None", "False", "0", "[]", "{}", "unsupported"]:
            assert_reject(build(metadata={key: value}), unknown)


def test_recursive_forbidden_requests_and_plans():
    sample = sorted(FORBIDDEN_REQUEST_FIELDS)
    for field in sample:
        assert_reject(build(metadata={field: True}), field)
        assert build(metadata={field: False})["situation_constraint_context_passed"]
        assert_reject(build(metadata={field: 1}), "malformed_forbidden_request_field")
        assert_reject(build(metadata={"nested": [{field: True}]}), field)
        assert_reject(build(constraint_clauses=[clause("x", **{field: True})]), field)
        assert_reject(build(constraint_clauses=[clause("x", unknown={"nested": [{field: True}]})]), field)
    p = build()
    builders = [build_constraint_context_to_situation_plan, build_constraint_context_to_snapshot_plan, build_constraint_context_to_transition_preflight_plan, build_constraint_context_to_memory_candidate_plan, build_constraint_context_to_appraisal_plan, build_constraint_context_to_agp_input_plan]
    for fn in builders:
        plan = fn(p)
        assert plan["ready"] is True and plan["candidate_only"] is True and plan["read_only"] is True and plan["constraint_candidate_only"] is True
        for key, value in plan.items():
            if key.endswith("_performed") or key.endswith("_allowed") or key in {"action_prevented", "action_permitted", "schedule_created", "resource_reserved", "resource_consumed", "constraint_enforced", "constraint_fact_asserted", "constraint_satisfied_asserted", "constraint_violated_asserted"}:
                assert value is False
        assert fn({})["ready"] is False


def test_full_payload_tampering_and_summary():
    p = build()
    mutations = []
    for key in ["situation_constraint_context_passed", "situation_constraint_context_status", "blocked_reasons", "constraint_context_id", "canonical_id_algorithm", "constraint_type", "constraint_boundary_classification", "constraint_confidence_state", "situation_id", "constraint_clauses", "metadata", "origin_summary", "fact_status_summary", "constraint_scope_summary", "constraint_conflict_summary", "enforcement_summary", "constraint_flags", "boundary_flags", "constraint_integrity_flags", "candidate_only_fields", "warnings"]:
        q = copy.deepcopy(p); q[key] = (False if q[key] is not False else True); mutations.append(q)
    q = copy.deepcopy(p); q["constraint_enforced"] = True; mutations.append(q)
    q = copy.deepcopy(p); q["read_only"] = False; mutations.append(q)
    q = copy.deepcopy(p); del q["warnings"]; mutations.append(q)
    q = copy.deepcopy(p); q["extra"] = 1; mutations.append(q)
    q = copy.deepcopy(p); q["memory_write_requested"] = True; mutations.append(q)
    q = copy.deepcopy(p); q["constraint_scope_summary"]["unary_clause_count"] = True; mutations.append(q)
    q = copy.deepcopy(p); q["constraint_clauses"][0]["strength_candidate"] = 1; mutations.append(q)
    for q in mutations:
        assert not validate_virtual_world_situation_constraint_context(q)
    summary = virtual_world_situation_constraint_context_schema_summary()
    assert summary["next_recommended_step"] == "read_only_virtual_world_situation_evidence_context_schema"
    assert summary["read_only"] is True and summary["candidate_only"] is True
