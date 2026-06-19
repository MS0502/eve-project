"""Operator report for round1381-1400 constraint context schema."""

import copy
import json
import sys

from adapters.virtual_world_situation_constraint_context_schema import (
    FORBIDDEN_REQUEST_FIELDS,
    SUPPORTED_BOUNDARIES,
    build_constraint_context_to_agp_input_plan,
    build_constraint_context_to_situation_plan,
    build_virtual_world_situation_constraint_context,
    virtual_world_situation_constraint_context_schema_summary,
    validate_virtual_world_situation_constraint_context,
)


def clause(cid="c1", kind="must_hold_candidate", subj="s1", obj=None, **extra):
    data = {"clause_id": cid, "situation_id": "sit-민석", "clause_kind": kind, "subject_ref_id": subj}
    if obj is not None:
        data["object_ref_id"] = obj
    data.update(extra)
    return data


def valid(**kw):
    args = {"constraint_type": "hard_constraint_candidate", "situation_id": "sit-민석", "constraint_clauses": [clause()], "metadata": {}}
    args.update(kw)
    return build_virtual_world_situation_constraint_context(**args)


def rejected(payload, reason):
    return payload["situation_constraint_context_passed"] is False and payload["blocked_reasons"] == [reason]


class RaisingDict(dict):
    def keys(self):
        raise RuntimeError("hostile")


class RaisingList(list):
    def __iter__(self):
        raise RuntimeError("hostile")


def deep_list(depth):
    value = "leaf"
    for _ in range(depth):
        value = [value]
    return value

def plan_safe(plan):
    return plan["ready"] is False and all(
        value is False
        for key, value in plan.items()
        if key.endswith("_performed") or key.endswith("_allowed") or key in {"constraint_enforced", "action_prevented", "action_permitted", "schedule_created", "resource_reserved", "resource_consumed"}
    )


def main():
    p = valid()
    conflict = valid(constraint_type="conflicting_constraint_candidate", constraint_clauses=[clause("a", "must_hold_candidate"), clause("b", "must_not_hold_candidate")])
    mixed = valid(constraint_type="mixed_unknown_constraint_candidate", constraint_clauses=[clause("a", "must_hold_candidate"), clause("b", "should_hold_candidate")])
    tampered = copy.deepcopy(p); tampered["constraint_scope_summary"]["unary_clause_count"] = True
    rid1 = valid(constraint_clauses=[{"z": False, **clause("b", "must_not_hold_candidate")}, {"a": False, **clause("a", "must_hold_candidate")}])
    rid2 = valid(constraint_clauses=[{"a": False, **clause("a", "must_hold_candidate")}, {"z": False, **clause("b", "must_not_hold_candidate")}])
    huge_invalid = valid(constraint_clauses=[clause("h", strength_candidate=10**1000)])
    deep_invalid = valid(metadata={"deep": deep_list(101)})
    hostile_invalid = valid(metadata=RaisingDict({"x": 1}))
    adversarial_sources = [
        valid(constraint_type="bad"), "not-a-dict", {"bad": {"x"}}, RaisingDict({"x": 1}), RaisingList([1]),
        huge_invalid, deep_invalid, hostile_invalid, valid(metadata={"constraint_boundary_classification": None}),
        valid(metadata={"constraint_confidence_state": None}), valid(metadata={"memory_write_requested": True}),
        valid(constraint_clauses=[clause("x", tool_execution_requested=True)]), valid(constraint_clauses=[clause("x"), clause("x", "must_not_hold_candidate")]),
        valid(constraint_clauses=[clause("x"), clause("y")]), valid(constraint_clauses=[clause("x", "must_hold_candidate", situation_id="other")]),
        valid(constraint_type="precondition_constraint_candidate", constraint_clauses=[clause("r", "requires_candidate")]),
        valid(constraint_type="soft_constraint_candidate"), valid(constraint_clauses=[clause("a", "must_hold_candidate"), clause("b", "must_not_hold_candidate")]),
        valid(constraint_type="conflicting_constraint_candidate", constraint_clauses=[clause("x")]),
        valid(constraint_clauses=[clause("a", memory_write_requested=True), clause("b", tool_execution_requested=True)]),
    ]
    results = {
        "valid_case_passed": p["situation_constraint_context_passed"] and validate_virtual_world_situation_constraint_context(p),
        "invalid_case_passed": rejected(valid(constraint_type="bad"), "unknown_constraint_type"),
        "deterministic_id_passed": p["constraint_context_id"] == valid()["constraint_context_id"] and p["constraint_context_id"] != valid(metadata={"note":"x"})["constraint_context_id"],
        "type_exact_tamper_detection_passed": not validate_virtual_world_situation_constraint_context(tampered),
        "clause_compatibility_passed": rejected(valid(constraint_type="soft_constraint_candidate"), "incompatible_constraint_type_clause_kind"),
        "reference_shape_validation_passed": rejected(valid(constraint_type="precondition_constraint_candidate", constraint_clauses=[clause("r", "requires_candidate")]), "missing_object_ref_id_for_relational_clause"),
        "clause_situation_coherence_passed": rejected(valid(constraint_clauses=[clause("x", "must_hold_candidate", situation_id="other")]), "constraint_clause_situation_mismatch"),
        "duplicate_semantic_clause_passed": rejected(valid(constraint_clauses=[clause("a"), clause("b")]), "duplicate_semantic_constraint_clause"),
        "conflict_detection_passed": conflict["constraint_conflict_summary"]["conflict_present"] is True,
        "conflict_type_coherence_passed": rejected(valid(constraint_clauses=[clause("a", "must_hold_candidate"), clause("b", "must_not_hold_candidate")]), "constraint_conflict_requires_conflicting_type"),
        "conflict_pair_determinism_passed": rid1["constraint_conflict_summary"] == rid2["constraint_conflict_summary"],
        "mixed_boundary_passed": mixed["constraint_boundary_classification"] == "mixed_virtual_external_constraint_boundary",
        "enforcement_remains_false_passed": p["enforcement_summary"]["enforced"] is False and p["action_permitted"] is False,
        "malformed_boundary_validation_passed": rejected(valid(metadata={"constraint_boundary_classification": None}), "malformed_constraint_boundary_class"),
        "malformed_confidence_validation_passed": rejected(valid(metadata={"constraint_confidence_state": None}), "malformed_constraint_confidence_state"),
        "malformed_unknown_enum_distinction_passed": rejected(valid(metadata={"constraint_boundary_classification": "None"}), "unknown_constraint_boundary_class") and all(valid(metadata={"constraint_boundary_classification": b})["situation_constraint_context_passed"] for b in SUPPORTED_BOUNDARIES),
        "recursive_forbidden_metadata_passed": rejected(valid(metadata={"nested": [{"memory_write_requested": True}]}), "memory_write_requested"),
        "recursive_forbidden_clause_passed": rejected(valid(constraint_clauses=[clause("x", extra={"tool_execution_requested": True})]), "tool_execution_requested"),
        "scheduling_alias_forbidden_passed": rejected(valid(metadata={"cron_expression": True}), "cron_expression"),
        "invalid_clause_reordering_deterministic_passed": valid(constraint_clauses=[{"tool_execution_requested": True}, {"memory_write_requested": True}]) == valid(constraint_clauses=[{"memory_write_requested": True}, {"tool_execution_requested": True}]),
        "non_json_fail_closed_passed": rejected(valid(metadata={"bad": {"x"}}), "non_json_serializable_semantic_input"),
        "huge_integer_strength_reason_passed": rejected(huge_invalid, "strength_candidate_out_of_range") and valid(constraint_clauses=[clause("one", strength_candidate=1)])["situation_constraint_context_passed"] and valid(constraint_clauses=[clause("onef", strength_candidate=1.0)])["situation_constraint_context_passed"],
        "hostile_container_fail_closed_passed": rejected(hostile_invalid, "non_json_serializable_semantic_input") and rejected(valid(constraint_clauses=RaisingList([clause()])), "non_json_serializable_semantic_input"),
        "deep_recursion_fail_closed_passed": rejected(deep_invalid, "non_json_serializable_semantic_input") and rejected(valid(constraint_clauses=[clause("deep", unknown=deep_list(101))]), "non_json_serializable_semantic_input"),
        "forbidden_string_value_passed": rejected(valid(metadata={"memory_write_requested": "yes"}), "malformed_forbidden_request_field") and rejected(valid(constraint_clauses=[clause("x", tool_execution_requested="yes")]), "malformed_forbidden_request_field"),
        "downstream_adversarial_coverage_passed": all(plan_safe(build_constraint_context_to_situation_plan(source)) and plan_safe(build_constraint_context_to_agp_input_plan(source)) for source in adversarial_sources),
        "no_side_effects_passed": all(p[k] is False for k in ("constraint_enforced", "memory_write_performed", "schedule_created", "vector_read_performed", "model_loaded", "network_action_performed", "agp_bypass_allowed", "fallback_bypass_allowed")) and build_constraint_context_to_situation_plan(p)["ready"] and not build_constraint_context_to_agp_input_plan({})["ready"],
        "schema_summary": virtual_world_situation_constraint_context_schema_summary(),
    }
    results["all_checks_passed"] = all(v for k, v in results.items() if k not in {"schema_summary", "all_checks_passed"})
    print(json.dumps(results, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    return 0 if results["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
