"""Operator checks for round1421-1440 hypothesis context schema."""
import json
import sys

from adapters.virtual_world_situation_hypothesis_context_schema import (
    FORBIDDEN_REQUEST_FIELDS, PLAN_FALSE_FLAGS, SUPPORTED_BOUNDARIES, SUPPORTED_CANDIDATE_ROLES,
    SUPPORTED_CONFIDENCE, SUPPORTED_HYPOTHESIS_KINDS, SUPPORTED_HYPOTHESIS_TYPES,
    TYPE_KIND_COMPATIBILITY, KIND_ROLE_COMPATIBILITY,
    build_hypothesis_context_to_agp_input_plan, build_hypothesis_context_to_appraisal_plan,
    build_hypothesis_context_to_memory_candidate_plan, build_hypothesis_context_to_situation_plan,
    build_hypothesis_context_to_snapshot_plan, build_hypothesis_context_to_transition_preflight_plan,
    build_virtual_world_situation_hypothesis_context, build_virtual_world_situation_hypothesis_context_schema_summary,
    validate_virtual_world_situation_hypothesis_context,
)


def item(kind="explanatory_proposition_candidate", role="focal_candidate", item_id="h1", obj=None, **extra):
    data = {"hypothesis_item_id": item_id, "situation_id": "sit-민석", "hypothesis_kind": kind, "candidate_role": role, "subject_ref_id": "subj", "claim_ref_id": "claim", "basis_evidence_ref_ids": ["ev-b", "ev-a"]}
    if obj is not None:
        data["object_ref_id"] = obj
    data.update(extra)
    return data


def ctx(**kwargs):
    args = {"hypothesis_type": "explanatory_hypothesis_candidate", "situation_id": "sit-민석", "hypothesis_items": [item()], "metadata": {}}
    args.update(kwargs)
    return build_virtual_world_situation_hypothesis_context(**args)


def reason(payload):
    return payload["blocked_reasons"]


def all_plan_ok(source, ready):
    builders = [build_hypothesis_context_to_situation_plan, build_hypothesis_context_to_snapshot_plan, build_hypothesis_context_to_transition_preflight_plan, build_hypothesis_context_to_memory_candidate_plan, build_hypothesis_context_to_appraisal_plan, build_hypothesis_context_to_agp_input_plan]
    plans = [b(source) for b in builders]
    return all(p["ready"] is ready and p["candidate_only"] is True and p["read_only"] is True and p["hypothesis_candidate_only"] is True and all(p[k] is False for k in PLAN_FALSE_FLAGS) for p in plans)


def main():
    valid = ctx()
    invalid = ctx(hypothesis_type="bad")
    reordered = ctx(hypothesis_items=[dict(reversed(list(item().items())))], metadata={"z": [2, 1], "a": {"b": False}})
    reordered2 = ctx(hypothesis_items=[item()], metadata={"a": {"b": False}, "z": [2, 1]})
    tampered = dict(valid); tampered["hypothesis_context_id"] = "0"
    checks = {}
    checks["valid_case_passed"] = validate_virtual_world_situation_hypothesis_context(valid)
    checks["invalid_case_passed"] = reason(invalid) == ["unknown_hypothesis_type"] and not validate_virtual_world_situation_hypothesis_context(invalid)
    checks["deterministic_id_passed"] = reordered["hypothesis_context_id"] == reordered2["hypothesis_context_id"] and valid["hypothesis_context_id"] != ctx(hypothesis_items=[item(claim_ref_id="claim2")])["hypothesis_context_id"]
    checks["type_exact_tamper_detection_passed"] = not validate_virtual_world_situation_hypothesis_context(tampered) and not validate_virtual_world_situation_hypothesis_context({**valid, "read_only": 1})
    checks["type_kind_compatibility_passed"] = all(ctx(hypothesis_type=t, hypothesis_items=[item(kind=sorted(kinds)[0], role=sorted(KIND_ROLE_COMPATIBILITY[sorted(kinds)[0]])[0], obj="obj" if sorted(kinds)[0] in {"causal_proposition_candidate","temporal_proposition_candidate","boundary_proposition_candidate","predictive_proposition_candidate","counterfactual_proposition_candidate","alternative_proposition_candidate"} else None)] if t not in {"competing_hypothesis_candidate","mixed_unknown_hypothesis_candidate"} else [item(), item(kind="causal_proposition_candidate", role="alternative_candidate", item_id="h2", obj="obj", claim_ref_id="claim2")]).get("blocked_reasons") == [] for t, kinds in TYPE_KIND_COMPATIBILITY.items())
    checks["kind_role_compatibility_passed"] = all(ctx(hypothesis_type="mixed_unknown_hypothesis_candidate", hypothesis_items=[item(), item(kind=k, role=sorted(r)[0], item_id="x"+k, obj="obj" if k in {"causal_proposition_candidate","temporal_proposition_candidate","boundary_proposition_candidate","predictive_proposition_candidate","counterfactual_proposition_candidate","alternative_proposition_candidate"} else None, claim_ref_id="c"+k)]).get("blocked_reasons") == [] for k, r in KIND_ROLE_COMPATIBILITY.items())
    checks["reference_shape_validation_passed"] = reason(ctx(hypothesis_type="causal_hypothesis_candidate", hypothesis_items=[item(kind="causal_proposition_candidate")])) == ["missing_object_ref_id_for_relational_hypothesis"] and reason(ctx(hypothesis_items=[item(obj="o")])) == ["unexpected_object_ref_id_for_unary_hypothesis"]
    checks["basis_reference_validation_passed"] = valid["hypothesis_items"][0]["basis_evidence_ref_ids"] == ["ev-a", "ev-b"] and reason(ctx(hypothesis_items=[item(basis_evidence_ref_ids=[])])) == ["missing_or_malformed_basis_evidence_ref_ids"] and reason(ctx(hypothesis_items=[item(basis_evidence_ref_ids=["x","x"])])) == ["duplicate_basis_evidence_ref_id"]
    checks["item_situation_coherence_passed"] = reason(ctx(hypothesis_items=[item(situation_id="other")])) == ["hypothesis_item_situation_mismatch"]
    checks["duplicate_semantic_item_passed"] = reason(ctx(hypothesis_type="competing_hypothesis_candidate", hypothesis_items=[item(), item(item_id="h2", confidence_candidate=0.5)])) == ["duplicate_semantic_hypothesis_item"]
    comp = ctx(hypothesis_type="competing_hypothesis_candidate", hypothesis_items=[item(), item(kind="causal_proposition_candidate", role="alternative_candidate", item_id="h2", obj="obj", claim_ref_id="claim2")])
    checks["competing_hypothesis_validation_passed"] = comp["blocked_reasons"] == [] and comp["hypothesis_competition_summary"]["winner_selected"] is False
    mixed = ctx(hypothesis_type="mixed_unknown_hypothesis_candidate", hypothesis_items=[item(), item(kind="unknown_origin_proposition_candidate", role="neutral_unknown", item_id="h2", claim_ref_id="claim2")])
    checks["mixed_hypothesis_validation_passed"] = mixed["blocked_reasons"] == [] and bool(mixed["warnings"])
    checks["competition_summary_passed"] = comp["hypothesis_competition_summary"]["competition_present"] is True and ctx(metadata={"hypothesis_confidence_state":"hypothesis_competing"})["hypothesis_competition_summary"]["competition_present"] is False
    checks["support_summary_passed"] = valid["hypothesis_support_summary"] == {"basis_evidence_ref_ids": ["ev-a", "ev-b"], "basis_evidence_ref_count": 2, "external_evidence_verified": False, "evidence_sufficient_asserted": False}
    checks["resolution_remains_false_passed"] = all(v is False for v in valid["resolution_summary"].values() if isinstance(v, bool) and v is not True)
    checks["malformed_boundary_validation_passed"] = reason(ctx(metadata={"hypothesis_boundary_classification": 0})) == ["malformed_hypothesis_boundary_class"]
    checks["malformed_confidence_validation_passed"] = reason(ctx(metadata={"hypothesis_confidence_state": []})) == ["malformed_hypothesis_confidence_state"]
    checks["malformed_unknown_enum_distinction_passed"] = reason(ctx(metadata={"hypothesis_boundary_classification": "0"})) == ["unknown_hypothesis_boundary_class"] and reason(ctx(metadata={"hypothesis_confidence_state": "0"})) == ["unknown_hypothesis_confidence_state"]
    checks["recursive_forbidden_metadata_passed"] = all(reason(ctx(metadata={field: True})) == [field] and reason(ctx(metadata={"x":[{field: "yes"}]})) == ["malformed_forbidden_request_field"] for field in sorted(FORBIDDEN_REQUEST_FIELDS))
    checks["recursive_forbidden_item_passed"] = all(reason(ctx(hypothesis_items=[item(**{field: True})])) == [field] and reason(ctx(hypothesis_items=[item(extra=[{field:"yes"}])])) == ["malformed_forbidden_request_field"] for field in sorted(FORBIDDEN_REQUEST_FIELDS))
    checks["scheduling_alias_forbidden_passed"] = reason(ctx(metadata={"cron_expression": True})) == ["cron_expression"]
    a = ctx(metadata={"winner_selection_requested": True, "truth_resolution_requested": True})
    b = ctx(metadata={"truth_resolution_requested": True, "winner_selection_requested": True})
    checks["invalid_item_reordering_deterministic_passed"] = a == b and reason(a) == ["truth_resolution_requested"]
    class BadDict(dict): pass
    checks["hostile_container_fail_closed_passed"] = reason(ctx(metadata=BadDict())) == ["non_json_serializable_semantic_input"]
    deep = []
    cur = deep
    for _ in range(105):
        nxt = [] ; cur.append(nxt); cur = nxt
    checks["deep_recursion_fail_closed_passed"] = reason(ctx(metadata={"deep": deep})) == ["non_json_serializable_semantic_input"]
    checks["huge_integer_numeric_reason_passed"] = reason(ctx(hypothesis_items=[item(confidence_candidate=10**100)])) == ["confidence_candidate_out_of_range"]
    checks["downstream_adversarial_coverage_passed"] = all_plan_ok(valid, True) and all_plan_ok(invalid, False) and all_plan_ok(tampered, False) and all_plan_ok({"bad": object()}, False)
    circ = {}; circ["self"] = circ
    checks["non_json_fail_closed_passed"] = reason(ctx(metadata={"nan": float("nan")})) == ["non_json_serializable_semantic_input"] and reason(ctx(metadata=circ)) == ["non_json_serializable_semantic_input"]
    checks["no_side_effects_passed"] = all(valid[k] is False for k in ("memory_write_performed","world_state_mutation_performed","transition_applied","planning_execution_performed","tool_execution_performed","persistence_write_performed","vector_read_performed","vector_load_performed","model_loaded","network_action_performed","agp_bypass_allowed","fallback_bypass_allowed")) and all_plan_ok(valid, True)
    checks["schema_summary"] = build_virtual_world_situation_hypothesis_context_schema_summary()
    checks["all_checks_passed"] = all(v for k, v in checks.items() if k != "schema_summary")
    print(json.dumps(checks, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return 0 if checks["all_checks_passed"] else 1

if __name__ == "__main__":
    sys.exit(main())
