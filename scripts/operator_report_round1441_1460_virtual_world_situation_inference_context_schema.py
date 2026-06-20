import copy
import json
import math

from adapters.virtual_world_situation_inference_context_schema import (
    build_inference_context_to_agp_input_plan,
    build_inference_context_to_appraisal_plan,
    build_inference_context_to_memory_candidate_plan,
    build_inference_context_to_situation_plan,
    build_inference_context_to_snapshot_plan,
    build_inference_context_to_transition_preflight_plan,
    build_virtual_world_situation_inference_context,
    build_virtual_world_situation_inference_context_schema_summary,
    validate_virtual_world_situation_inference_context,
)

PLAN_FALSE = ("external_inference_asserted", "inference_verified", "inference_fact_asserted", "inference_accepted", "inference_rejected", "conclusion_asserted", "truth_resolved", "winner_selected", "competition_resolved", "automatic_resolution_performed", "prediction_guaranteed", "future_outcome_asserted", "counterfactual_applied", "basis_verified", "basis_sufficient_asserted", "random_sampling_performed", "probability_draw_performed", "memory_write_performed", "relationship_update_performed", "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed", "world_state_mutation_performed", "transition_applied", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "schedule_created", "action_prevented", "action_permitted", "planning_execution_performed", "tool_execution_performed", "agp_bypass_allowed", "fallback_bypass_allowed")
FALSE_FLAGS = ("external_inference_asserted", "inference_verified", "inference_fact_asserted", "inference_accepted", "inference_rejected", "conclusion_asserted", "truth_resolved", "winner_selected", "competition_resolved", "automatic_resolution_performed", "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted", "counterfactual_applied", "basis_verified", "basis_sufficient_asserted", "random_sampling_performed", "probability_draw_performed", "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "artifact_created_or_staged", "schedule_created", "action_prevented", "action_permitted", "agp_bypass_allowed", "fallback_bypass_allowed")
TRUE_FLAGS = ("inference_candidate_only", "situation_inference_only", "read_only", "situation_review_required", "snapshot_review_required", "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required", "inference_integrity_review_required", "basis_review_required", "competition_review_required", "resolution_review_required", "appraisal_required", "agp_input_required")
PLANS = [build_inference_context_to_situation_plan, build_inference_context_to_snapshot_plan, build_inference_context_to_transition_preflight_plan, build_inference_context_to_memory_candidate_plan, build_inference_context_to_appraisal_plan, build_inference_context_to_agp_input_plan]

def item(kind="deductive_step_candidate", role="focal_conclusion_candidate", source="internal_hypothesis_context", iid="i1"):
    d = {"inference_item_id": iid, "situation_id": "s1", "inference_kind": kind, "candidate_role": role, "derivation_source_class": source, "subject_ref_id": "민석", "conclusion_ref_id": "c" + iid, "premise_ref_ids": ["p2", "p1"], "hypothesis_context_ref_ids": ["h2", "h1"], "evidence_ref_ids": ["e1"], "confidence_candidate": 1, "coherence_candidate": 0.5, "label": "후보"}
    if kind in {"causal_step_candidate", "temporal_step_candidate", "boundary_step_candidate", "predictive_step_candidate", "counterfactual_step_candidate", "alternative_step_candidate"}: d["object_ref_id"] = "o" + iid
    return d

def build(**kw):
    args = {"inference_type": "deductive_inference_candidate", "situation_id": "s1", "inference_items": [item()], "metadata": {}}
    args.update(kw); return build_virtual_world_situation_inference_context(**args)

def blocked(ctx, r): return ctx.get("blocked_reasons") == [r]

def main():
    summary = build_virtual_world_situation_inference_context_schema_summary()
    valid = build(metadata={"k": [2, 1]})
    rejected = build(inference_type=None)
    plans_valid = [fn(valid) for fn in PLANS]
    plans_invalid = [fn({**valid, "inference_context_id": "bad"}) for fn in PLANS]
    report = {}
    report["valid_case_passed"] = validate_virtual_world_situation_inference_context(valid) and valid["blocked_reasons"] == []
    report["invalid_case_passed"] = rejected["situation_inference_context_passed"] is False and rejected["blocked_reasons"] == ["missing_inference_type"]
    report["literal_schema_contract_passed"] = summary["next_recommended_step"] == "read_only_virtual_world_situation_conclusion_candidate_schema" and len(summary["supported_inference_types"]) == 17 and len(summary["supported_inference_kinds"]) == 16
    report["deterministic_id_passed"] = valid["inference_context_id"] == build(metadata={"k": [2, 1]}, inference_items=[dict(reversed(list(item().items())))])["inference_context_id"] and valid["inference_context_id"] != build(inference_items=[{**item(), "conclusion_ref_id": "changed"}])["inference_context_id"]
    tampered = copy.deepcopy(valid); tampered["inference_support_summary"]["premise_ref_count"] = True
    report["type_exact_tamper_detection_passed"] = not validate_virtual_world_situation_inference_context(tampered) and not validate_virtual_world_situation_inference_context({**valid, "extra": False})
    report["type_kind_compatibility_passed"] = blocked(build(inference_type="causal_inference_candidate"), "incompatible_inference_type_kind") and build(inference_type="causal_inference_candidate", inference_items=[item("causal_step_candidate", "focal_conclusion_candidate", "internal_causal_context")])["blocked_reasons"] == []
    report["kind_role_compatibility_passed"] = blocked(build(inference_items=[item("counterfactual_step_candidate", "focal_conclusion_candidate", "internal_simulation")], inference_type="mixed_unknown_inference_candidate"), "incompatible_inference_kind_candidate_role")
    report["kind_source_compatibility_passed"] = blocked(build(inference_items=[item("symbolic_step_candidate", "focal_conclusion_candidate", "internal_dmn")], inference_type="mixed_unknown_inference_candidate"), "incompatible_inference_kind_derivation_source")
    report["reference_shape_validation_passed"] = blocked(build(inference_items=[{k:v for k,v in item("causal_step_candidate", "focal_conclusion_candidate", "internal_causal_context").items() if k != "object_ref_id"}], inference_type="causal_inference_candidate"), "missing_object_ref_id_for_relational_inference") and blocked(build(inference_items=[{**item(), "object_ref_id":"o"}]), "unexpected_object_ref_id_for_unary_inference")
    report["basis_reference_validation_passed"] = blocked(build(inference_items=[{**item(), "premise_ref_ids":["p", "p"]}]), "duplicate_premise_ref_id") and blocked(build(inference_items=[{**item(), "hypothesis_context_ref_ids":[]}]), "missing_or_malformed_hypothesis_context_ref_ids")
    report["item_situation_coherence_passed"] = blocked(build(inference_items=[{**item(), "situation_id":"other"}]), "inference_item_situation_mismatch")
    report["duplicate_semantic_item_passed"] = blocked(build(inference_items=[item("deductive_step_candidate", iid="i1"), {**item("deductive_step_candidate", iid="i2"), "conclusion_ref_id":"ci1"}]), "duplicate_semantic_inference_item")
    competing = build(inference_type="competing_inference_candidate", inference_items=[item("deductive_step_candidate", "focal_conclusion_candidate", "internal_hypothesis_context", "i1"), item("abductive_step_candidate", "alternative_conclusion_candidate", "internal_hypothesis_context", "i2")])
    report["competing_inference_validation_passed"] = competing["blocked_reasons"] == [] and competing["winner_selected"] is False
    mixed = build(inference_type="mixed_unknown_inference_candidate", inference_items=[item("deductive_step_candidate", "focal_conclusion_candidate", "internal_hypothesis_context", "i1"), item("unknown_origin_step_candidate", "neutral_unknown", "unknown_origin_candidate", "i2")])
    report["mixed_inference_validation_passed"] = mixed["blocked_reasons"] == [] and mixed["resolution_summary"]["resolved"] is False
    report["competition_summary_passed"] = competing["inference_competition_summary"]["competition_present"] is True and build(metadata={"inference_confidence_state":"inference_competing"})["inference_competition_summary"]["competition_present"] is False
    report["support_summary_passed"] = valid["inference_support_summary"]["premise_ref_ids"] == ["p1", "p2"] and valid["inference_support_summary"]["external_basis_verified"] is False
    report["resolution_remains_false_passed"] = all(valid["resolution_summary"][k] is False for k in ["resolved", "accepted", "rejected", "winner_selected", "automatic_resolution_allowed"])
    report["malformed_boundary_validation_passed"] = blocked(build(metadata={"inference_boundary_classification": []}), "malformed_inference_boundary_class")
    report["malformed_confidence_validation_passed"] = blocked(build(metadata={"inference_confidence_state": []}), "malformed_inference_confidence_state")
    report["malformed_unknown_enum_distinction_passed"] = blocked(build(metadata={"inference_boundary_classification": "x"}), "unknown_inference_boundary_class") and blocked(build(metadata={"inference_confidence_state": "x"}), "unknown_inference_confidence_state")
    report["recursive_forbidden_metadata_passed"] = blocked(build(metadata={"x":[{"memory_write_requested": True}]}), "memory_write_requested") and blocked(build(metadata={"x":{"memory_write_requested": 1}}), "malformed_forbidden_request_field")
    report["recursive_forbidden_item_passed"] = blocked(build(inference_items=[{**item(), "x":[{"tool_execution_requested": True}]}]), "tool_execution_requested")
    report["scheduling_alias_forbidden_passed"] = blocked(build(metadata={"cron_expression": True}), "cron_expression") and blocked(build(inference_items=[{**item(), "schedule_requested": True}]), "schedule_requested")
    report["invalid_item_reordering_deterministic_passed"] = build(inference_items=[{**item("deductive_step_candidate", iid="a"), "memory_write_requested": True}, {**item("deductive_step_candidate", iid="b"), "tool_execution_requested": True}]) == build(inference_items=[{**item("deductive_step_candidate", iid="b"), "tool_execution_requested": True}, {**item("deductive_step_candidate", iid="a"), "memory_write_requested": True}])
    class D(dict): pass
    class L(list): pass
    hostile = [D(), L(), {1:"x"}, (1,), {1}, frozenset({1}), b"x", bytearray(b"x"), object()]
    report["hostile_container_fail_closed_passed"] = all(blocked(build(metadata={"h": h}), "non_json_serializable_semantic_input") for h in hostile)
    deep = v = {}
    for _ in range(105): v["x"] = {}; v = v["x"]
    report["deep_recursion_fail_closed_passed"] = blocked(build(metadata=deep), "non_json_serializable_semantic_input")
    report["huge_integer_numeric_reason_passed"] = blocked(build(inference_items=[{**item(), "confidence_candidate": 10**100}]), "confidence_candidate_out_of_range")
    report["downstream_adversarial_coverage_passed"] = all(p["ready"] is True for p in plans_valid) and all(p["ready"] is False and p["candidate_only"] is True and p["read_only"] is True and p["inference_candidate_only"] is True for p in plans_invalid) and all(all(p[f] is False for f in PLAN_FALSE) for p in plans_invalid)
    report["non_json_fail_closed_passed"] = blocked(build(metadata={"nan": math.nan}), "non_json_serializable_semantic_input") and validate_virtual_world_situation_inference_context(math.nan) is False
    report["no_side_effects_passed"] = (all(valid[f] is False and rejected[f] is False for f in FALSE_FLAGS) and all(valid[t] is True and rejected[t] is True for t in TRUE_FLAGS) and all(all(p[f] is False for f in PLAN_FALSE) for p in plans_valid + plans_invalid) and all(p["candidate_only"] is True and p["read_only"] is True and p["inference_candidate_only"] is True for p in plans_valid + plans_invalid))
    report["all_checks_passed"] = all(v for k, v in report.items() if k.endswith("_passed"))
    report["schema_summary"] = summary
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    raise SystemExit(0 if report["all_checks_passed"] else 1)

if __name__ == "__main__":
    main()
