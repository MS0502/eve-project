import copy
import json
import sys

from adapters.virtual_world_situation_evidence_context_schema import (
    build_evidence_context_to_agp_input_plan,
    build_evidence_context_to_appraisal_plan,
    build_evidence_context_to_memory_candidate_plan,
    build_evidence_context_to_situation_plan,
    build_evidence_context_to_snapshot_plan,
    build_evidence_context_to_transition_preflight_plan,
    build_virtual_world_situation_evidence_context,
    validate_virtual_world_situation_evidence_context,
    virtual_world_situation_evidence_context_schema_summary,
)

PLAN_FALSE = {
    "external_evidence_asserted", "evidence_verified", "source_verified", "evidence_fact_asserted", "conclusion_asserted", "truth_resolved", "winner_selected", "evidence_conflict_resolved", "automatic_resolution_performed", "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted", "random_sampling_performed", "probability_draw_performed", "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "schedule_created", "action_prevented", "action_permitted", "agp_bypass_allowed", "fallback_bypass_allowed", "relationship_update_performed", "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed",
}
IMMUTABLE_FALSE = {
    "external_evidence_asserted", "evidence_verified", "source_verified", "evidence_fact_asserted", "conclusion_asserted", "truth_resolved", "winner_selected", "evidence_conflict_resolved", "automatic_resolution_performed", "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted", "random_sampling_performed", "probability_draw_performed", "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "artifact_created_or_staged", "schedule_created", "action_prevented", "action_permitted", "agp_bypass_allowed", "fallback_bypass_allowed",
}
IMMUTABLE_TRUE = {"evidence_candidate_only", "situation_evidence_only", "read_only", "situation_review_required", "snapshot_review_required", "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required", "evidence_integrity_review_required", "provenance_review_required", "resolution_review_required", "appraisal_required", "agp_input_required"}
PLAN_BUILDERS = [build_evidence_context_to_situation_plan, build_evidence_context_to_snapshot_plan, build_evidence_context_to_transition_preflight_plan, build_evidence_context_to_memory_candidate_plan, build_evidence_context_to_appraisal_plan, build_evidence_context_to_agp_input_plan]


def item(eid="e1", kind="supporting_signal_candidate", stance="supports_candidate", source="internal_virtual_observation", sid="s1", source_ref_id="src-민석", claim_ref_id="claim-1", **extra):
    data = {"evidence_id": eid, "situation_id": sid, "evidence_kind": kind, "stance": stance, "source_class": source, "source_ref_id": source_ref_id, "claim_ref_id": claim_ref_id}
    data.update(extra)
    return data


def build(etype="supporting_evidence_candidate", sid="s1", items=None, metadata=None):
    return build_virtual_world_situation_evidence_context(etype, sid, [item()] if items is None else items, {} if metadata is None else metadata)


def reason(payload):
    return payload.get("blocked_reasons")


def valid_plan(plan):
    return plan.get("ready") is True and plan.get("candidate_only") is True and plan.get("read_only") is True and plan.get("evidence_candidate_only") is True and all(plan.get(k) is False for k in PLAN_FALSE)


def invalid_plan(plan):
    return plan.get("ready") is False and plan.get("candidate_only") is True and plan.get("read_only") is True and plan.get("evidence_candidate_only") is True and all(plan.get(k) is False for k in PLAN_FALSE)


def plan_sources(valid):
    deep = x = []
    for _ in range(105):
        y = []; x.append(y); x = y
    id_t = copy.deepcopy(valid); id_t["evidence_context_id"] = "x"
    alg_t = copy.deepcopy(valid); alg_t["canonical_id_algorithm"] = "x"
    sum_t = copy.deepcopy(valid); sum_t["resolution_summary"]["resolved"] = True
    extra = copy.deepcopy(valid); extra["extra"] = True
    false_t = copy.deepcopy(valid); false_t["evidence_verified"] = True
    true_f = copy.deepcopy(valid); true_f["read_only"] = False
    return [
        build(items=[item(stance="neutral_unknown")]), "x", {"bad": True}, {1: "bad"}, deep, id_t, alg_t, sum_t, extra, false_t, true_f,
        build(metadata={"evidence_boundary_classification": []}), build(metadata={"evidence_confidence_state": []}),
        build(metadata={"memory_write_requested": True}), build(items=[item(tool_execution_requested=True)]),
        build("mixed_unknown_evidence_candidate", items=[item("dup"), item("dup", claim_ref_id="b")]),
        build("mixed_unknown_evidence_candidate", items=[item("a"), item("b")]), build(items=[item(sid="s2")]),
        build(items=[item(stance="neutral_unknown")]), build("mixed_unknown_evidence_candidate", items=[item("a", "simulation_trace_candidate", "neutral_unknown", "internal_virtual_observation")]),
        build("temporal_evidence_candidate", items=[item()]), build("conflicting_evidence_candidate", items=[item()]),
        build("mixed_unknown_evidence_candidate", items=[item(memory_write_requested=True), item("b", tool_execution_requested=True, claim_ref_id="b")]),
        build(items=[item(confidence_candidate=10**100)]),
    ]


def main():
    valid = build()
    invalid = build(items=[item(stance="neutral_unknown")])
    original = build(items=[item(label="x")], metadata={"a": 1, "b": 2})
    reordered = build(items=[dict(reversed(list(item(label="x").items())))], metadata={"b": 2, "a": 1})
    tampered = copy.deepcopy(valid); tampered["read_only"] = 1
    conflict = build("conflicting_evidence_candidate", items=[item(), item("e2", "challenging_signal_candidate", "challenges_candidate", claim_ref_id="claim-2")])
    mixed = build("mixed_unknown_evidence_candidate", items=[item(), item("e2", "unknown_origin_trace_candidate", "neutral_unknown", "unknown_origin_candidate", claim_ref_id="claim-2")])
    invalid_order_items = [item("a", memory_write_requested=True), item("b", tool_execution_requested=True, claim_ref_id="b")]
    invalid_order_a = build("mixed_unknown_evidence_candidate", items=invalid_order_items)
    invalid_order_b = build("mixed_unknown_evidence_candidate", items=list(reversed(invalid_order_items)))
    adversarial_sources = plan_sources(valid)
    deep_payload = build(metadata={"deep": adversarial_sources[4]})
    plan_outputs = [fn(valid) for fn in PLAN_BUILDERS] + [fn(src) for fn in PLAN_BUILDERS for src in adversarial_sources]
    false_payloads = [valid, invalid, conflict, mixed]
    no_side_effect_payloads = all(all(p.get(k) is False for k in IMMUTABLE_FALSE) and all(p.get(k) is True for k in IMMUTABLE_TRUE) for p in false_payloads)
    checks = {
        "valid_case_passed": validate_virtual_world_situation_evidence_context(valid),
        "invalid_case_passed": reason(invalid) == ["incompatible_evidence_item_kind_stance"],
        "deterministic_id_passed": original.get("evidence_context_id") == reordered.get("evidence_context_id"),
        "type_exact_tamper_detection_passed": not validate_virtual_world_situation_evidence_context(tampered),
        "evidence_type_item_compatibility_passed": reason(build("temporal_evidence_candidate", items=[item()])) == ["incompatible_evidence_type_item_kind"],
        "item_stance_compatibility_passed": reason(invalid) == ["incompatible_evidence_item_kind_stance"],
        "item_source_compatibility_passed": reason(build("mixed_unknown_evidence_candidate", items=[item("e1", "simulation_trace_candidate", "neutral_unknown", "internal_virtual_observation")])) == ["incompatible_evidence_item_kind_source_class"],
        "item_situation_coherence_passed": reason(build(items=[item(sid="s2")])) == ["evidence_item_situation_mismatch"],
        "duplicate_semantic_item_passed": reason(build("mixed_unknown_evidence_candidate", items=[item(), item("e2")])) == ["duplicate_semantic_evidence_item"],
        "conflicting_evidence_validation_passed": reason(conflict) == [] and conflict["evidence_stance_summary"]["conflict_present"] is True,
        "mixed_evidence_validation_passed": reason(mixed) == [],
        "stance_summary_passed": valid["evidence_stance_summary"]["supporting_item_count"] == 1,
        "provenance_summary_passed": mixed["evidence_provenance_summary"]["unknown_origin_present"] is True,
        "resolution_remains_false_passed": valid["resolution_summary"] == {"resolution_candidate_only": True, "resolved": False, "conclusion_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False},
        "malformed_boundary_validation_passed": reason(build(metadata={"evidence_boundary_classification": []})) == ["malformed_evidence_boundary_class"],
        "malformed_confidence_validation_passed": reason(build(metadata={"evidence_confidence_state": []})) == ["malformed_evidence_confidence_state"],
        "malformed_unknown_enum_distinction_passed": reason(build(metadata={"evidence_boundary_classification": "unsupported"})) == ["unknown_evidence_boundary_class"] and reason(build(metadata={"evidence_confidence_state": "unsupported"})) == ["unknown_evidence_confidence_state"],
        "recursive_forbidden_metadata_passed": reason(build(metadata={"nested": {"memory_write_requested": True}})) == ["memory_write_requested"],
        "recursive_forbidden_item_passed": reason(build(items=[item(nested={"tool_execution_requested": True})])) == ["tool_execution_requested"],
        "scheduling_alias_forbidden_passed": reason(build(metadata={"cron_expression": True})) == ["cron_expression"],
        "invalid_item_reordering_deterministic_passed": invalid_order_a == invalid_order_b and reason(invalid_order_a) == ["memory_write_requested"],
        "hostile_container_fail_closed_passed": reason(build(metadata={1: "bad"})) == ["non_json_serializable_semantic_input"],
        "deep_recursion_fail_closed_passed": reason(deep_payload) == ["non_json_serializable_semantic_input"],
        "huge_integer_numeric_reason_passed": reason(build(items=[item(confidence_candidate=10**100)])) == ["confidence_candidate_out_of_range"],
        "downstream_adversarial_coverage_passed": all(valid_plan(fn(valid)) for fn in PLAN_BUILDERS) and all(invalid_plan(fn(src)) for fn in PLAN_BUILDERS for src in adversarial_sources),
        "non_json_fail_closed_passed": reason(build(items=[item(extra=float("nan"))])) == ["non_json_serializable_semantic_input"],
        "no_side_effects_passed": no_side_effect_payloads and all((valid_plan(p) or invalid_plan(p)) for p in plan_outputs),
        "schema_summary": virtual_world_situation_evidence_context_schema_summary(),
    }
    checks["all_checks_passed"] = all(value is True for key, value in checks.items() if key != "schema_summary")
    print(json.dumps(checks, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if checks["all_checks_passed"] else 1

if __name__ == "__main__":
    sys.exit(main())
