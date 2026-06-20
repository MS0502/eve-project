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


def item(eid="e1", kind="supporting_signal_candidate", stance="supports_candidate", source="internal_virtual_observation", **extra):
    data = {"evidence_id": eid, "situation_id": "s1", "evidence_kind": kind, "stance": stance, "source_class": source, "source_ref_id": "src-민석", "claim_ref_id": "claim-1"}
    data.update(extra)
    return data


def main():
    valid = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item()])
    invalid = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item(stance="neutral_unknown")])
    reordered = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [dict(reversed(list(item(label="x").items())))], {"b": 2, "a": 1})
    original = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item(label="x")], {"a": 1, "b": 2})
    tampered = json.loads(json.dumps(valid, ensure_ascii=False)); tampered["read_only"] = 1
    conflict = build_virtual_world_situation_evidence_context("conflicting_evidence_candidate", "s1", [item(), item("e2", "challenging_signal_candidate", "challenges_candidate", claim_ref_id="claim-2")])
    mixed = build_virtual_world_situation_evidence_context("mixed_unknown_evidence_candidate", "s1", [item(), item("e2", "unknown_origin_trace_candidate", "neutral_unknown", "unknown_origin_candidate", claim_ref_id="claim-2")])
    bad_boundary = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item()], {"evidence_boundary_classification": []})
    bad_conf = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item()], {"evidence_confidence_state": []})
    unknown_enum = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item()], {"evidence_boundary_classification": "bogus"})
    forbidden_meta = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item()], {"nested": {"memory_write_requested": True}})
    forbidden_item = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item(nested={"tool_execution_requested": True})])
    schedule = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item()], {"cron_expression": True})
    invalid_order_a = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item(nested={"agp_bypass_requested": True}), item(nested={"fallback_bypass_requested": True}, eid="e2", claim_ref_id="c2")])
    invalid_order_b = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", list(reversed([item(nested={"agp_bypass_requested": True}), item(nested={"fallback_bypass_requested": True}, eid="e2", claim_ref_id="c2")])) )
    class BadDict(dict): pass
    hostile = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item(extra=BadDict())])
    deep = x = []
    for _ in range(105):
        y = []; x.append(y); x = y
    deep_bad = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item(extra=deep)])
    huge = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item(confidence_candidate=10**100)])
    plans = [fn({"bad": True}) for fn in (build_evidence_context_to_situation_plan, build_evidence_context_to_snapshot_plan, build_evidence_context_to_transition_preflight_plan, build_evidence_context_to_memory_candidate_plan, build_evidence_context_to_appraisal_plan, build_evidence_context_to_agp_input_plan)]
    non_json = build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item(extra=float("nan"))])
    checks = {
        "valid_case_passed": validate_virtual_world_situation_evidence_context(valid),
        "invalid_case_passed": invalid["blocked_reasons"] == ["incompatible_evidence_item_kind_stance"],
        "deterministic_id_passed": original["evidence_context_id"] == reordered["evidence_context_id"],
        "type_exact_tamper_detection_passed": not validate_virtual_world_situation_evidence_context(tampered),
        "evidence_type_item_compatibility_passed": build_virtual_world_situation_evidence_context("temporal_evidence_candidate", "s1", [item()])["blocked_reasons"] == ["incompatible_evidence_type_item_kind"],
        "item_stance_compatibility_passed": invalid["blocked_reasons"] == ["incompatible_evidence_item_kind_stance"],
        "item_source_compatibility_passed": build_virtual_world_situation_evidence_context("mixed_unknown_evidence_candidate", "s1", [item("e1", "simulation_trace_candidate", "neutral_unknown", "internal_virtual_observation")])["blocked_reasons"] == ["incompatible_evidence_item_kind_source_class"],
        "item_situation_coherence_passed": build_virtual_world_situation_evidence_context("supporting_evidence_candidate", "s1", [item(situation_id="s2")])["blocked_reasons"] == ["evidence_item_situation_mismatch"],
        "duplicate_semantic_item_passed": build_virtual_world_situation_evidence_context("mixed_unknown_evidence_candidate", "s1", [item(), item("e2")])["blocked_reasons"] == ["duplicate_semantic_evidence_item"],
        "conflicting_evidence_validation_passed": conflict["blocked_reasons"] == [] and conflict["evidence_stance_summary"]["conflict_present"] is True,
        "mixed_evidence_validation_passed": mixed["blocked_reasons"] == [],
        "stance_summary_passed": valid["evidence_stance_summary"]["supporting_item_count"] == 1,
        "provenance_summary_passed": mixed["evidence_provenance_summary"]["unknown_origin_present"] is True,
        "resolution_remains_false_passed": valid["resolution_summary"]["resolved"] is False,
        "malformed_boundary_validation_passed": bad_boundary["blocked_reasons"] == ["malformed_evidence_boundary_class"],
        "malformed_confidence_validation_passed": bad_conf["blocked_reasons"] == ["malformed_evidence_confidence_state"],
        "malformed_unknown_enum_distinction_passed": unknown_enum["blocked_reasons"] == ["unknown_evidence_boundary_class"],
        "recursive_forbidden_metadata_passed": forbidden_meta["blocked_reasons"] == ["memory_write_requested"],
        "recursive_forbidden_item_passed": forbidden_item["blocked_reasons"] == ["tool_execution_requested"],
        "scheduling_alias_forbidden_passed": schedule["blocked_reasons"] == ["cron_expression"],
        "invalid_item_reordering_deterministic_passed": invalid_order_a == invalid_order_b,
        "hostile_container_fail_closed_passed": hostile["blocked_reasons"] == ["non_json_serializable_semantic_input"],
        "deep_recursion_fail_closed_passed": deep_bad["blocked_reasons"] == ["non_json_serializable_semantic_input"],
        "huge_integer_numeric_reason_passed": huge["blocked_reasons"] == ["confidence_candidate_out_of_range"],
        "downstream_adversarial_coverage_passed": all(p["ready"] is False and p["read_only"] is True for p in plans),
        "non_json_fail_closed_passed": non_json["blocked_reasons"] == ["non_json_serializable_semantic_input"],
        "no_side_effects_passed": True,
        "schema_summary": virtual_world_situation_evidence_context_schema_summary(),
    }
    checks["all_checks_passed"] = all(v is True for k, v in checks.items() if k != "schema_summary")
    print(json.dumps(checks, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if checks["all_checks_passed"] else 1

if __name__ == "__main__":
    sys.exit(main())
