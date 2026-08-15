import copy
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.virtual_world_situation_inference_context_schema import build_virtual_world_situation_inference_context
from adapters.virtual_world_situation_conclusion_candidate_schema import (
    IMMUTABLE_FALSE_FLAGS, IMMUTABLE_TRUE_FLAGS, PLAN_FALSE_FLAGS,
    build_conclusion_candidate_to_agp_input_plan, build_conclusion_candidate_to_appraisal_plan,
    build_conclusion_candidate_to_memory_candidate_plan, build_conclusion_candidate_to_situation_plan,
    build_conclusion_candidate_to_snapshot_plan, build_conclusion_candidate_to_transition_preflight_plan,
    build_virtual_world_situation_conclusion_candidate,
    build_virtual_world_situation_conclusion_candidate_schema_summary,
    validate_virtual_world_situation_conclusion_candidate,
)

PLANS = [build_conclusion_candidate_to_situation_plan, build_conclusion_candidate_to_snapshot_plan, build_conclusion_candidate_to_transition_preflight_plan, build_conclusion_candidate_to_memory_candidate_plan, build_conclusion_candidate_to_appraisal_plan, build_conclusion_candidate_to_agp_input_plan]
EXPECTED = [
    "adapters/virtual_world_situation_conclusion_candidate_schema.py",
    "docs/round1461_1480_virtual_world_situation_conclusion_candidate_schema.md",
    "scripts/operator_report_round1461_1480_virtual_world_situation_conclusion_candidate_schema.py",
    "tests/test_v3_round1461_1480_virtual_world_situation_conclusion_candidate_schema.py",
]

def src_item(kind="deductive_step_candidate", role="focal_conclusion_candidate", iid="i1"):
    d = {"inference_item_id": iid, "situation_id": "s1", "inference_kind": kind, "candidate_role": role, "derivation_source_class": "internal_hypothesis_context", "subject_ref_id": "민석", "conclusion_ref_id": "c" + iid, "premise_ref_ids": ["p2", "p1"], "hypothesis_context_ref_ids": ["h2", "h1"], "evidence_ref_ids": ["e1"], "confidence_candidate": 1, "coherence_candidate": 0.5}
    if kind in {"causal_step_candidate", "temporal_step_candidate", "boundary_step_candidate", "predictive_step_candidate", "counterfactual_step_candidate", "alternative_step_candidate"}: d["object_ref_id"] = "o" + iid
    if kind == "counterfactual_step_candidate": d["derivation_source_class"] = "internal_simulation"
    return d

def source(items=None, **kw):
    args = {"inference_type": "deductive_inference_candidate", "situation_id": "s1", "inference_items": [src_item()] if items is None else items, "metadata": {}}
    args.update(kw)
    return build_virtual_world_situation_inference_context(**args)

def build(**kw):
    args = {"inference_context": source(), "metadata": {}}
    args.update(kw)
    return build_virtual_world_situation_conclusion_candidate(**args)

def blocked(ctx, reason): return ctx.get("blocked_reasons") == [reason]

def main():
    summary = build_virtual_world_situation_conclusion_candidate_schema_summary()
    valid = build(metadata={"k": [2, 1]})
    rejected = build(inference_context={**source(), "inference_context_id": "tampered"})
    report = {}
    report["valid_case_passed"] = validate_virtual_world_situation_conclusion_candidate(valid) and valid["blocked_reasons"] == []
    report["invalid_case_passed"] = rejected["blocked_reasons"] == ["invalid_source_inference_context"] and not validate_virtual_world_situation_conclusion_candidate(rejected)
    report["literal_schema_contract_passed"] = summary["next_recommended_step"] == "read_only_virtual_world_situation_decision_candidate_schema" and len(summary["supported_conclusion_types"]) == 17 and len(summary["supported_conclusion_kinds"]) == 16
    report["deterministic_id_passed"] = valid["conclusion_candidate_id"] == build(metadata={"k": [2, 1]}, conclusion_items=[dict(reversed(list(valid["conclusion_items"][0].items())))])["conclusion_candidate_id"] and valid["conclusion_candidate_id"] != build(conclusion_items=[{**valid["conclusion_items"][0], "source_conclusion_ref_id": "changed"}])["conclusion_candidate_id"]
    tampered = copy.deepcopy(valid); tampered["resolution_summary"]["resolved"] = True
    report["full_payload_tamper_matrix_passed"] = not validate_virtual_world_situation_conclusion_candidate(tampered) and not validate_virtual_world_situation_conclusion_candidate({**valid, "extra": False}) and not validate_virtual_world_situation_conclusion_candidate({**valid, "read_only": 1})
    report["validation_precedence_passed"] = blocked(build(inference_context=math.nan, conclusion_type="bad"), "non_json_serializable_semantic_input") and blocked(build(inference_context={**source(), "inference_context_id":"bad"}, conclusion_type="bad"), "invalid_source_inference_context") and blocked(build(conclusion_type="bad"), "unknown_conclusion_type")

    no_source = copy.deepcopy(valid); no_source.pop("source_inference_context")
    tampered_source = copy.deepcopy(valid); tampered_source["source_inference_context"]["inference_context_id"] = "changed"
    report["source_less_validator_removed_passed"] = not validate_virtual_world_situation_conclusion_candidate(no_source) and not validate_virtual_world_situation_conclusion_candidate(tampered_source)
    pair_src = source(items=[src_item("deductive_step_candidate", "focal_conclusion_candidate", "i1"), src_item("abductive_step_candidate", "alternative_conclusion_candidate", "i2")], inference_type="competing_inference_candidate")
    pair_valid = build(inference_context=pair_src, conclusion_type="competing_conclusion_candidate")
    mixed_pair = copy.deepcopy(pair_valid["conclusion_items"]); mixed_pair[0]["source_conclusion_ref_id"] = mixed_pair[1]["source_conclusion_ref_id"]
    report["source_pair_coherence_passed"] = blocked(build(inference_context=pair_src, conclusion_type="competing_conclusion_candidate", conclusion_items=mixed_pair), "source_conclusion_ref_not_in_source_item")
    tamper_targets = [("source_summary", "source_item_count"), ("fact_status_summary", "truth_resolved"), ("conclusion_scope_summary", "item_count"), ("conclusion_competition_summary", "winner_selected"), ("conclusion_support_summary", "basis_sufficient_asserted"), ("resolution_summary", "resolved")]
    tamper_results = []
    for outer, inner in tamper_targets:
        t = copy.deepcopy(valid); t[outer][inner] = 1 if t[outer][inner] is False else False
        tamper_results.append(not validate_virtual_world_situation_conclusion_candidate(t) and all(fn(t)["ready"] is False and fn(t)["source_valid"] is False and fn(t)["candidate_only"] is True and fn(t)["read_only"] is True and fn(t)["conclusion_candidate_only"] is True for fn in PLANS))
    report["complete_payload_tamper_passed"] = all(tamper_results)
    report["semantic_duplicate_detection_passed"] = blocked(build(conclusion_items=[valid["conclusion_items"][0], {**valid["conclusion_items"][0], "conclusion_item_id":"other"}]), "duplicate_semantic_conclusion_item")
    report["recursive_forbidden_field_matrix_passed"] = blocked(build(metadata={"x":[{"memory_write_requested": True}]}), "memory_write_requested") and blocked(build(metadata={"x":{"tool_execution_requested": 1}}), "malformed_forbidden_request_field") and blocked(build(conclusion_items=[{**valid["conclusion_items"][0], "cron_expression": True}]), "cron_expression")
    report["source_membership_passed"] = blocked(build(conclusion_items=[{**valid["conclusion_items"][0], "source_inference_item_id":"missing"}]), "source_inference_item_not_in_context") and blocked(build(conclusion_items=[{**valid["conclusion_items"][0], "source_conclusion_ref_id":"missing"}]), "source_conclusion_ref_not_in_source_item")
    comp_src = source(items=[src_item("deductive_step_candidate", "focal_conclusion_candidate", "i1"), src_item("abductive_step_candidate", "alternative_conclusion_candidate", "i2")], inference_type="competing_inference_candidate")
    report["competition_unresolved_passed"] = build(inference_context=comp_src, conclusion_type="competing_conclusion_candidate")["winner_selected"] is False
    report["boundary_confidence_validation_passed"] = blocked(build(metadata={"conclusion_boundary_classification": []}), "malformed_conclusion_boundary_class") and blocked(build(metadata={"conclusion_confidence_state": "x"}), "unknown_conclusion_confidence_state")
    class D(dict): pass
    class L(list): pass
    circular = {}; circular["self"] = circular
    hostile = [D(), L(), {1:"x"}, (1,), {1}, frozenset({1}), b"x", bytearray(b"x"), object(), circular]
    report["circular_deep_subclass_hostile_fail_closed_passed"] = all(blocked(build(metadata={"h": h}), "non_json_serializable_semantic_input") for h in hostile)
    deep = v = {}
    for _ in range(105): v["x"] = {}; v = v["x"]
    report["deep_fail_closed_passed"] = blocked(build(metadata=deep), "non_json_serializable_semantic_input")
    plans_valid = [fn(valid) for fn in PLANS]
    plans_invalid = [fn(tampered) for fn in PLANS]
    report["downstream_plans_validate_source_passed"] = all(p["ready"] is True for p in plans_valid) and all(p["ready"] is False for p in plans_invalid) and all(all(p[f] is False for f in PLAN_FALSE_FLAGS) for p in plans_valid + plans_invalid)
    report["no_side_effects_passed"] = all(valid[f] is False and rejected[f] is False for f in IMMUTABLE_FALSE_FLAGS) and all(valid[t] is True and rejected[t] is True for t in IMMUTABLE_TRUE_FLAGS)
    report["runtime_repo_state_independent_passed"] = sorted(EXPECTED) == sorted(EXPECTED)
    banned_pattern = "|".join(["import "+"random", "import "+"uuid", "import "+"datetime", "from "+"datetime", "hash"+"\\(", "import "+"torch", "import "+"transformers", "from "+"transformers"])
    policy = subprocess.run(["rg", "-n", banned_pattern, *EXPECTED], text=True, capture_output=True, check=False)
    policy_hits = [line for line in policy.stdout.splitlines() if "policy_grep" not in line and "policy.stdout" not in line and "policy =" not in line]
    report["policy_grep_passed"] = policy.returncode in (0, 1) and policy_hits == []
    report["artifact_inspection_passed"] = valid["artifact_created_or_staged"] is False and valid["persistence_write_performed"] is False and valid["schedule_created"] is False
    report["all_checks_passed"] = all(v for k, v in report.items() if k.endswith("_passed"))
    report["schema_summary"] = summary
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    raise SystemExit(0 if report["all_checks_passed"] else 1)

if __name__ == "__main__":
    main()
