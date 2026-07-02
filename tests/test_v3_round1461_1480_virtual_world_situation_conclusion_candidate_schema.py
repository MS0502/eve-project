import copy
import json
import math
import subprocess

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
    source = "internal_hypothesis_context"
    if kind == "causal_step_candidate": source = "internal_causal_context"
    if kind == "counterfactual_step_candidate": source = "internal_simulation"
    d = {"inference_item_id": iid, "situation_id": "s1", "inference_kind": kind, "candidate_role": role, "derivation_source_class": source, "subject_ref_id": "민석", "conclusion_ref_id": "c" + iid, "premise_ref_ids": ["p2", "p1"], "hypothesis_context_ref_ids": ["h2", "h1"], "evidence_ref_ids": ["e1"], "confidence_candidate": 1, "coherence_candidate": 0.5}
    if kind in {"causal_step_candidate", "temporal_step_candidate", "boundary_step_candidate", "predictive_step_candidate", "counterfactual_step_candidate", "alternative_step_candidate"}: d["object_ref_id"] = "o" + iid
    return d

def source(items=None, **kw):
    args = {"inference_type": "deductive_inference_candidate", "situation_id": "s1", "inference_items": [src_item()] if items is None else items, "metadata": {}}
    args.update(kw)
    return build_virtual_world_situation_inference_context(**args)

def build(**kw):
    args = {"inference_context": source(), "metadata": {}}
    args.update(kw)
    return build_virtual_world_situation_conclusion_candidate(**args)

def blocked(ctx, reason):
    return ctx.get("blocked_reasons") == [reason]

def test_literal_independent_schema_contract_and_summary():
    summary = build_virtual_world_situation_conclusion_candidate_schema_summary()
    assert summary == {
        "schema_version": "1.0.0-round1461-1480",
        "schema_name": "read_only_virtual_world_situation_conclusion_candidate_schema",
        "canonical_id_algorithm": "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256",
        "supported_conclusion_types": summary["supported_conclusion_types"],
        "supported_conclusion_kinds": summary["supported_conclusion_kinds"],
        "read_only": True,
        "candidate_only": True,
        "strict_json_only": True,
        "source_must_validate": True,
        "side_effects_allowed": False,
        "next_recommended_step": "read_only_virtual_world_situation_decision_candidate_schema",
    }
    assert len(summary["supported_conclusion_types"]) == 17
    assert len(summary["supported_conclusion_kinds"]) == 16

def test_valid_candidate_is_unresolved_unaccepted_unrejected_non_factual():
    candidate = build()
    assert validate_virtual_world_situation_conclusion_candidate(candidate)
    assert candidate["situation_conclusion_candidate_status"] == "VALIDATED"
    assert candidate["fact_status_summary"] == {"candidate_only": True, "external_conclusion_fact_verified": False, "conclusion_fact_asserted": False, "conclusion_accepted": False, "conclusion_rejected": False, "truth_resolved": False}
    assert candidate["resolution_summary"] == {"resolution_candidate_only": True, "resolved": False, "accepted": False, "rejected": False, "winner_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False}
    assert json.loads(json.dumps(candidate, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)) == candidate

def test_exact_validation_precedence():
    assert blocked(build(inference_context=math.nan, conclusion_type="bad"), "non_json_serializable_semantic_input")
    assert blocked(build(inference_context={**source(), "inference_context_id":"bad"}, conclusion_type="bad"), "invalid_source_inference_context")
    assert blocked(build(conclusion_type=None), "unknown_conclusion_type") is False
    assert blocked(build(conclusion_type="bad"), "unknown_conclusion_type")
    assert blocked(build(metadata=[]), "metadata_not_object")
    assert blocked(build(metadata={"memory_write_requested": True}, conclusion_type="bad"), "unknown_conclusion_type")

def test_deterministic_canonical_sha256_id_and_semantic_changes():
    first = build(metadata={"b": [2, 1], "a": {"민석": True}})
    reordered_item = dict(reversed(list(first["conclusion_items"][0].items())))
    second = build(metadata={"a": {"민석": True}, "b": [2, 1]}, conclusion_items=[reordered_item])
    changed = build(conclusion_items=[{**first["conclusion_items"][0], "source_conclusion_ref_id": "changed"}])
    assert first["conclusion_candidate_id"] == second["conclusion_candidate_id"]
    assert first["conclusion_candidate_id"] != changed["conclusion_candidate_id"]
    assert len(first["conclusion_candidate_id"]) == 64
    assert all(c in "0123456789abcdef" for c in first["conclusion_candidate_id"])

def test_semantic_duplicate_detection_and_source_membership():
    valid = build()
    item = valid["conclusion_items"][0]
    assert blocked(build(conclusion_items=[item, {**item, "conclusion_item_id": "other"}]), "duplicate_semantic_conclusion_item")
    assert blocked(build(conclusion_items=[{**item, "source_inference_item_id": "missing"}]), "source_inference_item_not_in_context")
    assert blocked(build(conclusion_items=[{**item, "source_conclusion_ref_id": "missing"}]), "source_conclusion_ref_not_in_inference_context")

def test_full_payload_tamper_matrix_type_exact_integrity():
    base = build()
    for key in ["conclusion_candidate_id", "canonical_id_algorithm", "source_summary", "fact_status_summary", "conclusion_scope_summary", "resolution_summary"]:
        tampered = copy.deepcopy(base)
        tampered[key] = True
        assert not validate_virtual_world_situation_conclusion_candidate(tampered), key
    for key in ["read_only", "conclusion_candidate_only", "situation_conclusion_candidate_passed"]:
        tampered = copy.deepcopy(base); tampered[key] = 1
        assert not validate_virtual_world_situation_conclusion_candidate(tampered), key
    tampered = copy.deepcopy(base); tampered["conclusion_items"][0]["confidence_candidate"] = 1.0
    assert not validate_virtual_world_situation_conclusion_candidate(tampered)
    tampered = copy.deepcopy(base); tampered["extra"] = False
    assert not validate_virtual_world_situation_conclusion_candidate(tampered)
    tampered = copy.deepcopy(base); tampered.pop("warnings")
    assert not validate_virtual_world_situation_conclusion_candidate(tampered)

def test_recursive_forbidden_field_matrix():
    valid = build()
    forbidden = ["memory_write_requested", "identity_update_requested", "relationship_update_requested", "affect_transition_requested", "hormone_transition_requested", "world_state_mutation_requested", "transition_application_requested", "agp_bypass_requested", "fallback_bypass_requested", "tool_execution_requested", "schedule_requested", "vector_load_requested", "model_load_requested", "network_action_requested"]
    for field in forbidden:
        assert blocked(build(metadata={"nested": [{field: True}]}), field), field
        assert blocked(build(conclusion_items=[{**valid["conclusion_items"][0], "nested": [{field: True}]}]), field), field
    assert blocked(build(metadata={"cron_expression": "* * * * *"}), "malformed_forbidden_request_field")
    assert build(metadata={"memory_write_requested": False})["blocked_reasons"] == []

def test_circular_deep_subclass_and_hostile_object_fail_closed():
    class D(dict): pass
    class L(list): pass
    circular = {}; circular["self"] = circular
    hostile = [D(), L(), {1:"x"}, (1,), {1}, frozenset({1}), b"x", bytearray(b"x"), object(), circular, math.nan, math.inf]
    for value in hostile:
        assert blocked(build(metadata={"h": value}), "non_json_serializable_semantic_input")
    deep = v = {}
    for _ in range(105): v["x"] = {}; v = v["x"]
    assert blocked(build(metadata=deep), "non_json_serializable_semantic_input")

def test_competing_mixed_boundary_confidence_and_no_resolution():
    comp_source = source(items=[src_item("deductive_step_candidate", "focal_conclusion_candidate", "i1"), src_item("abductive_step_candidate", "alternative_conclusion_candidate", "i2")], inference_type="competing_inference_candidate")
    comp = build(inference_context=comp_source, conclusion_type="competing_conclusion_candidate")
    assert validate_virtual_world_situation_conclusion_candidate(comp)
    assert comp["conclusion_competition_summary"]["competition_present"] is True
    assert comp["winner_selected"] is False
    assert blocked(build(metadata={"conclusion_boundary_classification": []}), "malformed_conclusion_boundary_class")
    assert blocked(build(metadata={"conclusion_boundary_classification": "x"}), "unknown_conclusion_boundary_class")
    assert blocked(build(metadata={"conclusion_confidence_state": []}), "malformed_conclusion_confidence_state")
    assert blocked(build(metadata={"conclusion_confidence_state": "x"}), "unknown_conclusion_confidence_state")

def test_all_downstream_plans_validate_their_source_and_have_no_side_effects():
    valid = build()
    tampered = copy.deepcopy(valid); tampered["conclusion_candidate_id"] = "0" * 64
    valid_plans = [fn(valid) for fn in PLANS]
    invalid_plans = [fn(tampered) for fn in PLANS]
    assert all(p["ready"] is True and p["source_valid"] is True for p in valid_plans)
    assert all(p["ready"] is False and p["source_valid"] is False for p in invalid_plans)
    assert all(p["candidate_only"] is True and p["read_only"] is True and p["conclusion_candidate_only"] is True for p in valid_plans + invalid_plans)
    assert all(all(p[f] is False for f in PLAN_FALSE_FLAGS) for p in valid_plans + invalid_plans)

def test_no_side_effect_flags_and_exact_four_file_scope_policy_grep_artifacts():
    valid = build(); rejected = build(inference_context={**source(), "inference_context_id":"bad"})
    assert all(valid[f] is False and rejected[f] is False for f in IMMUTABLE_FALSE_FLAGS)
    assert all(valid[t] is True and rejected[t] is True for t in IMMUTABLE_TRUE_FLAGS)
    status_lines = subprocess.run(["git", "status", "--short"], text=True, capture_output=True, check=False).stdout.splitlines()
    changed = sorted(line[3:] for line in status_lines if line[:2].strip())
    assert changed == sorted(EXPECTED)
    banned_pattern = "|".join(["import "+"random", "import "+"uuid", "import "+"datetime", "from "+"datetime", "hash"+"\\(", "import "+"torch", "import "+"transformers", "from "+"transformers"])
    policy = subprocess.run(["rg", "-n", banned_pattern, *EXPECTED], text=True, capture_output=True, check=False)
    assert policy.returncode in (0, 1)
    policy_hits = [line for line in policy.stdout.splitlines() if "policy_grep" not in line and "policy.stdout" not in line and "policy =" not in line]
    assert policy_hits == []
    assert valid["artifact_created_or_staged"] is False and valid["persistence_write_performed"] is False and valid["schedule_created"] is False

def test_operator_report_emits_one_compact_json_object_with_actual_booleans():
    output = subprocess.check_output(["python", "scripts/operator_report_round1461_1480_virtual_world_situation_conclusion_candidate_schema.py"], text=True)
    lines = output.splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["all_checks_passed"] is True
    assert all(value is True for key, value in report.items() if key.endswith("_passed"))
