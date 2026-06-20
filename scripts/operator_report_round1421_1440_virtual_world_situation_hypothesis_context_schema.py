"""Operator checks for round1421-1440 hypothesis context schema."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.virtual_world_situation_hypothesis_context_schema import (
    build_hypothesis_context_to_agp_input_plan, build_hypothesis_context_to_appraisal_plan,
    build_hypothesis_context_to_memory_candidate_plan, build_hypothesis_context_to_situation_plan,
    build_hypothesis_context_to_snapshot_plan, build_hypothesis_context_to_transition_preflight_plan,
    build_virtual_world_situation_hypothesis_context, build_virtual_world_situation_hypothesis_context_schema_summary,
    validate_virtual_world_situation_hypothesis_context,
)

HYPOTHESIS_TYPES=["explanatory_hypothesis_candidate","causal_hypothesis_candidate","temporal_hypothesis_candidate","state_hypothesis_candidate","identity_hypothesis_candidate","boundary_hypothesis_candidate","simulation_hypothesis_candidate","symbolic_hypothesis_candidate","dmn_hypothesis_candidate","dream_hypothesis_candidate","predictive_hypothesis_candidate","counterfactual_hypothesis_candidate","competing_hypothesis_candidate","unknown_origin_hypothesis_candidate","mixed_unknown_hypothesis_candidate"]
HYPOTHESIS_KINDS=["explanatory_proposition_candidate","causal_proposition_candidate","temporal_proposition_candidate","state_proposition_candidate","identity_proposition_candidate","boundary_proposition_candidate","simulation_proposition_candidate","symbolic_proposition_candidate","dmn_proposition_candidate","dream_proposition_candidate","predictive_proposition_candidate","counterfactual_proposition_candidate","alternative_proposition_candidate","unknown_origin_proposition_candidate"]
ROLES=["focal_candidate","alternative_candidate","counter_candidate","neutral_unknown"]
BOUNDARIES=["internal_explanatory_hypothesis","causal_virtual_hypothesis","temporal_virtual_hypothesis","state_virtual_hypothesis","identity_virtual_hypothesis","boundary_virtual_hypothesis","simulation_hypothesis","symbolic_hypothesis","dmn_hypothesis","dream_hypothesis","predictive_virtual_hypothesis","counterfactual_virtual_hypothesis","unknown_origin_hypothesis","mixed_virtual_external_hypothesis_boundary"]
CONFIDENCE=["hypothesis_unverified","hypothesis_low_confidence","hypothesis_medium_confidence","hypothesis_high_confidence_but_not_fact","hypothesis_competing","hypothesis_origin_unknown"]
TYPE_KIND={"explanatory_hypothesis_candidate":["explanatory_proposition_candidate"],"causal_hypothesis_candidate":["causal_proposition_candidate"],"temporal_hypothesis_candidate":["temporal_proposition_candidate"],"state_hypothesis_candidate":["state_proposition_candidate"],"identity_hypothesis_candidate":["identity_proposition_candidate"],"boundary_hypothesis_candidate":["boundary_proposition_candidate"],"simulation_hypothesis_candidate":["simulation_proposition_candidate"],"symbolic_hypothesis_candidate":["symbolic_proposition_candidate"],"dmn_hypothesis_candidate":["dmn_proposition_candidate"],"dream_hypothesis_candidate":["dream_proposition_candidate"],"predictive_hypothesis_candidate":["predictive_proposition_candidate"],"counterfactual_hypothesis_candidate":["counterfactual_proposition_candidate"],"competing_hypothesis_candidate":["alternative_proposition_candidate","boundary_proposition_candidate","causal_proposition_candidate","counterfactual_proposition_candidate","explanatory_proposition_candidate","identity_proposition_candidate","predictive_proposition_candidate","simulation_proposition_candidate","state_proposition_candidate","symbolic_proposition_candidate","temporal_proposition_candidate"],"unknown_origin_hypothesis_candidate":["unknown_origin_proposition_candidate"],"mixed_unknown_hypothesis_candidate":["alternative_proposition_candidate","boundary_proposition_candidate","causal_proposition_candidate","counterfactual_proposition_candidate","dmn_proposition_candidate","dream_proposition_candidate","explanatory_proposition_candidate","identity_proposition_candidate","predictive_proposition_candidate","simulation_proposition_candidate","state_proposition_candidate","symbolic_proposition_candidate","temporal_proposition_candidate","unknown_origin_proposition_candidate"]}
KIND_ROLE={"explanatory_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"causal_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"temporal_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"state_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"identity_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"boundary_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"simulation_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"symbolic_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"dmn_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"dream_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"predictive_proposition_candidate":["alternative_candidate","focal_candidate"],"counterfactual_proposition_candidate":["counter_candidate"],"alternative_proposition_candidate":["alternative_candidate"],"unknown_origin_proposition_candidate":["neutral_unknown"]}
RELATIONAL=["alternative_proposition_candidate","boundary_proposition_candidate","causal_proposition_candidate","counterfactual_proposition_candidate","predictive_proposition_candidate","temporal_proposition_candidate"]
FORBIDDEN=["memory_write_requested","vector_load_requested","winner_selection_requested","truth_resolution_requested","cron_expression","agp_bypass_requested","fallback_bypass_requested"]
IMMUTABLE_FALSE=["external_hypothesis_asserted","hypothesis_verified","hypothesis_fact_asserted","hypothesis_accepted","hypothesis_rejected","conclusion_asserted","truth_resolved","winner_selected","competition_resolved","automatic_resolution_performed","automatic_resolution_allowed","prediction_guaranteed","future_outcome_asserted","counterfactual_applied","evidence_verified","evidence_sufficient_asserted","random_sampling_performed","probability_draw_performed","identity_asserted","memory_fact_asserted","memory_write_performed","memory_write_allowed","relationship_update_allowed","self_model_update_allowed","affect_transition_allowed","hormone_transition_allowed","world_state_mutation_performed","transition_applied","planning_execution_performed","tool_execution_performed","runtime_mutation_performed","persistence_write_performed","vector_read_performed","vector_load_performed","model_loaded","device_activation_performed","network_action_performed","artifact_created_or_staged","schedule_created","action_prevented","action_permitted","agp_bypass_allowed","fallback_bypass_allowed"]
IMMUTABLE_TRUE=["hypothesis_candidate_only","situation_hypothesis_only","read_only","situation_review_required","snapshot_review_required","transition_review_required","memory_gate_required","quarantine_required","origin_fact_status_required","hypothesis_integrity_review_required","evidence_review_required","competition_review_required","resolution_review_required","appraisal_required","agp_input_required"]
PLAN_FALSE=[x for x in IMMUTABLE_FALSE if x != "artifact_created_or_staged"]+["relationship_update_performed","self_model_update_performed","affect_transition_performed","hormone_transition_performed"]
TOP_KEYS=set(["situation_hypothesis_context_passed","situation_hypothesis_context_status","hypothesis_context_id","canonical_id_algorithm","hypothesis_type","hypothesis_boundary_classification","hypothesis_confidence_state","situation_id","hypothesis_items","metadata","origin_summary","fact_status_summary","hypothesis_scope_summary","hypothesis_competition_summary","hypothesis_support_summary","resolution_summary","hypothesis_flags","boundary_flags","hypothesis_integrity_flags","candidate_only_fields","blocked_reasons","warnings"]+IMMUTABLE_FALSE+IMMUTABLE_TRUE)
BUILDERS=[build_hypothesis_context_to_situation_plan,build_hypothesis_context_to_snapshot_plan,build_hypothesis_context_to_transition_preflight_plan,build_hypothesis_context_to_memory_candidate_plan,build_hypothesis_context_to_appraisal_plan,build_hypothesis_context_to_agp_input_plan]

def item(kind="explanatory_proposition_candidate", role="focal_candidate", item_id="h1", sit="sit-민석", obj=None, claim="claim", basis=None, **extra):
    d={"hypothesis_item_id":item_id,"situation_id":sit,"hypothesis_kind":kind,"candidate_role":role,"subject_ref_id":"subject","claim_ref_id":claim,"basis_evidence_ref_ids":["ev-b","ev-a"] if basis is None else basis}
    if obj is not None: d["object_ref_id"]=obj
    d.update(extra); return d

def build(**kw):
    args={"hypothesis_type":"explanatory_hypothesis_candidate","situation_id":"sit-민석","hypothesis_items":[item()],"metadata":{}}
    args.update(kw); return build_virtual_world_situation_hypothesis_context(**args)

def reason(payload): return payload["blocked_reasons"]
def obj_for(k): return "object" if k in RELATIONAL else None
def jsonable(p):
    try: json.dumps(p, ensure_ascii=False, allow_nan=False); return True
    except Exception: return False

def plan_ok(src, ready):
    plans=[fn(src) for fn in BUILDERS]
    return all(p["ready"] is ready and p["candidate_only"] is True and p["read_only"] is True and p["hypothesis_candidate_only"] is True and all(p.get(f) is False for f in PLAN_FALSE) for p in plans)

def main():
    valid=build(); invalid=build(hypothesis_type="bad"); tam=copy.deepcopy(valid); tam["hypothesis_context_id"]="0"
    checks={}
    summary=build_virtual_world_situation_hypothesis_context_schema_summary()
    checks["literal_schema_contract_passed"]=summary["supported_hypothesis_types"]==HYPOTHESIS_TYPES and summary["supported_hypothesis_kinds"]==HYPOTHESIS_KINDS and summary["supported_candidate_roles"]==ROLES and summary["supported_boundary_classes"]==BOUNDARIES and summary["supported_confidence_states"]==CONFIDENCE and summary["type_kind_compatibility"]==TYPE_KIND and summary["kind_role_compatibility"]==KIND_ROLE and summary["relational_hypothesis_kinds"]==RELATIONAL
    checks["valid_case_passed"]=validate_virtual_world_situation_hypothesis_context(valid) and set(valid)==TOP_KEYS
    checks["invalid_case_passed"]=reason(invalid)==["unknown_hypothesis_type"] and not validate_virtual_world_situation_hypothesis_context(invalid)
    checks["deterministic_id_passed"]=build(hypothesis_items=[dict(reversed(list(item().items())))],metadata={"z":1,"a":2})["hypothesis_context_id"]==build(hypothesis_items=[item(basis=["ev-a","ev-b"])],metadata={"a":2,"z":1})["hypothesis_context_id"] and valid["hypothesis_context_id"]!=build(hypothesis_items=[item(claim="changed")])["hypothesis_context_id"]
    checks["type_exact_tamper_detection_passed"]=not validate_virtual_world_situation_hypothesis_context(tam) and not validate_virtual_world_situation_hypothesis_context({**valid,"read_only":1})
    checks["type_kind_compatibility_passed"]=all(build(hypothesis_type=t,hypothesis_items=([item(kind=ks[0],role=KIND_ROLE[ks[0]][0],obj=obj_for(ks[0]))] if t not in {"competing_hypothesis_candidate","mixed_unknown_hypothesis_candidate"} else [item(),item(kind="causal_proposition_candidate",role="alternative_candidate",item_id="h2",obj="object",claim="c2")]))["blocked_reasons"]==[] for t,ks in TYPE_KIND.items())
    checks["kind_role_compatibility_passed"]=all(build(hypothesis_type="mixed_unknown_hypothesis_candidate",hypothesis_items=[item(kind="state_proposition_candidate",role="neutral_unknown",item_id="base",claim="base"),item(kind=k,role=r,item_id="x"+k+r,obj=obj_for(k),claim="c"+k+r)])["blocked_reasons"]==[] or (k=="state_proposition_candidate" and r=="neutral_unknown") for k,rs in KIND_ROLE.items() for r in rs)
    checks["reference_shape_validation_passed"]=reason(build(hypothesis_type="causal_hypothesis_candidate",hypothesis_items=[item(kind="causal_proposition_candidate")]))==["missing_object_ref_id_for_relational_hypothesis"] and reason(build(hypothesis_items=[item(obj="object")]))==["unexpected_object_ref_id_for_unary_hypothesis"]
    checks["basis_reference_validation_passed"]=valid["hypothesis_items"][0]["basis_evidence_ref_ids"]==["ev-a","ev-b"] and reason(build(hypothesis_items=[item(basis=[])]))==["missing_or_malformed_basis_evidence_ref_ids"] and reason(build(hypothesis_items=[item(basis=["x","x"])]))==["duplicate_basis_evidence_ref_id"]
    checks["item_situation_coherence_passed"]=reason(build(hypothesis_items=[item(sit="other")]))==["hypothesis_item_situation_mismatch"]
    checks["duplicate_semantic_item_passed"]=reason(build(hypothesis_items=[item(),item(item_id="h2")]))==["duplicate_semantic_hypothesis_item"]
    comp=build(hypothesis_type="competing_hypothesis_candidate",hypothesis_items=[item(),item(kind="causal_proposition_candidate",role="alternative_candidate",item_id="h2",obj="object",claim="c2")])
    mixed=build(hypothesis_type="mixed_unknown_hypothesis_candidate",hypothesis_items=[item(),item(kind="unknown_origin_proposition_candidate",role="neutral_unknown",item_id="h2",claim="c2")])
    checks["competing_hypothesis_validation_passed"]=reason(comp)==[] and comp["hypothesis_competition_summary"]["winner_selected"] is False
    checks["mixed_hypothesis_validation_passed"]=reason(mixed)==[] and bool(mixed["warnings"])
    checks["competition_summary_passed"]=comp["hypothesis_competition_summary"]["competition_present"] is True and build(metadata={"hypothesis_confidence_state":"hypothesis_competing"})["hypothesis_competition_summary"]["competition_present"] is False
    checks["support_summary_passed"]=valid["hypothesis_support_summary"]=={"basis_evidence_ref_ids":["ev-a","ev-b"],"basis_evidence_ref_count":2,"external_evidence_verified":False,"evidence_sufficient_asserted":False}
    checks["resolution_remains_false_passed"]=valid["resolution_summary"]=={"resolution_candidate_only":True,"resolved":False,"accepted":False,"rejected":False,"winner_selected":False,"external_resolution_verified":False,"automatic_resolution_allowed":False}
    checks["malformed_boundary_validation_passed"]=reason(build(metadata={"hypothesis_boundary_classification":0}))==["malformed_hypothesis_boundary_class"]
    checks["malformed_confidence_validation_passed"]=reason(build(metadata={"hypothesis_confidence_state":[]}))==["malformed_hypothesis_confidence_state"]
    checks["malformed_unknown_enum_distinction_passed"]=reason(build(metadata={"hypothesis_boundary_classification":"0"}))==["unknown_hypothesis_boundary_class"] and reason(build(metadata={"hypothesis_confidence_state":"0"}))==["unknown_hypothesis_confidence_state"]
    checks["recursive_forbidden_metadata_passed"]=all(reason(build(metadata={f:True}))==[f] and reason(build(metadata={f:False}))==[] and reason(build(metadata={"x":[{f:1}]}))==["malformed_forbidden_request_field"] for f in FORBIDDEN)
    checks["recursive_forbidden_item_passed"]=all(reason(build(hypothesis_items=[item(**{f:True})]))==[f] and reason(build(hypothesis_items=[item(**{f:False})]))==[] and reason(build(hypothesis_items=[item(nested={f:"yes"})]))==["malformed_forbidden_request_field"] for f in FORBIDDEN)
    checks["scheduling_alias_forbidden_passed"]=reason(build(metadata={"cron_expression":True}))==["cron_expression"]
    pairs=[([item(item_id="a",memory_write_requested=True),item(item_id="b",vector_load_requested=True,claim="c2")],"memory_write_requested"),([item(item_id="a",memory_write_requested="yes"),item(item_id="b",vector_load_requested=True,claim="c2")],"malformed_forbidden_request_field"),([{"memory_write_requested":True},item(item_id="b",claim="c2")],"missing_or_empty_hypothesis_item_id"),([1,{"hypothesis_item_id":""}],"malformed_hypothesis_item"),([item(item_id="x"),item(item_id="x",basis=[],claim="c2")],"duplicate_hypothesis_item_id"),([item(),item(item_id="h2")],"duplicate_semantic_hypothesis_item"),([item(item_id="a",basis=["x","x"],memory_write_requested=True),item(item_id="b",claim="c2")],"memory_write_requested"),([item(item_id="x",basis=[]),item(item_id="x",claim="c2")],"duplicate_hypothesis_item_id")]
    checks["invalid_item_reordering_deterministic_passed"]=all((lambda a,b,e: a==b and reason(a)==[e] and jsonable(a))(build(hypothesis_items=it),build(hypothesis_items=list(reversed(it))),exp) for it,exp in pairs)
    class DictSub(dict): pass
    deep=[]; cur=deep
    for _ in range(105): nxt=[]; cur.append(nxt); cur=nxt
    hostile=[(),set(),frozenset(),b"x",bytearray(b"x"),float("nan"),float("inf"),-float("inf"),{1:"x"},object(),DictSub(),deep]
    checks["hostile_container_fail_closed_passed"]=all(reason(build(metadata={"bad":v}))==["non_json_serializable_semantic_input"] and jsonable(build(metadata={"bad":v})) for v in hostile)
    checks["deep_recursion_fail_closed_passed"]=reason(build(metadata={"deep":deep}))==["non_json_serializable_semantic_input"] and reason(build(hypothesis_items=[item(deep=deep)]))==["non_json_serializable_semantic_input"]
    checks["huge_integer_numeric_reason_passed"]=reason(build(hypothesis_items=[item(confidence_candidate=10**100)]))==["confidence_candidate_out_of_range"]
    adversarial=[invalid,1,{"bad":object()},{"deep":deep},tam,{**valid,"canonical_id_algorithm":"x"},dict(valid,origin_summary={}),dict(valid,extra=False),dict(valid,memory_write_performed=True),dict(valid,read_only=False),build(metadata={"memory_write_requested":True}),build(hypothesis_items=[item(memory_write_requested=True)]),build(hypothesis_items=[item(),item(item_id="h1",claim="c2")]),build(hypothesis_items=[item(),item(item_id="h2")]),build(hypothesis_items=[item(sit="other")]),build(hypothesis_type="causal_hypothesis_candidate",hypothesis_items=[item(kind="causal_proposition_candidate")]),build(hypothesis_type="predictive_hypothesis_candidate",hypothesis_items=[item(kind="predictive_proposition_candidate",role="counter_candidate",obj="object")]),build(hypothesis_type="causal_hypothesis_candidate"),build(hypothesis_type="competing_hypothesis_candidate"),build(metadata={"hypothesis_boundary_classification":"bad"}),build(metadata={"hypothesis_confidence_state":"bad"}),build(hypothesis_items=[item(confidence_candidate=10**100)])]
    checks["downstream_adversarial_coverage_passed"]=plan_ok(valid,True) and all(plan_ok(src,False) for src in adversarial)
    checks["non_json_fail_closed_passed"]=all(reason(build(metadata={"bad":v}))==["non_json_serializable_semantic_input"] for v in [object(),float("nan")])
    valid_plans=[fn(valid) for fn in BUILDERS]; invalid_plans=[fn(invalid) for fn in BUILDERS]
    checks["no_side_effects_passed"]=all(valid[k] is False for k in IMMUTABLE_FALSE) and all(valid[k] is True for k in IMMUTABLE_TRUE) and all(all(p[k] is False for k in PLAN_FALSE) and p["candidate_only"] is True and p["read_only"] is True and p["hypothesis_candidate_only"] is True for p in valid_plans+invalid_plans)
    checks["schema_summary"]=summary
    checks["all_checks_passed"]=all(v for k,v in checks.items() if k != "schema_summary")
    print(json.dumps(checks, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return 0 if checks["all_checks_passed"] else 1

if __name__ == "__main__":
    sys.exit(main())
