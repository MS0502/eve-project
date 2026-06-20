import copy
import json
import math

from adapters.virtual_world_situation_hypothesis_context_schema import (
    build_hypothesis_context_to_agp_input_plan,
    build_hypothesis_context_to_appraisal_plan,
    build_hypothesis_context_to_memory_candidate_plan,
    build_hypothesis_context_to_situation_plan,
    build_hypothesis_context_to_snapshot_plan,
    build_hypothesis_context_to_transition_preflight_plan,
    build_virtual_world_situation_hypothesis_context,
    build_virtual_world_situation_hypothesis_context_schema_summary,
    validate_virtual_world_situation_hypothesis_context,
    virtual_world_situation_hypothesis_context_schema_summary,
)

HYPOTHESIS_TYPES = ["explanatory_hypothesis_candidate","causal_hypothesis_candidate","temporal_hypothesis_candidate","state_hypothesis_candidate","identity_hypothesis_candidate","boundary_hypothesis_candidate","simulation_hypothesis_candidate","symbolic_hypothesis_candidate","dmn_hypothesis_candidate","dream_hypothesis_candidate","predictive_hypothesis_candidate","counterfactual_hypothesis_candidate","competing_hypothesis_candidate","unknown_origin_hypothesis_candidate","mixed_unknown_hypothesis_candidate"]
HYPOTHESIS_KINDS = ["explanatory_proposition_candidate","causal_proposition_candidate","temporal_proposition_candidate","state_proposition_candidate","identity_proposition_candidate","boundary_proposition_candidate","simulation_proposition_candidate","symbolic_proposition_candidate","dmn_proposition_candidate","dream_proposition_candidate","predictive_proposition_candidate","counterfactual_proposition_candidate","alternative_proposition_candidate","unknown_origin_proposition_candidate"]
CANDIDATE_ROLES = ["focal_candidate","alternative_candidate","counter_candidate","neutral_unknown"]
BOUNDARIES = ["internal_explanatory_hypothesis","causal_virtual_hypothesis","temporal_virtual_hypothesis","state_virtual_hypothesis","identity_virtual_hypothesis","boundary_virtual_hypothesis","simulation_hypothesis","symbolic_hypothesis","dmn_hypothesis","dream_hypothesis","predictive_virtual_hypothesis","counterfactual_virtual_hypothesis","unknown_origin_hypothesis","mixed_virtual_external_hypothesis_boundary"]
CONFIDENCE = ["hypothesis_unverified","hypothesis_low_confidence","hypothesis_medium_confidence","hypothesis_high_confidence_but_not_fact","hypothesis_competing","hypothesis_origin_unknown"]
TYPE_KIND = {
"explanatory_hypothesis_candidate":["explanatory_proposition_candidate"],"causal_hypothesis_candidate":["causal_proposition_candidate"],"temporal_hypothesis_candidate":["temporal_proposition_candidate"],"state_hypothesis_candidate":["state_proposition_candidate"],"identity_hypothesis_candidate":["identity_proposition_candidate"],"boundary_hypothesis_candidate":["boundary_proposition_candidate"],"simulation_hypothesis_candidate":["simulation_proposition_candidate"],"symbolic_hypothesis_candidate":["symbolic_proposition_candidate"],"dmn_hypothesis_candidate":["dmn_proposition_candidate"],"dream_hypothesis_candidate":["dream_proposition_candidate"],"predictive_hypothesis_candidate":["predictive_proposition_candidate"],"counterfactual_hypothesis_candidate":["counterfactual_proposition_candidate"],"competing_hypothesis_candidate":["alternative_proposition_candidate","boundary_proposition_candidate","causal_proposition_candidate","counterfactual_proposition_candidate","explanatory_proposition_candidate","identity_proposition_candidate","predictive_proposition_candidate","simulation_proposition_candidate","state_proposition_candidate","symbolic_proposition_candidate","temporal_proposition_candidate"],"unknown_origin_hypothesis_candidate":["unknown_origin_proposition_candidate"],"mixed_unknown_hypothesis_candidate":["alternative_proposition_candidate","boundary_proposition_candidate","causal_proposition_candidate","counterfactual_proposition_candidate","dmn_proposition_candidate","dream_proposition_candidate","explanatory_proposition_candidate","identity_proposition_candidate","predictive_proposition_candidate","simulation_proposition_candidate","state_proposition_candidate","symbolic_proposition_candidate","temporal_proposition_candidate","unknown_origin_proposition_candidate"]}
KIND_ROLE = {"explanatory_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"causal_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"temporal_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"state_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"identity_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"boundary_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"simulation_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"symbolic_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"dmn_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"dream_proposition_candidate":["alternative_candidate","focal_candidate","neutral_unknown"],"predictive_proposition_candidate":["alternative_candidate","focal_candidate"],"counterfactual_proposition_candidate":["counter_candidate"],"alternative_proposition_candidate":["alternative_candidate"],"unknown_origin_proposition_candidate":["neutral_unknown"]}
RELATIONAL = ["alternative_proposition_candidate","boundary_proposition_candidate","causal_proposition_candidate","counterfactual_proposition_candidate","predictive_proposition_candidate","temporal_proposition_candidate"]
FORBIDDEN = ["external_hypothesis_assertion_requested","hypothesis_verification_requested","hypothesis_fact_assertion_requested","hypothesis_fact_promotion_requested","hypothesis_acceptance_requested","hypothesis_rejection_requested","conclusion_assertion_requested","conclusion_selection_requested","winner_selection_requested","competition_resolution_requested","truth_resolution_requested","automatic_resolution_requested","prediction_guarantee_requested","future_outcome_assertion_requested","counterfactual_application_requested","evidence_verification_requested","evidence_sufficiency_assertion_requested","certainty_promotion_requested","fact_promotion_requested","random_sampling_requested","probability_draw_requested","memory_write_requested","memory_fact_promotion_requested","relationship_update_requested","self_model_update_requested","identity_update_requested","affect_transition_requested","hormone_transition_requested","world_state_mutation_requested","transition_application_requested","planning_execution_requested","tool_execution_requested","runtime_mutation_requested","persistence_write_requested","vector_read_requested","vector_load_requested","model_load_requested","device_activation_requested","network_action_requested","artifact_creation_requested","intervention_requested","event_completion_assertion_requested","action_prevention_requested","action_permission_requested","agp_bypass_requested","fallback_bypass_requested","schedule_requested","scheduler_requested","timer_requested","alarm_requested","calendar_requested","reminder_requested","deadline_requested","cron_requested","cron_expression","scheduled_execution_requested","delayed_execution_requested","act_requested","execute_requested","apply_requested","commit_outcome_requested","promote_to_fact_requested"]
IMMUTABLE_FALSE = ["external_hypothesis_asserted","hypothesis_verified","hypothesis_fact_asserted","hypothesis_accepted","hypothesis_rejected","conclusion_asserted","truth_resolved","winner_selected","competition_resolved","automatic_resolution_performed","automatic_resolution_allowed","prediction_guaranteed","future_outcome_asserted","counterfactual_applied","evidence_verified","evidence_sufficient_asserted","random_sampling_performed","probability_draw_performed","identity_asserted","memory_fact_asserted","memory_write_performed","memory_write_allowed","relationship_update_allowed","self_model_update_allowed","affect_transition_allowed","hormone_transition_allowed","world_state_mutation_performed","transition_applied","planning_execution_performed","tool_execution_performed","runtime_mutation_performed","persistence_write_performed","vector_read_performed","vector_load_performed","model_loaded","device_activation_performed","network_action_performed","artifact_created_or_staged","schedule_created","action_prevented","action_permitted","agp_bypass_allowed","fallback_bypass_allowed"]
IMMUTABLE_TRUE = ["hypothesis_candidate_only","situation_hypothesis_only","read_only","situation_review_required","snapshot_review_required","transition_review_required","memory_gate_required","quarantine_required","origin_fact_status_required","hypothesis_integrity_review_required","evidence_review_required","competition_review_required","resolution_review_required","appraisal_required","agp_input_required"]
PLAN_FALSE = [x for x in IMMUTABLE_FALSE if x != "artifact_created_or_staged"] + ["relationship_update_performed","self_model_update_performed","affect_transition_performed","hormone_transition_performed"]
TOP_KEYS = set(["situation_hypothesis_context_passed","situation_hypothesis_context_status","hypothesis_context_id","canonical_id_algorithm","hypothesis_type","hypothesis_boundary_classification","hypothesis_confidence_state","situation_id","hypothesis_items","metadata","origin_summary","fact_status_summary","hypothesis_scope_summary","hypothesis_competition_summary","hypothesis_support_summary","resolution_summary","hypothesis_flags","boundary_flags","hypothesis_integrity_flags","candidate_only_fields","blocked_reasons","warnings"] + IMMUTABLE_FALSE + IMMUTABLE_TRUE)
BUILDERS = [build_hypothesis_context_to_situation_plan,build_hypothesis_context_to_snapshot_plan,build_hypothesis_context_to_transition_preflight_plan,build_hypothesis_context_to_memory_candidate_plan,build_hypothesis_context_to_appraisal_plan,build_hypothesis_context_to_agp_input_plan]

def item(kind="explanatory_proposition_candidate", role="focal_candidate", item_id="h1", sit="sit-민석", obj=None, claim="claim", basis=None, **extra):
    d={"hypothesis_item_id":item_id,"situation_id":sit,"hypothesis_kind":kind,"candidate_role":role,"subject_ref_id":"subject","claim_ref_id":claim,"basis_evidence_ref_ids":["ev-b","ev-a"] if basis is None else basis}
    if obj is not None: d["object_ref_id"]=obj
    d.update(extra); return d

def build(**kw):
    args={"hypothesis_type":"explanatory_hypothesis_candidate","situation_id":"sit-민석","hypothesis_items":[item()],"metadata":{}}
    args.update(kw); return build_virtual_world_situation_hypothesis_context(**args)

def reasons(**kw): return build(**kw)["blocked_reasons"]
def assert_jsonable(payload): json.dumps(payload, ensure_ascii=False, allow_nan=False)
def obj_for(kind): return "object" if kind in RELATIONAL else None

def test_schema_summary_uses_literal_contracts():
    s=build_virtual_world_situation_hypothesis_context_schema_summary(); s2=virtual_world_situation_hypothesis_context_schema_summary()
    assert s == s2
    assert s["supported_hypothesis_types"] == HYPOTHESIS_TYPES
    assert s["supported_hypothesis_kinds"] == HYPOTHESIS_KINDS
    assert s["supported_candidate_roles"] == CANDIDATE_ROLES
    assert s["supported_boundary_classes"] == BOUNDARIES
    assert s["supported_confidence_states"] == CONFIDENCE
    assert s["type_kind_compatibility"] == TYPE_KIND
    assert s["kind_role_compatibility"] == KIND_ROLE
    assert s["relational_hypothesis_kinds"] == RELATIONAL
    assert s["next_recommended_step"] == "read_only_virtual_world_situation_inference_context_schema"

def test_valid_payload_exact_contract_and_summaries():
    c=build(metadata={"hypothesis_confidence_state":"hypothesis_high_confidence_but_not_fact"})
    assert set(c.keys()) == TOP_KEYS
    assert validate_virtual_world_situation_hypothesis_context(c) is True
    assert c["hypothesis_items"][0]["basis_evidence_ref_ids"] == ["ev-a","ev-b"]
    assert c["origin_summary"] == {"origin_kind":"internal_virtual_hypothesis_candidate","external_origin_verified":False}
    assert c["fact_status_summary"] == {"candidate_only":True,"external_hypothesis_fact_verified":False,"hypothesis_fact_asserted":False,"hypothesis_accepted":False,"hypothesis_rejected":False,"conclusion_asserted":False,"truth_resolved":False}
    assert c["resolution_summary"] == {"resolution_candidate_only":True,"resolved":False,"accepted":False,"rejected":False,"winner_selected":False,"external_resolution_verified":False,"automatic_resolution_allowed":False}
    for k in IMMUTABLE_FALSE: assert c[k] is False
    for k in IMMUTABLE_TRUE: assert c[k] is True

def test_all_valid_enums_and_matrices_with_literal_expectations():
    assert set(HYPOTHESIS_TYPES) and set(HYPOTHESIS_KINDS) and set(CANDIDATE_ROLES)
    for t, kinds in TYPE_KIND.items():
        items=[item(kind=kinds[0], role=KIND_ROLE[kinds[0]][0], obj=obj_for(kinds[0]))]
        if t in {"competing_hypothesis_candidate","mixed_unknown_hypothesis_candidate"}:
            items=[item(), item(kind="causal_proposition_candidate", role="alternative_candidate", item_id="h2", obj="object", claim="claim2")]
        assert build(hypothesis_type=t, hypothesis_items=items)["blocked_reasons"] == []
    for kind, roles in KIND_ROLE.items():
        for role in roles:
            base=item(kind="state_proposition_candidate", role="neutral_unknown", item_id="base", claim="base")
            if kind == "state_proposition_candidate" and role == "neutral_unknown": base=item(kind="explanatory_proposition_candidate", role="focal_candidate", item_id="base", claim="base")
            assert build(hypothesis_type="mixed_unknown_hypothesis_candidate", hypothesis_items=[base,item(kind=kind, role=role, item_id="h2"+role+kind, obj=obj_for(kind), claim="c"+kind+role)])["blocked_reasons"] == []
    for b in BOUNDARIES: assert reasons(metadata={"hypothesis_boundary_classification":b}) == []
    for c in CONFIDENCE: assert reasons(metadata={"hypothesis_confidence_state":c}) == []

def test_required_failure_reasons_and_reference_shape():
    cases=[({"hypothesis_type":None},"missing_hypothesis_type"),({"hypothesis_type":"x"},"unknown_hypothesis_type"),({"situation_id":""},"missing_or_malformed_situation_id"),({"metadata":[]},"invalid_metadata"),({"hypothesis_items":None},"missing_hypothesis_items"),({"hypothesis_items":{}},"hypothesis_items_not_non_empty_list"),({"hypothesis_items":[]},"empty_hypothesis_items"),({"hypothesis_items":[1]},"malformed_hypothesis_item"),({"hypothesis_items":[item(hypothesis_item_id="")]},"missing_or_empty_hypothesis_item_id"),({"hypothesis_items":[item(hypothesis_kind="bad")]},"unknown_hypothesis_kind"),({"hypothesis_items":[item(candidate_role="bad")]},"unknown_candidate_role"),({"hypothesis_items":[item(subject_ref_id="")]},"missing_or_malformed_subject_ref_id"),({"hypothesis_items":[item(claim_ref_id="")]},"missing_or_malformed_claim_ref_id"),({"hypothesis_items":[item(basis=[])]},"missing_or_malformed_basis_evidence_ref_ids"),({"hypothesis_items":[item(basis=["x","x"])]},"duplicate_basis_evidence_ref_id"),({"hypothesis_items":[item(related_context_id="")]},"malformed_related_context_id"),({"hypothesis_items":[item(confidence_candidate=True)]},"malformed_confidence_candidate"),({"hypothesis_items":[item(confidence_candidate=10**100)]},"confidence_candidate_out_of_range"),({"hypothesis_items":[item(plausibility_candidate=True)]},"malformed_plausibility_candidate"),({"hypothesis_items":[item(plausibility_candidate=-1)]},"plausibility_candidate_out_of_range"),({"hypothesis_items":[item(label="")]},"malformed_hypothesis_item_label")]
    for kw, reason in cases: assert reasons(**kw) == [reason]
    assert reasons(hypothesis_type="causal_hypothesis_candidate", hypothesis_items=[item(kind="causal_proposition_candidate")]) == ["missing_object_ref_id_for_relational_hypothesis"]
    assert reasons(hypothesis_type="causal_hypothesis_candidate", hypothesis_items=[item(kind="causal_proposition_candidate", obj="")]) == ["malformed_object_ref_id"]
    assert reasons(hypothesis_type="causal_hypothesis_candidate", hypothesis_items=[item(kind="causal_proposition_candidate", obj="subject")]) == ["identical_hypothesis_refs"]
    assert reasons(hypothesis_items=[item(obj="object")]) == ["unexpected_object_ref_id_for_unary_hypothesis"]

def test_validation_precedence_singleton_reasons():
    assert build(hypothesis_type="bad", metadata={"x": object()})["blocked_reasons"] == ["non_json_serializable_semantic_input"]
    assert build(hypothesis_type="bad", situation_id="")["blocked_reasons"] == ["unknown_hypothesis_type"]
    assert build(metadata=[], hypothesis_items=[item(memory_write_requested=True)])["blocked_reasons"] == ["invalid_metadata"]
    assert build(metadata={"memory_write_requested": True}, hypothesis_items=[])["blocked_reasons"] == ["memory_write_requested"]
    assert build(hypothesis_items=[{"memory_write_requested": True}])["blocked_reasons"] == ["missing_or_empty_hypothesis_item_id"]
    assert build(hypothesis_items=[item(basis=["x","x"], memory_write_requested=True)])["blocked_reasons"] == ["memory_write_requested"]
    assert build(hypothesis_items=[item(basis=[]), item(item_id="h1", claim="c2")])["blocked_reasons"] == ["duplicate_hypothesis_item_id"]
    assert build(hypothesis_items=[item(basis=[]), item(item_id="h2")])["blocked_reasons"] == ["missing_or_malformed_basis_evidence_ref_ids"]
    assert build(hypothesis_items=[item(sit="other"), item(item_id="h2", sit="other")])["blocked_reasons"] == ["duplicate_semantic_hypothesis_item"]
    assert build(hypothesis_items=[item(sit="other", obj="object")])["blocked_reasons"] == ["hypothesis_item_situation_mismatch"]
    assert build(hypothesis_type="predictive_hypothesis_candidate", hypothesis_items=[item(kind="predictive_proposition_candidate", role="counter_candidate")])["blocked_reasons"] == ["missing_object_ref_id_for_relational_hypothesis"]
    assert build(hypothesis_type="causal_hypothesis_candidate", hypothesis_items=[item(kind="causal_proposition_candidate", role="counter_candidate", obj="object")])["blocked_reasons"] == ["incompatible_hypothesis_kind_candidate_role"]
    assert build(hypothesis_type="causal_hypothesis_candidate")["blocked_reasons"] == ["incompatible_hypothesis_type_kind"]
    assert build(hypothesis_type="competing_hypothesis_candidate", hypothesis_items=[item(metadata_boundary="x")], metadata={"hypothesis_boundary_classification":"bad"})["blocked_reasons"] == ["competing_hypothesis_requires_multiple_items"]
    assert build(metadata={"hypothesis_boundary_classification":"bad","hypothesis_confidence_state":"bad"})["blocked_reasons"] == ["unknown_hypothesis_boundary_class"]

def invalid_pair(items, expected, **kw):
    a=build(hypothesis_items=items, **kw); b=build(hypothesis_items=list(reversed(items)), **kw)
    assert a == b
    assert a["blocked_reasons"] == [expected]
    assert_jsonable(a)

def test_invalid_reordering_is_deterministic_for_required_scenarios():
    invalid_pair([item(item_id="a", memory_write_requested=True), item(item_id="b", vector_load_requested=True, claim="c2")], "memory_write_requested")
    invalid_pair([item(item_id="a", memory_write_requested="yes"), item(item_id="b", vector_load_requested=True, claim="c2")], "malformed_forbidden_request_field")
    invalid_pair([{"memory_write_requested": True}, item(item_id="b", claim="c2")], "missing_or_empty_hypothesis_item_id")
    invalid_pair([1, {"hypothesis_item_id":""}], "malformed_hypothesis_item")
    invalid_pair([item(item_id="x"), item(item_id="x", basis=[], claim="c2")], "duplicate_hypothesis_item_id")
    invalid_pair([item(kind="predictive_proposition_candidate", role="counter_candidate", obj="object"), item(kind="predictive_proposition_candidate", role="counter_candidate", obj="object", item_id="h2")], "duplicate_semantic_hypothesis_item")
    invalid_pair([item(item_id="a", basis=["x","x"], memory_write_requested=True), item(item_id="b", claim="c2")], "memory_write_requested")
    invalid_pair([item(item_id="x", basis=[]), item(item_id="x", claim="c2")], "duplicate_hypothesis_item_id")

def test_strict_json_hostile_values_all_entry_points_fail_closed():
    class DictSub(dict): pass
    class ListSub(list): pass
    class BadKeys(dict):
        def keys(self): raise RuntimeError("bad")
    class BadIter(list):
        def __iter__(self): raise RuntimeError("bad")
    class BadStr:
        def __str__(self): raise RuntimeError("bad")
    class BadRepr:
        def __repr__(self): raise RuntimeError("bad")
    circ_d={}; circ_d["self"]=circ_d
    circ_l=[]; circ_l.append(circ_l)
    deep=[]; cur=deep
    for _ in range(105):
        nxt=[]; cur.append(nxt); cur=nxt
    bad_values=[(), set(), frozenset(), b"x", bytearray(b"x"), float("nan"), float("inf"), -float("inf"), {1:"x"}, object(), circ_d, circ_l, deep, DictSub(), ListSub(), BadKeys(), BadIter(), BadStr(), BadRepr()]
    for v in bad_values:
        payload=build(metadata={"bad": v})
        assert payload["blocked_reasons"] == ["non_json_serializable_semantic_input"]
        assert_jsonable(payload)
        assert validate_virtual_world_situation_hypothesis_context(v) is False
        for fn in BUILDERS:
            p=fn(v); assert p["ready"] is False and p["candidate_only"] is True and p["read_only"] is True and p["hypothesis_candidate_only"] is True
            for f in PLAN_FALSE: assert p[f] is False
    assert build(hypothesis_items=[item(deep=deep)])["blocked_reasons"] == ["non_json_serializable_semantic_input"]
    assert build(hypothesis_items=[item(confidence_candidate=10**100)])["blocked_reasons"] == ["confidence_candidate_out_of_range"]

def test_recursive_forbidden_matrix_all_fields_locations_and_values():
    locations=[lambda f,v: {"metadata":{f:v}}, lambda f,v: {"metadata":{"nested":{f:v}}}, lambda f,v: {"metadata":{"lst":[{f:v}]}}, lambda f,v: {"hypothesis_items":[item(**{f:v})]}, lambda f,v: {"hypothesis_items":[item(nested={f:v})]}, lambda f,v: {"hypothesis_items":[item(lst=[{f:v}])]}]
    for f in FORBIDDEN:
        for loc in locations:
            assert build(**loc(f, True))["blocked_reasons"] == [f]
            assert build(**loc(f, False))["blocked_reasons"] == []
            assert build(**loc(f, 1))["blocked_reasons"] == ["malformed_forbidden_request_field"]
            assert build(**loc(f, "yes"))["blocked_reasons"] == ["malformed_forbidden_request_field"]

def test_payload_tamper_matrix_is_rejected():
    valid=build(metadata={"m":"v"})
    tamper_paths=[("situation_hypothesis_context_passed",False), ("situation_hypothesis_context_status","X"), ("blocked_reasons",["x"]), ("hypothesis_context_id","0"), ("canonical_id_algorithm","x"), ("hypothesis_type","causal_hypothesis_candidate"), ("hypothesis_boundary_classification","causal_virtual_hypothesis"), ("hypothesis_confidence_state","hypothesis_low_confidence"), ("situation_id","other"), ("hypothesis_items",[]), ("metadata",{}), ("origin_summary",{}), ("fact_status_summary",{}), ("hypothesis_scope_summary",{}), ("hypothesis_competition_summary",{}), ("hypothesis_support_summary",{}), ("resolution_summary",{}), ("hypothesis_flags",[]), ("boundary_flags",[]), ("hypothesis_integrity_flags",[]), ("candidate_only_fields",[]), ("warnings",["x"])]
    for key, value in tamper_paths:
        t=copy.deepcopy(valid); t[key]=value; assert validate_virtual_world_situation_hypothesis_context(t) is False
    for key in IMMUTABLE_FALSE:
        t=copy.deepcopy(valid); t[key]=True; assert validate_virtual_world_situation_hypothesis_context(t) is False
    for key in IMMUTABLE_TRUE:
        t=copy.deepcopy(valid); t[key]=False; assert validate_virtual_world_situation_hypothesis_context(t) is False
    for mutation in [lambda t: t.pop("read_only"), lambda t: t.update({"extra":False}), lambda t: t.update({"memory_write_requested":True}), lambda t: t.update({"read_only":1}), lambda t: t.update({"item_count":False}), lambda t: t["hypothesis_scope_summary"].update({"item_count":1.0}), lambda t: t["hypothesis_items"][0].update({"basis_evidence_ref_ids":["ev-b","ev-a"]}), lambda t: t["hypothesis_items"][0].update({"basis_evidence_ref_ids":["ev-a","changed"]}), lambda t: t["hypothesis_items"][0].update({"claim_ref_id":"changed"}), lambda t: t["resolution_summary"].update({"resolved":True})]:
        t=copy.deepcopy(valid); mutation(t); assert validate_virtual_world_situation_hypothesis_context(t) is False

def test_all_plan_builders_with_adversarial_sources_and_literal_false_fields():
    valid=build(); rejected=build(hypothesis_type="bad"); deep=[]; cur=deep
    for _ in range(105): nxt=[]; cur.append(nxt); cur=nxt
    invalid_sources=[rejected, 1, {"x": object()}, {"deep": deep}, {**valid,"hypothesis_context_id":"0"}, {**valid,"canonical_id_algorithm":"x"}, dict(valid, origin_summary={}), dict(valid, extra=False), dict(valid, memory_write_performed=True), dict(valid, read_only=False), build(metadata={"memory_write_requested":True}), build(hypothesis_items=[item(memory_write_requested=True)]), build(hypothesis_items=[item(), item(item_id="h1", claim="c2")]), build(hypothesis_items=[item(), item(item_id="h2")]), build(hypothesis_items=[item(sit="other")]), build(hypothesis_type="causal_hypothesis_candidate", hypothesis_items=[item(kind="causal_proposition_candidate")]), build(hypothesis_type="predictive_hypothesis_candidate", hypothesis_items=[item(kind="predictive_proposition_candidate", role="counter_candidate", obj="object")]), build(hypothesis_type="causal_hypothesis_candidate"), build(hypothesis_type="competing_hypothesis_candidate"), build(metadata={"hypothesis_boundary_classification":"bad"}), build(metadata={"hypothesis_confidence_state":"bad"}), build(hypothesis_items=[item(confidence_candidate=10**100)]), build(hypothesis_items=[item(basis=["x","x"], memory_write_requested=True), item(item_id="b", claim="c2")])]
    for fn in BUILDERS:
        p=fn(valid); assert p["ready"] is True and p["candidate_only"] is True and p["read_only"] is True and p["hypothesis_candidate_only"] is True
        for f in PLAN_FALSE: assert p[f] is False
        for src in invalid_sources:
            p=fn(src); assert p["ready"] is False and p["candidate_only"] is True and p["read_only"] is True and p["hypothesis_candidate_only"] is True
            for f in PLAN_FALSE: assert p[f] is False
