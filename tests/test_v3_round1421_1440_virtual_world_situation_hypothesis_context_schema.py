import json
import pytest

from adapters.virtual_world_situation_hypothesis_context_schema import *


def item(kind="explanatory_proposition_candidate", role="focal_candidate", item_id="h1", obj=None, **extra):
    d={"hypothesis_item_id":item_id,"situation_id":"sit-민석","hypothesis_kind":kind,"candidate_role":role,"subject_ref_id":"s","claim_ref_id":"c","basis_evidence_ref_ids":["b","a"]}
    if obj is not None: d["object_ref_id"]=obj
    d.update(extra); return d

def build(**kw):
    a={"hypothesis_type":"explanatory_hypothesis_candidate","situation_id":"sit-민석","hypothesis_items":[item()],"metadata":{}}
    a.update(kw); return build_virtual_world_situation_hypothesis_context(**a)

def reject(**kw): return build(**kw)["blocked_reasons"]

LITERAL_TOP_KEYS = {"situation_hypothesis_context_passed","situation_hypothesis_context_status","hypothesis_context_id","canonical_id_algorithm","hypothesis_type","hypothesis_boundary_classification","hypothesis_confidence_state","situation_id","hypothesis_items","metadata","origin_summary","fact_status_summary","hypothesis_scope_summary","hypothesis_competition_summary","hypothesis_support_summary","resolution_summary","hypothesis_flags","boundary_flags","hypothesis_integrity_flags","candidate_only_fields","blocked_reasons","warnings","external_hypothesis_asserted","hypothesis_verified","hypothesis_fact_asserted","hypothesis_accepted","hypothesis_rejected","conclusion_asserted","truth_resolved","winner_selected","competition_resolved","automatic_resolution_performed","automatic_resolution_allowed","prediction_guaranteed","future_outcome_asserted","counterfactual_applied","evidence_verified","evidence_sufficient_asserted","random_sampling_performed","probability_draw_performed","identity_asserted","memory_fact_asserted","memory_write_performed","memory_write_allowed","relationship_update_allowed","self_model_update_allowed","affect_transition_allowed","hormone_transition_allowed","world_state_mutation_performed","transition_applied","planning_execution_performed","tool_execution_performed","runtime_mutation_performed","persistence_write_performed","vector_read_performed","vector_load_performed","model_loaded","device_activation_performed","network_action_performed","artifact_created_or_staged","schedule_created","action_prevented","action_permitted","agp_bypass_allowed","fallback_bypass_allowed","hypothesis_candidate_only","situation_hypothesis_only","read_only","situation_review_required","snapshot_review_required","transition_review_required","memory_gate_required","quarantine_required","origin_fact_status_required","hypothesis_integrity_review_required","evidence_review_required","competition_review_required","resolution_review_required","appraisal_required","agp_input_required"}


def test_literal_payload_contract_and_summaries():
    c=build(metadata={"hypothesis_confidence_state":"hypothesis_high_confidence_but_not_fact"})
    assert set(c)==LITERAL_TOP_KEYS
    assert validate_virtual_world_situation_hypothesis_context(c)
    assert c["fact_status_summary"]=={"candidate_only":True,"external_hypothesis_fact_verified":False,"hypothesis_fact_asserted":False,"hypothesis_accepted":False,"hypothesis_rejected":False,"conclusion_asserted":False,"truth_resolved":False}
    assert c["resolution_summary"]=={"resolution_candidate_only":True,"resolved":False,"accepted":False,"rejected":False,"winner_selected":False,"external_resolution_verified":False,"automatic_resolution_allowed":False}
    assert c["hypothesis_support_summary"]["basis_evidence_ref_ids"]==["a","b"]
    json.dumps(c, allow_nan=False)

@pytest.mark.parametrize("hypothesis_type,kinds", sorted(TYPE_KIND_COMPATIBILITY.items()))
def test_every_valid_type_kind(hypothesis_type,kinds):
    k=sorted(kinds)[0]; r=sorted(KIND_ROLE_COMPATIBILITY[k])[0]
    items=[item(kind=k, role=r, obj="o" if k in RELATIONAL_KINDS else None)]
    if hypothesis_type in {"competing_hypothesis_candidate","mixed_unknown_hypothesis_candidate"}:
        items=[item(), item(kind="causal_proposition_candidate", role="alternative_candidate", item_id="h2", obj="o", claim_ref_id="c2")]
    assert build(hypothesis_type=hypothesis_type,hypothesis_items=items)["blocked_reasons"]==[]

@pytest.mark.parametrize("kind,roles", sorted(KIND_ROLE_COMPATIBILITY.items()))
def test_every_valid_kind_role(kind,roles):
    for role in roles:
        items=[item(kind="state_proposition_candidate", role="neutral_unknown", item_id="base", claim_ref_id="baseclaim"), item(kind=kind, role=role, item_id="h2"+role, obj="o" if kind in RELATIONAL_KINDS else None, claim_ref_id="c2"+role)]
        if kind == "state_proposition_candidate" and role == "neutral_unknown":
            items[0] = item(kind="explanatory_proposition_candidate", role="focal_candidate", item_id="base", claim_ref_id="baseclaim")
        assert build(hypothesis_type="mixed_unknown_hypothesis_candidate", hypothesis_items=items)["blocked_reasons"]==[]

@pytest.mark.parametrize("boundary", ["internal_explanatory_hypothesis","causal_virtual_hypothesis","temporal_virtual_hypothesis","state_virtual_hypothesis","identity_virtual_hypothesis","boundary_virtual_hypothesis","simulation_hypothesis","symbolic_hypothesis","dmn_hypothesis","dream_hypothesis","predictive_virtual_hypothesis","counterfactual_virtual_hypothesis","unknown_origin_hypothesis","mixed_virtual_external_hypothesis_boundary"])
def test_boundaries(boundary): assert build(metadata={"hypothesis_boundary_classification":boundary})["blocked_reasons"]==[]

@pytest.mark.parametrize("confidence", ["hypothesis_unverified","hypothesis_low_confidence","hypothesis_medium_confidence","hypothesis_high_confidence_but_not_fact","hypothesis_competing","hypothesis_origin_unknown"])
def test_confidence(confidence):
    c=build(metadata={"hypothesis_confidence_state":confidence})
    assert c["blocked_reasons"]==[] and c["hypothesis_accepted"] is False and c["hypothesis_verified"] is False

@pytest.mark.parametrize("kw,reason", [
    ({"hypothesis_type":None},"missing_hypothesis_type"),({"hypothesis_type":"x"},"unknown_hypothesis_type"),({"situation_id":""},"missing_or_malformed_situation_id"),({"metadata":[]},"invalid_metadata"),({"hypothesis_items":None},"missing_hypothesis_items"),({"hypothesis_items":{}},"hypothesis_items_not_non_empty_list"),({"hypothesis_items":[]},"empty_hypothesis_items"),({"hypothesis_items":[1]},"malformed_hypothesis_item"),({"hypothesis_items":[item(hypothesis_item_id="")]},"missing_or_empty_hypothesis_item_id"),({"hypothesis_items":[item(hypothesis_kind="bad")]},"unknown_hypothesis_kind"),({"hypothesis_items":[item(candidate_role="bad")]},"unknown_candidate_role"),({"hypothesis_items":[item(subject_ref_id="")]},"missing_or_malformed_subject_ref_id"),({"hypothesis_items":[item(claim_ref_id="")]},"missing_or_malformed_claim_ref_id"),({"hypothesis_items":[item(basis_evidence_ref_ids=["x","x"])]},"duplicate_basis_evidence_ref_id"),({"hypothesis_items":[item(related_context_id="")]},"malformed_related_context_id"),({"hypothesis_items":[item(confidence_candidate=True)]},"malformed_confidence_candidate"),({"hypothesis_items":[item(confidence_candidate=2)]},"confidence_candidate_out_of_range"),({"hypothesis_items":[item(plausibility_candidate=True)]},"malformed_plausibility_candidate"),({"hypothesis_items":[item(plausibility_candidate=-1)]},"plausibility_candidate_out_of_range"),({"hypothesis_items":[item(label="")]},"malformed_hypothesis_item_label")])
def test_fail_reasons(kw,reason): assert reject(**kw)==[reason]


def test_reference_duplicates_composition_and_ordering():
    assert reject(hypothesis_type="causal_hypothesis_candidate", hypothesis_items=[item(kind="causal_proposition_candidate")])==["missing_object_ref_id_for_relational_hypothesis"]
    assert reject(hypothesis_type="causal_hypothesis_candidate", hypothesis_items=[item(kind="causal_proposition_candidate", obj="")])==["malformed_object_ref_id"]
    assert reject(hypothesis_type="causal_hypothesis_candidate", hypothesis_items=[item(kind="causal_proposition_candidate", obj="s")])==["identical_hypothesis_refs"]
    assert reject(hypothesis_items=[item(obj="o")])==["unexpected_object_ref_id_for_unary_hypothesis"]
    assert reject(hypothesis_items=[item(), item(item_id="h1", claim_ref_id="c2")])==["duplicate_hypothesis_item_id"]
    assert reject(hypothesis_items=[item(), item(item_id="h2")])==["duplicate_semantic_hypothesis_item"]
    assert reject(hypothesis_items=[item(situation_id="other")])==["hypothesis_item_situation_mismatch"]
    assert reject(hypothesis_type="predictive_hypothesis_candidate", hypothesis_items=[item(kind="predictive_proposition_candidate", role="counter_candidate", obj="o")])==["incompatible_hypothesis_kind_candidate_role"]
    assert reject(hypothesis_type="causal_hypothesis_candidate") == ["incompatible_hypothesis_type_kind"]
    assert reject(hypothesis_type="competing_hypothesis_candidate") == ["competing_hypothesis_requires_multiple_items"]
    assert reject(hypothesis_type="mixed_unknown_hypothesis_candidate") == ["mixed_unknown_hypothesis_requires_multiple_items"]
    a=build(metadata={"z":1,"a":2}, hypothesis_items=[dict(reversed(list(item().items())))])
    b=build(metadata={"a":2,"z":1}, hypothesis_items=[item(basis_evidence_ref_ids=["a","b"])])
    assert a["hypothesis_context_id"]==b["hypothesis_context_id"]


def test_strict_json_forbidden_tamper_and_plans():
    for bad in [tuple(), set(), b"x", float("nan")]:
        assert reject(metadata={"bad":bad}) == ["non_json_serializable_semantic_input"]
    d={}; d["d"]=d
    assert reject(metadata=d)==["non_json_serializable_semantic_input"]
    assert reject(metadata={"winner_selection_requested":True})==["winner_selection_requested"]
    assert reject(metadata={"winner_selection_requested":"yes"})==["malformed_forbidden_request_field"]
    valid=build(); tam=dict(valid); tam["read_only"]=1
    assert not validate_virtual_world_situation_hypothesis_context(tam)
    tam=dict(valid); tam["extra"]=False
    assert not validate_virtual_world_situation_hypothesis_context(tam)
    tam=dict(valid); tam["hypothesis_items"]=[dict(valid["hypothesis_items"][0], basis_evidence_ref_ids=["b","a"])]
    assert not validate_virtual_world_situation_hypothesis_context(tam)
    builders=[build_hypothesis_context_to_situation_plan,build_hypothesis_context_to_snapshot_plan,build_hypothesis_context_to_transition_preflight_plan,build_hypothesis_context_to_memory_candidate_plan,build_hypothesis_context_to_appraisal_plan,build_hypothesis_context_to_agp_input_plan]
    for fn in builders:
        p=fn(valid); assert p["ready"] is True
        q=fn(tam); assert q["ready"] is False and q["candidate_only"] is True and q["read_only"] is True and q["hypothesis_candidate_only"] is True
        for f in PLAN_FALSE_FLAGS: assert q[f] is False


def test_schema_summary_literal_next_step():
    s=build_virtual_world_situation_hypothesis_context_schema_summary()
    assert s["next_recommended_step"]=="read_only_virtual_world_situation_inference_context_schema"
    assert s["supported_hypothesis_types"]==list(SUPPORTED_HYPOTHESIS_TYPES)
