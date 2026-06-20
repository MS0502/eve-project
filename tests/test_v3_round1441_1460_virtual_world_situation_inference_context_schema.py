import copy, json, math

from adapters.virtual_world_situation_inference_context_schema import (
    build_virtual_world_situation_inference_context,
    build_virtual_world_situation_inference_context_schema_summary,
    validate_virtual_world_situation_inference_context,
    build_inference_context_to_situation_plan,
    build_inference_context_to_snapshot_plan,
    build_inference_context_to_transition_preflight_plan,
    build_inference_context_to_memory_candidate_plan,
    build_inference_context_to_appraisal_plan,
    build_inference_context_to_agp_input_plan,
    virtual_world_situation_inference_context_schema_summary,
)

TYPES = ["deductive_inference_candidate", "abductive_inference_candidate", "inductive_inference_candidate", "causal_inference_candidate", "temporal_inference_candidate", "state_inference_candidate", "identity_inference_candidate", "boundary_inference_candidate", "simulation_inference_candidate", "symbolic_inference_candidate", "dmn_inference_candidate", "dream_inference_candidate", "predictive_inference_candidate", "counterfactual_inference_candidate", "competing_inference_candidate", "unknown_origin_inference_candidate", "mixed_unknown_inference_candidate"]
KINDS = ["deductive_step_candidate", "abductive_step_candidate", "inductive_step_candidate", "causal_step_candidate", "temporal_step_candidate", "state_step_candidate", "identity_step_candidate", "boundary_step_candidate", "simulation_step_candidate", "symbolic_step_candidate", "dmn_step_candidate", "dream_step_candidate", "predictive_step_candidate", "counterfactual_step_candidate", "alternative_step_candidate", "unknown_origin_step_candidate"]
ROLES = ["focal_conclusion_candidate", "alternative_conclusion_candidate", "counter_conclusion_candidate", "neutral_unknown"]
SOURCES = ["internal_hypothesis_context", "internal_evidence_context", "internal_constraint_context", "internal_uncertainty_context", "internal_temporal_context", "internal_causal_context", "internal_simulation", "internal_symbolic", "internal_dmn", "internal_dream", "external_unverified_candidate", "unknown_origin_candidate"]
BOUNDARIES = ["internal_deductive_inference", "internal_abductive_inference", "internal_inductive_inference", "causal_virtual_inference", "temporal_virtual_inference", "state_virtual_inference", "identity_virtual_inference", "boundary_virtual_inference", "simulation_inference", "symbolic_inference", "dmn_inference", "dream_inference", "predictive_virtual_inference", "counterfactual_virtual_inference", "unknown_origin_inference", "mixed_virtual_external_inference_boundary"]
CONFIDENCE = ["inference_unverified", "inference_low_confidence", "inference_medium_confidence", "inference_high_confidence_but_not_fact", "inference_competing", "inference_origin_unknown"]
TYPE_KIND = {"deductive_inference_candidate": {"deductive_step_candidate"}, "abductive_inference_candidate": {"abductive_step_candidate"}, "inductive_inference_candidate": {"inductive_step_candidate"}, "causal_inference_candidate": {"causal_step_candidate"}, "temporal_inference_candidate": {"temporal_step_candidate"}, "state_inference_candidate": {"state_step_candidate"}, "identity_inference_candidate": {"identity_step_candidate"}, "boundary_inference_candidate": {"boundary_step_candidate"}, "simulation_inference_candidate": {"simulation_step_candidate"}, "symbolic_inference_candidate": {"symbolic_step_candidate"}, "dmn_inference_candidate": {"dmn_step_candidate"}, "dream_inference_candidate": {"dream_step_candidate"}, "predictive_inference_candidate": {"predictive_step_candidate"}, "counterfactual_inference_candidate": {"counterfactual_step_candidate"}, "competing_inference_candidate": {"deductive_step_candidate", "abductive_step_candidate", "inductive_step_candidate", "causal_step_candidate", "temporal_step_candidate", "state_step_candidate", "identity_step_candidate", "boundary_step_candidate", "simulation_step_candidate", "symbolic_step_candidate", "predictive_step_candidate", "counterfactual_step_candidate", "alternative_step_candidate"}, "unknown_origin_inference_candidate": {"unknown_origin_step_candidate"}, "mixed_unknown_inference_candidate": set(KINDS)}
KIND_ROLE = {k: {"focal_conclusion_candidate", "alternative_conclusion_candidate", "neutral_unknown"} for k in KINDS[:12]}
KIND_ROLE.update({"predictive_step_candidate": {"focal_conclusion_candidate", "alternative_conclusion_candidate"}, "counterfactual_step_candidate": {"counter_conclusion_candidate"}, "alternative_step_candidate": {"alternative_conclusion_candidate"}, "unknown_origin_step_candidate": {"neutral_unknown"}})
KIND_SOURCE = {"deductive_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_constraint_context"}, "abductive_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_uncertainty_context", "unknown_origin_candidate"}, "inductive_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "external_unverified_candidate", "unknown_origin_candidate"}, "causal_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_causal_context"}, "temporal_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_temporal_context"}, "state_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_constraint_context"}, "identity_step_candidate": {"internal_hypothesis_context", "internal_evidence_context"}, "boundary_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_uncertainty_context"}, "simulation_step_candidate": {"internal_simulation"}, "symbolic_step_candidate": {"internal_symbolic"}, "dmn_step_candidate": {"internal_dmn"}, "dream_step_candidate": {"internal_dream"}, "predictive_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_temporal_context", "internal_causal_context"}, "counterfactual_step_candidate": {"internal_hypothesis_context", "internal_simulation"}, "alternative_step_candidate": {"internal_hypothesis_context", "internal_evidence_context", "internal_simulation"}, "unknown_origin_step_candidate": {"unknown_origin_candidate"}}
REL = {"causal_step_candidate", "temporal_step_candidate", "boundary_step_candidate", "predictive_step_candidate", "counterfactual_step_candidate", "alternative_step_candidate"}
PLAN_FALSE = {"external_inference_asserted", "inference_verified", "inference_fact_asserted", "inference_accepted", "inference_rejected", "conclusion_asserted", "truth_resolved", "winner_selected", "competition_resolved", "automatic_resolution_performed", "prediction_guaranteed", "future_outcome_asserted", "counterfactual_applied", "basis_verified", "basis_sufficient_asserted", "random_sampling_performed", "probability_draw_performed", "memory_write_performed", "relationship_update_performed", "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed", "world_state_mutation_performed", "transition_applied", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "schedule_created", "action_prevented", "action_permitted", "planning_execution_performed", "tool_execution_performed", "agp_bypass_allowed", "fallback_bypass_allowed"}


def item(kind="deductive_step_candidate", role="focal_conclusion_candidate", source="internal_hypothesis_context", iid="i1", situation="s1"):
    d = {"inference_item_id": iid, "situation_id": situation, "inference_kind": kind, "candidate_role": role, "derivation_source_class": source, "subject_ref_id": "민석", "conclusion_ref_id": "c1" + iid, "premise_ref_ids": ["p2", "p1"], "hypothesis_context_ref_ids": ["h2", "h1"], "evidence_ref_ids": ["e2", "e1"], "confidence_candidate": 1, "coherence_candidate": 0.5, "label": "후보"}
    if kind in REL:
        d["object_ref_id"] = "obj" + iid
    return d

def build(**kw):
    args = {"inference_type": "deductive_inference_candidate", "situation_id": "s1", "inference_items": [item()], "metadata": {}}
    args.update(kw)
    return build_virtual_world_situation_inference_context(**args)

def reason(ctx): return ctx["blocked_reasons"]


def test_summary_literal_contracts_independent_from_constants():
    s = build_virtual_world_situation_inference_context_schema_summary(); assert s == virtual_world_situation_inference_context_schema_summary()
    assert s["supported_inference_types"] == TYPES
    assert s["supported_inference_kinds"] == KINDS
    assert s["supported_candidate_roles"] == ROLES
    assert s["supported_derivation_source_classes"] == SOURCES
    assert s["supported_boundary_classes"] == BOUNDARIES
    assert s["supported_confidence_states"] == CONFIDENCE
    assert s["next_recommended_step"] == "read_only_virtual_world_situation_conclusion_candidate_schema"


def test_valid_payload_exact_keys_summaries_id_and_normalization():
    ctx = build(metadata={"z": [2, 1], "a": {"b": False}})
    assert validate_virtual_world_situation_inference_context(ctx)
    expected_keys = {"situation_inference_context_passed", "situation_inference_context_status", "inference_context_id", "canonical_id_algorithm", "inference_type", "inference_boundary_classification", "inference_confidence_state", "situation_id", "inference_items", "metadata", "origin_summary", "fact_status_summary", "inference_scope_summary", "inference_competition_summary", "inference_support_summary", "resolution_summary", "inference_flags", "boundary_flags", "inference_integrity_flags", "candidate_only_fields", "blocked_reasons", "warnings"} | {"external_inference_asserted", "inference_verified", "inference_fact_asserted", "inference_accepted", "inference_rejected", "conclusion_asserted", "truth_resolved", "winner_selected", "competition_resolved", "automatic_resolution_performed", "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted", "counterfactual_applied", "basis_verified", "basis_sufficient_asserted", "random_sampling_performed", "probability_draw_performed", "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "artifact_created_or_staged", "schedule_created", "action_prevented", "action_permitted", "agp_bypass_allowed", "fallback_bypass_allowed"} | {"inference_candidate_only", "situation_inference_only", "read_only", "situation_review_required", "snapshot_review_required", "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required", "inference_integrity_review_required", "basis_review_required", "competition_review_required", "resolution_review_required", "appraisal_required", "agp_input_required"}
    assert set(ctx) == expected_keys
    assert ctx["inference_items"][0]["premise_ref_ids"] == ["p1", "p2"]
    assert ctx["inference_support_summary"]["hypothesis_context_ref_ids"] == ["h1", "h2"]
    ctx2 = build(metadata={"a": {"b": False}, "z": [2, 1]}, inference_items=[dict(reversed(list(item().items())))])
    assert ctx["inference_context_id"] == ctx2["inference_context_id"]
    changed = build(inference_items=[{**item(), "conclusion_ref_id": "changed"}])
    assert changed["inference_context_id"] != ctx["inference_context_id"]


def test_all_enums_and_compatibility_matrices():
    for t, kinds in TYPE_KIND.items():
        k = sorted(kinds)[0]; role = sorted(KIND_ROLE[k])[0]; source = sorted(KIND_SOURCE[k])[0]
        items = [item(k, role, source)]
        if t == "competing_inference_candidate": items = [item("deductive_step_candidate", "focal_conclusion_candidate", "internal_hypothesis_context", "i1"), item("abductive_step_candidate", "alternative_conclusion_candidate", "internal_hypothesis_context", "i2")]
        if t == "mixed_unknown_inference_candidate": items = [item("deductive_step_candidate", "focal_conclusion_candidate", "internal_hypothesis_context", "i1"), item("unknown_origin_step_candidate", "neutral_unknown", "unknown_origin_candidate", "i2")]
        assert build(inference_type=t, inference_items=items)["situation_inference_context_passed"] is True
        if len(kinds) < len(KINDS):
            invalid_kind = next(x for x in KINDS if x not in kinds)
            bad_role = sorted(KIND_ROLE[invalid_kind])[0]; bad_source = sorted(KIND_SOURCE[invalid_kind])[0]
            assert reason(build(inference_type=t, inference_items=[item(invalid_kind, bad_role, bad_source)])) == ["incompatible_inference_type_kind"]
    for k in KINDS:
        for r in KIND_ROLE[k]:
            assert build(inference_type="mixed_unknown_inference_candidate", inference_items=[item(k, r, sorted(KIND_SOURCE[k])[0], "i1"), item("deductive_step_candidate", "focal_conclusion_candidate", "internal_hypothesis_context", "i2")])["blocked_reasons"] == []
        bad = next(r for r in ROLES if r not in KIND_ROLE[k])
        assert reason(build(inference_type="mixed_unknown_inference_candidate", inference_items=[item(k, bad, sorted(KIND_SOURCE[k])[0])])) == ["incompatible_inference_kind_candidate_role"]
        bads = next(s for s in SOURCES if s not in KIND_SOURCE[k])
        assert reason(build(inference_type="mixed_unknown_inference_candidate", inference_items=[item(k, sorted(KIND_ROLE[k])[0], bads)])) == ["incompatible_inference_kind_derivation_source"]


def test_reference_shape_duplicates_coherence_composition_boundary_confidence():
    assert reason(build(inference_items=[{k:v for k,v in item("causal_step_candidate", "focal_conclusion_candidate", "internal_causal_context").items() if k != "object_ref_id"}], inference_type="causal_inference_candidate")) == ["missing_object_ref_id_for_relational_inference"]
    assert reason(build(inference_items=[{**item(), "object_ref_id": "x"}])) == ["unexpected_object_ref_id_for_unary_inference"]
    assert reason(build(inference_items=[{**item(), "premise_ref_ids": ["p", "p"]}])) == ["duplicate_premise_ref_id"]
    assert reason(build(inference_items=[item(iid="same"), {**item(iid="same"), "conclusion_ref_id": "c2"}])) == ["duplicate_inference_item_id"]
    assert reason(build(inference_items=[item(iid="i1"), {**item(iid="i2"), "conclusion_ref_id": "c1i1"}])) == ["duplicate_semantic_inference_item"]
    assert reason(build(inference_items=[item(situation="other")])) == ["inference_item_situation_mismatch"]
    assert reason(build(inference_type="competing_inference_candidate")) == ["competing_inference_requires_multiple_items"]
    assert build(inference_type="competing_inference_candidate", inference_items=[item("deductive_step_candidate", "focal_conclusion_candidate", "internal_hypothesis_context", "i1"), item("abductive_step_candidate", "alternative_conclusion_candidate", "internal_hypothesis_context", "i2")])["inference_competition_summary"]["competition_present"] is True
    assert build(inference_type="mixed_unknown_inference_candidate", inference_items=[item("deductive_step_candidate", "focal_conclusion_candidate", "internal_hypothesis_context", "i1"), item("unknown_origin_step_candidate", "neutral_unknown", "unknown_origin_candidate", "i2")])["blocked_reasons"] == []
    for b in BOUNDARIES: assert build(metadata={"inference_boundary_classification": b})["inference_boundary_classification"] == b
    for bad in ["", None, False, 0, [], {}]: assert reason(build(metadata={"inference_boundary_classification": bad})) == ["malformed_inference_boundary_class"]
    assert reason(build(metadata={"inference_boundary_classification": "new"})) == ["unknown_inference_boundary_class"]
    for c in CONFIDENCE: assert build(metadata={"inference_confidence_state": c})["inference_confidence_state"] == c
    high = build(metadata={"inference_confidence_state": "inference_high_confidence_but_not_fact"}); assert high["inference_fact_asserted"] is False and high["resolution_summary"]["resolved"] is False
    comp = build(metadata={"inference_confidence_state": "inference_competing"}); assert comp["inference_competition_summary"]["competition_present"] is False
    assert reason(build(metadata={"inference_confidence_state": "x"})) == ["unknown_inference_confidence_state"]


def test_validation_precedence_and_invalid_ordering():
    base = item()
    sequence = [
        ({k:v for k,v in base.items() if k != "inference_item_id"}, "missing_or_empty_inference_item_id"),
        ({**base, "memory_write_requested": True}, "memory_write_requested"),
        ({**base, "inference_item_id": "dup"}, "duplicate_inference_item_id"),
        ({**base, "premise_ref_ids": []}, "missing_or_malformed_premise_ref_ids"),
        ({**base, "hypothesis_context_ref_ids": []}, "missing_or_malformed_hypothesis_context_ref_ids"),
        ({**base, "evidence_ref_ids": [""]}, "malformed_evidence_ref_ids"),
        (base, "duplicate_semantic_inference_item"),
        ({**base, "situation_id": "x"}, "inference_item_situation_mismatch"),
        ({**base, "object_ref_id": "x"}, "unexpected_object_ref_id_for_unary_inference"),
        ({**base, "candidate_role": "counter_conclusion_candidate"}, "incompatible_inference_kind_candidate_role"),
        ({**base, "derivation_source_class": "internal_dmn"}, "incompatible_inference_kind_derivation_source"),
    ]
    for bad, r in sequence:
        items = [bad, {**base, "inference_item_id": "dup" if r == "duplicate_inference_item_id" else "other"}]
        if r == "duplicate_semantic_inference_item": items = [base, {**base, "inference_item_id": "other"}]
        assert reason(build(inference_items=items)) == [r]
        assert build(inference_items=list(reversed(items))) == build(inference_items=items)


def test_strict_json_hostile_and_recursive_forbidden_matrix():
    class D(dict): pass
    class L(list): pass
    class BadStr:
        def __str__(self): raise RuntimeError("bad")
    hostile = [{1: "x"}, (1,), {1}, frozenset({1}), b"x", bytearray(b"x"), math.nan, math.inf, -math.inf, object(), D(), L()]
    circd = {}; circd["x"] = circd; circl = []; circl.append(circl); hostile += [circd, circl, BadStr()]
    for h in hostile:
        assert reason(build(metadata={"h": h})) == ["non_json_serializable_semantic_input"]
        assert validate_virtual_world_situation_inference_context(h) is False
        for fn in [build_inference_context_to_situation_plan, build_inference_context_to_snapshot_plan, build_inference_context_to_transition_preflight_plan, build_inference_context_to_memory_candidate_plan, build_inference_context_to_appraisal_plan, build_inference_context_to_agp_input_plan]:
            p = fn(h); assert p["ready"] is False and p["candidate_only"] is True and p["read_only"] is True and p["inference_candidate_only"] is True
    deep = v = {}
    for _ in range(105): v["x"] = {}; v = v["x"]
    assert reason(build(metadata=deep)) == ["non_json_serializable_semantic_input"]
    forbidden = ["external_inference_assertion_requested", "memory_write_requested", "schedule_requested", "cron_expression", "fallback_bypass_requested"]
    for f in forbidden:
        for loc in range(6):
            for val, exp in [(True, f), (False, None), (1, "malformed_forbidden_request_field"), ("yes", "malformed_forbidden_request_field")]:
                md = {}; it = item()
                if loc == 0: md[f] = val
                elif loc == 1: md["n"] = {f: val}
                elif loc == 2: md["l"] = [{f: val}]
                elif loc == 3: it[f] = val
                elif loc == 4: it["u"] = {f: val}
                else: it["ul"] = [{f: val}]
                ctx = build(metadata=md, inference_items=[it])
                if exp: assert reason(ctx) == [exp]
                else: assert ctx["blocked_reasons"] == []


def test_full_payload_tamper_matrix_and_plans():
    ctx = build(); assert validate_virtual_world_situation_inference_context(ctx)
    for key in list(ctx.keys()):
        bad = copy.deepcopy(ctx); bad[key] = (not bad[key]) if isinstance(bad[key], bool) else "tampered"
        assert validate_virtual_world_situation_inference_context(bad) is False
    for key in list(ctx.keys()):
        bad = copy.deepcopy(ctx); del bad[key]; assert validate_virtual_world_situation_inference_context(bad) is False
    bad = copy.deepcopy(ctx); bad["extra"] = False; assert validate_virtual_world_situation_inference_context(bad) is False
    paths = [("origin_summary", "external_origin_verified"), ("fact_status_summary", "candidate_only"), ("inference_scope_summary", "item_count"), ("inference_competition_summary", "winner_selected"), ("inference_support_summary", "premise_ref_ids"), ("resolution_summary", "resolved")]
    for a,b in paths:
        bad = copy.deepcopy(ctx); bad[a][b] = 1 if type(bad[a][b]) is bool else False; assert validate_virtual_world_situation_inference_context(bad) is False
    for fn in [build_inference_context_to_situation_plan, build_inference_context_to_snapshot_plan, build_inference_context_to_transition_preflight_plan, build_inference_context_to_memory_candidate_plan, build_inference_context_to_appraisal_plan, build_inference_context_to_agp_input_plan]:
        p = fn(ctx); assert p["ready"] is True and p["source_inference_context_id"] == ctx["inference_context_id"]
        q = fn({**ctx, "inference_context_id": "bad"}); assert q["ready"] is False and q["candidate_only"] is True and q["read_only"] is True and q["inference_candidate_only"] is True
        for f in PLAN_FALSE: assert q[f] is False
