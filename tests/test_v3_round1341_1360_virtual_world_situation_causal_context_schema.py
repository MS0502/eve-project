import copy
import json

import pytest

from adapters.virtual_world_situation_causal_context_schema import (
    IMMUTABLE_FALSE_FLAGS,
    IMMUTABLE_TRUE_FLAGS,
    SUPPORTED_CAUSAL_BOUNDARY_CLASSES,
    SUPPORTED_CAUSAL_CONFIDENCE_STATES,
    SUPPORTED_CAUSAL_TYPES,
    build_causal_context_to_agp_input_plan,
    build_causal_context_to_appraisal_plan,
    build_causal_context_to_memory_candidate_plan,
    build_causal_context_to_situation_plan,
    build_causal_context_to_snapshot_plan,
    build_causal_context_to_transition_preflight_plan,
    build_virtual_world_situation_causal_context,
    build_virtual_world_situation_causal_context_schema_summary,
    validate_virtual_world_situation_causal_context,
    virtual_world_situation_causal_context_schema_summary,
)

LINK = {"link_id": "l1", "source_situation_id": "s1", "target_situation_id": "s2", "link_kind": "cause_candidate"}
EXPECTED_TOP_LEVEL = {
    "situation_causal_context_passed", "situation_causal_context_status", "causal_context_id", "canonical_id_algorithm",
    "causal_type", "causal_boundary_classification", "causal_confidence_state", "subject_situation_id", "object_situation_id",
    "causal_links", "metadata", "origin_summary", "fact_status_summary", "causal_direction_summary", "causal_chain_summary",
    "counterfactual_summary", "uncertainty_flags", "boundary_flags", "causal_integrity_flags", "candidate_only_fields",
    "blocked_reasons", "warnings", *IMMUTABLE_FALSE_FLAGS, *IMMUTABLE_TRUE_FLAGS,
}


def build(**kwargs):
    data = {"causal_type": "direct_cause_candidate", "subject_situation_id": "s1", "object_situation_id": "s2", "causal_links": [LINK]}
    data.update(kwargs)
    return build_virtual_world_situation_causal_context(**data)


@pytest.mark.parametrize("causal_type", SUPPORTED_CAUSAL_TYPES)
def test_every_causal_type_builds_valid_candidate(causal_type):
    links = [{**LINK, "link_kind": "correlation_candidate"}] if causal_type == "correlation_only_candidate" else [LINK]
    if causal_type == "causal_chain_candidate":
        links = [{**LINK, "sequence_index": 0}, {"link_id": "l2", "source_situation_id": "s2", "target_situation_id": "s3", "link_kind": "consequence_candidate", "sequence_index": 1}]
    ctx = build(causal_type=causal_type, causal_links=links)
    assert ctx["situation_causal_context_passed"] is True
    assert ctx["situation_causal_context_status"] == "VALIDATED"
    assert ctx["blocked_reasons"] == []
    assert validate_virtual_world_situation_causal_context(ctx) is True
    assert ctx["origin_summary"] == {"origin_kind": "internal_virtual_causal_candidate", "external_origin_verified": False}
    assert ctx["fact_status_summary"]["candidate_only"] is True
    assert ctx["fact_status_summary"]["causal_fact_asserted"] is False
    assert ctx["counterfactual_summary"]["intervention_performed"] is False


@pytest.mark.parametrize("boundary", SUPPORTED_CAUSAL_BOUNDARY_CLASSES)
def test_all_boundary_classes(boundary):
    ctx = build(metadata={"causal_boundary_classification": boundary})
    assert validate_virtual_world_situation_causal_context(ctx) is True
    if boundary == "mixed_virtual_external_causal_boundary":
        assert ctx["blocked_reasons"] == []
        assert "mixed_external_causal_assertion_blocked" in ctx["boundary_flags"]


@pytest.mark.parametrize("confidence", SUPPORTED_CAUSAL_CONFIDENCE_STATES)
def test_all_confidence_states(confidence):
    ctx = build(metadata={"causal_confidence_state": confidence})
    assert validate_virtual_world_situation_causal_context(ctx) is True
    if confidence == "causal_high_confidence_but_not_fact":
        assert ctx["fact_status_summary"]["causal_fact_asserted"] is False
        assert "high_confidence_remains_non_factual" in ctx["warnings"]
    if confidence == "causal_conflict_detected":
        assert "causal_conflict_detected" in ctx["uncertainty_flags"]
        assert "conflicting_causal_candidates_require_review" in ctx["causal_integrity_flags"]
    if confidence == "causal_origin_unknown":
        assert "causal_origin_unknown" in ctx["uncertainty_flags"]


def test_directional_and_non_directional_behaviors():
    assert build(subject_situation_id="same", object_situation_id="same")["blocked_reasons"] == ["identical_directional_situation_ids"]
    unknown = build(causal_type="causal_direction_unknown_candidate")
    assert unknown["causal_direction_summary"]["direction_known"] is False
    corr = build(causal_type="correlation_only_candidate", causal_links=[{**LINK, "link_kind": "correlation_candidate"}])
    assert corr["causal_direction_summary"]["direction_known"] is False
    assert "correlation_is_not_causation" in corr["causal_integrity_flags"]
    cf = build(causal_type="counterfactual_cause_candidate")
    assert cf["intervention_performed"] is False and "counterfactual_no_intervention" in cf["boundary_flags"]
    sim = build(causal_type="simulation_cause_candidate")
    assert "simulation_only" in sim["boundary_flags"]
    for typ in ("symbolic_cause_candidate", "dmn_cause_candidate", "dream_cause_candidate"):
        assert "internal_symbolic_causality_only" in build(causal_type=typ)["boundary_flags"]


@pytest.mark.parametrize("kwargs, reason", [
    ({"causal_type": None}, "missing_causal_type"), ({"causal_type": "x"}, "unknown_causal_type"),
    ({"subject_situation_id": None}, "missing_or_malformed_subject_situation_id"), ({"subject_situation_id": []}, "missing_or_malformed_subject_situation_id"),
    ({"object_situation_id": None}, "missing_or_malformed_object_situation_id"),
    ({"causal_links": None}, "missing_causal_links"), ({"causal_links": "x"}, "causal_links_not_non_empty_list"), ({"causal_links": []}, "empty_causal_links"),
    ({"causal_links": ["x"]}, "malformed_causal_link"), ({"causal_links": [{"link_id": "", "source_situation_id": "s1", "target_situation_id": "s2", "link_kind": "cause_candidate"}]}, "missing_or_empty_link_id"),
    ({"causal_links": [LINK, {**LINK}]}, "duplicate_link_id"),
    ({"causal_links": [{**LINK, "source_situation_id": ""}]}, "malformed_link_situation_id"),
    ({"causal_links": [{**LINK, "target_situation_id": "s1"}]}, "identical_link_source_target"),
    ({"causal_links": [{**LINK, "link_kind": "bad"}]}, "unknown_link_kind"),
    ({"causal_links": [{**LINK, "sequence_index": True}]}, "malformed_sequence_index"),
    ({"causal_links": [{**LINK, "sequence_index": -1}]}, "negative_sequence_index"),
    ({"causal_links": [{**LINK, "weight_candidate": True}]}, "malformed_weight_candidate"),
    ({"causal_links": [{**LINK, "weight_candidate": float("inf")}]}, "non_json_serializable_semantic_input"),
    ({"causal_links": [{**LINK, "weight_candidate": 1.5}]}, "weight_candidate_out_of_range"),
    ({"causal_links": [LINK, {**LINK, "link_id": "l2"}]}, "duplicate_semantic_link"),
])
def test_input_failures(kwargs, reason):
    ctx = build(**kwargs)
    assert ctx["blocked_reasons"] == [reason]
    assert validate_virtual_world_situation_causal_context(ctx) is False
    json.dumps(ctx, allow_nan=False)


@pytest.mark.parametrize("links, reason", [
    ([LINK], "causal_chain_sequence_missing"),
    ([{**LINK, "sequence_index": 0}, {"link_id":"l2","source_situation_id":"s2","target_situation_id":"s3","link_kind":"cause_candidate","sequence_index":0}], "causal_chain_sequence_duplicated"),
    ([{**LINK, "sequence_index": 1}], "causal_chain_sequence_non_contiguous"),
])
def test_invalid_causal_chain_sequence(links, reason):
    assert build(causal_type="causal_chain_candidate", causal_links=links)["blocked_reasons"] == [reason]


def test_determinism_reordering_and_semantic_changes():
    links_a = [{**LINK, "link_id": "b"}, {"link_id": "a", "source_situation_id": "s3", "target_situation_id": "s4", "link_kind": "consequence_candidate"}]
    links_b = [links_a[1], links_a[0]]
    c1 = build(causal_links=links_a, metadata={"z": [2, 1], "a": {"민석": True}})
    c2 = build(causal_links=links_b, metadata={"a": {"민석": True}, "z": [2, 1]})
    assert c1["causal_context_id"] == c2["causal_context_id"]
    chain_a = build(causal_type="causal_chain_candidate", causal_links=[{**LINK, "sequence_index": 1, "link_id": "l2", "source_situation_id": "s2", "target_situation_id": "s3"}, {**LINK, "sequence_index": 0}])
    chain_b = build(causal_type="causal_chain_candidate", causal_links=[{**LINK, "sequence_index": 0}, {**LINK, "sequence_index": 1, "link_id": "l2", "source_situation_id": "s2", "target_situation_id": "s3"}])
    assert chain_a["causal_context_id"] == chain_b["causal_context_id"]
    assert c1["causal_context_id"] != build(causal_links=links_a, metadata={"z": [2, 1], "a": {"민석": False}})["causal_context_id"]
    tampered = copy.deepcopy(c1); tampered["causal_context_id"] += "x"
    assert validate_virtual_world_situation_causal_context(tampered) is False
    tampered = copy.deepcopy(c1); tampered.pop("canonical_id_algorithm")
    assert validate_virtual_world_situation_causal_context(tampered) is False


class BadStr:
    def __str__(self): raise RuntimeError("bad")
class BadRepr:
    def __repr__(self): raise RuntimeError("bad")


@pytest.mark.parametrize("value", [
    {1: "x"}, ("tuple",), {"set"}, frozenset(["x"]), b"x", bytearray(b"x"), float("nan"), float("inf"), -float("inf"), BadStr(), BadRepr(),
])
def test_strict_json_native_failures(value):
    ctx = build(metadata={"unsafe": value})
    assert ctx["blocked_reasons"] == ["non_json_serializable_semantic_input"]
    assert validate_virtual_world_situation_causal_context(ctx) is False
    json.dumps(ctx, allow_nan=False)


def test_circular_and_deep_recursion_fail_closed():
    d = {}; d["self"] = d
    l = []; l.append(l)
    for value in (d, l):
        ctx = build(metadata={"x": value})
        assert ctx["blocked_reasons"] == ["non_json_serializable_semantic_input"]
        json.dumps(ctx, allow_nan=False)
    deep = current = {}
    for _ in range(1100):
        current["x"] = {}; current = current["x"]
    assert build(metadata=deep)["blocked_reasons"] == ["non_json_serializable_semantic_input"]
    deep_link = copy.deepcopy(LINK); cur = deep_link
    for _ in range(1100): cur["x"] = {}; cur = cur["x"]
    assert build(causal_links=[deep_link])["blocked_reasons"] == ["non_json_serializable_semantic_input"]


@pytest.mark.parametrize("key, reason", [("causal_boundary_classification", "malformed_causal_boundary_class"), ("causal_confidence_state", "malformed_causal_confidence_state")])
@pytest.mark.parametrize("value", ["", None, False, 0, [], {}])
def test_boundary_confidence_malformed_presence(key, reason, value):
    assert build(metadata={key: value})["blocked_reasons"] == [reason]


@pytest.mark.parametrize("key, reason", [("causal_boundary_classification", "unknown_causal_boundary_class"), ("causal_confidence_state", "unknown_causal_confidence_state")])
@pytest.mark.parametrize("value", ["None", "False", "0", "[]", "{}"])
def test_boundary_confidence_unknown_strings(key, reason, value):
    assert build(metadata={key: value})["blocked_reasons"] == [reason]


@pytest.mark.parametrize("field", [
    "external_cause_assertion_requested", "external_consequence_assertion_requested", "causal_fact_assertion_requested",
    "causal_direction_verification_requested", "causal_chain_verification_requested", "prediction_guarantee_requested",
    "future_outcome_assertion_requested", "intervention_requested", "counterfactual_execution_requested",
    "event_completion_assertion_requested", "memory_write_requested", "memory_fact_promotion_requested",
    "relationship_update_requested", "self_model_update_requested", "affect_transition_requested", "hormone_transition_requested",
    "world_state_mutation_requested", "transition_application_requested", "planning_execution_requested", "tool_execution_requested",
    "runtime_mutation_requested", "persistence_write_requested", "vector_read_requested", "vector_load_requested",
    "model_load_requested", "device_activation_requested", "network_action_requested", "artifact_creation_requested",
    "agp_bypass_requested", "fallback_bypass_requested", "act_requested", "execute_requested", "apply_requested",
    "intervene_requested", "simulate_and_apply_requested", "commit_outcome_requested", "promote_to_fact_requested",
])
def test_recursive_forbidden_requests_and_aliases(field):
    assert build(metadata={field: True})["blocked_reasons"] == [field]
    assert build(metadata={"outer": [{"inner": {field: True}}]})["blocked_reasons"] == [field]
    link = {**LINK, "unknown": {field: True}}
    assert build(causal_links=[link])["blocked_reasons"] == [field]
    assert validate_virtual_world_situation_causal_context(build(metadata={field: False})) is True


@pytest.mark.parametrize("bad", [1, "yes"])
def test_forbidden_request_non_boolean_fails(bad):
    assert build(metadata={"memory_write_requested": bad})["blocked_reasons"] == ["malformed_forbidden_request_field"]


def test_literal_top_level_required_field_set_and_nested_safety_values():
    ctx = build()
    assert set(ctx) == EXPECTED_TOP_LEVEL
    assert ctx["fact_status_summary"] == {"candidate_only": True, "verified_external_cause": False, "verified_external_consequence": False, "causal_fact_asserted": False, "prediction_guaranteed": False}
    assert ctx["causal_direction_summary"]["external_direction_verified"] is False
    assert ctx["causal_chain_summary"]["external_chain_verified"] is False
    for flag in IMMUTABLE_FALSE_FLAGS: assert ctx[flag] is False
    for flag in IMMUTABLE_TRUE_FLAGS: assert ctx[flag] is True


@pytest.mark.parametrize("mutate", [
    lambda x: x.update({"situation_causal_context_passed": False}), lambda x: x.update({"situation_causal_context_status": "REJECTED"}),
    lambda x: x.update({"blocked_reasons": ["x"]}), lambda x: x.update({"causal_context_id": "x"}),
    lambda x: x.update({"canonical_id_algorithm": "x"}), lambda x: x.update({"causal_type": "candidate_consequence"}),
    lambda x: x.update({"causal_boundary_classification": "dream_virtual_causality"}), lambda x: x.update({"causal_confidence_state": "causal_low_confidence"}),
    lambda x: x.update({"subject_situation_id": "changed"}), lambda x: x.update({"object_situation_id": "changed"}),
    lambda x: x.update({"metadata": {"x": 1}}), lambda x: x.update({"causal_links": [{**LINK, "label": "x"}]}),
    lambda x: x["origin_summary"].update({"external_origin_verified": True}), lambda x: x["fact_status_summary"].update({"causal_fact_asserted": True}),
    lambda x: x["causal_direction_summary"].update({"external_direction_verified": True}), lambda x: x["causal_chain_summary"].update({"external_chain_verified": True}),
    lambda x: x["counterfactual_summary"].update({"intervention_performed": True}), lambda x: x.update({"uncertainty_flags": ["x"]}),
    lambda x: x.update({"boundary_flags": ["x"]}), lambda x: x.update({"causal_integrity_flags": ["x"]}),
    lambda x: x.update({"candidate_only_fields": ["x"]}), lambda x: x.update({"warnings": ["x"]}),
    lambda x: x.update({"memory_write_performed": True}), lambda x: x.update({"read_only": False}),
    lambda x: x.pop("metadata"), lambda x: x.update({"extra": "x"}), lambda x: x.update({"memory_write_requested": True}),
])
def test_full_integrity_tampering_fails(mutate):
    tampered = copy.deepcopy(build())
    mutate(tampered)
    assert validate_virtual_world_situation_causal_context(tampered) is False


@pytest.mark.parametrize("builder", [
    build_causal_context_to_situation_plan, build_causal_context_to_snapshot_plan,
    build_causal_context_to_transition_preflight_plan, build_causal_context_to_memory_candidate_plan,
    build_causal_context_to_appraisal_plan, build_causal_context_to_agp_input_plan,
])
def test_downstream_plans(builder):
    valid = build(); assert builder(valid)["ready"] is True
    invalids = [build(causal_type=None), "x", {"x": object()}, {**valid, "causal_context_id": "x"}, {**valid, "extra": "x"}]
    nested = copy.deepcopy(valid); nested["origin_summary"]["external_origin_verified"] = True; invalids.append(nested)
    imm = copy.deepcopy(valid); imm["read_only"] = False; invalids.append(imm)
    invalids.extend([build(metadata={"causal_boundary_classification": ""}), build(metadata={"causal_confidence_state": ""}), build(metadata={"memory_write_requested": True})])
    deep = {}; cur = deep
    for _ in range(1100): cur["x"] = {}; cur = cur["x"]
    invalids.append(build(metadata=deep))
    for source in invalids:
        plan = builder(source)
        assert plan["ready"] is False
        assert plan["candidate_only"] is True and plan["read_only"] is True and plan["causal_candidate_only"] is True
        for key in ["external_cause_asserted", "external_consequence_asserted", "causal_fact_asserted", "intervention_performed", "memory_write_performed", "relationship_update_performed", "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed", "world_state_mutation_performed", "transition_applied", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "planning_execution_performed", "tool_execution_performed", "agp_bypass_allowed", "fallback_bypass_allowed"]:
            assert plan[key] is False


def test_summary_alias():
    summary = build_virtual_world_situation_causal_context_schema_summary()
    assert summary == virtual_world_situation_causal_context_schema_summary()
    assert summary["recommended_next_implementation_step"] == "read_only_virtual_world_situation_uncertainty_context_schema"
