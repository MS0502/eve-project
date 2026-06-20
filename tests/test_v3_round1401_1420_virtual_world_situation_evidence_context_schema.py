import json
import math

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


def item(eid="e1", kind="supporting_signal_candidate", stance="supports_candidate", source="internal_virtual_observation", sid="s1", **extra):
    data = {"evidence_id": eid, "situation_id": sid, "evidence_kind": kind, "stance": stance, "source_class": source, "source_ref_id": "src-민석", "claim_ref_id": "claim-1"}
    data.update(extra)
    return data


def build(**kw):
    return build_virtual_world_situation_evidence_context(**kw)


def valid_support():
    return build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item()])


def assert_reject(payload, reason):
    assert payload["situation_evidence_context_passed"] is False
    assert payload["situation_evidence_context_status"] == "REJECTED"
    assert payload["blocked_reasons"] == [reason]
    json.dumps(payload, allow_nan=False)


def test_valid_literal_contract_and_summary():
    payload = valid_support()
    assert payload["situation_evidence_context_passed"] is True
    assert payload["situation_evidence_context_status"] == "VALIDATED"
    assert validate_virtual_world_situation_evidence_context(payload)
    assert set(["situation_evidence_context_passed", "situation_evidence_context_status", "evidence_context_id", "canonical_id_algorithm", "evidence_type", "evidence_boundary_classification", "evidence_confidence_state", "situation_id", "evidence_items", "metadata", "origin_summary", "fact_status_summary", "evidence_scope_summary", "evidence_stance_summary", "evidence_provenance_summary", "resolution_summary", "evidence_flags", "boundary_flags", "evidence_integrity_flags", "candidate_only_fields", "blocked_reasons", "warnings"]).issubset(payload)
    assert payload["origin_summary"] == {"origin_kind": "internal_virtual_evidence_candidate", "external_origin_verified": False}
    assert payload["fact_status_summary"] == {"candidate_only": True, "external_evidence_fact_verified": False, "evidence_fact_asserted": False, "conclusion_asserted": False, "truth_resolved": False}
    assert payload["resolution_summary"] == {"resolution_candidate_only": True, "resolved": False, "conclusion_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False}
    summary = virtual_world_situation_evidence_context_schema_summary()
    assert summary["next_recommended_step"] == "read_only_virtual_world_situation_hypothesis_context_schema"


def test_all_evidence_types_boundary_and_confidence_states():
    cases = {
        "supporting_evidence_candidate": [item()],
        "challenging_evidence_candidate": [item(kind="challenging_signal_candidate", stance="challenges_candidate")],
        "conflicting_evidence_candidate": [item(), item("e2", "challenging_signal_candidate", "challenges_candidate", claim_ref_id="claim-2")],
        "neutral_evidence_candidate": [item(kind="neutral_signal_candidate", stance="neutral_unknown")],
        "source_provenance_evidence_candidate": [item(kind="source_provenance_candidate", stance="neutral_unknown", source="internal_symbolic")],
        "temporal_evidence_candidate": [item(kind="temporal_trace_candidate", stance="neutral_unknown", source="internal_simulation")],
        "causal_evidence_candidate": [item(kind="causal_trace_candidate", stance="neutral_unknown", source="internal_simulation")],
        "state_evidence_candidate": [item(kind="state_trace_candidate", stance="neutral_unknown", source="internal_tool_state")],
        "identity_evidence_candidate": [item(kind="identity_trace_candidate", stance="neutral_unknown", source="internal_simulation")],
        "boundary_evidence_candidate": [item(kind="boundary_trace_candidate", stance="neutral_unknown", source="internal_simulation")],
        "simulation_evidence_candidate": [item(kind="simulation_trace_candidate", stance="neutral_unknown", source="internal_simulation")],
        "symbolic_evidence_candidate": [item(kind="symbolic_trace_candidate", stance="neutral_unknown", source="internal_symbolic")],
        "dmn_evidence_candidate": [item(kind="dmn_trace_candidate", stance="neutral_unknown", source="internal_dmn")],
        "dream_evidence_candidate": [item(kind="dream_trace_candidate", stance="neutral_unknown", source="internal_dream")],
        "tool_state_evidence_candidate": [item(kind="tool_state_trace_candidate", stance="neutral_unknown", source="internal_tool_state")],
        "unknown_origin_evidence_candidate": [item(kind="unknown_origin_trace_candidate", stance="neutral_unknown", source="unknown_origin_candidate")],
        "mixed_unknown_evidence_candidate": [item(), item("e2", "unknown_origin_trace_candidate", "neutral_unknown", "unknown_origin_candidate", claim_ref_id="claim-2")],
    }
    for etype, items in cases.items():
        assert build(evidence_type=etype, situation_id="s1", evidence_items=items)["blocked_reasons"] == []
    for boundary in virtual_world_situation_evidence_context_schema_summary()["supported_boundary_classes"]:
        assert build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item()], metadata={"evidence_boundary_classification": boundary})["evidence_boundary_classification"] == boundary
    for confidence in virtual_world_situation_evidence_context_schema_summary()["supported_confidence_states"]:
        p = build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item()], metadata={"evidence_confidence_state": confidence})
        assert p["evidence_confidence_state"] == confidence
        assert p["fact_status_summary"]["evidence_fact_asserted"] is False


def test_fail_reasons_and_precedence():
    assert_reject(build(situation_id="s1", evidence_items=[item()]), "missing_evidence_type")
    assert_reject(build(evidence_type="bad", situation_id="s1", evidence_items=[item()]), "unknown_evidence_type")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="", evidence_items=[item()]), "missing_or_malformed_situation_id")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", metadata=[]), "invalid_metadata")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", metadata={"memory_write_requested": True}, evidence_items=[{}]), "memory_write_requested")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1"), "missing_evidence_items")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items={}), "evidence_items_not_non_empty_list")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[]), "empty_evidence_items")
    bad = item(); bad.pop("evidence_id")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[bad]), "missing_or_empty_evidence_id")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(eid="")]), "missing_or_empty_evidence_id")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(kind="bad")]), "unknown_evidence_kind")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(stance="bad")]), "unknown_evidence_stance")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(source="bad")]), "unknown_source_class")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(confidence_candidate=True)]), "malformed_confidence_candidate")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(confidence_candidate=2)]), "confidence_candidate_out_of_range")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(weight_candidate="1")]), "malformed_weight_candidate")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(weight_candidate=-1)]), "weight_candidate_out_of_range")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(label="")]), "malformed_evidence_item_label")


def test_duplicates_coherence_compatibility_composition_order():
    dup_id = [item(), item(eid="e1", source_ref_id="src2", claim_ref_id="claim2")]
    assert_reject(build(evidence_type="mixed_unknown_evidence_candidate", situation_id="s1", evidence_items=dup_id), "duplicate_evidence_id")
    dup_sem = [item(), item(eid="e2")]
    assert_reject(build(evidence_type="mixed_unknown_evidence_candidate", situation_id="s1", evidence_items=dup_sem), "duplicate_semantic_evidence_item")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(sid="s2")]), "evidence_item_situation_mismatch")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(kind="supporting_signal_candidate", stance="neutral_unknown")]), "incompatible_evidence_item_kind_stance")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(kind="simulation_trace_candidate", stance="neutral_unknown", source="internal_virtual_observation")]), "incompatible_evidence_item_kind_source_class")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(kind="neutral_signal_candidate", stance="neutral_unknown")]), "incompatible_evidence_type_item_kind")
    assert_reject(build(evidence_type="conflicting_evidence_candidate", situation_id="s1", evidence_items=[item()]), "conflicting_evidence_requires_multiple_items")
    assert_reject(build(evidence_type="mixed_unknown_evidence_candidate", situation_id="s1", evidence_items=[item(), item("e2", claim_ref_id="claim-2")]), "mixed_unknown_evidence_requires_distinct_evidence_dimensions")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item()], metadata={"evidence_boundary_classification": [] , "evidence_confidence_state": []}), "malformed_evidence_boundary_class")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item()], metadata={"evidence_boundary_classification": "bogus", "evidence_confidence_state": []}), "unknown_evidence_boundary_class")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item()], metadata={"evidence_confidence_state": []}), "malformed_evidence_confidence_state")
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item()], metadata={"evidence_confidence_state": "bogus"}), "unknown_evidence_confidence_state")


def test_deterministic_id_reordering_and_mutation():
    a = item(unknown={"b": 2, "a": [1, False]})
    b = dict(reversed(list(a.items())))
    p1 = build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[a], metadata={"z": 1, "a": 2})
    p2 = build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[b], metadata={"a": 2, "z": 1})
    assert p1["evidence_context_id"] == p2["evidence_context_id"]
    changed = build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(label="x")], metadata={"a": 2, "z": 1})
    assert p1["evidence_context_id"] != changed["evidence_context_id"]


def test_strict_json_hostile_and_forbidden_aliases():
    class BadDict(dict): pass
    class BadStr:
        def __str__(self): raise RuntimeError("x")
        def __repr__(self): raise RuntimeError("x")
    circular = []; circular.append(circular)
    for value in [float("nan"), float("inf"), (1,), {1}, b"x", BadDict(), BadStr(), circular]:
        assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(extra=value)]), "non_json_serializable_semantic_input")
    deep = x = []
    for _ in range(105):
        y = []; x.append(y); x = y
    assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(extra=deep)]), "non_json_serializable_semantic_input")
    for key in ["schedule_requested", "cron_expression", "execute_requested", "agp_bypass_requested", "fallback_bypass_requested"]:
        assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item()], metadata={key: True}), key)
        assert build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item({} if False else "e1")], metadata={key: False})["blocked_reasons"] == []
        assert_reject(build(evidence_type="supporting_evidence_candidate", situation_id="s1", evidence_items=[item(nested={key: "yes"})]), "malformed_forbidden_request_field")


def test_payload_tampering_and_type_exactness_and_plans():
    p = valid_support()
    for k, v in [("evidence_context_id", "x"), ("read_only", 1), ("item_count", None)]:
        q = json.loads(json.dumps(p))
        if k == "item_count":
            q["evidence_scope_summary"]["item_count"] = 1.0
        else:
            q[k] = v
        assert not validate_virtual_world_situation_evidence_context(q)
    for extra in ["extra"]:
        q = json.loads(json.dumps(p)); q[extra] = False; assert not validate_virtual_world_situation_evidence_context(q)
    for remove in ["warnings", "resolution_summary"]:
        q = json.loads(json.dumps(p)); q.pop(remove); assert not validate_virtual_world_situation_evidence_context(q)
    builders = [build_evidence_context_to_situation_plan, build_evidence_context_to_snapshot_plan, build_evidence_context_to_transition_preflight_plan, build_evidence_context_to_memory_candidate_plan, build_evidence_context_to_appraisal_plan, build_evidence_context_to_agp_input_plan]
    for fn in builders:
        vp = fn(p); assert vp["ready"] is True and vp["read_only"] is True
        ip = fn({"bad": True}); assert ip["ready"] is False
        for key, val in ip.items():
            if key.endswith("performed") or key.endswith("allowed") or key in {"schedule_created", "action_prevented", "action_permitted", "model_loaded"}:
                assert val is False


def test_summaries_counts_unknown_origin_conflict_without_fabrication():
    p = build(evidence_type="conflicting_evidence_candidate", situation_id="s1", evidence_items=[item(), item("e2", "challenging_signal_candidate", "challenges_candidate", claim_ref_id="claim-2")])
    assert p["evidence_stance_summary"] == {"supporting_item_count": 1, "challenging_item_count": 1, "neutral_item_count": 0, "conflict_present": True, "winner_selected": False}
    assert p["blocked_reasons"] == []
    u = build(evidence_type="unknown_origin_evidence_candidate", situation_id="s1", evidence_items=[item(kind="unknown_origin_trace_candidate", stance="neutral_unknown", source="unknown_origin_candidate")])
    assert u["evidence_provenance_summary"]["unknown_origin_present"] is True
    c = build(evidence_type="neutral_evidence_candidate", situation_id="s1", evidence_items=[item(kind="neutral_signal_candidate", stance="neutral_unknown")], metadata={"evidence_confidence_state": "evidence_conflict_detected"})
    assert c["evidence_stance_summary"]["conflict_present"] is False
