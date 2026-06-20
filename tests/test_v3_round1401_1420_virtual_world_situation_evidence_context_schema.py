import copy
import json
import math

import pytest

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

EVIDENCE_TYPES = {
    "supporting_evidence_candidate", "challenging_evidence_candidate", "conflicting_evidence_candidate",
    "neutral_evidence_candidate", "source_provenance_evidence_candidate", "temporal_evidence_candidate",
    "causal_evidence_candidate", "state_evidence_candidate", "identity_evidence_candidate",
    "boundary_evidence_candidate", "simulation_evidence_candidate", "symbolic_evidence_candidate",
    "dmn_evidence_candidate", "dream_evidence_candidate", "tool_state_evidence_candidate",
    "unknown_origin_evidence_candidate", "mixed_unknown_evidence_candidate",
}
ITEM_KINDS = {
    "direct_observation_candidate", "indirect_observation_candidate", "supporting_signal_candidate",
    "challenging_signal_candidate", "neutral_signal_candidate", "source_provenance_candidate",
    "temporal_trace_candidate", "causal_trace_candidate", "state_trace_candidate", "identity_trace_candidate",
    "boundary_trace_candidate", "simulation_trace_candidate", "symbolic_trace_candidate", "dmn_trace_candidate",
    "dream_trace_candidate", "tool_state_trace_candidate", "unknown_origin_trace_candidate",
}
STANCES = {"supports_candidate", "challenges_candidate", "neutral_unknown"}
SOURCES = {
    "internal_virtual_observation", "internal_memory_replay", "internal_imagination", "internal_simulation",
    "internal_symbolic", "internal_dmn", "internal_dream", "internal_tool_state",
    "external_unverified_candidate", "unknown_origin_candidate",
}
BOUNDARIES = {
    "internal_virtual_evidence", "memory_replay_evidence", "imagination_evidence", "simulation_evidence",
    "symbolic_evidence", "dmn_evidence", "dream_evidence", "tool_state_evidence",
    "external_unverified_evidence", "unknown_origin_evidence", "mixed_virtual_external_evidence_boundary",
}
CONFIDENCE_STATES = {
    "evidence_unverified", "evidence_low_confidence", "evidence_medium_confidence",
    "evidence_high_confidence_but_not_fact", "evidence_conflict_detected", "evidence_origin_unknown",
}
TYPE_ITEM = {
    "supporting_evidence_candidate": {"direct_observation_candidate", "indirect_observation_candidate", "supporting_signal_candidate"},
    "challenging_evidence_candidate": {"direct_observation_candidate", "indirect_observation_candidate", "challenging_signal_candidate"},
    "conflicting_evidence_candidate": {"direct_observation_candidate", "indirect_observation_candidate", "supporting_signal_candidate", "challenging_signal_candidate", "neutral_signal_candidate"},
    "neutral_evidence_candidate": {"direct_observation_candidate", "indirect_observation_candidate", "neutral_signal_candidate"},
    "source_provenance_evidence_candidate": {"source_provenance_candidate"},
    "temporal_evidence_candidate": {"temporal_trace_candidate"},
    "causal_evidence_candidate": {"causal_trace_candidate"},
    "state_evidence_candidate": {"state_trace_candidate"},
    "identity_evidence_candidate": {"identity_trace_candidate"},
    "boundary_evidence_candidate": {"boundary_trace_candidate"},
    "simulation_evidence_candidate": {"simulation_trace_candidate"},
    "symbolic_evidence_candidate": {"symbolic_trace_candidate"},
    "dmn_evidence_candidate": {"dmn_trace_candidate"},
    "dream_evidence_candidate": {"dream_trace_candidate"},
    "tool_state_evidence_candidate": {"tool_state_trace_candidate"},
    "unknown_origin_evidence_candidate": {"unknown_origin_trace_candidate"},
    "mixed_unknown_evidence_candidate": ITEM_KINDS,
}
KIND_STANCE = {
    "direct_observation_candidate": STANCES, "indirect_observation_candidate": STANCES,
    "supporting_signal_candidate": {"supports_candidate"}, "challenging_signal_candidate": {"challenges_candidate"},
    "neutral_signal_candidate": {"neutral_unknown"}, "source_provenance_candidate": {"neutral_unknown"},
    "temporal_trace_candidate": {"neutral_unknown"}, "causal_trace_candidate": {"neutral_unknown"},
    "state_trace_candidate": {"neutral_unknown"}, "identity_trace_candidate": {"neutral_unknown"},
    "boundary_trace_candidate": {"neutral_unknown"}, "simulation_trace_candidate": {"neutral_unknown"},
    "symbolic_trace_candidate": {"neutral_unknown"}, "dmn_trace_candidate": {"neutral_unknown"},
    "dream_trace_candidate": {"neutral_unknown"}, "tool_state_trace_candidate": {"neutral_unknown"},
    "unknown_origin_trace_candidate": {"neutral_unknown"},
}
ALL_SOURCES = SOURCES
VMSU = {"internal_virtual_observation", "internal_memory_replay", "internal_simulation", "external_unverified_candidate", "unknown_origin_candidate"}
KIND_SOURCE = {
    "direct_observation_candidate": {"internal_virtual_observation", "internal_tool_state", "external_unverified_candidate"},
    "indirect_observation_candidate": {"internal_memory_replay", "internal_imagination", "internal_simulation", "external_unverified_candidate", "unknown_origin_candidate"},
    "supporting_signal_candidate": ALL_SOURCES, "challenging_signal_candidate": ALL_SOURCES,
    "neutral_signal_candidate": ALL_SOURCES, "source_provenance_candidate": ALL_SOURCES,
    "temporal_trace_candidate": VMSU, "causal_trace_candidate": VMSU,
    "state_trace_candidate": {"internal_virtual_observation", "internal_memory_replay", "internal_simulation", "internal_tool_state", "external_unverified_candidate", "unknown_origin_candidate"},
    "identity_trace_candidate": VMSU, "boundary_trace_candidate": VMSU,
    "simulation_trace_candidate": {"internal_simulation"}, "symbolic_trace_candidate": {"internal_symbolic"},
    "dmn_trace_candidate": {"internal_dmn"}, "dream_trace_candidate": {"internal_dream"},
    "tool_state_trace_candidate": {"internal_tool_state"}, "unknown_origin_trace_candidate": {"unknown_origin_candidate"},
}
FORBIDDEN_FIELDS = {
    "external_evidence_assertion_requested", "evidence_verification_requested", "source_verification_requested", "source_trust_promotion_requested", "evidence_fact_assertion_requested", "evidence_fact_promotion_requested", "conclusion_assertion_requested", "conclusion_selection_requested", "winner_selection_requested", "truth_resolution_requested", "evidence_conflict_resolution_requested", "automatic_resolution_requested", "certainty_promotion_requested", "fact_promotion_requested", "prediction_guarantee_requested", "future_outcome_assertion_requested", "random_sampling_requested", "probability_draw_requested", "memory_write_requested", "memory_fact_promotion_requested", "relationship_update_requested", "self_model_update_requested", "identity_update_requested", "affect_transition_requested", "hormone_transition_requested", "world_state_mutation_requested", "transition_application_requested", "planning_execution_requested", "tool_execution_requested", "runtime_mutation_requested", "persistence_write_requested", "vector_read_requested", "vector_load_requested", "model_load_requested", "device_activation_requested", "network_action_requested", "artifact_creation_requested", "intervention_requested", "event_completion_assertion_requested", "action_prevention_requested", "action_permission_requested", "agp_bypass_requested", "fallback_bypass_requested", "schedule_requested", "scheduler_requested", "timer_requested", "alarm_requested", "calendar_requested", "reminder_requested", "deadline_requested", "cron_requested", "cron_expression", "scheduled_execution_requested", "delayed_execution_requested", "act_requested", "execute_requested", "apply_requested", "commit_outcome_requested", "promote_to_fact_requested",
}
IMMUTABLE_FALSE = {
    "external_evidence_asserted", "evidence_verified", "source_verified", "evidence_fact_asserted", "conclusion_asserted", "truth_resolved", "winner_selected", "evidence_conflict_resolved", "automatic_resolution_performed", "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted", "random_sampling_performed", "probability_draw_performed", "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "artifact_created_or_staged", "schedule_created", "action_prevented", "action_permitted", "agp_bypass_allowed", "fallback_bypass_allowed",
}
IMMUTABLE_TRUE = {
    "evidence_candidate_only", "situation_evidence_only", "read_only", "situation_review_required", "snapshot_review_required", "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required", "evidence_integrity_review_required", "provenance_review_required", "resolution_review_required", "appraisal_required", "agp_input_required",
}
TOP_KEYS = {
    "situation_evidence_context_passed", "situation_evidence_context_status", "evidence_context_id", "canonical_id_algorithm", "evidence_type", "evidence_boundary_classification", "evidence_confidence_state", "situation_id", "evidence_items", "metadata", "origin_summary", "fact_status_summary", "evidence_scope_summary", "evidence_stance_summary", "evidence_provenance_summary", "resolution_summary", "evidence_flags", "boundary_flags", "evidence_integrity_flags", "candidate_only_fields", "blocked_reasons", "warnings",
    *IMMUTABLE_FALSE, *IMMUTABLE_TRUE,
}
PLAN_FALSE = {
    "external_evidence_asserted", "evidence_verified", "source_verified", "evidence_fact_asserted", "conclusion_asserted", "truth_resolved", "winner_selected", "evidence_conflict_resolved", "automatic_resolution_performed", "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted", "random_sampling_performed", "probability_draw_performed", "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "schedule_created", "action_prevented", "action_permitted", "agp_bypass_allowed", "fallback_bypass_allowed", "relationship_update_performed", "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed",
}
PLAN_BUILDERS = [build_evidence_context_to_situation_plan, build_evidence_context_to_snapshot_plan, build_evidence_context_to_transition_preflight_plan, build_evidence_context_to_memory_candidate_plan, build_evidence_context_to_appraisal_plan, build_evidence_context_to_agp_input_plan]


def complete_item(eid="e1", kind="supporting_signal_candidate", stance="supports_candidate", source="internal_virtual_observation", sid="s1", source_ref_id="src-민석", claim_ref_id="claim-1", **extra):
    data = {"evidence_id": eid, "situation_id": sid, "evidence_kind": kind, "stance": stance, "source_class": source, "source_ref_id": source_ref_id, "claim_ref_id": claim_ref_id}
    data.update(extra)
    return data


def build(etype="supporting_evidence_candidate", sid="s1", items=None, metadata=None):
    return build_virtual_world_situation_evidence_context(etype, sid, [complete_item()] if items is None else items, {} if metadata is None else metadata)


def reject_reason(payload):
    assert payload["situation_evidence_context_passed"] is False
    assert payload["situation_evidence_context_status"] == "REJECTED"
    assert len(payload["blocked_reasons"]) == 1
    json.dumps(payload, allow_nan=False)
    return payload["blocked_reasons"][0]


def assert_reason(payload, reason):
    assert reject_reason(payload) == reason


def valid_payload():
    return build()


def valid_item_for_kind(kind, eid="e1"):
    stance = sorted(KIND_STANCE[kind])[0]
    source = sorted(KIND_SOURCE[kind])[0]
    return complete_item(eid=eid, kind=kind, stance=stance, source=source)


def valid_items_for_type(etype):
    if etype == "conflicting_evidence_candidate":
        return [complete_item("e1", "supporting_signal_candidate", "supports_candidate"), complete_item("e2", "challenging_signal_candidate", "challenges_candidate", claim_ref_id="claim-2")]
    if etype == "mixed_unknown_evidence_candidate":
        return [complete_item("e1", "supporting_signal_candidate", "supports_candidate"), complete_item("e2", "unknown_origin_trace_candidate", "neutral_unknown", "unknown_origin_candidate", claim_ref_id="claim-2")]
    kind = sorted(TYPE_ITEM[etype])[0]
    if etype == "supporting_evidence_candidate":
        return [complete_item("e1", kind, "supports_candidate", sorted(KIND_SOURCE[kind])[0])]
    if etype == "challenging_evidence_candidate":
        return [complete_item("e1", kind, "challenges_candidate", sorted(KIND_SOURCE[kind])[0])]
    if etype == "neutral_evidence_candidate":
        return [complete_item("e1", kind, "neutral_unknown", sorted(KIND_SOURCE[kind])[0])]
    return [valid_item_for_kind(kind)]


def test_literal_top_level_contract_and_nested_summaries():
    payload = valid_payload()
    assert set(payload.keys()) == TOP_KEYS
    assert payload["origin_summary"] == {"origin_kind": "internal_virtual_evidence_candidate", "external_origin_verified": False}
    assert payload["fact_status_summary"] == {"candidate_only": True, "external_evidence_fact_verified": False, "evidence_fact_asserted": False, "conclusion_asserted": False, "truth_resolved": False}
    assert payload["evidence_scope_summary"] == {"scope_candidate_only": True, "external_scope_verified": False, "evidence_kinds": ["supporting_signal_candidate"], "source_classes": ["internal_virtual_observation"], "claim_ref_ids": ["claim-1"], "item_count": 1}
    assert payload["evidence_stance_summary"] == {"supporting_item_count": 1, "challenging_item_count": 0, "neutral_item_count": 0, "conflict_present": False, "winner_selected": False}
    assert payload["evidence_provenance_summary"] == {"source_ref_ids": ["src-민석"], "source_count": 1, "external_source_verified": False, "unknown_origin_present": False}
    assert payload["resolution_summary"] == {"resolution_candidate_only": True, "resolved": False, "conclusion_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False}
    assert all(payload[k] is False for k in IMMUTABLE_FALSE)
    assert all(payload[k] is True for k in IMMUTABLE_TRUE)


def test_literal_summary_enums_and_every_item_kind_valid_combination():
    summary = virtual_world_situation_evidence_context_schema_summary()
    assert set(summary["supported_evidence_types"]) == EVIDENCE_TYPES
    assert set(summary["supported_evidence_item_kinds"]) == ITEM_KINDS
    assert set(summary["supported_evidence_stances"]) == STANCES
    assert set(summary["supported_source_classes"]) == SOURCES
    assert set(summary["supported_boundary_classes"]) == BOUNDARIES
    assert set(summary["supported_confidence_states"]) == CONFIDENCE_STATES
    assert summary["next_recommended_step"] == "read_only_virtual_world_situation_hypothesis_context_schema"
    for kind in ITEM_KINDS:
        payload = build("mixed_unknown_evidence_candidate", items=[valid_item_for_kind(kind), complete_item("mix", "supporting_signal_candidate", "supports_candidate", claim_ref_id="mix")])
        assert payload["blocked_reasons"] == []
    assert build("mixed_unknown_evidence_candidate", items=[complete_item("a", "direct_observation_candidate", "neutral_unknown", "external_unverified_candidate"), complete_item("b", "indirect_observation_candidate", "neutral_unknown", "internal_imagination", claim_ref_id="b")])["blocked_reasons"] == []
    assert build("mixed_unknown_evidence_candidate", items=[complete_item("a", "indirect_observation_candidate", "neutral_unknown", "internal_memory_replay"), complete_item("b", "indirect_observation_candidate", "neutral_unknown", "internal_imagination", claim_ref_id="b")])["blocked_reasons"] == []


def test_every_evidence_type_valid_and_all_boundary_confidence_values():
    for etype in EVIDENCE_TYPES:
        assert build(etype, items=valid_items_for_type(etype))["blocked_reasons"] == []
    for boundary in BOUNDARIES:
        payload = build(metadata={"evidence_boundary_classification": boundary})
        assert payload["blocked_reasons"] == [] and payload["evidence_boundary_classification"] == boundary
    for confidence in CONFIDENCE_STATES:
        payload = build(metadata={"evidence_confidence_state": confidence})
        assert payload["blocked_reasons"] == [] and payload["evidence_confidence_state"] == confidence
        assert payload["fact_status_summary"]["evidence_fact_asserted"] is False


def test_type_item_matrix_every_valid_and_representative_invalid_no_rewrite():
    for etype, kinds in TYPE_ITEM.items():
        for kind in kinds:
            if etype in {"conflicting_evidence_candidate", "mixed_unknown_evidence_candidate"}:
                continue
            payload = build(etype, items=[valid_item_for_kind(kind)])
            if etype == "supporting_evidence_candidate" and payload["blocked_reasons"]:
                continue
            if etype == "challenging_evidence_candidate" and payload["blocked_reasons"]:
                continue
            if etype == "neutral_evidence_candidate" and payload["blocked_reasons"]:
                continue
            assert payload["blocked_reasons"] == []
            assert payload["evidence_items"][0]["evidence_kind"] == kind
        if ITEM_KINDS - kinds:
            invalid_kind = sorted(ITEM_KINDS - kinds)[0]
            invalid = build(etype, items=[valid_item_for_kind(invalid_kind)])
            if reject_reason(invalid) not in {"supporting_evidence_requires_support", "challenging_evidence_requires_challenge", "neutral_evidence_requires_neutral_items"}:
                assert_reason(invalid, "incompatible_evidence_type_item_kind")


def test_kind_stance_and_kind_source_matrices_every_valid_and_invalid():
    for kind, stances in KIND_STANCE.items():
        for stance in stances:
            payload = build("mixed_unknown_evidence_candidate", items=[complete_item("a", kind, stance, sorted(KIND_SOURCE[kind])[0]), complete_item("b", "supporting_signal_candidate", "supports_candidate", claim_ref_id="b")])
            assert payload["blocked_reasons"] == []
        invalid_stance = sorted(STANCES - stances)[0] if STANCES - stances else None
        if invalid_stance:
            assert_reason(build("mixed_unknown_evidence_candidate", items=[complete_item("a", kind, invalid_stance, sorted(KIND_SOURCE[kind])[0])]), "incompatible_evidence_item_kind_stance")
    for kind, sources in KIND_SOURCE.items():
        for source in sources:
            companion = complete_item("b", "unknown_origin_trace_candidate", "neutral_unknown", "unknown_origin_candidate", claim_ref_id="b")
            if kind == "unknown_origin_trace_candidate":
                companion = complete_item("b", "supporting_signal_candidate", "supports_candidate", "internal_virtual_observation", claim_ref_id="b")
            payload = build("mixed_unknown_evidence_candidate", items=[complete_item("a", kind, sorted(KIND_STANCE[kind])[0], source), companion])
            assert payload["blocked_reasons"] == []
        invalid_source = sorted(SOURCES - sources)[0] if SOURCES - sources else None
        if invalid_source:
            assert_reason(build("mixed_unknown_evidence_candidate", items=[complete_item("a", kind, sorted(KIND_STANCE[kind])[0], invalid_source)]), "incompatible_evidence_item_kind_source_class")


@pytest.mark.parametrize("field,reason", [
    ("evidence_id", "missing_or_empty_evidence_id"), ("situation_id", "malformed_evidence_item_situation_id"),
    ("evidence_kind", "unknown_evidence_kind"), ("stance", "unknown_evidence_stance"),
    ("source_class", "unknown_source_class"), ("source_ref_id", "missing_or_malformed_source_ref_id"),
    ("claim_ref_id", "missing_or_malformed_claim_ref_id"),
])
def test_required_item_fields_missing_and_malformed(field, reason):
    malformed_values = ["", "   ", 1.5, [], {}, True, 1]
    item = complete_item(); item.pop(field)
    assert_reason(build(items=[item]), reason)
    for value in malformed_values:
        item = complete_item(); item[field] = value
        payload = build(items=[item])
        assert_reason(payload, reason)


def test_optional_fields_valid_boundaries_and_malformed_numeric_reasons():
    for value in ["rel", " 민석 "]:
        assert build(items=[complete_item(related_context_id=value)])["blocked_reasons"] == []
    for value in ["", " ", [], {}, True, 1]:
        assert_reason(build(items=[complete_item(related_context_id=value)]), "malformed_related_context_id")
    for field, malformed, out_range in [("confidence_candidate", "malformed_confidence_candidate", "confidence_candidate_out_of_range"), ("weight_candidate", "malformed_weight_candidate", "weight_candidate_out_of_range")]:
        for value in [0, 0.0, 1, 1.0, 0.5]:
            assert build(items=[complete_item(**{field: value})])["blocked_reasons"] == []
        for value in [True, False, "0.5", [], {}]:
            assert_reason(build(items=[complete_item(**{field: value})]), malformed)
        for value in [10**100, -(10**100), 1.1, -0.1]:
            assert_reason(build(items=[complete_item(**{field: value})]), out_range)
        for value in [float("nan"), float("inf"), -float("inf")]:
            assert_reason(build(items=[complete_item(**{field: value})]), "non_json_serializable_semantic_input")
    assert build(items=[complete_item(label="label")])["blocked_reasons"] == []
    for value in ["", " ", [], {}, True, 1]:
        assert_reason(build(items=[complete_item(label=value)]), "malformed_evidence_item_label")


def test_type_specific_composition_failures_and_valid_summaries():
    assert_reason(build("supporting_evidence_candidate", items=[complete_item("a", "direct_observation_candidate", "neutral_unknown")]), "supporting_evidence_requires_support")
    assert_reason(build("challenging_evidence_candidate", items=[complete_item("a", "direct_observation_candidate", "neutral_unknown")]), "challenging_evidence_requires_challenge")
    assert_reason(build("conflicting_evidence_candidate", items=[complete_item()]), "conflicting_evidence_requires_multiple_items")
    assert_reason(build("conflicting_evidence_candidate", items=[complete_item("a", "neutral_signal_candidate", "neutral_unknown"), complete_item("b", "challenging_signal_candidate", "challenges_candidate", claim_ref_id="b")]), "conflicting_evidence_missing_support")
    assert_reason(build("conflicting_evidence_candidate", items=[complete_item(), complete_item("b", "neutral_signal_candidate", "neutral_unknown", claim_ref_id="b")]), "conflicting_evidence_missing_challenge")
    assert_reason(build("neutral_evidence_candidate", items=[complete_item("a", "direct_observation_candidate", "supports_candidate")]), "neutral_evidence_requires_neutral_items")
    assert_reason(build("mixed_unknown_evidence_candidate", items=[complete_item()]), "mixed_unknown_evidence_requires_multiple_items")
    assert_reason(build("mixed_unknown_evidence_candidate", items=[complete_item(), complete_item("b", claim_ref_id="b")]), "mixed_unknown_evidence_requires_distinct_evidence_dimensions")
    conflict = build("conflicting_evidence_candidate", items=[complete_item(), complete_item("b", "challenging_signal_candidate", "challenges_candidate", claim_ref_id="b")])
    assert conflict["blocked_reasons"] == [] and conflict["evidence_stance_summary"]["conflict_present"] is True
    assert build("mixed_unknown_evidence_candidate", items=[complete_item(), complete_item("b", "neutral_signal_candidate", "neutral_unknown", claim_ref_id="b")])["blocked_reasons"] == []
    assert build("mixed_unknown_evidence_candidate", items=[complete_item(), complete_item("b", source="internal_dmn", claim_ref_id="b")])["blocked_reasons"] == []
    high = build(metadata={"evidence_confidence_state": "evidence_high_confidence_but_not_fact"})
    assert high["fact_status_summary"]["evidence_fact_asserted"] is False
    conf_state = build("neutral_evidence_candidate", items=[complete_item("a", "neutral_signal_candidate", "neutral_unknown")], metadata={"evidence_confidence_state": "evidence_conflict_detected"})
    assert conf_state["evidence_stance_summary"]["conflict_present"] is False
    unknown = build(metadata={"evidence_confidence_state": "evidence_origin_unknown"})
    assert "origin_unknown" in unknown["evidence_flags"]
    origin = build("unknown_origin_evidence_candidate", items=[complete_item("a", "unknown_origin_trace_candidate", "neutral_unknown", "unknown_origin_candidate")])
    assert origin["evidence_provenance_summary"]["unknown_origin_present"] is True
    assert all(origin["resolution_summary"][k] is False for k in ["resolved", "conclusion_selected", "external_resolution_verified", "automatic_resolution_allowed"])


def test_validation_precedence_multiple_errors_single_reason():
    assert_reason(build(etype=None, sid="", items=[complete_item(extra=float("nan"))], metadata={"memory_write_requested": True}), "non_json_serializable_semantic_input")
    assert_reason(build(etype="bad", sid="", metadata=[]), "unknown_evidence_type")
    assert_reason(build(sid="", metadata=[]), "missing_or_malformed_situation_id")
    assert_reason(build(metadata={"memory_write_requested": True}, items=None), "memory_write_requested")
    assert_reason(build(items=[{"memory_write_requested": True}]), "missing_or_empty_evidence_id")
    dup = [complete_item("same"), complete_item("same")]
    assert_reason(build("mixed_unknown_evidence_candidate", items=dup), "duplicate_evidence_id")
    sem = [complete_item("a"), complete_item("b"), complete_item("c", sid="s2", claim_ref_id="c")]
    assert_reason(build("mixed_unknown_evidence_candidate", items=sem), "duplicate_semantic_evidence_item")
    assert_reason(build(items=[complete_item(sid="s2", stance="neutral_unknown")]), "evidence_item_situation_mismatch")
    assert_reason(build(items=[complete_item(stance="neutral_unknown", source="bad")]), "unknown_source_class")
    assert_reason(build("mixed_unknown_evidence_candidate", items=[complete_item("a", "supporting_signal_candidate", "neutral_unknown", "bad")]), "unknown_source_class")
    assert_reason(build("temporal_evidence_candidate", items=[complete_item("a", "supporting_signal_candidate", "supports_candidate")], metadata={"evidence_boundary_classification": []}), "incompatible_evidence_type_item_kind")
    assert_reason(build("conflicting_evidence_candidate", items=[complete_item()], metadata={"evidence_boundary_classification": []}), "conflicting_evidence_requires_multiple_items")
    assert_reason(build(metadata={"evidence_boundary_classification": [], "evidence_confidence_state": []}), "malformed_evidence_boundary_class")


def test_deterministic_invalid_item_ordering_required_scenarios():
    scenarios = [
        ([complete_item("a", memory_write_requested=True), complete_item("b", tool_execution_requested=True, claim_ref_id="b")], "memory_write_requested"),
        ([complete_item("a", cron_expression="yes"), complete_item("b", memory_write_requested=True, claim_ref_id="b")], "malformed_forbidden_request_field"),
        ([{k: v for k, v in complete_item("a").items() if k != "evidence_id"}, complete_item("b", memory_write_requested=True, claim_ref_id="b")], "missing_or_empty_evidence_id"),
        ([[], {"evidence_id": "b"}], "malformed_evidence_item"),
        ([complete_item("dup"), complete_item("dup", stance="neutral_unknown", claim_ref_id="b")], "duplicate_evidence_id"),
        ([complete_item("a"), complete_item("b"), complete_item("c", "supporting_signal_candidate", "neutral_unknown", claim_ref_id="c")], "duplicate_semantic_evidence_item"),
    ]
    for items, reason in scenarios:
        a = build("mixed_unknown_evidence_candidate", items=items)
        b = build("mixed_unknown_evidence_candidate", items=list(reversed(items)))
        assert a == b
        assert a["situation_evidence_context_status"] == b["situation_evidence_context_status"] == "REJECTED"
        assert a["situation_evidence_context_passed"] is b["situation_evidence_context_passed"] is False
        assert a["blocked_reasons"] == b["blocked_reasons"] == [reason]


def test_strict_json_hostile_inputs_builder_validator_and_plans_never_raise():
    class Custom: pass
    class DictSub(dict): pass
    class ListSub(list): pass
    class BadKeys(dict):
        def keys(self): raise RuntimeError("keys")
    class BadIter(list):
        def __iter__(self): raise RuntimeError("iter")
    class BadStr:
        def __str__(self): raise RuntimeError("str")
    class BadRepr:
        def __repr__(self): raise RuntimeError("repr")
    cd = {}; cd["self"] = cd
    cl = []; cl.append(cl)
    values = [{1: "x"}, (1,), {1}, frozenset({1}), b"x", bytearray(b"x"), float("nan"), float("inf"), -float("inf"), Custom(), cd, cl, DictSub(), ListSub(), BadKeys(), BadIter(), BadStr(), BadRepr()]
    deep = x = {}
    for _ in range(105):
        y = {}; x["x"] = y; x = y
    values.append(deep)
    for value in values:
        for kwargs in [dict(metadata={"x": value}), dict(items=[complete_item(extra=value)])]:
            try:
                payload = build(**kwargs)
                assert_reason(payload, "non_json_serializable_semantic_input")
                json.dumps(payload, allow_nan=False)
                assert validate_virtual_world_situation_evidence_context(value) is False
                for fn in PLAN_BUILDERS:
                    plan = fn(value)
                    assert plan["ready"] is False
            except Exception as exc:  # pragma: no cover
                pytest.fail(f"public API raised for hostile input {type(value)}: {exc}")


@pytest.mark.parametrize("key,malformed_reason,unknown_reason", [("evidence_boundary_classification", "malformed_evidence_boundary_class", "unknown_evidence_boundary_class"), ("evidence_confidence_state", "malformed_evidence_confidence_state", "unknown_evidence_confidence_state")])
def test_malformed_versus_unknown_enum_values(key, malformed_reason, unknown_reason):
    for value in ["", None, False, 0, [], {}]:
        assert_reason(build(metadata={key: value}), malformed_reason)
    for value in ["None", "False", "0", "[]", "{}", "unsupported"]:
        assert_reason(build(metadata={key: value}), unknown_reason)


def test_recursive_forbidden_field_matrix_all_fields_locations_and_values():
    locations = [
        lambda k, v: dict(metadata={k: v}),
        lambda k, v: dict(metadata={"nested": {k: v}}),
        lambda k, v: dict(metadata={"list": [{k: v}]}),
        lambda k, v: dict(items=[complete_item(**{k: v})]),
        lambda k, v: dict(items=[complete_item(nested={k: v})]),
        lambda k, v: dict(items=[complete_item(list_field=[{k: v}])]),
    ]
    for field in FORBIDDEN_FIELDS:
        for loc in locations:
            assert_reason(build(**loc(field, True)), field)
            assert build(**loc(field, False))["blocked_reasons"] == []
            assert_reason(build(**loc(field, 1)), "malformed_forbidden_request_field")
            assert_reason(build(**loc(field, "yes")), "malformed_forbidden_request_field")


def test_payload_integrity_mutation_matrix_every_protected_area():
    base = valid_payload()
    mutations = {
        "situation_evidence_context_passed": False, "situation_evidence_context_status": "REJECTED", "blocked_reasons": ["x"],
        "evidence_context_id": "x", "canonical_id_algorithm": "x", "evidence_type": "neutral_evidence_candidate",
        "evidence_boundary_classification": "unknown_origin_evidence", "evidence_confidence_state": "evidence_origin_unknown", "situation_id": "s2",
        "evidence_items": [], "metadata": {"x": 1}, "origin_summary": {}, "fact_status_summary": {}, "evidence_scope_summary": {},
        "evidence_stance_summary": {}, "evidence_provenance_summary": {}, "resolution_summary": {}, "evidence_flags": [], "boundary_flags": [],
        "evidence_integrity_flags": [], "candidate_only_fields": [], "warnings": ["x"],
    }
    for key, value in mutations.items():
        p = copy.deepcopy(base); p[key] = value; assert not validate_virtual_world_situation_evidence_context(p), key
    for key in IMMUTABLE_FALSE:
        p = copy.deepcopy(base); p[key] = True; assert not validate_virtual_world_situation_evidence_context(p), key
    for key in IMMUTABLE_TRUE:
        p = copy.deepcopy(base); p[key] = False; assert not validate_virtual_world_situation_evidence_context(p), key
    for key in TOP_KEYS:
        p = copy.deepcopy(base); p.pop(key); assert not validate_virtual_world_situation_evidence_context(p), key
    p = copy.deepcopy(base); p["extra"] = False; assert not validate_virtual_world_situation_evidence_context(p)
    p = copy.deepcopy(base); p["memory_write_requested"] = True; assert not validate_virtual_world_situation_evidence_context(p)
    p = copy.deepcopy(base); p["read_only"] = 1; assert not validate_virtual_world_situation_evidence_context(p)
    p = copy.deepcopy(base); p["evidence_scope_summary"]["item_count"] = True; assert not validate_virtual_world_situation_evidence_context(p)
    p = copy.deepcopy(base); p["evidence_items"][0]["confidence_candidate"] = 1.0; assert not validate_virtual_world_situation_evidence_context(p)
    p = copy.deepcopy(base); p.pop("canonical_id_algorithm"); assert not validate_virtual_world_situation_evidence_context(p)
    p = copy.deepcopy(base); p["canonical_id_algorithm"] = "altered"; assert not validate_virtual_world_situation_evidence_context(p)


def assert_invalid_plan(plan):
    assert plan["ready"] is False
    assert plan["candidate_only"] is True
    assert plan["read_only"] is True
    assert plan["evidence_candidate_only"] is True
    for key in PLAN_FALSE:
        assert plan[key] is False


def test_downstream_plan_adversarial_coverage_literal_false_fields():
    valid = valid_payload()
    rejected = build(items=[complete_item(stance="neutral_unknown")])
    hostile = {1: "bad"}
    deep = x = []
    for _ in range(105):
        y = []; x.append(y); x = y
    tampered_id = copy.deepcopy(valid); tampered_id["evidence_context_id"] = "x"
    tampered_alg = copy.deepcopy(valid); tampered_alg["canonical_id_algorithm"] = "x"
    tampered_summary = copy.deepcopy(valid); tampered_summary["resolution_summary"]["resolved"] = True
    extra = copy.deepcopy(valid); extra["extra"] = True
    false_true = copy.deepcopy(valid); false_true["evidence_verified"] = True
    true_false = copy.deepcopy(valid); true_false["read_only"] = False
    invalid_sources = [
        rejected, "x", {"bad": True}, hostile, deep, tampered_id, tampered_alg, tampered_summary, extra, false_true, true_false,
        build(metadata={"evidence_boundary_classification": []}), build(metadata={"evidence_confidence_state": []}),
        build(metadata={"memory_write_requested": True}), build(items=[complete_item(tool_execution_requested=True)]),
        build("mixed_unknown_evidence_candidate", items=[complete_item("dup"), complete_item("dup", claim_ref_id="b")]),
        build("mixed_unknown_evidence_candidate", items=[complete_item("a"), complete_item("b")]),
        build(items=[complete_item(sid="s2")]), build(items=[complete_item(stance="neutral_unknown")]),
        build("mixed_unknown_evidence_candidate", items=[complete_item("a", "simulation_trace_candidate", "neutral_unknown", "internal_virtual_observation")]),
        build("temporal_evidence_candidate", items=[complete_item()]), build("conflicting_evidence_candidate", items=[complete_item()]),
        build("mixed_unknown_evidence_candidate", items=[complete_item(memory_write_requested=True), complete_item("b", tool_execution_requested=True, claim_ref_id="b")]),
        build(items=[complete_item(confidence_candidate=10**100)]),
    ]
    for fn in PLAN_BUILDERS:
        plan = fn(valid)
        assert plan["ready"] is True and plan["candidate_only"] is True and plan["read_only"] is True and plan["evidence_candidate_only"] is True
        for key in PLAN_FALSE:
            assert plan[key] is False
        for source in invalid_sources:
            assert_invalid_plan(fn(source))
