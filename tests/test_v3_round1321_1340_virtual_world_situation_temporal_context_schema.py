import copy
import inspect

import pytest

import adapters.virtual_world_situation_temporal_context_schema as schema
from adapters.virtual_world_situation_temporal_context_schema import (
    IMMUTABLE_FALSE_FLAGS,
    IMMUTABLE_TRUE_FLAGS,
    SUPPORTED_TEMPORAL_TYPES,
    build_temporal_context_to_agp_input_plan,
    build_temporal_context_to_appraisal_plan,
    build_temporal_context_to_memory_candidate_plan,
    build_temporal_context_to_situation_plan,
    build_temporal_context_to_snapshot_plan,
    build_temporal_context_to_transition_preflight_plan,
    build_virtual_world_situation_temporal_context,
    build_virtual_world_situation_temporal_context_schema_summary,
    validate_virtual_world_situation_temporal_context,
    virtual_world_situation_temporal_context_schema_summary,
)

ANCHOR = {"anchor_id": "anchor_1", "anchor_kind": "logical_step", "label": "민석 logical step", "sequence_index": 0}
REFERENCE_TYPES = {"situation_before_candidate", "situation_after_candidate", "situation_simultaneous_candidate", "situation_sequence_candidate"}


def build(**kwargs):
    data = {"temporal_type": "situation_ongoing_candidate", "situation_id": "situation_a", "temporal_anchor": ANCHOR}
    data.update(kwargs)
    if data["temporal_type"] in REFERENCE_TYPES and "reference_situation_id" not in data:
        data["reference_situation_id"] = "situation_b"
    return build_virtual_world_situation_temporal_context(**data)


@pytest.mark.parametrize("temporal_type", SUPPORTED_TEMPORAL_TYPES)
def test_every_supported_temporal_type_builds_valid_read_only_candidate(temporal_type):
    context = build(temporal_type=temporal_type)
    assert context["situation_temporal_context_passed"] is True
    assert context["situation_temporal_context_status"] == "VALIDATED"
    assert context["blocked_reasons"] == []
    assert validate_virtual_world_situation_temporal_context(context) is True
    assert context["temporal_candidate_only"] is True
    assert context["logical_time_only"] is True


def test_missing_and_unknown_temporal_types_fail_closed():
    missing = build_virtual_world_situation_temporal_context(situation_id="s", temporal_anchor=ANCHOR)
    assert missing["blocked_reasons"] == ["missing_temporal_type"]
    assert validate_virtual_world_situation_temporal_context(missing) is False
    unknown = build(temporal_type="unknown")
    assert unknown["blocked_reasons"] == ["unknown_temporal_type"]
    assert validate_virtual_world_situation_temporal_context(unknown) is False


def test_missing_situation_id_and_anchor_fail_closed():
    assert build(situation_id=None)["blocked_reasons"] == ["missing_or_malformed_situation_id"]
    assert build(temporal_anchor=None)["blocked_reasons"] == ["missing_temporal_anchor"]


@pytest.mark.parametrize("anchor, reason", [
    ({"anchor_id": "", "anchor_kind": "logical_step", "label": "x"}, "malformed_temporal_anchor"),
    ({"anchor_id": "a", "anchor_kind": "bad", "label": "x"}, "malformed_temporal_anchor"),
    ({"anchor_id": "a", "anchor_kind": "logical_step", "label": "x", "logical_start": "0"}, "invalid_logical_start"),
])
def test_malformed_anchor_fails_closed(anchor, reason):
    context = build(temporal_anchor=anchor)
    assert context["blocked_reasons"] == [reason]
    assert validate_virtual_world_situation_temporal_context(context) is False


@pytest.mark.parametrize("metadata, reason", [
    ({"temporal_boundary_classification": "unknown_boundary"}, "unknown_temporal_boundary_class"),
    ({"temporal_confidence_state": "unknown_confidence"}, "unknown_temporal_confidence_state"),
])
def test_unknown_boundary_and_confidence_values_fail_closed(metadata, reason):
    context = build(metadata=metadata)
    assert context["blocked_reasons"] == [reason]
    assert validate_virtual_world_situation_temporal_context(context) is False


@pytest.mark.parametrize("temporal_type", sorted(REFERENCE_TYPES))
def test_required_reference_situation_is_enforced(temporal_type):
    context = build_virtual_world_situation_temporal_context(temporal_type=temporal_type, situation_id="situation_a", temporal_anchor=ANCHOR)
    assert context["blocked_reasons"] == ["missing_reference_situation_id"]
    assert validate_virtual_world_situation_temporal_context(context) is False


@pytest.mark.parametrize("temporal_type", ["situation_before_candidate", "situation_after_candidate"])
def test_same_situation_ids_fail_for_before_after(temporal_type):
    context = build(temporal_type=temporal_type, reference_situation_id="situation_a")
    assert context["blocked_reasons"] == ["same_situation_invalid_for_order_relation"]


@pytest.mark.parametrize("anchor, reason", [
    ({"anchor_id": "a", "anchor_kind": "bounded_logical_interval", "label": "x", "logical_start": -1}, "negative_logical_start"),
    ({"anchor_id": "a", "anchor_kind": "bounded_logical_interval", "label": "x", "logical_end": -1}, "negative_logical_end"),
    ({"anchor_id": "a", "anchor_kind": "bounded_logical_interval", "label": "x", "logical_start": 3, "logical_end": 2}, "logical_end_before_logical_start"),
    ({"anchor_id": "a", "anchor_kind": "logical_step", "label": "x", "sequence_index": -1}, "negative_sequence_index"),
])
def test_invalid_logical_intervals_and_negative_sequence_indexes_fail_closed(anchor, reason):
    context = build(temporal_anchor=anchor)
    assert context["blocked_reasons"] == [reason]
    assert validate_virtual_world_situation_temporal_context(context) is False


def test_deterministic_ids_stable_reordered_keys_and_semantic_changes():
    c1 = build(metadata={"b": {"z": 1, "a": 2}, "a": "x"})
    c2 = build(metadata={"a": "x", "b": {"a": 2, "z": 1}})
    assert c1["temporal_context_id"] == c2["temporal_context_id"]
    assert validate_virtual_world_situation_temporal_context(c1) is True
    changed = build(metadata={"a": "changed", "b": {"a": 2, "z": 1}})
    assert c1["temporal_context_id"] != changed["temporal_context_id"]


def test_malformed_optional_reference_situation_id_fails_for_all_temporal_types():
    non_reference = build(reference_situation_id=["not", "string"])
    assert non_reference["blocked_reasons"] == ["malformed_reference_situation_id"]
    assert validate_virtual_world_situation_temporal_context(non_reference) is False
    dict_reference = build(reference_situation_id={"situation_id": "situation_b"})
    assert dict_reference["blocked_reasons"] == ["malformed_reference_situation_id"]
    assert validate_virtual_world_situation_temporal_context(dict_reference) is False


def test_validator_rejects_tampered_fact_status_summary():
    context = build()
    tampered = copy.deepcopy(context)
    tampered["fact_status_summary"] = dict(tampered["fact_status_summary"])
    tampered["fact_status_summary"]["confirmed_external_time_fact"] = True
    assert validate_virtual_world_situation_temporal_context(tampered) is False
    tampered = copy.deepcopy(context)
    tampered["fact_status_summary"] = dict(tampered["fact_status_summary"])
    tampered["fact_status_summary"]["completed_external_event"] = True
    assert validate_virtual_world_situation_temporal_context(tampered) is False


def test_validator_rejects_tampered_duration_candidate_and_sequence_constraints():
    duration = build(temporal_type="waiting_duration_candidate")
    tampered_duration = copy.deepcopy(duration)
    tampered_duration["duration_candidate"]["external_duration_asserted"] = True
    assert validate_virtual_world_situation_temporal_context(tampered_duration) is False
    tampered_duration = copy.deepcopy(duration)
    tampered_duration["duration_candidate"]["duration_candidate_only"] = False
    assert validate_virtual_world_situation_temporal_context(tampered_duration) is False

    ordered = build(temporal_type="situation_before_candidate")
    tampered_sequence = copy.deepcopy(ordered)
    tampered_sequence["sequence_constraints"][0]["external_order_asserted"] = True
    assert validate_virtual_world_situation_temporal_context(tampered_sequence) is False
    tampered_sequence = copy.deepcopy(ordered)
    tampered_sequence["sequence_constraints"][0]["candidate_only"] = False
    assert validate_virtual_world_situation_temporal_context(tampered_sequence) is False


def test_validator_rejects_removed_or_altered_required_temporal_flags():
    before = build(temporal_type="situation_before_candidate")
    missing_boundary = copy.deepcopy(before)
    missing_boundary["boundary_flags"] = []
    assert validate_virtual_world_situation_temporal_context(missing_boundary) is False
    altered_boundary = copy.deepcopy(before)
    altered_boundary["boundary_flags"] = ["unexpected_boundary_flag"]
    assert validate_virtual_world_situation_temporal_context(altered_boundary) is False

    future = build(temporal_type="situation_future_window_candidate")
    missing_uncertainty = copy.deepcopy(future)
    missing_uncertainty["uncertainty_flags"] = []
    assert validate_virtual_world_situation_temporal_context(missing_uncertainty) is False
    altered_uncertainty = copy.deepcopy(future)
    altered_uncertainty["uncertainty_flags"] = ["future_outcome_asserted"]
    assert validate_virtual_world_situation_temporal_context(altered_uncertainty) is False

    high = build(metadata={"temporal_confidence_state": "temporal_high_confidence_but_not_fact"})
    removed_warning = copy.deepcopy(high)
    removed_warning["warnings"] = []
    assert validate_virtual_world_situation_temporal_context(removed_warning) is False

    conflict = build(metadata={"temporal_confidence_state": "temporal_conflict_detected"})
    removed_integrity = copy.deepcopy(conflict)
    removed_integrity["temporal_integrity_flags"] = []
    assert validate_virtual_world_situation_temporal_context(removed_integrity) is False


def test_validator_rejects_tampered_origin_summary_and_candidate_only_fields():
    context = build()
    tampered_origin = copy.deepcopy(context)
    tampered_origin["origin_summary"]["external_origin_verified"] = True
    assert validate_virtual_world_situation_temporal_context(tampered_origin) is False
    tampered_fields = copy.deepcopy(context)
    tampered_fields["candidate_only_fields"] = ["temporal_type"]
    assert validate_virtual_world_situation_temporal_context(tampered_fields) is False


def test_id_tampering_is_rejected():
    context = build()
    tampered = copy.deepcopy(context)
    tampered["temporal_context_id"] = tampered["temporal_context_id"] + "x"
    assert validate_virtual_world_situation_temporal_context(tampered) is False


def test_required_temporal_type_behaviors_do_not_assert_external_time_or_mutation():
    assert "current_external_activity_not_asserted" in build(temporal_type="situation_ongoing_candidate")["uncertainty_flags"]
    assert "external_start_time_not_verified" in build(temporal_type="situation_started_candidate")["uncertainty_flags"]
    assert "external_completion_not_asserted" in build(temporal_type="situation_ended_candidate")["uncertainty_flags"]
    assert "logical_order_candidate_only" in build(temporal_type="situation_before_candidate")["boundary_flags"]
    assert "logical_order_candidate_only" in build(temporal_type="situation_after_candidate")["boundary_flags"]
    assert "real_world_simultaneity_not_asserted" in build(temporal_type="situation_simultaneous_candidate")["boundary_flags"]
    assert "schedule_creation_blocked" in build(temporal_type="situation_recurring_pattern_candidate")["boundary_flags"]
    assert "future_outcome_not_asserted" in build(temporal_type="situation_future_window_candidate")["uncertainty_flags"]
    assert "duration_unverified_candidate" in build(temporal_type="waiting_duration_candidate")["uncertainty_flags"]
    assert "duration_unverified_candidate" in build(temporal_type="traveling_duration_candidate")["uncertainty_flags"]
    assert "memory_write_blocked" in build(temporal_type="learning_session_candidate")["boundary_flags"]


@pytest.mark.parametrize("temporal_type", ["symbolic_time_candidate", "dmn_time_candidate", "simulation_time_candidate", "dream_time_candidate"])
def test_symbolic_dmn_simulation_and_dream_remain_internal(temporal_type):
    context = build(temporal_type=temporal_type)
    assert "internal_symbolic_temporal_candidate" in context["boundary_flags"]
    assert context["external_time_asserted"] is False
    assert validate_virtual_world_situation_temporal_context(context) is True


def test_mixed_boundary_valid_high_confidence_non_factual_conflict_and_unknown_origin_flags():
    mixed = build(temporal_type="mixed_unknown_temporal_candidate")
    assert validate_virtual_world_situation_temporal_context(mixed) is True
    assert mixed["blocked_reasons"] == []
    assert "mixed_external_time_assertion_blocked" in mixed["boundary_flags"]
    high = build(metadata={"temporal_confidence_state": "temporal_high_confidence_but_not_fact"})
    assert "high_confidence_remains_non_factual" in high["warnings"]
    conflict = build(metadata={"temporal_confidence_state": "temporal_conflict_detected"})
    assert "temporal_conflict_detected" in conflict["uncertainty_flags"]
    assert "conflict_requires_review" in conflict["temporal_integrity_flags"]
    unknown = build(metadata={"temporal_confidence_state": "temporal_origin_unknown"})
    assert "temporal_origin_unknown" in unknown["uncertainty_flags"]


def test_immutable_false_flags_false_and_required_true_gates_true():
    context = build()
    for flag in IMMUTABLE_FALSE_FLAGS:
        assert context[flag] is False
    for flag in IMMUTABLE_TRUE_FLAGS:
        assert context[flag] is True


@pytest.mark.parametrize("field", sorted(schema.FORBIDDEN_REQUEST_FIELDS))
def test_forbidden_request_metadata_fails_closed(field):
    context = build(metadata={field: True})
    assert context["situation_temporal_context_passed"] is False
    assert context["blocked_reasons"] == [field]
    assert validate_virtual_world_situation_temporal_context(context) is False


def test_passed_rejected_status_and_blocked_reasons_coherent():
    valid = build()
    passed_with_block = copy.deepcopy(valid)
    passed_with_block["blocked_reasons"] = ["bad"]
    assert validate_virtual_world_situation_temporal_context(passed_with_block) is False
    rejected = build(temporal_type="unknown")
    rejected["blocked_reasons"] = []
    assert validate_virtual_world_situation_temporal_context(rejected) is False


def test_every_plan_builder_candidate_only_and_read_only():
    context = build()
    for builder in (
        build_temporal_context_to_situation_plan,
        build_temporal_context_to_snapshot_plan,
        build_temporal_context_to_transition_preflight_plan,
        build_temporal_context_to_memory_candidate_plan,
        build_temporal_context_to_appraisal_plan,
        build_temporal_context_to_agp_input_plan,
    ):
        plan = builder(context)
        assert plan["ready"] is True
        assert plan["candidate_only"] is True
        assert plan["read_only"] is True
        assert plan["current_time_read_performed"] is False
        assert plan["scheduler_action_performed"] is False
        assert plan["mutation_performed"] is False


def test_non_json_serializable_semantic_input_fails_closed_without_raise():
    cases = [
        {"metadata": {"bad_set": {"x"}}},
        {"metadata": {"bad_bytes": b"x"}},
        {"metadata": {1: "int-key", "mixed": "string-key"}},
        {"temporal_anchor": {**ANCHOR, "unknown_set": {"x"}}},
        {"temporal_anchor": {**ANCHOR, "unknown_bytes": b"x"}},
    ]
    for kwargs in cases:
        context = build(**kwargs)
        assert context["situation_temporal_context_passed"] is False
        assert context["blocked_reasons"] == ["non_json_serializable_semantic_input"]
        assert validate_virtual_world_situation_temporal_context(context) is False


def test_validator_returns_false_for_non_json_serializable_semantic_input_without_raise():
    valid = build()
    for field, value in (
        ("metadata", {"bad_set": {"x"}}),
        ("metadata", {"bad_bytes": b"x"}),
        ("metadata", {1: "int-key", "mixed": "string-key"}),
        ("temporal_anchor", {**ANCHOR, "unknown_set": {"x"}}),
        ("temporal_anchor", {**ANCHOR, "unknown_bytes": b"x"}),
    ):
        tampered = copy.deepcopy(valid)
        tampered[field] = value
        assert validate_virtual_world_situation_temporal_context(tampered) is False


REQUIRED_PUBLIC_OUTPUT_FIELDS = [
    "situation_temporal_context_passed",
    "situation_temporal_context_status",
    "temporal_context_id",
    "temporal_type",
    "temporal_boundary_classification",
    "temporal_confidence_state",
    "situation_id",
    "reference_situation_id",
    "temporal_anchor",
    "metadata",
    "origin_summary",
    "fact_status_summary",
    "sequence_constraints",
    "duration_candidate",
    "uncertainty_flags",
    "boundary_flags",
    "temporal_integrity_flags",
    "candidate_only_fields",
    "blocked_reasons",
    "warnings",
]


def test_all_required_public_output_fields_are_present_by_literal_name():
    context = build()
    for field in REQUIRED_PUBLIC_OUTPUT_FIELDS:
        assert field in context


def test_every_plan_builder_not_ready_for_adversarial_sources():
    valid = build()
    tampered_id = copy.deepcopy(valid)
    tampered_id["temporal_context_id"] += "tampered"
    tampered_fact = copy.deepcopy(valid)
    tampered_fact["fact_status_summary"]["confirmed_external_time_fact"] = True
    rejected = build(temporal_type="unknown")
    non_dict = ["not", "dict"]
    for source in (tampered_id, tampered_fact, rejected, non_dict):
        for builder in (
            build_temporal_context_to_situation_plan,
            build_temporal_context_to_snapshot_plan,
            build_temporal_context_to_transition_preflight_plan,
            build_temporal_context_to_memory_candidate_plan,
            build_temporal_context_to_appraisal_plan,
            build_temporal_context_to_agp_input_plan,
        ):
            plan = builder(source)
            assert plan["ready"] is False
            assert plan["candidate_only"] is True
            assert plan["read_only"] is True


def test_no_clock_time_api_is_imported_or_called():
    source = inspect.getsource(schema)
    forbidden = ["import time", "from time", "datetime", "time.time", "monotonic", "perf_counter", "now()"]
    for token in forbidden:
        assert token not in source


def test_schema_summary_alias():
    summary = build_virtual_world_situation_temporal_context_schema_summary()
    assert virtual_world_situation_temporal_context_schema_summary() == summary
    assert summary["recommended_next_implementation_step"] == "read_only_virtual_world_situation_causal_context_schema"
