from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

from core.m3_b_agp_bounded_expression_action_source_binding import (
    APPRAISAL_SCHEMA_VERSION,
    BINDING_SCHEMA_VERSION,
    EXPRESSION_ACTION_AXES,
    RAW_SCHEMA_VERSION,
    SOURCE_FAMILY,
    AGPBoundedExpressionActionRawRecord,
    AGPBoundedExpressionActionSourceBindingError,
    agp_bounded_expression_action_raw_observation_digest,
    agp_bounded_expression_action_source_bindings,
    derive_agp_bounded_expression_action_axis_evidence,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = min(0.08, ((tick - 1) // 2) * 0.02)
    if axis == "expression_pressure":
        return (
            ("agp_anchor_coverage", 0.88 - offset),
            ("context_relevance", 0.82 - offset),
            ("pending_expression_count", 1 + (tick - 1) // 2),
            ("recurrence_count", 1 + (tick - 1) // 2),
            ("salience_score", 0.78 - offset),
        )
    if axis == "expression_inhibition":
        return (
            ("agp_failure_count", (tick - 1) // 2),
            ("conflict_risk", 0.24 + offset),
            ("disclosure_risk", 0.20 + offset),
            ("fallback_required", tick > 1),
            ("uncertainty_score", 0.26 + offset),
        )
    if axis == "action_readiness":
        return (
            ("authorization_status", "authorized"),
            ("capability_available", True),
            ("feasible_action_count", 2 + (tick - 1) // 2),
            ("reversibility", 0.86 - offset),
            ("selected_action_confidence", 0.80 - offset),
        )
    if axis == "risk_tolerance":
        return (
            ("authorization_scope", "bounded-expression-action-v1"),
            ("expected_cost", 0.22 + offset),
            ("reversibility", 0.84 - offset),
            ("safety_margin", 0.76 - offset),
            ("uncertainty_score", 0.24 + offset),
        )
    if axis == "patience_level":
        return (
            ("alternative_action_count", 2 + (tick - 1) // 2),
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("cooldown_remaining", 1 + (tick - 1) // 2),
            ("deadline_pressure", 0.20 + offset),
            ("uncertainty_resolution_gain", 0.74 - offset),
        )
    if axis == "conflict_avoidance":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("boundary_cost", 0.64 + offset),
            ("conflict_probability", 0.34 + offset),
            ("deescalation_option_count", 2 + (tick - 1) // 2),
            ("harm_avoidance_gain", 0.78 - offset),
        )
    raise AssertionError(axis)


def _record(
    axis: str,
    tick: int,
    *,
    observation_id: str | None = None,
    agp_status: str = "passed",
) -> AGPBoundedExpressionActionRawRecord:
    observation_id = observation_id or f"test:{axis}:observation:{tick}"
    source_instance_id = "test:expression-action-source:v1"
    source_snapshot_id = f"test:{axis}:snapshot:{tick}"
    source_schema_version = "test.expression-action-source.v1"
    source_integrity_digest = _sha(f"source:{axis}:{tick}")
    agp_trace_id = f"test:{axis}:agp:{tick}"
    agp_input_digest = _sha(f"agp-input:{axis}:{tick}")
    agp_integrity_digest = _sha(f"agp-integrity:{axis}:{tick}")
    appraisal_trace_id = f"test:{axis}:appraisal:{tick}"
    appraisal_input_digest = agp_integrity_digest
    appraisal_integrity_digest = _sha(f"appraisal-integrity:{axis}:{tick}")
    raw_values = _raw_values(axis, tick)
    raw_observation_digest = agp_bounded_expression_action_raw_observation_digest(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        agp_trace_id=agp_trace_id,
        agp_input_digest=agp_input_digest,
        agp_integrity_digest=agp_integrity_digest,
        agp_status=agp_status,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_values=raw_values,
    )
    return AGPBoundedExpressionActionRawRecord(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        agp_trace_id=agp_trace_id,
        agp_input_digest=agp_input_digest,
        agp_integrity_digest=agp_integrity_digest,
        agp_status=agp_status,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_observation_digest=raw_observation_digest,
        raw_values=raw_values,
    )


def _records(axis: str) -> tuple[AGPBoundedExpressionActionRawRecord, ...]:
    return (_record(axis, 1), _record(axis, 2))


def test_binding_set_completes_exact_thirty_seven_source_bindings_without_claiming_observation():
    binding_set = agp_bounded_expression_action_source_bindings()
    assert tuple(item.axis for item in binding_set.bindings) == EXPRESSION_ACTION_AXES
    assert binding_set.appraised_binding_count == 6
    assert binding_set.total_bound_axis_count == 37
    assert binding_set.remaining_axis_count == 0
    assert binding_set.retained_real_observation_count == 0
    assert binding_set.positive_confidence_real_observation_count == 0
    assert binding_set.blockers == (
        "REGISTRY_RETAINED_REAL_OBSERVATION_CAPTURE_ABSENT",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
    )
    for binding in binding_set.bindings:
        assert (binding.minimum_raw_record_count, binding.minimum_logical_span_ticks) == (2, 1)
        assert binding.authority == "shadow_only"
        assert binding.production_capture_present is False
        assert binding.expression_or_action_executed is False
        assert binding.memory_write_performed is False
        assert binding.observation_window_started is False
        assert binding.m3_b_complete is False
        assert binding.m3_c_open is False
        assert binding.m3_e_authority_open is False
        assert binding.cutover_authorized is False


@pytest.mark.parametrize("axis", EXPRESSION_ACTION_AXES)
def test_agp_bounded_records_derive_deterministic_positive_confidence(axis: str):
    first = derive_agp_bounded_expression_action_axis_evidence(_records(axis))
    second = derive_agp_bounded_expression_action_axis_evidence(_records(axis))
    assert first == second
    assert first.axis == axis
    assert 0.0 <= first.value <= 1.0
    assert 0.0 < first.confidence <= 1.0
    assert first.source_family == SOURCE_FAMILY
    assert first.source_schema_version == RAW_SCHEMA_VERSION
    assert first.model_or_rule_version == f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1"
    assert first.observation_kind == "verified_current_value_observation"
    assert first.verification_status == "verified"
    assert first.genesis_derived is False
    assert first.baseline_derived is False
    assert first.default_derived is False
    assert first.synthetic is False
    assert first.proposal_only is False
    assert first.recalculable_reference_present is True


def test_failed_agp_trace_can_only_be_observed_as_explicit_bounded_failure():
    record = _record("expression_inhibition", 2, agp_status="failed_bounded")
    evidence = derive_agp_bounded_expression_action_axis_evidence(
        (_record("expression_inhibition", 1), record)
    )
    assert evidence.confidence > 0.0
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="agp_status"):
        replace(record, agp_status="failed_unbounded")


def test_agp_and_appraisal_chain_is_digest_bound_and_fail_closed():
    record = _record("expression_pressure", 1)
    assert record.raw_observation_digest == record.recalculated_raw_observation_digest
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="raw observation digest"):
        replace(record, observation_id="changed")
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="exact verified AGP output"):
        replace(record, appraisal_input_digest=_sha("other-agp-output"))
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="requires exact AGP and appraisal"):
        replace(record, agp_trace_verified=False)
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="requires exact AGP and appraisal"):
        replace(record, appraisal_verified=False)


@pytest.mark.parametrize(
    "field",
    (
        "raw_social_feedback_source",
        "hardware_direct_input",
        "synthetic",
        "proposal_only",
        "registry_owner_source",
        "runtime_polled",
        "expression_or_action_executed",
        "memory_write_performed",
        "cutover_authorized",
    ),
)
def test_forbidden_expression_action_origins_execution_and_authority_fail_closed(field: str):
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="cannot use"):
        replace(_record("expression_pressure", 1), **{field: True})


def test_field_order_appraisal_version_authorization_and_boolean_types_are_exact():
    pressure = _record("expression_pressure", 1)
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="canonical expression-action source plan"):
        replace(pressure, raw_values=tuple(reversed(pressure.raw_values)))

    patience = _record("patience_level", 1)
    bad_version = tuple(
        (field, "other") if field == "appraisal_version" else (field, value)
        for field, value in patience.raw_values
    )
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="appraisal_version"):
        replace(patience, raw_values=bad_version)

    readiness = _record("action_readiness", 1)
    bad_authorization = tuple(
        (field, "implicit") if field == "authorization_status" else (field, value)
        for field, value in readiness.raw_values
    )
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="authorization_status"):
        replace(readiness, raw_values=bad_authorization)

    bad_capability = tuple(
        (field, 1) if field == "capability_available" else (field, value)
        for field, value in readiness.raw_values
    )
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="capability_available"):
        replace(readiness, raw_values=bad_capability)


def test_derivation_enforces_two_records_one_tick_unique_identity_and_one_source():
    records = _records("expression_pressure")
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="insufficient raw record count"):
        derive_agp_bounded_expression_action_axis_evidence(records[:1])
    same_tick = (_record("expression_pressure", 1), _record("expression_pressure", 1, observation_id="other"))
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="sorted and unique"):
        derive_agp_bounded_expression_action_axis_evidence(same_tick)
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="sorted and unique"):
        derive_agp_bounded_expression_action_axis_evidence(tuple(reversed(records)))
    duplicate = _record("expression_pressure", 2, observation_id=records[0].observation_id)
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="observation_id values must be unique"):
        derive_agp_bounded_expression_action_axis_evidence((records[0], duplicate))


def test_expression_action_binding_objects_are_frozen_and_cannot_claim_execution_or_authority():
    binding_set = agp_bounded_expression_action_source_bindings()
    with pytest.raises(FrozenInstanceError):
        binding_set.m3_b_complete = True  # type: ignore[misc]
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="cannot claim"):
        replace(binding_set, observation_window_started=True)
    with pytest.raises(AGPBoundedExpressionActionSourceBindingError, match="cannot claim"):
        replace(binding_set.bindings[0], expression_or_action_executed=True)
