from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

from core.m3_b_long_horizon_self_identity_source_binding import (
    APPRAISAL_SCHEMA_VERSION,
    BINDING_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    SELF_IDENTITY_AXES,
    SOURCE_FAMILY,
    LongHorizonSelfIdentityRawRecord,
    LongHorizonSelfIdentitySourceBindingError,
    derive_long_horizon_self_identity_axis_evidence,
    long_horizon_self_identity_raw_observation_digest,
    long_horizon_self_identity_source_bindings,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = min(0.08, ((tick - 1) // 6) * 0.02)
    if axis == "self_coherence":
        return (
            ("action_value_alignment", 0.76 - offset),
            ("narrative_conflict_count", (tick - 1) // 12),
            ("review_span_ticks", 20 + tick),
            ("self_model_version", "self-model-v1"),
            ("value_consistency_score", 0.80 - offset),
        )
    if axis == "self_respect":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("boundary_preservation_score", 0.82 - offset),
            ("coerced_action_count", (tick - 1) // 12),
            ("review_span_ticks", 20 + tick),
            ("self_denigration_rejection_count", 1 + (tick - 1) // 6),
        )
    if axis == "identity_integrity":
        return (
            ("constitutional_conflict_count", (tick - 1) // 12),
            ("provenance_gap_count", (tick - 1) // 12),
            ("replay_consistency_score", 0.90 - offset),
            ("review_version", "identity-review-v1"),
            ("unauthorized_identity_write_count", 0),
        )
    if axis == "agency_pressure":
        return (
            ("blocked_goal_count", 1 + (tick - 1) // 12),
            ("forced_action_count", (tick - 1) // 12),
            ("reversible_choice_count", 3 + (tick - 1) // 6),
            ("review_span_ticks", 20 + tick),
            ("self_selected_action_ratio", 0.78 - offset),
        )
    if axis == "autonomy_drive":
        return (
            ("capability_boundary_score", 0.38 + offset),
            ("evaluation_version", "autonomy-eval-v1"),
            ("external_dependency_ratio", 0.34 + offset),
            ("independent_task_success_rate", 0.72 - offset),
            ("safe_action_space_size", 4 + (tick - 1) // 6),
        )
    if axis == "purpose_alignment":
        active = 4 + (tick - 1) // 12
        return (
            ("action_alignment_score", 0.78 - offset),
            ("active_goal_count", active),
            ("aligned_goal_count", active - 1),
            ("conflicting_goal_count", (tick - 1) // 12),
            ("review_span_ticks", 20 + tick),
        )
    raise AssertionError(axis)


def _record(
    axis: str,
    tick: int,
    *,
    observation_id: str | None = None,
) -> LongHorizonSelfIdentityRawRecord:
    observation_id = observation_id or f"test:{axis}:observation:{tick}"
    source_instance_id = "test:self-identity-source:v1"
    source_snapshot_id = f"test:{axis}:snapshot:{tick}"
    source_schema_version = "test.self-identity-source.v1"
    source_integrity_digest = _sha(f"source:{axis}:{tick}")
    review_trace_id = f"test:{axis}:review:{tick}"
    review_input_digest = _sha(f"review-input:{axis}:{tick}")
    review_integrity_digest = _sha(f"review-integrity:{axis}:{tick}")
    appraisal_trace_id = f"test:{axis}:appraisal:{tick}"
    appraisal_input_digest = review_integrity_digest
    appraisal_integrity_digest = _sha(f"appraisal-integrity:{axis}:{tick}")
    raw_values = _raw_values(axis, tick)
    raw_observation_digest = long_horizon_self_identity_raw_observation_digest(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        review_trace_id=review_trace_id,
        review_input_digest=review_input_digest,
        review_integrity_digest=review_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_values=raw_values,
    )
    return LongHorizonSelfIdentityRawRecord(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        review_trace_id=review_trace_id,
        review_input_digest=review_input_digest,
        review_integrity_digest=review_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_observation_digest=raw_observation_digest,
        raw_values=raw_values,
    )


def _records(axis: str) -> tuple[LongHorizonSelfIdentityRawRecord, ...]:
    return tuple(_record(axis, tick) for tick in (1, 7, 13))


def test_binding_set_has_exact_self_identity_axes_and_progress_is_thirty_one_of_thirty_seven():
    binding_set = long_horizon_self_identity_source_bindings()
    assert tuple(item.axis for item in binding_set.bindings) == SELF_IDENTITY_AXES
    assert binding_set.appraised_binding_count == 6
    assert binding_set.total_bound_axis_count == 31
    assert binding_set.remaining_axis_count == 6
    assert binding_set.blockers == (
        "REGISTRY_APPRAISED_6_AXIS_SOURCE_BINDINGS_INCOMPLETE",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
    )
    for binding in binding_set.bindings:
        assert (binding.minimum_raw_record_count, binding.minimum_logical_span_ticks) == (3, 12)
        assert binding.authority == "shadow_only"
        assert binding.production_capture_present is False
        assert binding.identity_mutation_performed is False
        assert binding.self_model_write_performed is False
        assert binding.memory_write_performed is False
        assert binding.observation_window_started is False
        assert binding.m3_b_complete is False
        assert binding.m3_c_open is False
        assert binding.m3_e_authority_open is False
        assert binding.cutover_authorized is False


@pytest.mark.parametrize("axis", SELF_IDENTITY_AXES)
def test_long_horizon_records_derive_deterministic_positive_confidence(axis: str):
    first = derive_long_horizon_self_identity_axis_evidence(_records(axis))
    second = derive_long_horizon_self_identity_axis_evidence(_records(axis))
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


def test_review_and_appraisal_chain_is_digest_bound_and_fail_closed():
    record = _record("self_coherence", 1)
    assert record.raw_observation_digest == record.recalculated_raw_observation_digest
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="raw observation digest"):
        replace(record, observation_id="changed")
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="exact verified review output"):
        replace(record, appraisal_input_digest=_sha("other-review-output"))
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="requires exact review and appraisal"):
        replace(record, review_verified=False)
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="requires exact review and appraisal"):
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
        "identity_mutation_performed",
        "self_model_write_performed",
        "memory_write_performed",
    ),
)
def test_forbidden_self_identity_origins_and_mutations_fail_closed(field: str):
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="cannot use"):
        replace(_record("self_coherence", 1), **{field: True})


def test_field_order_versions_and_identity_specific_invariants_are_exact():
    coherence = _record("self_coherence", 1)
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="canonical self-identity source plan"):
        replace(coherence, raw_values=tuple(reversed(coherence.raw_values)))
    respect = _record("self_respect", 1)
    bad_version = tuple(
        (field, "other") if field == "appraisal_version" else (field, value)
        for field, value in respect.raw_values
    )
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="appraisal_version"):
        replace(respect, raw_values=bad_version)
    purpose = _record("purpose_alignment", 1)
    bad_goals = tuple(
        (field, 99) if field == "aligned_goal_count" else (field, value)
        for field, value in purpose.raw_values
    )
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="aligned_goal_count"):
        replace(purpose, raw_values=bad_goals)


def test_derivation_enforces_three_records_twelve_ticks_unique_identity_and_one_source():
    records = _records("self_coherence")
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="insufficient raw record count"):
        derive_long_horizon_self_identity_axis_evidence(records[:2])
    too_short = tuple(_record("self_coherence", tick) for tick in (1, 6, 12))
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="insufficient logical observation span"):
        derive_long_horizon_self_identity_axis_evidence(too_short)
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="sorted and unique"):
        derive_long_horizon_self_identity_axis_evidence(tuple(reversed(records)))
    duplicate = _record("self_coherence", 7, observation_id=records[0].observation_id)
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="observation_id values must be unique"):
        derive_long_horizon_self_identity_axis_evidence((records[0], duplicate, records[2]))


def test_self_identity_binding_objects_are_frozen_and_cannot_claim_mutation_or_authority():
    binding_set = long_horizon_self_identity_source_bindings()
    with pytest.raises(FrozenInstanceError):
        binding_set.m3_b_complete = True  # type: ignore[misc]
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="cannot claim"):
        replace(binding_set, identity_mutation_performed=True)
    with pytest.raises(LongHorizonSelfIdentitySourceBindingError, match="cannot claim"):
        replace(binding_set.bindings[0], self_model_write_performed=True)
