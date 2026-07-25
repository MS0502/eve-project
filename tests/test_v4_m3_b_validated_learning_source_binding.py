from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

from core.m3_b_validated_learning_source_binding import (
    ACQUISITION_METHOD,
    APPRAISAL_SCHEMA_VERSION,
    BINDING_SCHEMA_VERSION,
    LEARNING_EXPLORATION_AXES,
    RAW_SCHEMA_VERSION,
    SOURCE_FAMILY,
    VERIFICATION_METHOD,
    ValidatedLearningRawRecord,
    ValidatedLearningSourceBindingError,
    derive_validated_learning_axis_evidence,
    validated_learning_raw_observation_digest,
    validated_learning_source_bindings,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ticks(axis: str) -> tuple[int, ...]:
    if axis == "competence_drive":
        return (1, 3, 5)
    if axis == "prediction_error_pressure":
        return (1, 2)
    return (1, 3)


def _raw_values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = min(0.08, (tick - 1) * 0.02)
    if axis == "curiosity_drive":
        return (
            ("exploration_cost", 0.26 + offset),
            ("information_gain_estimate", 0.68 - offset / 2),
            ("relevance_score", 0.72 - offset / 2),
            ("sampling_window_ticks", 4 + tick),
            ("unknown_count", 1 + (tick - 1) // 2),
        )
    if axis == "novelty_seeking":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("expected_information_gain", 0.62 + offset),
            ("novelty_score", 0.58 + offset),
            ("reversibility", 0.82 - offset),
            ("safety_score", 0.84 - offset),
        )
    if axis == "learning_pressure":
        return (
            ("available_training_signal", 0.70 - offset),
            ("competence_gap", 0.44 + offset),
            ("error_recurrence", 1 + (tick - 1) // 2),
            ("task_relevance", 0.76 - offset / 2),
            ("validation_status", "verified" if tick == 1 else "operator_validated"),
        )
    if axis == "memory_consolidation_pressure":
        return (
            ("causal_relevance", 0.62 + offset),
            ("emotional_relevance", 0.48 + offset),
            ("provenance_completeness", 0.88 - offset),
            ("recurrence_count", 1 + (tick - 1) // 2),
            ("salience_score", 0.66 + offset / 2),
        )
    if axis == "prediction_error_pressure":
        return (
            ("model_version", "test-model-v1"),
            ("normalized_error", 0.24 + offset),
            ("observed_value_digest", _sha(f"observed:{tick}")),
            ("predicted_value_digest", _sha(f"predicted:{tick}")),
            ("verification_status", "verified"),
        )
    if axis == "competence_drive":
        return (
            ("calibrated_error_rate", 0.34 + offset),
            ("evaluation_version", "test-eval-v1"),
            ("learning_progress", 0.52 + offset / 2),
            ("skill_gap", 0.46 + offset),
            ("success_rate", 0.66 - offset),
        )
    raise AssertionError(axis)


def _record(axis: str, tick: int) -> ValidatedLearningRawRecord:
    observation_id = f"test:{axis}:observation:{tick}"
    source_instance_id = "test:validated-learning-source:v1"
    source_snapshot_id = f"test:{axis}:snapshot:{tick}"
    source_schema_version = "test.validated-learning-source.v1"
    source_integrity_digest = _sha(f"source:{axis}:{tick}")
    validation_trace_id = f"test:{axis}:validation:{tick}"
    validation_input_digest = _sha(f"validation-input:{axis}:{tick}")
    validation_integrity_digest = _sha(f"validation-integrity:{axis}:{tick}")
    appraisal_trace_id = f"test:{axis}:appraisal:{tick}"
    appraisal_input_digest = validation_integrity_digest
    appraisal_integrity_digest = _sha(f"appraisal-integrity:{axis}:{tick}")
    raw_values = _raw_values(axis, tick)
    raw_observation_digest = validated_learning_raw_observation_digest(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        validation_trace_id=validation_trace_id,
        validation_input_digest=validation_input_digest,
        validation_integrity_digest=validation_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_values=raw_values,
    )
    return ValidatedLearningRawRecord(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        validation_trace_id=validation_trace_id,
        validation_input_digest=validation_input_digest,
        validation_integrity_digest=validation_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_observation_digest=raw_observation_digest,
        raw_values=raw_values,
    )


def _rebuilt(record: ValidatedLearningRawRecord, **changes: object) -> ValidatedLearningRawRecord:
    values = {
        "axis": changes.get("axis", record.axis),
        "logical_tick": changes.get("logical_tick", record.logical_tick),
        "observation_id": changes.get("observation_id", record.observation_id),
        "source_instance_id": changes.get("source_instance_id", record.source_instance_id),
        "source_snapshot_id": changes.get("source_snapshot_id", record.source_snapshot_id),
        "source_schema_version": changes.get("source_schema_version", record.source_schema_version),
        "source_integrity_digest": changes.get("source_integrity_digest", record.source_integrity_digest),
        "validation_trace_id": changes.get("validation_trace_id", record.validation_trace_id),
        "validation_input_digest": changes.get("validation_input_digest", record.validation_input_digest),
        "validation_integrity_digest": changes.get("validation_integrity_digest", record.validation_integrity_digest),
        "appraisal_trace_id": changes.get("appraisal_trace_id", record.appraisal_trace_id),
        "appraisal_input_digest": changes.get("appraisal_input_digest", record.appraisal_input_digest),
        "appraisal_integrity_digest": changes.get("appraisal_integrity_digest", record.appraisal_integrity_digest),
        "raw_values": changes.get("raw_values", record.raw_values),
    }
    digest = validated_learning_raw_observation_digest(**values)  # type: ignore[arg-type]
    return replace(record, **changes, raw_observation_digest=digest)


def test_binding_set_has_exact_six_axes_and_progress_is_twenty_five_of_thirty_seven():
    binding_set = validated_learning_source_bindings()
    assert tuple(item.axis for item in binding_set.bindings) == LEARNING_EXPLORATION_AXES
    assert binding_set.appraised_binding_count == 6
    assert binding_set.total_bound_axis_count == 25
    assert binding_set.remaining_axis_count == 12
    assert binding_set.blockers == (
        "REGISTRY_APPRAISED_12_AXIS_SOURCE_BINDINGS_INCOMPLETE",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
    )
    for binding in binding_set.bindings:
        if binding.axis == "competence_drive":
            assert (binding.minimum_raw_record_count, binding.minimum_logical_span_ticks) == (3, 4)
        elif binding.axis == "prediction_error_pressure":
            assert (binding.minimum_raw_record_count, binding.minimum_logical_span_ticks) == (2, 1)
        else:
            assert (binding.minimum_raw_record_count, binding.minimum_logical_span_ticks) == (2, 2)
        assert binding.authority == "shadow_only"
        assert binding.appraisal_required is True
        assert binding.quarantine_required_for_social_feedback is True
        assert binding.hardware_direct_input_allowed is False
        assert binding.production_capture_present is False
        assert binding.learning_mutation_performed is False
        assert binding.memory_write_performed is False
        assert binding.observation_window_started is False
        assert binding.m3_b_complete is False
        assert binding.m3_c_open is False
        assert binding.m3_e_authority_open is False
        assert binding.cutover_authorized is False


@pytest.mark.parametrize("axis", LEARNING_EXPLORATION_AXES)
def test_validated_learning_records_derive_deterministic_positive_confidence(axis: str):
    records = tuple(_record(axis, tick) for tick in _ticks(axis))
    first = derive_validated_learning_axis_evidence(records)
    second = derive_validated_learning_axis_evidence(records)
    assert first == second
    assert first.axis == axis
    assert 0.0 <= first.value <= 1.0
    assert 0.0 < first.confidence <= 1.0
    assert first.source_family == SOURCE_FAMILY
    assert first.source_schema_version == RAW_SCHEMA_VERSION
    assert first.acquisition_method == ACQUISITION_METHOD
    assert first.verification_method == VERIFICATION_METHOD
    assert first.model_or_rule_version == f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1"
    assert first.observation_kind == "verified_current_value_observation"
    assert first.verification_status == "verified"
    assert first.recalculable_reference_present is True


def test_validation_and_appraisal_chain_is_digest_bound_and_fail_closed():
    record = _record("curiosity_drive", 1)
    assert record.raw_observation_digest == record.recalculated_raw_observation_digest
    with pytest.raises(ValidatedLearningSourceBindingError, match="raw observation digest"):
        replace(record, observation_id="changed")
    with pytest.raises(ValidatedLearningSourceBindingError, match="exact verified validation output"):
        replace(record, appraisal_input_digest=_sha("different-validation-output"))
    with pytest.raises(ValidatedLearningSourceBindingError, match="requires exact validation and appraisal"):
        replace(record, validation_verified=False)
    with pytest.raises(ValidatedLearningSourceBindingError, match="requires exact validation and appraisal"):
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
        "learning_mutation_performed",
        "memory_write_performed",
    ),
)
def test_forbidden_learning_origins_and_mutations_fail_closed(field: str):
    with pytest.raises(ValidatedLearningSourceBindingError, match="cannot use"):
        replace(_record("curiosity_drive", 1), **{field: True})


def test_raw_field_order_versions_statuses_digests_and_counts_are_exact():
    curiosity = _record("curiosity_drive", 1)
    with pytest.raises(ValidatedLearningSourceBindingError, match="canonical learning source plan"):
        replace(curiosity, raw_values=tuple(reversed(curiosity.raw_values)))
    novelty = _record("novelty_seeking", 1)
    bad_version = tuple(
        (field, "other") if field == "appraisal_version" else (field, value)
        for field, value in novelty.raw_values
    )
    with pytest.raises(ValidatedLearningSourceBindingError, match="appraisal_version"):
        _rebuilt(novelty, raw_values=bad_version)
    learning = _record("learning_pressure", 1)
    bad_status = tuple(
        (field, "claimed") if field == "validation_status" else (field, value)
        for field, value in learning.raw_values
    )
    with pytest.raises(ValidatedLearningSourceBindingError, match="validation_status"):
        _rebuilt(learning, raw_values=bad_status)
    prediction = _record("prediction_error_pressure", 1)
    bad_digest = tuple(
        (field, "0" * 64) if field == "observed_value_digest" else (field, value)
        for field, value in prediction.raw_values
    )
    with pytest.raises(ValidatedLearningSourceBindingError, match="observed_value_digest"):
        _rebuilt(prediction, raw_values=bad_digest)


def test_derivation_enforces_minimum_count_span_uniqueness_and_one_source():
    first = _record("curiosity_drive", 1)
    second = _record("curiosity_drive", 3)
    with pytest.raises(ValidatedLearningSourceBindingError, match="insufficient raw record count"):
        derive_validated_learning_axis_evidence((first,))
    too_short = _record("curiosity_drive", 2)
    with pytest.raises(ValidatedLearningSourceBindingError, match="insufficient logical observation span"):
        derive_validated_learning_axis_evidence((first, too_short))
    with pytest.raises(ValidatedLearningSourceBindingError, match="ticks must be sorted"):
        derive_validated_learning_axis_evidence((second, first))
    duplicate = _rebuilt(second, observation_id=first.observation_id)
    with pytest.raises(ValidatedLearningSourceBindingError, match="observation_id values must be unique"):
        derive_validated_learning_axis_evidence((first, duplicate))
    changed_source = _rebuilt(second, source_instance_id="other-learning-source")
    with pytest.raises(ValidatedLearningSourceBindingError, match="share one source_instance_id"):
        derive_validated_learning_axis_evidence((first, changed_source))


def test_learning_binding_objects_are_frozen_and_cannot_claim_mutation_or_authority():
    binding_set = validated_learning_source_bindings()
    with pytest.raises(FrozenInstanceError):
        binding_set.m3_b_complete = True  # type: ignore[misc]
    with pytest.raises(ValidatedLearningSourceBindingError, match="cannot claim"):
        replace(binding_set, learning_mutation_performed=True)
    with pytest.raises(ValidatedLearningSourceBindingError, match="cannot claim"):
        replace(binding_set.bindings[0], memory_write_performed=True)
