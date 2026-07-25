from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from core.m3_b_prediction_error_runtime_source_bridge import (
    AXIS,
    MODEL_VERSION,
    OBSERVATION_WINDOW_BLOCKER,
    POSITIVE_CONFIDENCE_BLOCKER,
    PRODUCTION_VERIFIER_BLOCKER,
    RUNTIME_PROVENANCE_BLOCKER,
    PredictionErrorRuntimeSourceBridgeError,
    derive_detached_prediction_error_evidence,
    prediction_error_runtime_snapshot_from_mappings,
    prediction_error_runtime_source_bridge_capability,
    read_prediction_error_runtime_source,
)


def _prediction(
    *,
    prediction_id: str,
    mood_valence: float = 0.2,
    observed: bool = True,
) -> dict:
    return {
        "id": prediction_id,
        "time": 0.0,
        "horizon": 1.0,
        "target_time": 1.0,
        "expected_state": {
            "time": 0.0,
            "mood_valence": mood_valence,
            "mood_arousal": 0.4,
            "primary_hormone": "dopamine",
            "primary_level": 0.6,
            "hormones": {"cortisol": 0.3, "dopamine": 0.6},
        },
        "expected_outcome": "neutral",
        "confidence": 0.0,
        "outcome_scores": {"bad": 0.0, "good": 0.0, "neutral": 0.0},
        "observed": observed,
    }


def _error(*, prediction_id: str, mood_error: float) -> dict:
    return {
        "prediction_id": prediction_id,
        "time": 0.0,
        "mood_error": mood_error,
        "hormone_errors": {"cortisol": 0.1, "dopamine": -0.05},
        "predicted_outcome": "neutral",
        "actual_outcome": "neutral",
        "outcome_match": True,
        "pred_confidence": 0.0,
    }


@dataclass(frozen=True, slots=True)
class _RuntimeAI:
    errors: tuple[dict, ...]
    observe_count: int


@dataclass(frozen=True, slots=True)
class _RuntimeAdapter:
    _last_prediction: dict | None
    ai: _RuntimeAI


def test_runtime_shaped_snapshot_matches_manifest_fields_and_is_deterministic():
    prediction = _prediction(prediction_id="pred_0000")
    error = _error(prediction_id="pred_0000", mood_error=0.6)
    first = prediction_error_runtime_snapshot_from_mappings(
        prediction=prediction,
        error=error,
        observe_count=1,
        source_instance_id="runtime:ai-adapter:primary",
        fixture_only=True,
    )
    second = prediction_error_runtime_snapshot_from_mappings(
        prediction=prediction,
        error=error,
        observe_count=1,
        source_instance_id="runtime:ai-adapter:primary",
        fixture_only=True,
    )
    assert first == second
    assert first.normalized_error == pytest.approx(0.2)
    assert first.raw_values[0] == ("model_version", MODEL_VERSION)
    assert first.raw_values[1][0] == "normalized_error"
    assert first.raw_values[1][1] == pytest.approx(0.2)
    assert first.raw_values[2] == ("observed_value_digest", first.observed_value_digest)
    assert first.raw_values[3] == ("predicted_value_digest", first.predicted_value_digest)
    assert first.raw_values[4] == ("verification_status", "verified")
    record = first.to_validated_learning_raw_record()
    assert record.axis == AXIS
    assert tuple(field for field, _ in record.raw_values) == (
        "model_version",
        "normalized_error",
        "observed_value_digest",
        "predicted_value_digest",
        "verification_status",
    )
    assert record.raw_observation_digest == record.recalculated_raw_observation_digest
    assert first.production_origin_verified is False
    assert first.production_verifier_registered is False
    assert first.retained_real_observation is False


def test_two_runtime_shaped_snapshots_derive_detached_positive_confidence_evidence():
    first = prediction_error_runtime_snapshot_from_mappings(
        prediction=_prediction(prediction_id="pred_0000"),
        error=_error(prediction_id="pred_0000", mood_error=0.6),
        observe_count=1,
        source_instance_id="runtime:ai-adapter:primary",
        fixture_only=True,
    )
    second = prediction_error_runtime_snapshot_from_mappings(
        prediction=_prediction(prediction_id="pred_0001", mood_valence=0.1),
        error=_error(prediction_id="pred_0001", mood_error=0.3),
        observe_count=2,
        source_instance_id="runtime:ai-adapter:primary",
        fixture_only=True,
    )
    evidence = derive_detached_prediction_error_evidence((first, second))
    assert evidence.axis == AXIS
    assert evidence.observed_tick == 2
    assert evidence.value == pytest.approx(0.15)
    assert evidence.confidence == pytest.approx(0.9975)
    assert evidence.confidence > 0.0
    assert first.fixture_only is True
    assert second.fixture_only is True
    assert prediction_error_runtime_source_bridge_capability().retained_real_observation_count == 0


def test_runtime_reader_reads_existing_trace_without_mutation_or_runtime_calls():
    prediction = _prediction(prediction_id="pred_0007")
    error = _error(prediction_id="pred_0007", mood_error=0.45)
    adapter = _RuntimeAdapter(
        _last_prediction=prediction,
        ai=_RuntimeAI(errors=(error,), observe_count=7),
    )
    before = json.dumps(
        {"prediction": prediction, "error": error, "observe_count": adapter.ai.observe_count},
        sort_keys=True,
        separators=(",", ":"),
    )
    snapshot = read_prediction_error_runtime_source(
        adapter,
        source_instance_id="runtime:ai-adapter:primary",
        fixture_only=True,
    )
    after = json.dumps(
        {"prediction": prediction, "error": error, "observe_count": adapter.ai.observe_count},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert snapshot is not None
    assert snapshot.logical_tick == 7
    assert snapshot.prediction_id == "pred_0007"
    assert before == after


def test_runtime_reader_returns_none_until_an_observed_error_exists():
    adapter = _RuntimeAdapter(
        _last_prediction=None,
        ai=_RuntimeAI(errors=(), observe_count=0),
    )
    assert (
        read_prediction_error_runtime_source(
            adapter,
            source_instance_id="runtime:ai-adapter:primary",
            fixture_only=True,
        )
        is None
    )


def test_runtime_bridge_fails_closed_on_unobserved_or_mismatched_trace():
    with pytest.raises(PredictionErrorRuntimeSourceBridgeError, match="already-observed"):
        prediction_error_runtime_snapshot_from_mappings(
            prediction=_prediction(prediction_id="pred_0000", observed=False),
            error=_error(prediction_id="pred_0000", mood_error=0.2),
            observe_count=1,
            source_instance_id="runtime:ai-adapter:primary",
            fixture_only=True,
        )

    with pytest.raises(PredictionErrorRuntimeSourceBridgeError, match="does not match"):
        prediction_error_runtime_snapshot_from_mappings(
            prediction=_prediction(prediction_id="pred_0001"),
            error=_error(prediction_id="pred_other", mood_error=0.2),
            observe_count=1,
            source_instance_id="runtime:ai-adapter:primary",
            fixture_only=True,
        )


def test_non_fixture_label_is_not_production_origin_proof():
    snapshot = prediction_error_runtime_snapshot_from_mappings(
        prediction=_prediction(prediction_id="pred_0000"),
        error=_error(prediction_id="pred_0000", mood_error=0.2),
        observe_count=1,
        source_instance_id="runtime:ai-adapter:primary",
        fixture_only=False,
    )
    assert snapshot.fixture_only is False
    assert snapshot.production_origin_verified is False
    assert snapshot.production_verifier_registered is False
    assert snapshot.retained_real_observation is False


def test_bridge_capability_keeps_production_and_m3_boundaries_closed():
    status = prediction_error_runtime_source_bridge_capability()
    assert status.axis == AXIS
    assert status.source_bridge_present is True
    assert status.runtime_hook_installed is False
    assert status.trusted_production_runtime_provenance_present is False
    assert status.production_source_verifier_registered is False
    assert status.retained_real_observation_count == 0
    assert status.positive_confidence_real_observation_count == 0
    assert status.observation_window_eligible is False
    assert status.observation_window_started is False
    assert status.m3_b_complete is False
    assert status.m3_c_open is False
    assert status.m3_e_authority_open is False
    assert status.cutover_authorized is False
    assert status.blockers == (
        RUNTIME_PROVENANCE_BLOCKER,
        PRODUCTION_VERIFIER_BLOCKER,
        POSITIVE_CONFIDENCE_BLOCKER,
        OBSERVATION_WINDOW_BLOCKER,
    )
