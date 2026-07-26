from __future__ import annotations

import json

import pytest

from core.m3_b_phone_prediction_error_witness import (
    DEFAULT_SOURCE_INSTANCE_ID,
    ENTRYPOINT_ID,
    PhonePredictionErrorWitness,
    PhonePredictionErrorWitnessError,
    build_phone_prediction_error_witness,
)
from core.m3_b_prediction_error_runtime_source_bridge import (
    prediction_error_runtime_snapshot_from_mappings,
)

HEAD = "a09ebb8abbbf68c9235795a7c89d8b8ea5d75378"
NONCE = b"operator-private-c2-witness-nonce-32bytes!!"
RUNTIME_ID = "runtime:phone:primary-c2-witness-01"
ATTESTATION_ID = "operator-attestation:c2-witness-01"


def _prediction(*, prediction_id: str, mood_valence: float) -> dict:
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
        "observed": True,
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


def _snapshot(*, prediction_id: str, tick: int, mood_error: float, source: str = DEFAULT_SOURCE_INSTANCE_ID):
    return prediction_error_runtime_snapshot_from_mappings(
        prediction=_prediction(prediction_id=prediction_id, mood_valence=0.2 / tick),
        error=_error(prediction_id=prediction_id, mood_error=mood_error),
        observe_count=tick,
        source_instance_id=source,
        fixture_only=False,
    )


def _witness() -> PhonePredictionErrorWitness:
    return build_phone_prediction_error_witness(
        private_nonce=NONCE,
        runtime_instance_id=RUNTIME_ID,
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        repository_head_sha=HEAD,
        launch_attestation_id=ATTESTATION_ID,
        snapshots=(
            _snapshot(prediction_id="pred_real_0001", tick=1, mood_error=0.6),
            _snapshot(prediction_id="pred_real_0002", tick=2, mood_error=0.3),
        ),
    )


def test_phone_witness_binds_c1_attestation_and_two_real_runtime_snapshots():
    witness = _witness()
    assert witness.attestation.fixture_only is False
    assert witness.attestation.repository_head_sha == HEAD
    assert witness.attestation.entrypoint_id == ENTRYPOINT_ID
    assert witness.attestation.runtime_instance_id == RUNTIME_ID
    assert witness.attestation.source_instance_id == DEFAULT_SOURCE_INSTANCE_ID
    assert witness.evidence.axis == "prediction_error_pressure"
    assert witness.evidence.observed_tick == 2
    assert witness.evidence.confidence > 0.0
    assert tuple(item.logical_tick for item in witness.snapshots) == (1, 2)


def test_public_review_is_digest_only_and_keeps_all_authority_closed():
    witness = _witness()
    public = witness.public_review_mapping()
    encoded = json.dumps(public, sort_keys=True)
    assert "prediction" not in public
    assert "error" not in public
    assert "snapshots" not in public
    assert NONCE.decode("ascii") not in encoded
    assert public["private_raw_location"] == "operator_private_companion_only"
    assert public["raw_record_count"] == 2
    assert len(public["private_material_digest"]) == 64
    assert len(public["public_review_digest"]) == 64
    assert public["reviewed_attestation_registered"] is False
    assert public["runtime_provenance_verifier_registered"] is False
    assert public["production_source_verifier_registered"] is False
    assert public["retained_real_observation"] is False
    assert public["observation_window_started"] is False
    assert public["m3_b_complete"] is False
    assert public["m3_c_open"] is False
    assert public["m3_e_authority_open"] is False
    assert public["cutover_authorized"] is False


def test_private_mapping_retains_recalculable_raw_trace_but_not_private_nonce():
    witness = _witness()
    private = witness.private_mapping()
    encoded = json.dumps(private, sort_keys=True)
    assert len(private["snapshots"]) == 2
    assert "prediction" in private["snapshots"][0]
    assert "error" in private["snapshots"][0]
    assert private["evidence"]["raw_observation_digest"] == witness.evidence.raw_observation_digest
    assert NONCE.decode("ascii") not in encoded


def test_witness_fails_closed_on_wrong_count_span_or_source_binding():
    first = _snapshot(prediction_id="pred_real_0001", tick=1, mood_error=0.6)
    with pytest.raises(PhonePredictionErrorWitnessError, match="exactly two"):
        build_phone_prediction_error_witness(
            private_nonce=NONCE,
            runtime_instance_id=RUNTIME_ID,
            source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
            repository_head_sha=HEAD,
            launch_attestation_id=ATTESTATION_ID,
            snapshots=(first,),
        )

    same_tick = _snapshot(prediction_id="pred_real_0002", tick=1, mood_error=0.3)
    with pytest.raises(PhonePredictionErrorWitnessError, match="strictly increasing"):
        build_phone_prediction_error_witness(
            private_nonce=NONCE,
            runtime_instance_id=RUNTIME_ID,
            source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
            repository_head_sha=HEAD,
            launch_attestation_id=ATTESTATION_ID,
            snapshots=(first, same_tick),
        )

    other_source = _snapshot(
        prediction_id="pred_real_0002",
        tick=2,
        mood_error=0.3,
        source="runtime:ai-adapter:other",
    )
    with pytest.raises(PhonePredictionErrorWitnessError, match="one source instance"):
        build_phone_prediction_error_witness(
            private_nonce=NONCE,
            runtime_instance_id=RUNTIME_ID,
            source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
            repository_head_sha=HEAD,
            launch_attestation_id=ATTESTATION_ID,
            snapshots=(first, other_source),
        )
