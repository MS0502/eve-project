import json
from pathlib import Path

import pytest

from core.m3_b_c2_energy_budget_retention_activation import (
    append_reviewed_energy_budget_observation,
)
from core.m3_b_c2_fatigue_pressure_retention_activation import (
    append_reviewed_fatigue_pressure_observation,
)
from core.m3_b_c2_recovery_need_retention_activation import (
    append_reviewed_recovery_need_observation,
)
from core.m3_b_c2_retention_activation import append_reviewed_observation
from core.m3_b_c2_reviewed_stress_load_integration import (
    ATTESTATION_DIGEST,
    EVIDENCE_DIGEST,
    PUBLIC_REVIEW_DIGEST,
    C2ReviewedStressLoadIntegrationError,
    build_reviewed_capture,
    integration_status,
    verify_public_review,
)
from core.m3_b_c2_stress_load_retention_activation import (
    C2StressLoadRetentionActivationError,
    append_reviewed_stress_load_observation,
)
from core.sqlite_shadow_store import SQLiteShadowStore

ROOT = Path(__file__).resolve().parents[1]


def _review(path: str) -> dict:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return value["public_review"]


def _stress_review() -> dict:
    return _review("docs/audit/M3_B_C2_REVIEWED_STRESS_LOAD_WITNESS.json")


def _seed_four(store: SQLiteShadowStore) -> None:
    append_reviewed_observation(
        store,
        _review("docs/audit/M3_B_C2_REVIEWED_PHONE_WITNESS.json"),
    )
    append_reviewed_energy_budget_observation(
        store,
        _review("docs/audit/M3_B_C2_REVIEWED_ENERGY_BUDGET_WITNESS.json"),
    )
    append_reviewed_fatigue_pressure_observation(
        store,
        _review("docs/audit/M3_B_C2_REVIEWED_FATIGUE_PRESSURE_WITNESS.json"),
    )
    append_reviewed_recovery_need_observation(
        store,
        _review("docs/audit/M3_B_C2_REVIEWED_RECOVERY_NEED_WITNESS.json"),
    )


def test_reviewed_stress_load_public_review_matches_exact_pins() -> None:
    attestation, evidence = verify_public_review(_stress_review())
    assert attestation.attestation_digest == ATTESTATION_DIGEST
    assert evidence.evidence_digest == EVIDENCE_DIGEST
    assert _stress_review()["public_review_digest"] == PUBLIC_REVIEW_DIGEST
    assert evidence.axis == "stress_load"
    assert evidence.synthetic is False
    assert evidence.confidence > 0.0


def test_reviewed_stress_load_integration_advances_reviewed_not_retained_boundary() -> None:
    status = integration_status(_stress_review())
    capture = build_reviewed_capture(_stress_review())
    assert status.reviewed_real_operator_attestation_count == 5
    assert status.registered_runtime_provenance_verifier_count == 5
    assert status.registered_production_source_verifier_count == 5
    assert status.verified_positive_confidence_candidate_count == 5
    assert status.retained_real_observation_count == 4
    assert capture.retained_real_observation_eligible is True
    assert capture.retained_real_observation is False
    assert capture.observation_window_started is False
    assert capture.m3_c_open is False
    assert capture.m3_e_authority_open is False
    assert capture.cutover_authorized is False


def test_reviewed_stress_load_preserves_two_stage_appraisal_provenance() -> None:
    review = _stress_review()
    capture = build_reviewed_capture(review)
    assert review["provenance_boundary"] == {
        "appraisal_bridge_output_detached": True,
        "appraisal_output_kind": "detached_verified_appraisal_trace",
        "canonical_appraised_record_hardware_direct_input": False,
        "canonical_appraised_record_runtime_polled": False,
        "raw_runtime_metrics_publicly_retained": False,
        "runtime_input_kind": "operator_private_real_runtime_metrics",
        "runtime_metrics_used_as_appraisal_input": True,
    }
    source = capture.source_verification.to_mapping()
    assert source["appraisal_bridge_output_detached"] is True
    assert source["runtime_metrics_used_as_appraisal_input"] is True
    assert source["production_origin_verified"] is True
    assert source["synthetic"] is False


def test_stress_load_sequence_five_append_requires_exact_prior_chain(tmp_path: Path) -> None:
    store = SQLiteShadowStore(tmp_path / "retained.sqlite3")
    store.initialize()
    _seed_four(store)

    receipt = append_reviewed_stress_load_observation(store, _stress_review())

    assert receipt.axis == "stress_load"
    assert receipt.sequence == 5
    assert receipt.store_ordinal == 5
    assert receipt.store_before_count == 4
    assert receipt.store_after_count == 5
    assert receipt.retained_real_observation_count_after_append == 5
    assert receipt.readback_verified is True
    assert receipt.observation_window_started is False
    assert receipt.m3_b_complete is False
    assert receipt.m3_c_open is False
    assert receipt.m3_e_authority_open is False
    assert receipt.cutover_authorized is False
    events = store.events()
    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]
    assert [event.payload["axis"] for event in events] == [
        "prediction_error_pressure",
        "energy_budget",
        "fatigue_pressure",
        "recovery_need",
        "stress_load",
    ]


def test_stress_load_sequence_five_cannot_replay_same_witness(tmp_path: Path) -> None:
    store = SQLiteShadowStore(tmp_path / "retained.sqlite3")
    store.initialize()
    _seed_four(store)
    append_reviewed_stress_load_observation(store, _stress_review())

    with pytest.raises(
        C2StressLoadRetentionActivationError,
        match="requires exactly four prior retained events",
    ):
        append_reviewed_stress_load_observation(store, _stress_review())


def test_reviewed_stress_load_public_review_fails_closed_on_tamper() -> None:
    review = _stress_review()
    review["evidence"]["value"] = 0.5
    with pytest.raises(
        C2ReviewedStressLoadIntegrationError,
        match="canonical digest mismatch",
    ):
        verify_public_review(review)
