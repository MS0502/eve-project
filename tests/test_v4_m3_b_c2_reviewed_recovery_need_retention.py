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
    C2RecoveryNeedRetentionActivationError,
    append_reviewed_recovery_need_observation,
)
from core.m3_b_c2_retention_activation import append_reviewed_observation
from core.m3_b_c2_reviewed_recovery_need_integration import (
    ATTESTATION_DIGEST,
    EVIDENCE_DIGEST,
    PUBLIC_REVIEW_DIGEST,
    C2ReviewedRecoveryNeedIntegrationError,
    build_reviewed_capture,
    integration_status,
    verify_public_review,
)
from core.sqlite_shadow_store import SQLiteShadowStore

ROOT = Path(__file__).resolve().parents[1]


def _review(path: str) -> dict:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return value["public_review"]


def _recovery_review() -> dict:
    return _review("docs/audit/M3_B_C2_REVIEWED_RECOVERY_NEED_WITNESS.json")


def _seed_three(store: SQLiteShadowStore) -> None:
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


def test_reviewed_recovery_public_review_matches_exact_pins() -> None:
    attestation, evidence = verify_public_review(_recovery_review())
    assert attestation.attestation_digest == ATTESTATION_DIGEST
    assert evidence.evidence_digest == EVIDENCE_DIGEST
    assert _recovery_review()["public_review_digest"] == PUBLIC_REVIEW_DIGEST
    assert evidence.axis == "recovery_need"
    assert evidence.synthetic is False


def test_reviewed_recovery_integration_advances_reviewed_not_retained_boundary() -> None:
    status = integration_status(_recovery_review())
    capture = build_reviewed_capture(_recovery_review())
    assert status.reviewed_real_operator_attestation_count == 4
    assert status.registered_runtime_provenance_verifier_count == 4
    assert status.registered_production_source_verifier_count == 4
    assert status.verified_positive_confidence_candidate_count == 4
    assert status.retained_real_observation_count == 3
    assert capture.retained_real_observation_eligible is True
    assert capture.retained_real_observation is False
    assert capture.observation_window_started is False
    assert capture.m3_c_open is False
    assert capture.m3_e_authority_open is False
    assert capture.cutover_authorized is False


def test_recovery_sequence_four_append_requires_exact_prior_chain(tmp_path: Path) -> None:
    store = SQLiteShadowStore(tmp_path / "retained.sqlite3")
    store.initialize()
    _seed_three(store)

    receipt = append_reviewed_recovery_need_observation(store, _recovery_review())

    assert receipt.axis == "recovery_need"
    assert receipt.sequence == 4
    assert receipt.store_ordinal == 4
    assert receipt.store_before_count == 3
    assert receipt.store_after_count == 4
    assert receipt.retained_real_observation_count_after_append == 4
    assert receipt.readback_verified is True
    assert receipt.observation_window_started is False
    assert receipt.m3_b_complete is False
    assert receipt.m3_c_open is False
    assert receipt.m3_e_authority_open is False
    assert receipt.cutover_authorized is False
    events = store.events()
    assert [event.sequence for event in events] == [1, 2, 3, 4]
    assert [event.payload["axis"] for event in events] == [
        "prediction_error_pressure",
        "energy_budget",
        "fatigue_pressure",
        "recovery_need",
    ]


def test_recovery_sequence_four_cannot_replay_same_witness(tmp_path: Path) -> None:
    store = SQLiteShadowStore(tmp_path / "retained.sqlite3")
    store.initialize()
    _seed_three(store)
    append_reviewed_recovery_need_observation(store, _recovery_review())

    with pytest.raises(
        C2RecoveryNeedRetentionActivationError,
        match="requires exactly three prior retained events",
    ):
        append_reviewed_recovery_need_observation(store, _recovery_review())


def test_reviewed_recovery_public_review_fails_closed_on_tamper() -> None:
    review = _recovery_review()
    review["evidence"]["value"] = 0.5
    with pytest.raises(
        C2ReviewedRecoveryNeedIntegrationError,
        match="canonical digest mismatch",
    ):
        verify_public_review(review)
