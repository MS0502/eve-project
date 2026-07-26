from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from core.m3_b_c2_energy_budget_retention_activation import (
    EVENT_ID,
    PRIOR_EVENT_ENVELOPE_DIGEST,
    SEQUENCE,
    C2EnergyBudgetRetentionActivationError,
    append_reviewed_energy_budget_observation,
)
from core.m3_b_c2_retention_activation import append_reviewed_observation
from core.m3_b_c2_reviewed_energy_budget_integration import (
    ATTESTATION_DIGEST,
    EVIDENCE_DIGEST,
    PUBLIC_REVIEW_DIGEST,
    C2_ENERGY_PRODUCTION_SOURCE_VERIFIERS,
    C2_ENERGY_REVIEWED_OPERATOR_ATTESTATIONS,
    C2_ENERGY_RUNTIME_PROVENANCE_VERIFIERS,
    C2ReviewedEnergyBudgetIntegrationError,
    build_reviewed_capture,
    integration_status,
    verify_energy_budget_source,
    verify_public_review,
    verify_runtime_provenance,
)
from core.sqlite_shadow_store import SQLiteShadowStore

ROOT = Path(__file__).resolve().parents[1]
ENERGY_REVIEW_RECORD = ROOT / "docs/audit/M3_B_C2_REVIEWED_ENERGY_BUDGET_WITNESS.json"
FIRST_REVIEW_RECORD = ROOT / "docs/audit/M3_B_C2_REVIEWED_PHONE_WITNESS.json"


def _energy_review() -> dict:
    payload = json.loads(ENERGY_REVIEW_RECORD.read_text(encoding="utf-8"))
    return payload["public_review"]


def _first_review() -> dict:
    payload = json.loads(FIRST_REVIEW_RECORD.read_text(encoding="utf-8"))
    return payload["public_review"]


def _seed_first_retention(store: SQLiteShadowStore):
    return append_reviewed_observation(store, _first_review())


def test_reviewed_energy_budget_pin_reconstructs_exact_material_and_cumulative_status():
    review = _energy_review()
    attestation, evidence = verify_public_review(review)
    assert attestation.attestation_digest == ATTESTATION_DIGEST
    assert evidence.evidence_digest == EVIDENCE_DIGEST
    assert review["public_review_digest"] == PUBLIC_REVIEW_DIGEST
    assert review["cpu_measurement_methods"] == ["kernel_loadavg_1m_headroom_v1"]
    assert review["memory_measurement_methods"] == ["proc_meminfo_available_v1"]
    assert review["battery_measurement_methods"] == ["termux_api_battery_status_v1"]
    assert review["raw_record_count"] == 3
    assert review["fixture_only"] is False
    assert review["evidence"]["synthetic"] is False
    assert review["evidence"]["confidence"] > 0.0

    assert type(C2_ENERGY_REVIEWED_OPERATOR_ATTESTATIONS) is MappingProxyType
    assert type(C2_ENERGY_RUNTIME_PROVENANCE_VERIFIERS) is MappingProxyType
    assert type(C2_ENERGY_PRODUCTION_SOURCE_VERIFIERS) is MappingProxyType
    assert len(C2_ENERGY_REVIEWED_OPERATOR_ATTESTATIONS) == 1
    assert len(C2_ENERGY_RUNTIME_PROVENANCE_VERIFIERS) == 1
    assert len(C2_ENERGY_PRODUCTION_SOURCE_VERIFIERS) == 1

    status = integration_status(review)
    assert status.reviewed_real_operator_attestation_count == 2
    assert status.registered_runtime_provenance_verifier_count == 2
    assert status.verified_production_runtime_anchor_count == 2
    assert status.registered_production_source_verifier_count == 2
    assert status.verified_positive_confidence_candidate_count == 2
    assert status.retained_real_observation_count == 1
    assert status.observation_window_eligible is False
    assert status.observation_window_started is False
    assert status.m3_b_complete is False
    assert status.m3_c_open is False
    assert status.m3_e_authority_open is False
    assert status.cutover_authorized is False


def test_energy_budget_public_review_tampering_fails_before_verification():
    review = _energy_review()
    tampered = json.loads(json.dumps(review))
    tampered["evidence"]["value"] = 0.25
    with pytest.raises(
        C2ReviewedEnergyBudgetIntegrationError, match="canonical digest mismatch"
    ):
        verify_public_review(tampered)

    tampered = json.loads(json.dumps(review))
    tampered["battery_measurement_methods"] = ["sysfs_capacity_v1"]
    with pytest.raises(
        C2ReviewedEnergyBudgetIntegrationError, match="canonical digest mismatch"
    ):
        verify_public_review(tampered)


def test_energy_budget_runtime_source_and_capture_are_token_issued_and_exactly_bound():
    review = _energy_review()
    runtime = verify_runtime_provenance(review)
    source = verify_energy_budget_source(review, runtime)
    capture = build_reviewed_capture(review)

    assert runtime.counts_as_production is True
    assert runtime.attestation_digest == ATTESTATION_DIGEST
    assert source.counts_as_real is True
    assert source.observation_evidence_digest == EVIDENCE_DIGEST
    assert source.runtime_provenance_verification_digest == runtime.verification_digest
    assert capture.retained_real_observation_eligible is True
    assert capture.retained_real_observation is False
    assert capture.observation_window_started is False

    with pytest.raises(
        C2ReviewedEnergyBudgetIntegrationError,
        match="issued by reviewed energy-budget verifier",
    ):
        replace(runtime, verifier_trace_digest="1" * 64)
    with pytest.raises(
        C2ReviewedEnergyBudgetIntegrationError,
        match="issued by registered energy-budget verifier",
    ):
        replace(source, verifier_trace_digest="2" * 64)


def test_sequence_two_retention_requires_exact_pinned_sequence_one_and_opens_no_authority(
    tmp_path,
):
    store = SQLiteShadowStore(tmp_path / "retained.sqlite3")
    store.initialize()

    with pytest.raises(
        C2EnergyBudgetRetentionActivationError,
        match="requires exactly the prior sequence-1 event",
    ):
        append_reviewed_energy_budget_observation(store, _energy_review())

    first = _seed_first_retention(store)
    assert first.event_envelope_digest == PRIOR_EVENT_ENVELOPE_DIGEST
    assert first.store_before_count == 0
    assert first.store_after_count == 1

    receipt = append_reviewed_energy_budget_observation(store, _energy_review())
    assert receipt.event_id == EVENT_ID
    assert receipt.sequence == SEQUENCE
    assert receipt.store_ordinal == 2
    assert receipt.store_before_count == 1
    assert receipt.store_after_count == 2
    assert receipt.retained_real_observation_delta == 1
    assert receipt.retained_real_observation_count_after_append == 2
    assert receipt.readback_verified is True
    assert receipt.observation_window_started is False
    assert receipt.m3_b_complete is False
    assert receipt.m3_c_open is False
    assert receipt.m3_e_authority_open is False
    assert receipt.cutover_authorized is False

    with pytest.raises(
        C2EnergyBudgetRetentionActivationError,
        match="requires exactly the prior sequence-1 event",
    ):
        append_reviewed_energy_budget_observation(store, _energy_review())
    with pytest.raises(
        C2EnergyBudgetRetentionActivationError,
        match="issued by durable append",
    ):
        replace(receipt, store_after_chain_digest="3" * 64)


def test_sequence_two_retention_rejects_wrong_prior_history(tmp_path):
    store = SQLiteShadowStore(tmp_path / "retained.sqlite3")
    store.initialize()
    first = _seed_first_retention(store)
    assert first.readback_verified is True

    events = store.events()
    assert len(events) == 1
    assert events[0].digest == PRIOR_EVENT_ENVELOPE_DIGEST

    wrong_root = SQLiteShadowStore(tmp_path / "wrong.sqlite3")
    wrong_root.initialize()
    with pytest.raises(
        C2EnergyBudgetRetentionActivationError,
        match="requires exactly the prior sequence-1 event",
    ):
        append_reviewed_energy_budget_observation(wrong_root, _energy_review())


def test_operator_sequence_two_script_executes_against_same_private_store(tmp_path):
    first_review_path = tmp_path / "first-public-review.json"
    first_review_path.write_text(
        json.dumps(
            _first_review(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )
    energy_review_path = tmp_path / "energy-public-review.json"
    energy_review_path.write_text(
        json.dumps(
            _energy_review(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )
    private_root = tmp_path / "private-retention"
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    subprocess.run(
        [
            sys.executable,
            "scripts/operator/m3_b_c2_retain_reviewed_prediction_error.py",
            "--public-review-file",
            str(first_review_path),
            "--private-root",
            str(private_root),
            "--expected-head",
            head,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/operator/m3_b_c2_retain_reviewed_energy_budget.py",
            "--public-review-file",
            str(energy_review_path),
            "--private-root",
            str(private_root),
            "--expected-head",
            head,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    public = json.loads(result.stdout)
    assert public["activation_repository_head_sha"] == head
    assert public["database_location"] == "operator_private_companion_only"
    assert public["retained_real_observation_count_after_append"] == 2
    assert public["receipt"]["event_id"] == EVENT_ID
    assert public["receipt"]["sequence"] == 2
    assert public["receipt"]["store_before_count"] == 1
    assert public["receipt"]["store_after_count"] == 2
    assert public["receipt"]["readback_verified"] is True
    assert public["receipt"]["observation_window_started"] is False
    assert public["receipt"]["cutover_authorized"] is False
    assert len(public["receipt_digest"]) == 64

    second = subprocess.run(
        [
            sys.executable,
            "scripts/operator/m3_b_c2_retain_reviewed_energy_budget.py",
            "--public-review-file",
            str(energy_review_path),
            "--private-root",
            str(private_root),
            "--expected-head",
            head,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert second.returncode != 0
    assert "requires exactly the prior sequence-1 event" in second.stderr
