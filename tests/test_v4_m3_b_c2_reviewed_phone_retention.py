from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from core.m3_b_c2_reviewed_phone_integration import (
    ATTESTATION_DIGEST,
    EVIDENCE_DIGEST,
    PUBLIC_REVIEW_DIGEST,
    C2_PRODUCTION_SOURCE_VERIFIERS,
    C2_REVIEWED_OPERATOR_ATTESTATIONS,
    C2_RUNTIME_PROVENANCE_VERIFIERS,
    C2ReviewedPhoneIntegrationError,
    build_reviewed_capture,
    integration_status,
    verify_prediction_error_source,
    verify_public_review,
    verify_runtime_provenance,
)
from core.m3_b_c2_retention_activation import (
    C2RetentionActivationError,
    append_reviewed_observation,
)
from core.sqlite_shadow_store import SQLiteShadowStore

ROOT = Path(__file__).resolve().parents[1]
REVIEW_RECORD = ROOT / "docs/audit/M3_B_C2_REVIEWED_PHONE_WITNESS.json"


def _review() -> dict:
    payload = json.loads(REVIEW_RECORD.read_text(encoding="utf-8"))
    return payload["public_review"]


def test_reviewed_phone_pin_reconstructs_exact_attestation_evidence_and_status():
    review = _review()
    attestation, evidence = verify_public_review(review)
    assert attestation.attestation_digest == ATTESTATION_DIGEST
    assert evidence.evidence_digest == EVIDENCE_DIGEST
    assert review["public_review_digest"] == PUBLIC_REVIEW_DIGEST
    assert type(C2_REVIEWED_OPERATOR_ATTESTATIONS) is MappingProxyType
    assert type(C2_RUNTIME_PROVENANCE_VERIFIERS) is MappingProxyType
    assert type(C2_PRODUCTION_SOURCE_VERIFIERS) is MappingProxyType
    assert len(C2_REVIEWED_OPERATOR_ATTESTATIONS) == 1
    assert len(C2_RUNTIME_PROVENANCE_VERIFIERS) == 1
    assert len(C2_PRODUCTION_SOURCE_VERIFIERS) == 1

    status = integration_status(review)
    assert status.reviewed_real_operator_attestation_count == 1
    assert status.registered_runtime_provenance_verifier_count == 1
    assert status.verified_production_runtime_anchor_count == 1
    assert status.registered_production_source_verifier_count == 1
    assert status.verified_positive_confidence_candidate_count == 1
    assert status.retained_real_observation_count == 0
    assert status.observation_window_started is False
    assert status.m3_b_complete is False
    assert status.m3_c_open is False
    assert status.m3_e_authority_open is False
    assert status.cutover_authorized is False


def test_tampered_public_review_fails_before_any_verification():
    review = _review()
    tampered = json.loads(json.dumps(review))
    tampered["evidence"]["value"] = 0.25
    with pytest.raises(C2ReviewedPhoneIntegrationError, match="canonical digest mismatch"):
        verify_public_review(tampered)


def test_c2_runtime_and_source_verification_are_token_issued_and_exactly_bound():
    review = _review()
    runtime = verify_runtime_provenance(review)
    source = verify_prediction_error_source(review, runtime)
    capture = build_reviewed_capture(review)

    assert runtime.counts_as_production is True
    assert runtime.attestation_digest == ATTESTATION_DIGEST
    assert source.counts_as_real is True
    assert source.observation_evidence_digest == EVIDENCE_DIGEST
    assert source.runtime_provenance_verification_digest == runtime.verification_digest
    assert capture.retained_real_observation_eligible is True
    assert capture.retained_real_observation is False
    assert capture.observation_window_started is False

    with pytest.raises(C2ReviewedPhoneIntegrationError, match="issued by C2 reviewed verifier"):
        replace(runtime, verifier_trace_digest="1" * 64)
    with pytest.raises(C2ReviewedPhoneIntegrationError, match="issued by C2 registered verifier"):
        replace(source, verifier_trace_digest="2" * 64)


def test_disposable_ci_retention_proves_exactly_one_append_but_opens_no_authority(tmp_path):
    review = _review()
    store = SQLiteShadowStore(tmp_path / "retained.sqlite3")
    store.initialize()

    receipt = append_reviewed_observation(store, review)
    assert receipt.retained_real_observation_delta == 1
    assert receipt.store_before_count == 0
    assert receipt.store_after_count == 1
    assert receipt.readback_verified is True
    assert receipt.observation_window_started is False
    assert receipt.m3_b_complete is False
    assert receipt.m3_c_open is False
    assert receipt.m3_e_authority_open is False
    assert receipt.cutover_authorized is False

    with pytest.raises(C2RetentionActivationError, match="already non-empty"):
        append_reviewed_observation(store, review)

    with pytest.raises(C2RetentionActivationError, match="issued by durable append"):
        replace(receipt, store_after_chain_digest="3" * 64)


def test_operator_retention_script_executes_as_real_script_entrypoint(tmp_path):
    review_path = tmp_path / "public-review.json"
    review_path.write_text(
        json.dumps(_review(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    private_root = tmp_path / "private-retention"
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/operator/m3_b_c2_retain_reviewed_prediction_error.py",
            "--public-review-file",
            str(review_path),
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
    assert public["retained_real_observation_count_after_append"] == 1
    assert public["receipt"]["readback_verified"] is True
    assert public["receipt"]["observation_window_started"] is False
    assert public["receipt"]["cutover_authorized"] is False
    assert len(public["receipt_digest"]) == 64

    second = subprocess.run(
        [
            sys.executable,
            "scripts/operator/m3_b_c2_retain_reviewed_prediction_error.py",
            "--public-review-file",
            str(review_path),
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
    assert "already non-empty" in second.stderr
