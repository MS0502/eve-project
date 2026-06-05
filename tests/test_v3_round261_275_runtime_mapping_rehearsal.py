"""EVE v3 rounds261-275 — isolated no-persistence runtime-mapping rehearsal."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.operator_plan_runtime_mapping_acceptance_handoff as handoff
import scripts.operator_rehearse_runtime_mapping_no_persistence as rehearsal


def _green_suite_report() -> dict:
    mapping_measurement = {
        "target_word": "민석",
        "negative_token": "EVE",
        "target_would_map": True,
        "target_precheck_ready": True,
        "target_mapped_in_controlled_smoke": True,
        "negative_token_blocked": True,
        "rollback_complete": True,
        "runtime_mapping_enabled_after": False,
        "enforcement_enabled_after": False,
        "runtime_mapping_persisted": False,
        "gate": {"would_map_tokens": ["민석"], "blocked_tokens": ["EVE"]},
        "precheck": {"ready_tokens": ["민석"]},
        "controlled_smoke": {"mapped_tokens": ["민석"], "rollback": {"rollback_complete": True}},
    }
    return {
        "status": "operator_local_validation_suite_green",
        "success": True,
        "exit_code": 0,
        "operator_prompt_summary": {
            "exit_code": 0,
            "status": "operator_local_validation_suite_green",
            "target_word": "민석",
            "selected_cluster_id": "concept_runtime_mapping_after_eve_self_learning_guarded_medium30k",
            "vector_lookup_after_commit": True,
            "wrapper_primary_loaded": True,
            "git_status_safety_safe": True,
            "production_persistence_enabled": False,
            "runtime_mapping_enabled_default": False,
        },
        "steps": [{}, {}, {"parsed_json": {"mapping_measurement": mapping_measurement}}],
    }


def _handoff_packet() -> dict:
    report = handoff.build_round236_260_report(_green_suite_report())
    assert report["success"] is True
    return report


def test_round261_275_authorization_guard_blocks_before_rehearsal_scope() -> None:
    report = rehearsal.run_isolated_no_persistence_rehearsal(_handoff_packet())

    assert report["version"] == rehearsal.ROUND261_275_VERSION
    assert report["rounds_completed"] == list(range(261, 276))
    assert report["success"] is False
    assert report["rehearsal_executed"] is False
    assert report["authorization_guard"]["authorized"] is False
    assert "authorization_authorization_token_ok" in report["blockers"]
    assert "authorization_operator_authorized_flag" in report["blockers"]
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False
    assert report["no_persistence_proof"]["runtime_mapping_persisted"] is False


def test_round261_275_authorized_rehearsal_rolls_back_and_keeps_no_persistence() -> None:
    report = rehearsal.run_isolated_no_persistence_rehearsal(
        _handoff_packet(),
        operator_authorized=True,
        authorization_token=rehearsal.AUTHORIZATION_TOKEN,
    )

    assert report["success"] is True
    assert report["status"] == "no_persistence_runtime_mapping_rehearsal_green"
    smoke = report["round268_271_rehearsal_smoke"]
    assert smoke["runtime_mapping_enabled_before"] is False
    assert smoke["runtime_mapping_enabled_during_smoke"] is True
    assert smoke["runtime_mapping_enabled_after_rollback"] is False
    assert smoke["enforcement_enabled_during_smoke"] is False
    assert smoke["enforcement_enabled_after_rollback"] is False
    assert smoke["mapped_tokens"] == ["민석"]
    assert smoke["blocked_tokens"] == []
    assert smoke["mutation_checks"]["runtime_mapping_persisted"] is False
    assert report["rollback_proof"]["rollback_verified"] is True
    assert report["no_persistence_proof"]["passed"] is True
    assert report["runtime_mapping_enabled_default"] is False
    assert report["production_persistence_enabled"] is False
    assert report["enforcement_enabled"] is False


def test_round261_275_rehearsal_does_not_mutate_vectors_or_call_embedding_lookup() -> None:
    report = rehearsal.run_isolated_no_persistence_rehearsal(
        _handoff_packet(),
        operator_authorized=True,
        authorization_token=rehearsal.AUTHORIZATION_TOKEN,
    )

    vector_proof = report["no_vector_mutation_proof"]
    assert vector_proof["passed"] is True
    checks = vector_proof["checks"]
    assert checks["vector_store_stats_unchanged"] is True
    assert checks["wrapper_telemetry_unchanged"] is True
    assert checks["eve_specific_vector_commit_not_called"] is True
    assert checks["fasttext_seed_not_mutated"] is True
    assert checks["dummy_vectors_not_created"] is True
    smoke_checks = report["round268_271_rehearsal_smoke"]["mutation_checks"]
    assert smoke_checks["embedding_lookup_called_during_enable_smoke"] is False
    assert smoke_checks["eve_specific_vector_commit_called"] is False


def test_round261_275_rehearsal_preserves_korean_fixtures_and_minsok_exactly() -> None:
    report = rehearsal.run_isolated_no_persistence_rehearsal(
        _handoff_packet(),
        operator_authorized=True,
        authorization_token=rehearsal.AUTHORIZATION_TOKEN,
    )

    preservation = report["korean_fixture_preservation"]
    assert preservation["preserved"] is True
    assert preservation["contains_minsok_exact"] is True
    assert preservation["contains_project_fixture"] is True
    assert preservation["before"]["texts"] == preservation["after"]["texts"]
    assert preservation["before"]["categories"] == preservation["after"]["categories"]
    assert "EVE 프로젝트" in preservation["after"]["texts"]
    assert "minsok" in preservation["after"]["categories"]


def test_round261_275_rehearsal_uses_handoff_packet_and_keeps_blocked_control_out_of_smoke() -> None:
    report = rehearsal.run_isolated_no_persistence_rehearsal(
        _handoff_packet(),
        operator_authorized=True,
        authorization_token=rehearsal.AUTHORIZATION_TOKEN,
    )

    assert report["source_handoff"]["handoff_version"] == handoff.ROUND236_260_VERSION
    assert report["source_handoff"]["accepted_tokens"] == ["민석"]
    assert report["source_handoff"]["blocked_control_tokens"] == ["EVE"]
    precheck = report["precheck"]
    assert precheck["ready_tokens"] == ["민석"]
    assert precheck["blocked_tokens"] == ["EVE"]
    assert report["round268_271_rehearsal_smoke"]["mapped_tokens"] == ["민석"]
    assert "EVE" not in report["round268_271_rehearsal_smoke"]["mapped_tokens"]
    assert "EVE_blocked_control_requires_concept_SA_AGP_evidence" in report["round275_operator_handoff"]["remaining_taxonomy"]


def test_round261_275_cli_requires_explicit_authorization(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(_handoff_packet(), ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "scripts/operator_rehearse_runtime_mapping_no_persistence.py", "--handoff-json", str(handoff_path)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["success"] is False
    assert payload["rehearsal_executed"] is False
    assert "authorization_operator_authorized_flag" in payload["blockers"]
    assert "authorization_authorization_token_ok" in payload["blockers"]


def test_round261_275_cli_writes_json_only_when_authorized(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.json"
    output_path = tmp_path / "rehearsal.json"
    handoff_path.write_text(json.dumps(_handoff_packet(), ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/operator_rehearse_runtime_mapping_no_persistence.py",
            "--handoff-json",
            str(handoff_path),
            "--operator-authorized",
            "--authorization-token",
            rehearsal.AUTHORIZATION_TOKEN,
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert payload["no_persistence_proof"]["passed"] is True
    assert payload["no_vector_mutation_proof"]["passed"] is True
    assert output_path.suffix == ".json"
