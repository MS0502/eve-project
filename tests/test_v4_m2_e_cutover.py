from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from core.m2_e_cutover import (
    ACCEPTED_M2_D_PACKET_DIGEST,
    BOUNDED_STATE_SCHEMA,
    BOUNDED_STREAM,
    EVENT_STORE_BOUNDED_AUTHORITY,
    EVENT_STORE_SHADOW_AUTHORITY,
    LEGACY_EVIDENCE_MODE,
    M2ECutoverError,
    HumanCutoverDecision,
    authorize_cutover,
    evaluate_cutover_candidate,
    record_post_cutover_observation,
    scenario_ids,
)
from scripts.audit.m2_d_rehearsal import generate_packet

ROOT = Path(__file__).resolve().parents[1]
EMPTY = {"calls": [], "learned": []}
PAIR_ONE = ["alpha", "beta", 0.4]
PAIR_TWO = ["gamma", "delta", 0.6]
AFTER_TWO = {"calls": [PAIR_ONE, PAIR_TWO], "learned": [PAIR_ONE, PAIR_TWO]}


@pytest.fixture(scope="module")
def accepted_m2_d_record(tmp_path_factory: pytest.TempPathFactory) -> dict:
    workspace = tmp_path_factory.mktemp("m2-e-prerequisite") / "rehearsal"
    packet = generate_packet(workspace)
    assert packet.packet_digest == ACCEPTED_M2_D_PACKET_DIGEST
    return packet.canonical_record


def _accepted_decision(candidate_packet_digest: str) -> HumanCutoverDecision:
    return HumanCutoverDecision.create(
        decision_id="m2-e:human-decision:test",
        candidate_head="1" * 40,
        workflow=123456789,
        artifact_sha256="2" * 64,
        candidate_packet_digest=candidate_packet_digest,
        reviewer="project-owner",
        accepted=True,
    )


def test_candidate_reuses_exact_accepted_m2_d_packet(accepted_m2_d_record: dict) -> None:
    candidate = evaluate_cutover_candidate(accepted_m2_d_record)
    assert candidate.machine_passed is True
    assert candidate.eligible_for_human_review is True
    assert candidate.human_accepted is False
    assert candidate.cutover_authorized is False
    assert candidate.event_store_authority == EVENT_STORE_SHADOW_AUTHORITY
    assert candidate.legacy_sidecars_read_only is False
    assert candidate.runtime_integrated is False
    assert candidate.production_defaults_changed is False
    assert candidate.observations["m2_d_packet_digest"] == ACCEPTED_M2_D_PACKET_DIGEST
    assert scenario_ids(candidate.canonical_record) == (
        "snapshot_restore",
        "full_replay_equivalence",
        "corrupt_snapshot_fallback",
        "corrupt_event_fail_closed",
        "forced_termination",
        "rollback_rehearsal",
    )


def test_candidate_is_deterministic(accepted_m2_d_record: dict) -> None:
    first = evaluate_cutover_candidate(accepted_m2_d_record)
    second = evaluate_cutover_candidate(json.loads(json.dumps(accepted_m2_d_record)))
    assert first.canonical_record == second.canonical_record
    assert first.packet_digest == second.packet_digest


def test_tampered_m2_d_packet_is_not_eligible(accepted_m2_d_record: dict) -> None:
    tampered = json.loads(json.dumps(accepted_m2_d_record))
    tampered["packet_digest"] = "0" * 64
    candidate = evaluate_cutover_candidate(tampered)
    assert candidate.machine_passed is False
    assert candidate.eligible_for_human_review is False
    assert candidate.checks["accepted_m2_d_packet_digest"] is False


def test_candidate_value_cannot_self_promote(accepted_m2_d_record: dict) -> None:
    candidate = evaluate_cutover_candidate(accepted_m2_d_record)
    with pytest.raises(M2ECutoverError, match="cannot self-promote"):
        replace(candidate, human_accepted=True, cutover_authorized=True)


def test_external_human_decision_is_digest_bound(accepted_m2_d_record: dict) -> None:
    candidate = evaluate_cutover_candidate(accepted_m2_d_record)
    decision = _accepted_decision(candidate.packet_digest)
    assert decision.human_accepted is True
    assert decision.cutover_authorized is True
    assert decision.event_store_authority == EVENT_STORE_BOUNDED_AUTHORITY
    assert decision.legacy_sidecars_mode == LEGACY_EVIDENCE_MODE
    with pytest.raises(M2ECutoverError, match="digest mismatch"):
        replace(decision, decision_digest="0" * 64)


def test_rejected_decision_cannot_authorize_cutover(accepted_m2_d_record: dict) -> None:
    candidate = evaluate_cutover_candidate(accepted_m2_d_record)
    rejected = HumanCutoverDecision.create(
        decision_id="m2-e:human-decision:rejected",
        candidate_head="3" * 40,
        workflow=123456790,
        artifact_sha256="4" * 64,
        candidate_packet_digest=candidate.packet_digest,
        reviewer="project-owner",
        accepted=False,
    )
    with pytest.raises(M2ECutoverError, match="did not authorize"):
        authorize_cutover(candidate, rejected)


def test_authorization_is_exact_and_bounded(accepted_m2_d_record: dict) -> None:
    candidate = evaluate_cutover_candidate(accepted_m2_d_record)
    authorization = authorize_cutover(candidate, _accepted_decision(candidate.packet_digest))
    assert authorization.stream_id == BOUNDED_STREAM
    assert authorization.state_schema == BOUNDED_STATE_SCHEMA
    assert authorization.event_store_authority == EVENT_STORE_BOUNDED_AUTHORITY
    assert authorization.legacy_sidecars_mode == LEGACY_EVIDENCE_MODE
    assert authorization.authoritative_recovery is True
    assert authorization.rollback_available is True
    assert authorization.runtime_integrated is False
    assert authorization.production_defaults_changed is False


def test_decision_for_another_candidate_is_rejected(accepted_m2_d_record: dict) -> None:
    candidate = evaluate_cutover_candidate(accepted_m2_d_record)
    wrong = _accepted_decision("5" * 64)
    with pytest.raises(M2ECutoverError, match="does not pin"):
        authorize_cutover(candidate, wrong)


def test_post_cutover_observation_proves_replay_and_rollback(
    accepted_m2_d_record: dict,
) -> None:
    candidate = evaluate_cutover_candidate(accepted_m2_d_record)
    authorization = authorize_cutover(candidate, _accepted_decision(candidate.packet_digest))
    evidence = record_post_cutover_observation(
        authorization=authorization,
        event_count=2,
        before_state=EMPTY,
        authoritative_state=AFTER_TWO,
        replay_state=AFTER_TWO,
        rollback_state=EMPTY,
    )
    assert all(evidence.checks.values())
    assert evidence.event_count == 2
    assert evidence.stream_id == BOUNDED_STREAM
    assert evidence.state_schema == BOUNDED_STATE_SCHEMA
    assert evidence.event_store_authority == EVENT_STORE_BOUNDED_AUTHORITY
    assert evidence.rollback_available is True


def test_post_cutover_replay_mismatch_fails_closed(accepted_m2_d_record: dict) -> None:
    candidate = evaluate_cutover_candidate(accepted_m2_d_record)
    authorization = authorize_cutover(candidate, _accepted_decision(candidate.packet_digest))
    with pytest.raises(M2ECutoverError, match="did not pass"):
        record_post_cutover_observation(
            authorization=authorization,
            event_count=2,
            before_state=EMPTY,
            authoritative_state=AFTER_TWO,
            replay_state=EMPTY,
            rollback_state=EMPTY,
        )


def test_cli_reuses_existing_m2_d_json(
    accepted_m2_d_record: dict,
    tmp_path: Path,
) -> None:
    source = tmp_path / "m2-d.json"
    output = tmp_path / "m2-e.json"
    source.write_text(json.dumps(accepted_m2_d_record), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit/m2_e_cutover.py",
            "--m2-d-packet",
            str(source),
            "--output",
            str(output),
            "--pretty",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["machine_passed"] is True
    assert record["human_accepted"] is False
    assert record["cutover_authorized"] is False
    assert record["event_store_authority"] == EVENT_STORE_SHADOW_AUTHORITY


def test_module_has_no_runtime_io_or_automatic_activation_surface() -> None:
    path = ROOT / "core/m2_e_cutover.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported.isdisjoint(
        {
            "main",
            "language",
            "sqlite3",
            "pathlib",
            "pickle",
            "gzip",
            "random",
            "time",
            "threading",
            "subprocess",
        }
    )
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint({"open", "write_text", "write_bytes", "connect", "unlink", "replace"})
