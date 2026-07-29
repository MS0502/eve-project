from __future__ import annotations

import ast
import hashlib
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import core.m3_c_h_dormant_goal_lifecycle_writer as writer_module
import core.m3_c_j_goal_lifecycle_observation_window as window_module
from core.m3_c_b_goal_selection_kernel import (
    ALLOWED_DRIVES,
    DriveSample,
    GoalCandidate,
    select_goal_proposal,
)
from core.m3_c_c_goal_lifecycle_kernel import (
    GoalLifecycleState,
    LifecycleEvidence,
    evaluate_lifecycle_transition,
)
from core.m3_c_d_goal_lifecycle_event_preflight import (
    EVENT_STREAM,
    GoalLifecycleReducerSnapshot,
    build_event_envelope_candidate,
)
from core.m3_c_e_goal_lifecycle_substrate_binding_preflight import (
    build_substrate_binding_candidates,
)
from core.m3_c_h_dormant_goal_lifecycle_writer import (
    DormantGoalLifecycleWriter,
    GoalLifecycleWriterAuthorizationPacket,
    WriterStorageLimits,
    WriterValidationPins,
    database_path_digest,
)
from core.m3_c_j_goal_lifecycle_observation_window import (
    M3CObservationWindowAuthorizationError,
    M3CObservationWindowEvidenceError,
    ObservationWindowAuthorizationPacket,
    ObservationWindowBaseline,
    RollbackPreservationEvidence,
    build_observation_window_authorization_candidate,
    evaluate_observation_window,
    verify_active_observation_window_authorization,
)
from core.sqlite_shadow_store import GENESIS_DIGEST, SQLiteShadowStore, ShadowStoragePolicy

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_j_goal_lifecycle_observation_window.py"
WINDOW_HEAD = "3" * 40
WRITER_HEAD = "1" * 40


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate() -> GoalCandidate:
    return GoalCandidate(
        semantic_goal_id="recover_operating_margin",
        decision_epoch=0,
        evidence_digest=_digest("m3-c-j-observation-evidence"),
        base_value=0.30,
        expected_value=0.0,
        urgency=0.0,
        continuity=0.0,
        cost=0.0,
        risk=0.0,
        drive_alignment={
            drive: {
                "energy": -0.90,
                "safety": -0.80,
                "curiosity": -0.10,
            }.get(drive, 0.0)
            for drive in ALLOWED_DRIVES
        },
        drive_confidence={drive: 1.0 for drive in ALLOWED_DRIVES},
    )


def _samples() -> dict[str, DriveSample]:
    values = {"energy": -0.70, "safety": -0.80, "curiosity": -0.20}
    return {
        drive: DriveSample(
            drive=drive,
            value=values.get(drive, 0.0),
            lower_bound=-1.0,
            upper_bound=1.0,
            sample_digest=_digest(f"m3-c-j-sample:{drive}"),
            replay_elapsed_seconds=0.0,
        )
        for drive in ALLOWED_DRIVES
    }


def _bindings():
    candidate = _candidate()
    selection = select_goal_proposal([candidate], _samples())
    score = selection.scored_candidates[0]
    state = GoalLifecycleState(
        candidate_id=candidate.candidate_id,
        semantic_goal_id=candidate.semantic_goal_id,
        decision_epoch=candidate.decision_epoch,
        evidence_digest=candidate.evidence_digest,
    )
    transitions = []
    for evidence in (
        LifecycleEvidence(candidate_score=score, logical_step=1),
        LifecycleEvidence(
            candidate_score=score,
            logical_step=2,
            validation_status="passed",
        ),
        LifecycleEvidence(candidate_score=score, logical_step=3),
        LifecycleEvidence(
            candidate_score=score,
            logical_step=4,
            selection_receipt=selection,
        ),
    ):
        decision = evaluate_lifecycle_transition(state, evidence)
        assert decision.transition is not None
        transitions.append(decision.transition)
        state = decision.transition.next_state()
    sources = tuple(build_event_envelope_candidate(item) for item in transitions)
    return build_substrate_binding_candidates(sources)


def _policy() -> ShadowStoragePolicy:
    return ShadowStoragePolicy(
        snapshot_interval_events=2,
        max_event_count=16,
        max_event_bytes=1_048_576,
        max_snapshot_count=8,
        max_snapshot_bytes=1_048_576,
        max_backups=2,
    )


def _writer_packet(path: Path, policy: ShadowStoragePolicy):
    return GoalLifecycleWriterAuthorizationPacket(
        validation=WriterValidationPins(
            implementation_head=WRITER_HEAD,
            exact_run=900001,
            focused_passed=16,
            full_passed=3304,
            forward_gate_errors=0,
            artifact_sha256="2" * 64,
            m2e_run=900002,
        ),
        storage_limits=WriterStorageLimits.from_policy(policy),
        database_path_digest=database_path_digest(path),
    )


def _window_packet(
    writer_packet: GoalLifecycleWriterAuthorizationPacket,
    *,
    max_window_events: int = 32,
):
    return ObservationWindowAuthorizationPacket(
        window_implementation_head=WINDOW_HEAD,
        writer_authorization_digest=writer_packet.authorization_digest,
        writer_implementation_head=writer_packet.implementation_head,
        database_path_digest=writer_packet.database_path_digest,
        storage_limits=writer_packet.storage_limits,
        max_window_events=max_window_events,
    )


def _activate_test_pins(
    writer_packet: GoalLifecycleWriterAuthorizationPacket,
    window_packet: ObservationWindowAuthorizationPacket,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        writer_module,
        "_ACTIVE_REVIEWED_IMPLEMENTATION_HEAD",
        writer_packet.implementation_head,
    )
    monkeypatch.setattr(
        writer_module,
        "_ACTIVE_REVIEWED_AUTHORIZATION_DIGEST",
        writer_packet.authorization_digest,
    )
    monkeypatch.setattr(
        writer_module,
        "_ACTIVE_REVIEWED_DATABASE_PATH_DIGEST",
        writer_packet.database_path_digest,
    )
    monkeypatch.setattr(
        window_module,
        "_ACTIVE_REVIEWED_WINDOW_IMPLEMENTATION_HEAD",
        window_packet.window_implementation_head,
    )
    monkeypatch.setattr(
        window_module,
        "_ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST",
        window_packet.authorization_digest,
    )


def _evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    database_path = tmp_path / "private" / "goal-lifecycle.sqlite3"
    policy = _policy()
    writer_packet = _writer_packet(database_path, policy)
    window_packet = _window_packet(writer_packet)
    _activate_test_pins(writer_packet, window_packet, monkeypatch)

    baseline_store = SQLiteShadowStore(database_path, policy=policy)
    baseline_store.initialize()
    baseline_integrity = baseline_store.integrity_check()
    assert baseline_integrity.valid is True
    backup = baseline_store.create_backup(tmp_path / "backups", backup_ordinal=1)
    empty_snapshot = GoalLifecycleReducerSnapshot.empty()
    baseline = ObservationWindowBaseline(
        authorization_digest=window_packet.authorization_digest,
        database_path_digest=writer_packet.database_path_digest,
        start_sequence=0,
        start_event_count=0,
        start_chain_digest=GENESIS_DIGEST,
        start_reducer_snapshot_digest=empty_snapshot.snapshot_digest,
        integrity_report_digest=baseline_integrity.report_digest,
        backup_sha256=backup.backup_sha256,
        backup_path_digest=database_path_digest(backup.backup_path),
    )

    writer = DormantGoalLifecycleWriter(database_path, policy=policy)
    receipts = tuple(
        writer.append(binding, authorization_packet=writer_packet)
        for binding in _bindings()
    )
    assert all(item.production_authoritative_append_performed for item in receipts)
    assert all(not item.disposable_or_test_path_only for item in receipts)
    final_store = SQLiteShadowStore(database_path, policy=policy)
    final_store.initialize()
    final_integrity = final_store.integrity_check()

    restore_path = tmp_path / "restore" / "goal-lifecycle.sqlite3"
    restore_path.parent.mkdir(parents=True)
    shutil.copy2(backup.backup_path, restore_path)
    restored_store = SQLiteShadowStore(restore_path, policy=policy)
    restored_store.initialize()
    restored_integrity = restored_store.integrity_check()
    assert restored_integrity.valid is True
    assert restored_store.events(stream_id=EVENT_STREAM) == ()
    rollback = RollbackPreservationEvidence(
        authorization_digest=window_packet.authorization_digest,
        database_path_digest=writer_packet.database_path_digest,
        backup_sha256=backup.backup_sha256,
        backup_path_digest=database_path_digest(backup.backup_path),
        restore_path_digest=database_path_digest(restore_path),
        pre_window_snapshot_digest=empty_snapshot.snapshot_digest,
        restored_snapshot_digest=empty_snapshot.snapshot_digest,
        restored_integrity_report_digest=restored_integrity.report_digest,
    )
    return window_packet, baseline, receipts, final_integrity, rollback


def _evaluate(evidence):
    packet, baseline, receipts, integrity, rollback = evidence
    return evaluate_observation_window(
        packet,
        baseline=baseline,
        append_receipts=receipts,
        final_integrity_report=integrity,
        final_reducer_snapshot_digest=receipts[-1].reducer_snapshot_digest,
        rollback_evidence=rollback,
    )


def test_candidate_packet_pins_exact_m3_c_i_evidence_without_activation():
    packet = build_observation_window_authorization_candidate(
        window_implementation_head=WINDOW_HEAD,
    )
    assert packet.window_implementation_head == WINDOW_HEAD
    assert packet.prerequisite_exact_head == (
        "bec44a796834e037c41fbb941d090de416cf1e23"
    )
    assert packet.prerequisite_exact_run == 30447974882
    assert packet.prerequisite_focused_passed == 16
    assert packet.prerequisite_full_passed == 3304
    assert packet.prerequisite_artifact_sha256 == (
        "650d11a611b9ae8dcf49fe540b117a26e49fedab5576c366f332eda9d7b92f0f"
    )
    assert packet.prerequisite_m2e_run == 30447974661
    assert packet.prerequisite_merge_sha == (
        "51f682e00059698cbb301a75983e11dd4812f574"
    )
    with pytest.raises(M3CObservationWindowAuthorizationError, match="absent"):
        verify_active_observation_window_authorization(packet)


def test_exact_test_pins_accept_four_event_window_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    receipt = _evaluate(_evidence(tmp_path, monkeypatch))
    assert receipt.observed_event_count == 4
    assert (receipt.first_sequence, receipt.last_sequence) == (1, 4)
    assert receipt.duplicate_acceptance_count == 0
    assert receipt.conflict_acceptance_count == 0
    assert receipt.contiguous_sequences_verified is True
    assert receipt.chain_continuity_verified is True
    assert receipt.direct_replay_equivalent is True
    assert receipt.rollback_preservation_verified is True
    assert receipt.production_append_executed_by_evaluator is False
    assert receipt.runtime_integration_performed is False
    assert receipt.legacy_goal_authority_transferred is False
    assert receipt.m3_e_authority_open is False
    assert len(receipt.receipt_digest) == 64


def test_duplicate_envelope_identity_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    packet, baseline, receipts, integrity, rollback = _evidence(tmp_path, monkeypatch)
    duplicate = replace(
        receipts[1],
        event_envelope_digest=receipts[0].event_envelope_digest,
    )
    with pytest.raises(M3CObservationWindowEvidenceError, match="duplicate event"):
        evaluate_observation_window(
            packet,
            baseline=baseline,
            append_receipts=(receipts[0], duplicate, *receipts[2:]),
            final_integrity_report=integrity,
            final_reducer_snapshot_digest=receipts[-1].reducer_snapshot_digest,
            rollback_evidence=rollback,
        )


def test_chain_discontinuity_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    packet, baseline, receipts, integrity, rollback = _evidence(tmp_path, monkeypatch)
    broken = replace(receipts[1], before_chain_digest="f" * 64)
    with pytest.raises(M3CObservationWindowEvidenceError, match="chain"):
        evaluate_observation_window(
            packet,
            baseline=baseline,
            append_receipts=(receipts[0], broken, *receipts[2:]),
            final_integrity_report=integrity,
            final_reducer_snapshot_digest=receipts[-1].reducer_snapshot_digest,
            rollback_evidence=rollback,
        )


def test_disposable_receipt_is_not_production_window_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    packet, baseline, receipts, integrity, rollback = _evidence(tmp_path, monkeypatch)
    disposable = replace(
        receipts[0],
        disposable_or_test_path_only=True,
        production_authoritative_append_performed=False,
    )
    with pytest.raises(M3CObservationWindowEvidenceError, match="production-path"):
        evaluate_observation_window(
            packet,
            baseline=baseline,
            append_receipts=(disposable, *receipts[1:]),
            final_integrity_report=integrity,
            final_reducer_snapshot_digest=receipts[-1].reducer_snapshot_digest,
            rollback_evidence=rollback,
        )


def test_window_bound_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    packet, baseline, receipts, integrity, rollback = _evidence(tmp_path, monkeypatch)
    bounded = replace(packet, max_window_events=2)
    monkeypatch.setattr(
        window_module,
        "_ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST",
        bounded.authorization_digest,
    )
    with pytest.raises(M3CObservationWindowEvidenceError, match="outside"):
        evaluate_observation_window(
            bounded,
            baseline=replace(
                baseline,
                authorization_digest=bounded.authorization_digest,
            ),
            append_receipts=receipts,
            final_integrity_report=integrity,
            final_reducer_snapshot_digest=receipts[-1].reducer_snapshot_digest,
            rollback_evidence=replace(
                rollback,
                authorization_digest=bounded.authorization_digest,
            ),
        )


def test_final_replay_divergence_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    packet, baseline, receipts, integrity, rollback = _evidence(tmp_path, monkeypatch)
    with pytest.raises(M3CObservationWindowEvidenceError, match="reducer replay"):
        evaluate_observation_window(
            packet,
            baseline=baseline,
            append_receipts=receipts,
            final_integrity_report=integrity,
            final_reducer_snapshot_digest="e" * 64,
            rollback_evidence=rollback,
        )


def test_rollback_restore_must_match_pre_window_snapshot():
    with pytest.raises(M3CObservationWindowEvidenceError, match="snapshot differs"):
        RollbackPreservationEvidence(
            authorization_digest="1" * 64,
            database_path_digest="2" * 64,
            backup_sha256="3" * 64,
            backup_path_digest="4" * 64,
            restore_path_digest="5" * 64,
            pre_window_snapshot_digest="6" * 64,
            restored_snapshot_digest="7" * 64,
            restored_integrity_report_digest="8" * 64,
        )


def test_scope_escape_authorization_is_rejected(tmp_path: Path):
    writer_packet = _writer_packet(tmp_path / "goal.sqlite3", _policy())
    with pytest.raises(M3CObservationWindowAuthorizationError, match="escaped"):
        ObservationWindowAuthorizationPacket(
            window_implementation_head=WINDOW_HEAD,
            writer_authorization_digest=writer_packet.authorization_digest,
            writer_implementation_head=writer_packet.implementation_head,
            database_path_digest=writer_packet.database_path_digest,
            storage_limits=writer_packet.storage_limits,
            action_authorized=True,
        )


def test_packet_and_receipt_are_deterministic_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evidence = _evidence(tmp_path, monkeypatch)
    first = evidence[0]
    second = replace(first)
    assert first == second
    assert first.authorization_digest == second.authorization_digest
    with pytest.raises(FrozenInstanceError):
        first.human_reviewed = False
    receipt = _evaluate(evidence)
    with pytest.raises(FrozenInstanceError):
        receipt.observed_event_count = 0


def test_checked_in_module_is_io_free_and_window_pins_are_absent():
    text = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(MODULE))
    assignments = {}
    imported_modules = set()
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node.value
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    for name in (
        "_ACTIVE_REVIEWED_WINDOW_IMPLEMENTATION_HEAD",
        "_ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST",
    ):
        assert isinstance(assignments[name], ast.Constant)
        assert assignments[name].value is None
    assert not imported_modules & {"os", "pathlib", "shutil", "sqlite3", "subprocess"}
    assert not called & {
        "SQLiteShadowStore",
        "DormantGoalLifecycleWriter",
        "append",
        "create_backup",
        "initialize",
        "open",
        "restore_verified",
    }
    assert "if __name__" not in text
    assert "os.environ" not in text
    assert "sequences 1 through 5" not in text
