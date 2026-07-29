from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.event_kernel import EventEnvelope
from core.m2_e_cutover_activation import (
    build_operational_rollback_record,
    resolve_cutover_authority,
)
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
    build_event_envelope_candidate,
)
from core.m3_c_e_goal_lifecycle_substrate_binding_preflight import (
    build_substrate_binding_candidates,
)
from core.m3_c_g_disposable_sqlite_rehearsal import (
    FORWARD_DATABASE_NAME,
    PREREQUISITE_PR,
    RESTORED_DATABASE_NAME,
    M3CDisposableSQLiteRehearsalError,
    run_disposable_sqlite_rehearsal,
)
from core.sqlite_shadow_store import (
    AppendOnlyViolation,
    SQLiteShadowStore,
    ShadowStoragePolicy,
    StoragePolicyExceeded,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_g_disposable_sqlite_rehearsal.py"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate() -> GoalCandidate:
    return GoalCandidate(
        semantic_goal_id="recover_operating_margin",
        decision_epoch=0,
        evidence_digest=_digest("m3-c-g-recovery-evidence"),
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
            sample_digest=_digest(f"m3-c-g-sample:{drive}"),
            replay_elapsed_seconds=0.0,
        )
        for drive in ALLOWED_DRIVES
    }


def _sources():
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
    return tuple(build_event_envelope_candidate(item) for item in transitions)


def _paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "forward" / FORWARD_DATABASE_NAME,
        root / "restored" / RESTORED_DATABASE_NAME,
    )


def _policy() -> ShadowStoragePolicy:
    return ShadowStoragePolicy(
        snapshot_interval_events=2,
        max_event_count=4,
        max_event_bytes=1_048_576,
        max_snapshot_count=1,
        max_snapshot_bytes=1_048_576,
        max_backups=1,
    )


def test_disposable_sqlite_rehearsal_proves_full_boundary(tmp_path: Path):
    receipt = run_disposable_sqlite_rehearsal(
        _sources(),
        rehearsal_root=tmp_path,
        checkpoint_sequence=2,
    )
    forward_path, restored_path = _paths(tmp_path)
    assert forward_path.is_file()
    assert restored_path.is_file()
    assert forward_path != restored_path
    assert receipt.prerequisite_pr == PREREQUISITE_PR == 223
    assert receipt.appended_event_count == 4
    assert receipt.forward_event_count == 4
    assert receipt.restored_event_count == 2
    assert receipt.forward_direct_snapshot_digest == receipt.forward_sqlite_snapshot_digest
    assert receipt.forward_direct_snapshot_digest == receipt.snapshot_suffix_snapshot_digest
    assert receipt.checkpoint_snapshot_digest == receipt.restored_checkpoint_snapshot_digest
    assert receipt.one_event_per_append_call is True
    assert receipt.precommit_readback_verified is True
    assert receipt.postcommit_readback_verified is True
    assert receipt.event_chain_verified is True
    assert receipt.rollback_checkpoint_restored is True
    assert receipt.disposable_sqlite_write_performed is True
    assert receipt.production_authoritative_append_performed is False
    assert receipt.live_writer_installed is False
    assert receipt.production_integration_performed is False
    assert receipt.legacy_goal_authority_transferred is False
    assert receipt.m3_e_authority_open is False
    assert receipt.writer_operationally_enabled is False


def test_forward_and_restored_databases_reopen_with_verified_counts(tmp_path: Path):
    run_disposable_sqlite_rehearsal(
        _sources(),
        rehearsal_root=tmp_path,
        checkpoint_sequence=2,
    )
    forward_path, restored_path = _paths(tmp_path)
    forward = SQLiteShadowStore(forward_path, policy=_policy())
    restored = SQLiteShadowStore(restored_path, policy=_policy())
    forward.initialize()
    restored.initialize()
    assert len(forward.events(stream_id=EVENT_STREAM)) == 4
    assert len(restored.events(stream_id=EVENT_STREAM)) == 2
    assert forward.integrity_check().valid is True
    assert restored.integrity_check().valid is True


def test_rehearsal_uses_exactly_one_envelope_per_store_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    original = SQLiteShadowStore.append_many
    observed_batch_sizes: list[int] = []

    def recording_append_many(self, envelopes):
        values = tuple(envelopes)
        observed_batch_sizes.append(len(values))
        return original(self, values)

    monkeypatch.setattr(SQLiteShadowStore, "append_many", recording_append_many)
    run_disposable_sqlite_rehearsal(
        _sources(),
        rehearsal_root=tmp_path,
        checkpoint_sequence=2,
    )
    assert observed_batch_sizes == [1, 1, 1, 1]


def test_rolled_back_m2e_authority_refuses_before_any_file_is_created(
    tmp_path: Path,
):
    rollback = build_operational_rollback_record(
        requested_by="m3-c-g-test",
        reason="bounded-test",
    )
    rolled_back = resolve_cutover_authority(rollback)
    with pytest.raises(ValueError, match="not active"):
        run_disposable_sqlite_rehearsal(
            _sources(),
            rehearsal_root=tmp_path,
            checkpoint_sequence=2,
            authority_state=rolled_back,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("checkpoint", [False, 0, 4, 5])
def test_checkpoint_must_be_inside_synthetic_chain(
    tmp_path: Path,
    checkpoint,
):
    with pytest.raises(
        M3CDisposableSQLiteRehearsalError,
        match="non-empty suffix",
    ):
        run_disposable_sqlite_rehearsal(
            _sources(),
            rehearsal_root=tmp_path,
            checkpoint_sequence=checkpoint,
        )
    assert list(tmp_path.iterdir()) == []


def test_root_must_be_explicit_absolute_and_caller_created(tmp_path: Path):
    with pytest.raises(M3CDisposableSQLiteRehearsalError, match="absolute"):
        run_disposable_sqlite_rehearsal(
            _sources(),
            rehearsal_root=Path("relative-rehearsal"),
            checkpoint_sequence=2,
        )
    missing = tmp_path / "missing"
    with pytest.raises(M3CDisposableSQLiteRehearsalError, match="caller-created"):
        run_disposable_sqlite_rehearsal(
            _sources(),
            rehearsal_root=missing,
            checkpoint_sequence=2,
        )
    assert not missing.exists()


def test_preexisting_target_is_never_reused_or_overwritten(tmp_path: Path):
    forward_path, _restored_path = _paths(tmp_path)
    forward_path.parent.mkdir()
    forward_path.write_bytes(b"do-not-overwrite")
    with pytest.raises(M3CDisposableSQLiteRehearsalError, match="must not already exist"):
        run_disposable_sqlite_rehearsal(
            _sources(),
            rehearsal_root=tmp_path,
            checkpoint_sequence=2,
        )
    assert forward_path.read_bytes() == b"do-not-overwrite"


def test_duplicate_rehearsal_event_conflicts_without_partial_write(tmp_path: Path):
    run_disposable_sqlite_rehearsal(
        _sources(),
        rehearsal_root=tmp_path,
        checkpoint_sequence=2,
    )
    forward_path, _restored_path = _paths(tmp_path)
    store = SQLiteShadowStore(forward_path, policy=_policy())
    store.initialize()
    first = build_substrate_binding_candidates(_sources())[0].event_envelope
    with pytest.raises(AppendOnlyViolation, match="duplicate event_id"):
        store.append(first)
    assert len(store.events(stream_id=EVENT_STREAM)) == 4


def test_unknown_causation_conflicts_without_partial_write(tmp_path: Path):
    run_disposable_sqlite_rehearsal(
        _sources(),
        rehearsal_root=tmp_path,
        checkpoint_sequence=2,
    )
    forward_path, _restored_path = _paths(tmp_path)
    store = SQLiteShadowStore(forward_path, policy=_policy())
    store.initialize()
    source = build_substrate_binding_candidates(_sources())[0].event_envelope
    conflict = EventEnvelope.create(
        event_id="m3c:goal-lifecycle:unknown-causation-test",
        event_type=source.event_type,
        stream_id=source.stream_id,
        sequence=5,
        producer=source.producer,
        producer_version=source.producer_version,
        correlation_id=source.correlation_id,
        causation_id="m3c:goal-lifecycle:missing-cause",
        payload=source.payload,
        causal_context=source.causal_context,
    )
    with pytest.raises(AppendOnlyViolation, match="unknown causation"):
        store.append(conflict)
    assert len(store.events(stream_id=EVENT_STREAM)) == 4


def test_bounded_capacity_failure_preserves_existing_history(tmp_path: Path):
    run_disposable_sqlite_rehearsal(
        _sources(),
        rehearsal_root=tmp_path,
        checkpoint_sequence=2,
    )
    forward_path, _restored_path = _paths(tmp_path)
    store = SQLiteShadowStore(forward_path, policy=_policy())
    store.initialize()
    bindings = build_substrate_binding_candidates(_sources())
    last = bindings[-1].event_envelope
    overflow = EventEnvelope.create(
        event_id="m3c:goal-lifecycle:capacity-test",
        event_type=last.event_type,
        stream_id=last.stream_id,
        sequence=5,
        producer=last.producer,
        producer_version=last.producer_version,
        correlation_id=last.correlation_id,
        causation_id=last.event_id,
        payload=last.payload,
        causal_context=last.causal_context,
    )
    with pytest.raises(StoragePolicyExceeded, match="bounded storage policy"):
        store.append(overflow)
    assert len(store.events(stream_id=EVENT_STREAM)) == 4
    assert store.integrity_check().valid is True


def test_failure_after_forward_write_preserves_database_for_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    original = SQLiteShadowStore.restore_verified

    def fail_forward_restore(self, **kwargs):
        if self.database_path.name == FORWARD_DATABASE_NAME:
            raise RuntimeError("injected post-commit replay mismatch")
        return original(self, **kwargs)

    monkeypatch.setattr(
        SQLiteShadowStore,
        "restore_verified",
        fail_forward_restore,
    )
    with pytest.raises(RuntimeError, match="post-commit replay mismatch"):
        run_disposable_sqlite_rehearsal(
            _sources(),
            rehearsal_root=tmp_path,
            checkpoint_sequence=2,
        )
    forward_path, _restored_path = _paths(tmp_path)
    assert forward_path.is_file()
    store = SQLiteShadowStore(forward_path, policy=_policy())
    store.initialize()
    assert len(store.events(stream_id=EVENT_STREAM)) == 4
    assert store.integrity_check().valid is True


def test_receipt_is_deterministic_and_immutable(tmp_path: Path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = run_disposable_sqlite_rehearsal(
        _sources(),
        rehearsal_root=first_root,
        checkpoint_sequence=2,
    )
    second = run_disposable_sqlite_rehearsal(
        _sources(),
        rehearsal_root=second_root,
        checkpoint_sequence=2,
    )
    assert first == second
    assert first.receipt_digest == second.receipt_digest
    with pytest.raises(FrozenInstanceError):
        first.writer_operationally_enabled = True


def test_module_has_no_default_path_runtime_hook_or_legacy_goal_surface():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    imported_modules = {
        (node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not any(
        name.startswith(
            (
                "main",
                "streaming",
                "language.speech_hub",
                "adapters.goal_adapter",
                "adapters.memory_adapter",
                "adapters.hormone_adapter",
            )
        )
        for name in imported_modules
    )
    assert not imported_names & {
        "GoalManagement",
        "StreamingEngine",
        "build_full_engine",
    }
    assert not called & {
        "build_full_engine",
        "chat_stream",
        "schedule",
        "speak",
    }
    text = MODULE.read_text(encoding="utf-8")
    assert '":memory:"' not in text
    assert "os.environ" not in text
    assert "getenv" not in text
