from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import core.m3_c_h_dormant_goal_lifecycle_writer as dormant_module
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
    GoalLifecycleReducerSnapshot,
    build_event_envelope_candidate,
)
from core.m3_c_e_goal_lifecycle_substrate_binding_preflight import (
    build_substrate_binding_candidates,
)
from core.m3_c_h_dormant_goal_lifecycle_writer import (
    DormantGoalLifecycleWriter,
    GoalLifecycleWriterAuthorizationPacket,
    M3CDormantWriterAuthorizationError,
    M3CDormantWriterConflictError,
    M3CDormantWriterRecoveryRequired,
    WriterStorageLimits,
    WriterValidationPins,
    active_reviewed_writer_authorization_packet,
    build_dormant_writer_rollback_control,
    verify_active_writer_authorization,
)
from core.sqlite_shadow_store import (
    SQLiteShadowStore,
    ShadowStoragePolicy,
    StoragePolicyExceeded,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_h_dormant_goal_lifecycle_writer.py"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate() -> GoalCandidate:
    return GoalCandidate(
        semantic_goal_id="recover_operating_margin",
        decision_epoch=0,
        evidence_digest=_digest("m3-c-h-recovery-evidence"),
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
            sample_digest=_digest(f"m3-c-h-sample:{drive}"),
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


def _bindings():
    return build_substrate_binding_candidates(_sources())


def _policy(
    *,
    snapshot_interval_events: int = 100,
    max_event_count: int = 16,
    max_snapshot_count: int = 4,
) -> ShadowStoragePolicy:
    return ShadowStoragePolicy(
        snapshot_interval_events=snapshot_interval_events,
        max_event_count=max_event_count,
        max_event_bytes=1_048_576,
        max_snapshot_count=max_snapshot_count,
        max_snapshot_bytes=1_048_576,
        max_backups=1,
    )


def _writer(path: Path, *, policy: ShadowStoragePolicy | None = None):
    return DormantGoalLifecycleWriter(path, policy=policy or _policy())


def _packet(writer: DormantGoalLifecycleWriter):
    return GoalLifecycleWriterAuthorizationPacket(
        validation=WriterValidationPins(
            implementation_head="1" * 40,
            exact_run=900001,
            focused_passed=15,
            full_passed=3303,
            forward_gate_errors=0,
            artifact_sha256="2" * 64,
            m2e_run=900002,
        ),
        storage_limits=WriterStorageLimits.from_policy(writer.policy),
        database_path_digest=writer.database_path_digest,
    )


def _activate_for_test(
    packet: GoalLifecycleWriterAuthorizationPacket,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dormant_module,
        "_ACTIVE_REVIEWED_IMPLEMENTATION_HEAD",
        packet.implementation_head,
    )
    monkeypatch.setattr(
        dormant_module,
        "_ACTIVE_REVIEWED_AUTHORIZATION_DIGEST",
        packet.authorization_digest,
    )


def _open_store(writer: DormantGoalLifecycleWriter) -> SQLiteShadowStore:
    store = SQLiteShadowStore(writer.database_path, policy=writer.policy)
    store.initialize()
    return store


def test_construction_is_io_free_and_requires_explicit_absolute_path(tmp_path: Path):
    path = tmp_path / "not-created" / "goal-lifecycle.sqlite3"
    writer = _writer(path)
    assert writer.database_path == path
    assert not path.exists()
    assert not path.parent.exists()
    with pytest.raises(M3CDormantWriterAuthorizationError, match="absolute"):
        _writer(Path("relative.sqlite3"))
    with pytest.raises(M3CDormantWriterAuthorizationError, match="concrete"):
        _writer(Path(":memory:"))


def test_missing_packet_is_first_failure_before_store_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(tmp_path / "goal-lifecycle.sqlite3")

    def forbidden_store(*args, **kwargs):
        raise AssertionError("store construction must be unreachable")

    monkeypatch.setattr(dormant_module, "SQLiteShadowStore", forbidden_store)
    with pytest.raises(M3CDormantWriterAuthorizationError, match="authorization packet"):
        writer.append(_bindings()[0])
    assert not writer.database_path.exists()


def test_structurally_valid_nonreviewed_packet_is_rejected_before_io(tmp_path: Path):
    writer = _writer(tmp_path / "goal-lifecycle.sqlite3")
    packet = _packet(writer)
    with pytest.raises(M3CDormantWriterAuthorizationError, match="active reviewed head"):
        writer.append(_bindings()[0], authorization_packet=packet)
    assert not writer.database_path.exists()


def test_active_packet_path_mismatch_refuses_before_file_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(tmp_path / "first.sqlite3")
    other = _writer(tmp_path / "second.sqlite3")
    packet = _packet(other)
    _activate_for_test(packet, monkeypatch)
    with pytest.raises(M3CDormantWriterAuthorizationError, match="path"):
        writer.append(_bindings()[0], authorization_packet=packet)
    assert not writer.database_path.exists()
    assert not other.database_path.exists()


def test_m2e_operational_rollback_refuses_before_file_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(tmp_path / "goal-lifecycle.sqlite3")
    packet = _packet(writer)
    _activate_for_test(packet, monkeypatch)
    rollback = build_operational_rollback_record(
        requested_by="m3-c-h-test",
        reason="bounded-test",
    )
    rolled_back = resolve_cutover_authority(rollback)
    with pytest.raises(M3CDormantWriterAuthorizationError, match="not active"):
        writer.append(
            _bindings()[0],
            authorization_packet=packet,
            authority_state=rolled_back,
        )
    assert not writer.database_path.exists()


def test_test_only_pin_exercises_one_verified_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(tmp_path / "goal-lifecycle.sqlite3")
    packet = _packet(writer)
    _activate_for_test(packet, monkeypatch)
    receipt = writer.append(_bindings()[0], authorization_packet=packet)
    assert writer.database_path.is_file()
    assert receipt.sequence == 1
    assert receipt.inserted_rows == 1
    assert receipt.transaction_committed is True
    assert receipt.precommit_readback_verified is True
    assert receipt.postcommit_readback_verified is True
    assert receipt.direct_reducer_equivalent is True
    assert receipt.sqlite_write_performed is True
    assert receipt.disposable_or_test_path_only is True
    assert receipt.production_authoritative_append_performed is False
    assert receipt.live_writer_installed is False
    assert receipt.production_integration_performed is False
    assert receipt.legacy_goal_authority_transferred is False
    assert receipt.m3_e_authority_open is False
    store = _open_store(writer)
    assert len(store.events(stream_id=EVENT_STREAM)) == 1
    assert store.integrity_check().valid is True


def test_four_appends_produce_bounded_snapshots_and_replay_equivalence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(
        tmp_path / "goal-lifecycle.sqlite3",
        policy=_policy(
            snapshot_interval_events=2,
            max_event_count=4,
            max_snapshot_count=2,
        ),
    )
    packet = _packet(writer)
    _activate_for_test(packet, monkeypatch)
    receipts = [
        writer.append(binding, authorization_packet=packet)
        for binding in _bindings()
    ]
    assert [item.sequence for item in receipts] == [1, 2, 3, 4]
    assert receipts[0].snapshot_digest is None
    assert receipts[1].snapshot_digest is not None
    assert receipts[2].snapshot_digest is None
    assert receipts[3].snapshot_digest is not None
    store = _open_store(writer)
    selection = store.latest_valid_snapshot(EVENT_STREAM)
    assert selection.selected is not None
    assert selection.selected.through_sequence == 4
    assert selection.selected.state_digest == receipts[-1].reducer_snapshot_digest
    assert store.integrity_check().valid is True


def test_duplicate_request_is_conflict_not_idempotent_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(tmp_path / "goal-lifecycle.sqlite3")
    packet = _packet(writer)
    _activate_for_test(packet, monkeypatch)
    first = _bindings()[0]
    writer.append(first, authorization_packet=packet)
    with pytest.raises(M3CDormantWriterConflictError, match="sequence"):
        writer.append(first, authorization_packet=packet)
    store = _open_store(writer)
    assert len(store.events(stream_id=EVENT_STREAM)) == 1


def test_wrong_causation_is_conflict_without_partial_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(tmp_path / "goal-lifecycle.sqlite3")
    packet = _packet(writer)
    _activate_for_test(packet, monkeypatch)
    bindings = _bindings()
    writer.append(bindings[0], authorization_packet=packet)
    wrong = replace(bindings[1], causation_event_id="m3c:goal-lifecycle:missing")
    with pytest.raises(M3CDormantWriterConflictError, match="causation"):
        writer.append(wrong, authorization_packet=packet)
    store = _open_store(writer)
    assert len(store.events(stream_id=EVENT_STREAM)) == 1


def test_bounded_capacity_failure_preserves_prior_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(
        tmp_path / "goal-lifecycle.sqlite3",
        policy=_policy(max_event_count=1),
    )
    packet = _packet(writer)
    _activate_for_test(packet, monkeypatch)
    bindings = _bindings()
    writer.append(bindings[0], authorization_packet=packet)
    with pytest.raises(StoragePolicyExceeded, match="bounded storage policy"):
        writer.append(bindings[1], authorization_packet=packet)
    assert writer.operationally_disabled is False
    store = _open_store(writer)
    assert len(store.events(stream_id=EVENT_STREAM)) == 1
    assert store.integrity_check().valid is True


def test_reviewed_rollback_control_disables_without_deleting_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(tmp_path / "goal-lifecycle.sqlite3")
    packet = _packet(writer)
    _activate_for_test(packet, monkeypatch)
    bindings = _bindings()
    writer.append(bindings[0], authorization_packet=packet)
    control = build_dormant_writer_rollback_control(
        packet,
        database_path=writer.database_path,
        requested_by="m3-c-h-test",
        reason="verified-disable",
    )
    assert writer.apply_rollback(control) == control.control_digest
    assert writer.operationally_disabled is True
    assert writer.disable_reason == "reviewed_operational_rollback"
    with pytest.raises(M3CDormantWriterAuthorizationError, match="disabled"):
        writer.append(bindings[1], authorization_packet=packet)
    store = _open_store(writer)
    assert len(store.events(stream_id=EVENT_STREAM)) == 1
    assert store.integrity_check().valid is True


def test_postcommit_replay_mismatch_disables_and_preserves_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(tmp_path / "goal-lifecycle.sqlite3")
    packet = _packet(writer)
    _activate_for_test(packet, monkeypatch)
    original = DormantGoalLifecycleWriter._replay_persisted_events

    def mismatched_replay(events, *, authority_state_digest):
        values = tuple(events)
        if values:
            return GoalLifecycleReducerSnapshot.empty()
        return original(values, authority_state_digest=authority_state_digest)

    monkeypatch.setattr(
        DormantGoalLifecycleWriter,
        "_replay_persisted_events",
        staticmethod(mismatched_replay),
    )
    with pytest.raises(M3CDormantWriterRecoveryRequired, match="replay mismatch"):
        writer.append(_bindings()[0], authorization_packet=packet)
    assert writer.operationally_disabled is True
    assert writer.database_path.is_file()
    store = _open_store(writer)
    assert len(store.events(stream_id=EVENT_STREAM)) == 1
    assert store.integrity_check().valid is True


def test_authorization_packet_and_receipt_are_deterministic_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = _writer(tmp_path / "goal-lifecycle.sqlite3")
    first = _packet(writer)
    second = _packet(writer)
    assert first == second
    assert first.authorization_digest == second.authorization_digest
    with pytest.raises(FrozenInstanceError):
        first.human_reviewed = False
    _activate_for_test(first, monkeypatch)
    receipt = writer.append(_bindings()[0], authorization_packet=first)
    assert len(receipt.receipt_digest) == 64
    with pytest.raises(FrozenInstanceError):
        receipt.transaction_committed = False


def test_scope_escape_packet_is_rejected():
    validation = WriterValidationPins(
        implementation_head="1" * 40,
        exact_run=1,
        focused_passed=1,
        full_passed=1,
        forward_gate_errors=0,
        artifact_sha256="2" * 64,
        m2e_run=2,
    )
    limits = WriterStorageLimits.from_policy(_policy())
    with pytest.raises(M3CDormantWriterAuthorizationError, match="escaped"):
        GoalLifecycleWriterAuthorizationPacket(
            validation=validation,
            storage_limits=limits,
            database_path_digest="3" * 64,
            action_authorized=True,
        )


def test_checked_in_reviewed_packet_matches_exact_h_evidence_and_bounds():
    packet = active_reviewed_writer_authorization_packet()
    assert packet.implementation_head == "68efeca10c6819cb74ccc884e3c0c784e0b44c95"
    assert packet.validation.exact_run == 30444371019
    assert packet.validation.focused_passed == 15
    assert packet.validation.full_passed == 3303
    assert packet.validation.artifact_sha256 == (
        "79f7f6a2034ced8b04dfb3ae3ed69f56cdd6eb6c8f0da3cb740fc900f4ef80be"
    )
    assert packet.validation.m2e_run == 30444371035
    assert packet.storage_limits.to_mapping() == {
        "snapshot_interval_events": 32,
        "max_event_count": 4096,
        "max_event_bytes": 16_777_216,
        "max_snapshot_count": 128,
        "max_snapshot_bytes": 16_777_216,
        "max_backups": 3,
    }
    assert packet.database_path_digest == (
        "cfcc91e8bab89beceff3ce8f5ecbc325705bd33b256e9d47ca8bdb9008833b80"
    )
    assert packet.authorization_digest == (
        "ab050d04f7ae7a6f920e94696d5b0988e4ad5331e9082d5ec61c30548166c111"
    )
    assert verify_active_writer_authorization(packet) == packet.authorization_digest


def test_checked_in_module_has_exact_pins_and_no_activation_heuristic():
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
    expected = {
        "_ACTIVE_REVIEWED_IMPLEMENTATION_HEAD": (
            "68efeca10c6819cb74ccc884e3c0c784e0b44c95"
        ),
        "_ACTIVE_REVIEWED_AUTHORIZATION_DIGEST": (
            "ab050d04f7ae7a6f920e94696d5b0988e4ad5331e9082d5ec61c30548166c111"
        ),
    }
    for name, value in expected.items():
        assert isinstance(assignments[name], ast.Constant)
        assert assignments[name].value == value
    assert "/data/data/" not in text
    assert "os" not in imported_modules
    assert "sys" not in imported_modules
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
    assert not called & {
        "getenv",
        "build_full_engine",
        "chat_stream",
        "schedule",
        "speak",
    }
    assert "os.environ" not in text
    assert "if __name__" not in text
    assert "sequences 1 through 5" not in text
