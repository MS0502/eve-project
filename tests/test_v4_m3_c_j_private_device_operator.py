from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import core.m3_c_h_dormant_goal_lifecycle_writer as writer_module
import core.m3_c_j_goal_lifecycle_observation_window as window_module
import core.m3_c_j_private_device_operator as operator_module
from core.m3_c_b_goal_selection_kernel import ALLOWED_DRIVES, DriveSample, GoalCandidate
from core.m3_c_h_dormant_goal_lifecycle_writer import (
    GoalLifecycleWriterAuthorizationPacket,
    WriterStorageLimits,
    WriterValidationPins,
    database_path_digest,
)
from core.m3_c_j_goal_lifecycle_observation_window import (
    ObservationWindowAuthorizationPacket,
)
from core.m3_c_j_private_device_operator import (
    M3CPrivateDeviceOperatorAuthorizationError,
    M3CPrivateDeviceOperatorExecutionError,
    M3CPrivateDeviceOperatorInputError,
    PrivateDeviceGoalInput,
    PrivateDeviceOperatorAuthorizationPacket,
    build_private_device_operator_authorization_candidate,
    build_reviewed_lifecycle_bindings,
    execute_private_device_observation_window,
    verify_active_private_device_operator_authorization,
)
from core.sqlite_shadow_store import SQLiteShadowStore, ShadowStoragePolicy

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_j_private_device_operator.py"
SCRIPT = ROOT / "scripts/operator/m3_c_j_private_device_window.py"
WRITER_HEAD = "1" * 40
WINDOW_HEAD = "3" * 40
OPERATOR_HEAD = "4" * 40


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _input() -> PrivateDeviceGoalInput:
    candidate = GoalCandidate(
        semantic_goal_id="recover_operating_margin",
        decision_epoch=1,
        evidence_digest=_digest("m3-c-j-private-device-evidence"),
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
    values = {"energy": -0.70, "safety": -0.80, "curiosity": -0.20}
    samples = tuple(
        DriveSample(
            drive=drive,
            value=values.get(drive, 0.0),
            lower_bound=-1.0,
            upper_bound=1.0,
            sample_digest=_digest(f"m3-c-j-private-device-sample:{drive}"),
            replay_elapsed_seconds=0.0,
        )
        for drive in ALLOWED_DRIVES
    )
    return PrivateDeviceGoalInput(candidate=candidate, drive_samples=samples)


def _policy() -> ShadowStoragePolicy:
    return ShadowStoragePolicy(
        snapshot_interval_events=2,
        max_event_count=16,
        max_event_bytes=1_048_576,
        max_snapshot_count=8,
        max_snapshot_bytes=1_048_576,
        max_backups=2,
    )


def _packets(database_path: Path):
    policy = _policy()
    writer = GoalLifecycleWriterAuthorizationPacket(
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
        database_path_digest=database_path_digest(database_path),
    )
    window = ObservationWindowAuthorizationPacket(
        window_implementation_head=WINDOW_HEAD,
        writer_authorization_digest=writer.authorization_digest,
        writer_implementation_head=writer.implementation_head,
        database_path_digest=writer.database_path_digest,
        storage_limits=writer.storage_limits,
        max_window_events=32,
    )
    operator = PrivateDeviceOperatorAuthorizationPacket(
        operator_implementation_head=OPERATOR_HEAD,
        window_authorization_digest=window.authorization_digest,
        window_implementation_head=window.window_implementation_head,
        writer_authorization_digest=writer.authorization_digest,
        writer_implementation_head=writer.implementation_head,
        database_path_digest=writer.database_path_digest,
        max_window_events=window.max_window_events,
    )
    return writer, window, operator


def _activate(
    database_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer, window, operator = _packets(database_path)
    monkeypatch.setattr(
        writer_module,
        "_ACTIVE_REVIEWED_IMPLEMENTATION_HEAD",
        writer.implementation_head,
    )
    monkeypatch.setattr(
        writer_module,
        "_ACTIVE_REVIEWED_AUTHORIZATION_DIGEST",
        writer.authorization_digest,
    )
    monkeypatch.setattr(
        writer_module,
        "_ACTIVE_REVIEWED_DATABASE_PATH_DIGEST",
        writer.database_path_digest,
    )
    monkeypatch.setattr(
        window_module,
        "_ACTIVE_REVIEWED_WINDOW_IMPLEMENTATION_HEAD",
        window.window_implementation_head,
    )
    monkeypatch.setattr(
        window_module,
        "_ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST",
        window.authorization_digest,
    )
    monkeypatch.setattr(
        operator_module,
        "_ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD",
        operator.operator_implementation_head,
    )
    monkeypatch.setattr(
        operator_module,
        "_ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST",
        operator.authorization_digest,
    )
    monkeypatch.setattr(
        operator_module,
        "active_reviewed_writer_authorization_packet",
        lambda: writer,
    )
    monkeypatch.setattr(
        operator_module,
        "active_reviewed_observation_window_authorization_packet",
        lambda: window,
    )
    return writer, window, operator


def test_input_round_trip_is_canonical_and_new_material_only():
    original = _input()
    restored = PrivateDeviceGoalInput.from_mapping(original.to_mapping())
    assert restored == original
    assert restored.input_digest == original.input_digest
    assert len(restored.private_binding_digest(b"n" * 32)) == 64
    with pytest.raises(FrozenInstanceError):
        restored.new_window_material = False
    with pytest.raises(M3CPrivateDeviceOperatorInputError, match="scope"):
        replace(restored, retained_sequences_replayed=True)


def test_reviewed_input_derives_exact_four_transition_chain():
    selection, bindings = build_reviewed_lifecycle_bindings(_input())
    assert selection.decision_kind == "initial_selection"
    assert selection.selected_candidate_id == _input().candidate.candidate_id
    assert len(bindings) == 4
    assert [item.sequence for item in bindings] == [1, 2, 3, 4]
    assert bindings[0].causation_event_id is None
    assert all(item.causation_event_id is not None for item in bindings[1:])
    assert [
        item.source.transition.after_state for item in bindings
    ] == ["proposed", "validated", "eligible", "selected"]


def test_below_threshold_input_is_rejected():
    original = _input()
    weak = replace(
        original,
        candidate=replace(
            original.candidate,
            base_value=-1.0,
            expected_value=-1.0,
            urgency=0.0,
            continuity=-1.0,
            cost=1.0,
            risk=1.0,
            drive_alignment={drive: 0.0 for drive in ALLOWED_DRIVES},
        ),
    )
    with pytest.raises(M3CPrivateDeviceOperatorInputError, match="initial-selection"):
        build_reviewed_lifecycle_bindings(weak)


def test_candidate_packet_pins_exact_m3_c_j_authorization_without_activation():
    packet = build_private_device_operator_authorization_candidate(
        operator_implementation_head=OPERATOR_HEAD,
    )
    assert packet.operator_implementation_head == OPERATOR_HEAD
    assert packet.prerequisite_exact_head == (
        "532c595158ee68eb3268f75414bf6eaa23a79ffb"
    )
    assert packet.prerequisite_exact_run == 30451436253
    assert packet.prerequisite_focused_passed == 11
    assert packet.prerequisite_full_passed == 3315
    assert packet.prerequisite_artifact_sha256 == (
        "e488f98d0d60a4572ea1f64c383ee8f3a0d91d23b22477c431695b16e9d9d12d"
    )
    assert packet.prerequisite_m2e_run == 30451436272
    assert packet.prerequisite_merge_sha == (
        "361ed88be399ed7650a946b58e713bc14253384e"
    )
    assert packet.required_transition_count == 4
    assert packet.production_append_authorized is True
    assert packet.runtime_integration_authorized is False
    with pytest.raises(M3CPrivateDeviceOperatorAuthorizationError, match="absent"):
        verify_active_private_device_operator_authorization(packet)


def test_temp_path_execution_proves_single_use_backup_append_disable_and_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "private-db" / "goal_lifecycle.sqlite3"
    _writer, _window, operator = _activate(database, monkeypatch)
    bundle = execute_private_device_observation_window(
        operator,
        operator_input=_input(),
        private_nonce=b"x" * 32,
        repository_head=operator.operator_implementation_head,
        launch_attestation_id="launch:test:001",
        runtime_instance_id="runtime:test:001",
        database_path=database,
        backup_directory=tmp_path / "private-output" / "backups",
        restore_path=tmp_path / "private-output" / "restore" / "baseline.sqlite3",
    )
    assert database.is_file()
    assert bundle.operator_receipt.exact_transition_count == 4
    assert bundle.operator_receipt.production_append_performed is True
    assert bundle.operator_receipt.writer_disabled_after_append is True
    assert bundle.operator_receipt.separate_restore_verified is True
    assert bundle.window_receipt.observed_event_count == 4
    assert bundle.window_receipt.duplicate_acceptance_count == 0
    assert bundle.window_receipt.conflict_acceptance_count == 0
    assert bundle.window_receipt.runtime_integration_performed is False
    assert bundle.window_receipt.legacy_goal_authority_transferred is False
    assert bundle.window_receipt.m3_e_authority_open is False
    store = SQLiteShadowStore(database, policy=_policy())
    store.initialize()
    assert store.integrity_check().valid is True
    assert len(store.events()) == 4
    public = bundle.public_review_mapping()
    assert public["operator_input_public"] is False
    assert public["database_path_plaintext_public"] is False
    assert public["phone_witness_replayed"] is False
    assert public["retained_sequences_replayed"] is False


def test_existing_database_blocks_rerun_before_second_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "private-db" / "goal_lifecycle.sqlite3"
    _writer, _window, operator = _activate(database, monkeypatch)
    kwargs = dict(
        authorization_packet=operator,
        operator_input=_input(),
        private_nonce=b"x" * 32,
        repository_head=operator.operator_implementation_head,
        launch_attestation_id="launch:test:002",
        runtime_instance_id="runtime:test:002",
        database_path=database,
        backup_directory=tmp_path / "first" / "backups",
        restore_path=tmp_path / "first" / "restore" / "baseline.sqlite3",
    )
    execute_private_device_observation_window(**kwargs)
    with pytest.raises(M3CPrivateDeviceOperatorExecutionError, match="absent database"):
        execute_private_device_observation_window(
            **{
                **kwargs,
                "backup_directory": tmp_path / "second" / "backups",
                "restore_path": tmp_path / "second" / "restore" / "baseline.sqlite3",
            }
        )
    store = SQLiteShadowStore(database, policy=_policy())
    store.initialize()
    assert len(store.events()) == 4


def test_wrong_database_digest_fails_before_database_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    reviewed = tmp_path / "reviewed" / "goal.sqlite3"
    wrong = tmp_path / "wrong" / "goal.sqlite3"
    _writer, _window, operator = _activate(reviewed, monkeypatch)
    with pytest.raises(M3CPrivateDeviceOperatorAuthorizationError, match="path"):
        execute_private_device_observation_window(
            operator,
            operator_input=_input(),
            private_nonce=b"x" * 32,
            repository_head=operator.operator_implementation_head,
            launch_attestation_id="launch:test:003",
            runtime_instance_id="runtime:test:003",
            database_path=wrong,
            backup_directory=tmp_path / "output" / "backups",
            restore_path=tmp_path / "output" / "restore" / "baseline.sqlite3",
        )
    assert not reviewed.exists()
    assert not wrong.exists()


def test_scope_escape_operator_authorization_is_rejected(tmp_path: Path):
    _writer, window, operator = _packets(tmp_path / "goal.sqlite3")
    with pytest.raises(M3CPrivateDeviceOperatorAuthorizationError, match="scope"):
        replace(operator, action_authorized=True)
    with pytest.raises(M3CPrivateDeviceOperatorAuthorizationError, match="four"):
        replace(operator, required_transition_count=3)
    assert window.max_window_events == 32


def test_checked_in_operator_module_is_dormant_and_script_is_explicit():
    text = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(MODULE))
    assignments = {}
    top_level_calls = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node.value
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            top_level_calls.append(node.value)
    for name in (
        "_ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD",
        "_ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST",
    ):
        assert isinstance(assignments[name], ast.Constant)
        assert assignments[name].value is None
    assert top_level_calls == []
    script = SCRIPT.read_text(encoding="utf-8")
    assert "--expected-head" in script
    assert "--database-path" in script
    assert "--private-root" in script
    assert "--nonce-file" in script
    assert "--input-file" in script
    assert "git\", \"status\", \"--porcelain" in script
    assert "single-use operator output already exists" in script
    assert "build_full_engine" not in script
    assert "sequences 1 through 5" not in script
