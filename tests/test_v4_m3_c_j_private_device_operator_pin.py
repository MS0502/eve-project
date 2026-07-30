from __future__ import annotations

import hashlib
from dataclasses import replace
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
    PrivateDeviceGoalInput,
    PrivateDeviceOperatorAuthorizationPacket,
    active_reviewed_private_device_operator_authorization_packet,
    execute_private_device_observation_window,
    verify_active_private_device_operator_authorization,
)
from core.sqlite_shadow_store import ShadowStoragePolicy

PINNED_IMPLEMENTATION_HEAD = "d8eb3c2d6b576cc313712f831f8b2f1556cdefb2"
PINNED_AUTHORIZATION_DIGEST = (
    "e360c0e669af3ba89a6f552c81c67e3b3d908171665ed20b510a0044003d13a5"
)
LAUNCH_HEAD = "5" * 40


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _input() -> PrivateDeviceGoalInput:
    candidate = GoalCandidate(
        semantic_goal_id="recover_operating_margin",
        decision_epoch=1,
        evidence_digest=_digest("pin-evidence"),
        base_value=0.30,
        expected_value=0.0,
        urgency=0.0,
        continuity=0.0,
        cost=0.0,
        risk=0.0,
        drive_alignment={
            drive: {"energy": -0.90, "safety": -0.80, "curiosity": -0.10}.get(
                drive, 0.0
            )
            for drive in ALLOWED_DRIVES
        },
        drive_confidence={drive: 1.0 for drive in ALLOWED_DRIVES},
    )
    values = {"energy": -0.70, "safety": -0.80, "curiosity": -0.20}
    return PrivateDeviceGoalInput(
        candidate=candidate,
        drive_samples=tuple(
            DriveSample(
                drive=drive,
                value=values.get(drive, 0.0),
                lower_bound=-1.0,
                upper_bound=1.0,
                sample_digest=_digest(f"pin-sample:{drive}"),
                replay_elapsed_seconds=0.0,
            )
            for drive in ALLOWED_DRIVES
        ),
    )


def _policy() -> ShadowStoragePolicy:
    return ShadowStoragePolicy(
        snapshot_interval_events=2,
        max_event_count=16,
        max_event_bytes=1_048_576,
        max_snapshot_count=8,
        max_snapshot_bytes=1_048_576,
        max_backups=2,
    )


def _temporary_packets(database_path: Path):
    writer = GoalLifecycleWriterAuthorizationPacket(
        validation=WriterValidationPins(
            implementation_head="1" * 40,
            exact_run=900101,
            focused_passed=16,
            full_passed=3324,
            forward_gate_errors=0,
            artifact_sha256="2" * 64,
            m2e_run=900102,
        ),
        storage_limits=WriterStorageLimits.from_policy(_policy()),
        database_path_digest=database_path_digest(database_path),
    )
    window = ObservationWindowAuthorizationPacket(
        window_implementation_head="3" * 40,
        writer_authorization_digest=writer.authorization_digest,
        writer_implementation_head=writer.implementation_head,
        database_path_digest=writer.database_path_digest,
        storage_limits=writer.storage_limits,
        max_window_events=32,
    )
    operator = PrivateDeviceOperatorAuthorizationPacket(
        operator_implementation_head="4" * 40,
        window_authorization_digest=window.authorization_digest,
        window_implementation_head=window.window_implementation_head,
        writer_authorization_digest=writer.authorization_digest,
        writer_implementation_head=writer.implementation_head,
        database_path_digest=writer.database_path_digest,
        max_window_events=window.max_window_events,
    )
    return writer, window, operator


def _activate_temporary_packets(
    database_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer, window, operator = _temporary_packets(database_path)
    monkeypatch.setattr(
        writer_module, "_ACTIVE_REVIEWED_IMPLEMENTATION_HEAD", writer.implementation_head
    )
    monkeypatch.setattr(
        writer_module, "_ACTIVE_REVIEWED_AUTHORIZATION_DIGEST", writer.authorization_digest
    )
    monkeypatch.setattr(
        writer_module, "_ACTIVE_REVIEWED_DATABASE_PATH_DIGEST", writer.database_path_digest
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
        operator_module, "active_reviewed_writer_authorization_packet", lambda: writer
    )
    monkeypatch.setattr(
        operator_module,
        "active_reviewed_observation_window_authorization_packet",
        lambda: window,
    )
    return operator


def test_exact_reviewed_operator_packet_is_pinned_without_io():
    packet = active_reviewed_private_device_operator_authorization_packet()
    assert packet.operator_implementation_head == PINNED_IMPLEMENTATION_HEAD
    assert packet.authorization_digest == PINNED_AUTHORIZATION_DIGEST
    assert packet.production_append_authorized is True
    assert packet.runtime_integration_authorized is False
    assert packet.action_authorized is False
    assert packet.scheduler_authorized is False
    assert packet.speech_authorized is False
    assert packet.legacy_goal_authority_transferred is False
    assert packet.legacy_migration_authorized is False
    assert packet.m3_e_authority_open is False
    assert verify_active_private_device_operator_authorization(packet) == (
        PINNED_AUTHORIZATION_DIGEST
    )


def test_nonreviewed_packet_and_scope_escape_fail_closed():
    packet = active_reviewed_private_device_operator_authorization_packet()
    with pytest.raises(M3CPrivateDeviceOperatorAuthorizationError, match="active reviewed"):
        verify_active_private_device_operator_authorization(
            replace(packet, operator_implementation_head="6" * 40)
        )
    with pytest.raises(M3CPrivateDeviceOperatorAuthorizationError, match="scope"):
        replace(packet, runtime_integration_authorized=True)


def test_launch_head_is_receipt_bound_separately_from_implementation_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "database" / "goal.sqlite3"
    operator = _activate_temporary_packets(database, monkeypatch)
    bundle = execute_private_device_observation_window(
        operator,
        operator_input=_input(),
        private_nonce=b"p" * 32,
        repository_head=LAUNCH_HEAD,
        launch_attestation_id="launch:pin-test:001",
        runtime_instance_id="runtime:pin-test:001",
        database_path=database,
        backup_directory=tmp_path / "output" / "backups",
        restore_path=tmp_path / "output" / "restore" / "baseline.sqlite3",
    )
    assert bundle.operator_receipt.operator_implementation_head == "4" * 40
    assert bundle.operator_receipt.repository_head == LAUNCH_HEAD
    assert bundle.operator_receipt.exact_transition_count == 4
    assert bundle.operator_receipt.writer_disabled_after_append is True
    assert bundle.operator_receipt.separate_restore_verified is True


def test_operator_command_uses_active_packet_and_exact_launch_checkout():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/operator/m3_c_j_private_device_window.py"
    ).read_text(encoding="utf-8")
    assert "active_reviewed_private_device_operator_authorization_packet" in source
    assert "build_private_device_operator_authorization_candidate" not in source
    assert "repository_head = _repository_head(args.expected_head)" in source
    assert source.index("repository_head = _repository_head(args.expected_head)") < source.index(
        "database_path = _outside_repository"
    )
    assert "phone_witness_replayed\": False" in source
    assert "retained_sequences_replayed\": False" in source
