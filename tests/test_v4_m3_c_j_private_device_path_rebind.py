from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import core.m3_c_h_dormant_goal_lifecycle_writer as writer_module
import core.m3_c_j_goal_lifecycle_observation_window as window_module
import core.m3_c_j_private_device_operator as operator_module
import core.m3_c_j_private_device_path_rebind as rebind
import scripts.operator.m3_c_j_private_device_window as base_script
import scripts.operator.m3_c_j_private_device_window_rebound as rebound_script
from core.m3_c_j_private_device_operator import (
    M3CPrivateDeviceOperatorAuthorizationError,
)
from core.m3_c_j_private_device_operator_pin import (
    REVIEWED_OPERATOR_IMPLEMENTATION_HEAD,
    active_reviewed_private_device_operator_authorization_packet,
)


@dataclass(frozen=True)
class _FakeReceipt:
    repository_head: str


@dataclass(frozen=True)
class _FakeBundle:
    operator_receipt: _FakeReceipt


def test_rebound_packet_chain_changes_only_private_path_bindings():
    original_writer = writer_module.active_reviewed_writer_authorization_packet()
    original_window = (
        window_module.active_reviewed_observation_window_authorization_packet()
    )
    original_operator = (
        active_reviewed_private_device_operator_authorization_packet()
    )

    rebound_writer = rebind.active_rebound_writer_authorization_packet()
    rebound_window = rebind.active_rebound_window_authorization_packet()
    rebound_operator = (
        rebind.active_rebound_private_device_operator_authorization_packet()
    )

    assert rebound_writer.authorization_digest == rebind.REBOUND_WRITER_AUTHORIZATION_DIGEST
    assert rebound_window.authorization_digest == rebind.REBOUND_WINDOW_AUTHORIZATION_DIGEST
    assert rebound_operator.authorization_digest == rebind.REBOUND_OPERATOR_AUTHORIZATION_DIGEST
    assert rebound_operator.database_path_digest == rebind.REBOUND_DATABASE_PATH_DIGEST

    assert replace(
        rebound_writer,
        database_path_digest=rebind.ORIGINAL_DATABASE_PATH_DIGEST,
    ) == original_writer
    assert replace(
        rebound_window,
        writer_authorization_digest=original_writer.authorization_digest,
        database_path_digest=rebind.ORIGINAL_DATABASE_PATH_DIGEST,
    ) == original_window
    assert replace(
        rebound_operator,
        writer_authorization_digest=original_writer.authorization_digest,
        window_authorization_digest=original_window.authorization_digest,
        database_path_digest=rebind.ORIGINAL_DATABASE_PATH_DIGEST,
    ) == original_operator


def test_rebound_execution_scopes_all_pins_and_restores_them(monkeypatch):
    packet = rebind.active_rebound_private_device_operator_authorization_packet()
    launch_head = "9" * 40

    previous = (
        writer_module._ACTIVE_REVIEWED_AUTHORIZATION_DIGEST,
        writer_module._ACTIVE_REVIEWED_DATABASE_PATH_DIGEST,
        window_module._ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST,
        operator_module._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD,
        operator_module._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST,
    )

    monkeypatch.setattr(
        rebind,
        "database_path_digest",
        lambda _path: rebind.REBOUND_DATABASE_PATH_DIGEST,
    )

    def fake_execute(authorization_packet, **kwargs):
        assert authorization_packet == packet
        assert kwargs["repository_head"] == REVIEWED_OPERATOR_IMPLEMENTATION_HEAD
        active_writer = writer_module.active_reviewed_writer_authorization_packet()
        active_window = (
            window_module.active_reviewed_observation_window_authorization_packet()
        )
        assert active_writer.authorization_digest == rebind.REBOUND_WRITER_AUTHORIZATION_DIGEST
        assert active_window.authorization_digest == rebind.REBOUND_WINDOW_AUTHORIZATION_DIGEST
        assert (
            operator_module.verify_active_private_device_operator_authorization(packet)
            == rebind.REBOUND_OPERATOR_AUTHORIZATION_DIGEST
        )
        return _FakeBundle(
            _FakeReceipt(repository_head=REVIEWED_OPERATOR_IMPLEMENTATION_HEAD)
        )

    monkeypatch.setattr(
        operator_module,
        "execute_private_device_observation_window",
        fake_execute,
    )

    result = rebind.execute_rebound_private_device_observation_window(
        packet,
        operator_input=object(),
        private_nonce=b"n" * 32,
        repository_head=launch_head,
        launch_attestation_id="launch-attestation",
        runtime_instance_id="runtime-instance",
        database_path=Path("/private/redacted.sqlite3"),
        backup_directory=Path("/private/backups"),
        restore_path=Path("/private/restore/baseline.sqlite3"),
    )

    assert result.operator_receipt.repository_head == launch_head
    assert (
        writer_module._ACTIVE_REVIEWED_AUTHORIZATION_DIGEST,
        writer_module._ACTIVE_REVIEWED_DATABASE_PATH_DIGEST,
        window_module._ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST,
        operator_module._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD,
        operator_module._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST,
    ) == previous


def test_unreviewed_database_path_fails_before_scope():
    packet = rebind.active_rebound_private_device_operator_authorization_packet()
    with pytest.raises(
        M3CPrivateDeviceOperatorAuthorizationError,
        match="database path differs",
    ):
        rebind.execute_rebound_private_device_observation_window(
            packet,
            operator_input=object(),
            private_nonce=b"n" * 32,
            repository_head="8" * 40,
            launch_attestation_id="launch-attestation",
            runtime_instance_id="runtime-instance",
            database_path=Path("/tmp/not-reviewed.sqlite3"),
            backup_directory=Path("/tmp/backups"),
            restore_path=Path("/tmp/restore/baseline.sqlite3"),
        )
    assert operator_module._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD is None
    assert operator_module._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST is None


def test_rebound_script_restores_base_script_globals_on_failure(monkeypatch):
    previous = (
        base_script.active_reviewed_private_device_operator_authorization_packet,
        base_script.execute_exact_reviewed_private_device_observation_window,
        base_script.verify_reviewed_private_device_operator_authorization,
    )

    def fail(_argv):
        assert (
            base_script.active_reviewed_private_device_operator_authorization_packet
            is rebind.active_rebound_private_device_operator_authorization_packet
        )
        assert (
            base_script.execute_exact_reviewed_private_device_observation_window
            is rebind.execute_rebound_private_device_observation_window
        )
        assert (
            base_script.verify_reviewed_private_device_operator_authorization
            is rebind.verify_rebound_private_device_operator_authorization
        )
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(base_script, "main", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        rebound_script.main([])

    assert (
        base_script.active_reviewed_private_device_operator_authorization_packet,
        base_script.execute_exact_reviewed_private_device_observation_window,
        base_script.verify_reviewed_private_device_operator_authorization,
    ) == previous
