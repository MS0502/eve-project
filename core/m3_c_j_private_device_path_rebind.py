"""Exact-reviewed private-path rebind for one M3-C-J device command.

The historical PR #225/#229/#230/#231 implementations and their source-bound
evidence remain byte-for-byte unchanged. This adapter replaces only the lost
caller-owned database-path digest in memory for one synchronous explicit call,
then restores every reviewed writer/window/operator pin in ``finally``.

Import performs no filesystem, SQLite, backup, restore, append, runtime, action,
scheduler, speech, legacy-migration, goal-authority-transfer, or M3-E operation.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final

import core.m3_c_h_dormant_goal_lifecycle_writer as writer_module
import core.m3_c_j_goal_lifecycle_observation_window as window_module
import core.m3_c_j_private_device_operator as operator_module
from core.m3_c_h_dormant_goal_lifecycle_writer import (
    GoalLifecycleWriterAuthorizationPacket,
    database_path_digest,
)
from core.m3_c_j_goal_lifecycle_observation_window import (
    ObservationWindowAuthorizationPacket,
)
from core.m3_c_j_private_device_operator import (
    M3CPrivateDeviceOperatorAuthorizationError,
    PrivateDeviceGoalInput,
    PrivateDeviceOperatorAuthorizationPacket,
    PrivateDeviceOperatorBundle,
)
from core.m3_c_j_private_device_operator_pin import (
    REVIEWED_OPERATOR_IMPLEMENTATION_HEAD,
)

ORIGINAL_DATABASE_PATH_DIGEST: Final = (
    "cfcc91e8bab89beceff3ce8f5ecbc325705bd33b256e9d47ca8bdb9008833b80"
)
ORIGINAL_WRITER_AUTHORIZATION_DIGEST: Final = (
    "ab050d04f7ae7a6f920e94696d5b0988e4ad5331e9082d5ec61c30548166c111"
)
ORIGINAL_WINDOW_AUTHORIZATION_DIGEST: Final = (
    "803780b19f0c496adb0a3a68ba32bd296a356a8ea3eeaf2fe6a33cb3476510fb"
)

REBOUND_DATABASE_PATH_DIGEST: Final = (
    "269c89e0e6d5614e2ca86ae5e68b261f3bb0d67bc12bf2045957052cf82ef715"
)
REBOUND_WRITER_AUTHORIZATION_DIGEST: Final = (
    "852e20984a9d670ec2a690106984ebc5d0071daae63bac0c6ebf7f7b255bb1d4"
)
REBOUND_WINDOW_AUTHORIZATION_DIGEST: Final = (
    "7347ae8a0e9cf8b5c44e519728847e8a2e2cb87bd4ea7fc2baf63880f3f30e69"
)
REBOUND_OPERATOR_AUTHORIZATION_DIGEST: Final = (
    "a344de2cb41a2ffcf3923680b57c297ba340127b9f92e11d4a6ead72deffc7bb"
)


def _require_launch_head(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "launch repository head must be lowercase 40-character hex"
        )
    return value


def active_rebound_writer_authorization_packet(
) -> GoalLifecycleWriterAuthorizationPacket:
    """Build the reviewed writer packet with only the private path digest replaced."""

    original = writer_module.active_reviewed_writer_authorization_packet()
    if (
        original.authorization_digest != ORIGINAL_WRITER_AUTHORIZATION_DIGEST
        or original.database_path_digest != ORIGINAL_DATABASE_PATH_DIGEST
    ):
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "historical writer packet differs from the reviewed prerequisite"
        )
    rebound = replace(original, database_path_digest=REBOUND_DATABASE_PATH_DIGEST)
    if rebound.authorization_digest != REBOUND_WRITER_AUTHORIZATION_DIGEST:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "rebound writer packet digest is inconsistent"
        )
    return rebound


def active_rebound_window_authorization_packet(
) -> ObservationWindowAuthorizationPacket:
    """Build the reviewed window packet over the rebound writer packet."""

    original = window_module.active_reviewed_observation_window_authorization_packet()
    if (
        original.authorization_digest != ORIGINAL_WINDOW_AUTHORIZATION_DIGEST
        or original.database_path_digest != ORIGINAL_DATABASE_PATH_DIGEST
    ):
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "historical window packet differs from the reviewed prerequisite"
        )
    writer = active_rebound_writer_authorization_packet()
    rebound = replace(
        original,
        writer_authorization_digest=writer.authorization_digest,
        database_path_digest=REBOUND_DATABASE_PATH_DIGEST,
    )
    if rebound.authorization_digest != REBOUND_WINDOW_AUTHORIZATION_DIGEST:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "rebound window packet digest is inconsistent"
        )
    return rebound


def active_rebound_private_device_operator_authorization_packet(
) -> PrivateDeviceOperatorAuthorizationPacket:
    """Build the reviewed operator packet over the rebound writer/window packets."""

    writer = active_rebound_writer_authorization_packet()
    window = active_rebound_window_authorization_packet()
    packet = PrivateDeviceOperatorAuthorizationPacket(
        operator_implementation_head=REVIEWED_OPERATOR_IMPLEMENTATION_HEAD,
        window_authorization_digest=window.authorization_digest,
        window_implementation_head=window.window_implementation_head,
        writer_authorization_digest=writer.authorization_digest,
        writer_implementation_head=writer.implementation_head,
        database_path_digest=REBOUND_DATABASE_PATH_DIGEST,
        max_window_events=window.max_window_events,
    )
    if packet.authorization_digest != REBOUND_OPERATOR_AUTHORIZATION_DIGEST:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "rebound operator packet digest is inconsistent"
        )
    return packet


def verify_rebound_private_device_operator_authorization(
    packet: PrivateDeviceOperatorAuthorizationPacket | None,
) -> str:
    """Verify the rebound packet without opening any execution scope."""

    if not isinstance(packet, PrivateDeviceOperatorAuthorizationPacket):
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "packet must be PrivateDeviceOperatorAuthorizationPacket"
        )
    if packet.operator_implementation_head != REVIEWED_OPERATOR_IMPLEMENTATION_HEAD:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "operator implementation head is not the reviewed head"
        )
    if packet.database_path_digest != REBOUND_DATABASE_PATH_DIGEST:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "operator database path digest is not the reviewed rebound path"
        )
    if packet.authorization_digest != REBOUND_OPERATOR_AUTHORIZATION_DIGEST:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "operator authorization digest is not the reviewed rebound packet"
        )
    return packet.authorization_digest


def execute_rebound_private_device_observation_window(
    authorization_packet: PrivateDeviceOperatorAuthorizationPacket | None,
    *,
    operator_input: PrivateDeviceGoalInput,
    private_nonce: bytes,
    repository_head: str,
    launch_attestation_id: str,
    runtime_instance_id: str,
    database_path: str | Path,
    backup_directory: str | Path,
    restore_path: str | Path,
) -> PrivateDeviceOperatorBundle:
    """Execute one explicit call with a synchronous writer/window/operator rebind."""

    packet_digest = verify_rebound_private_device_operator_authorization(
        authorization_packet
    )
    launch_head = _require_launch_head(repository_head)
    if database_path_digest(database_path) != REBOUND_DATABASE_PATH_DIGEST:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "database path differs from the reviewed rebound path digest"
        )
    assert authorization_packet is not None

    writer_packet = active_rebound_writer_authorization_packet()
    window_packet = active_rebound_window_authorization_packet()

    previous_writer_digest = writer_module._ACTIVE_REVIEWED_AUTHORIZATION_DIGEST
    previous_writer_path = writer_module._ACTIVE_REVIEWED_DATABASE_PATH_DIGEST
    previous_window_digest = (
        window_module._ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST
    )
    previous_operator_implementation = (
        operator_module._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD
    )
    previous_operator_digest = (
        operator_module._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST
    )

    if (
        previous_writer_digest != ORIGINAL_WRITER_AUTHORIZATION_DIGEST
        or previous_writer_path != ORIGINAL_DATABASE_PATH_DIGEST
        or previous_window_digest != ORIGINAL_WINDOW_AUTHORIZATION_DIGEST
        or previous_operator_implementation is not None
        or previous_operator_digest is not None
    ):
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "private-path rebind prerequisites or operator scope are not pristine"
        )

    writer_module._ACTIVE_REVIEWED_AUTHORIZATION_DIGEST = (
        writer_packet.authorization_digest
    )
    writer_module._ACTIVE_REVIEWED_DATABASE_PATH_DIGEST = (
        REBOUND_DATABASE_PATH_DIGEST
    )
    window_module._ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST = (
        window_packet.authorization_digest
    )
    operator_module._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD = (
        REVIEWED_OPERATOR_IMPLEMENTATION_HEAD
    )
    operator_module._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST = packet_digest
    try:
        bundle = operator_module.execute_private_device_observation_window(
            authorization_packet,
            operator_input=operator_input,
            private_nonce=private_nonce,
            repository_head=REVIEWED_OPERATOR_IMPLEMENTATION_HEAD,
            launch_attestation_id=launch_attestation_id,
            runtime_instance_id=runtime_instance_id,
            database_path=database_path,
            backup_directory=backup_directory,
            restore_path=restore_path,
        )
    finally:
        operator_module._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD = (
            previous_operator_implementation
        )
        operator_module._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST = (
            previous_operator_digest
        )
        window_module._ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST = (
            previous_window_digest
        )
        writer_module._ACTIVE_REVIEWED_DATABASE_PATH_DIGEST = previous_writer_path
        writer_module._ACTIVE_REVIEWED_AUTHORIZATION_DIGEST = previous_writer_digest

    receipt = replace(bundle.operator_receipt, repository_head=launch_head)
    return replace(bundle, operator_receipt=receipt)
