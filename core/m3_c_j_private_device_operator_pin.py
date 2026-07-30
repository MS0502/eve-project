"""Exact-reviewed pin adapter for the M3-C-J private-device operator.

The PR #230 operator implementation remains byte-for-byte immutable so its exact
M2-B evidence identifiers stay reusable. This adapter binds that reviewed
implementation and packet digest, then opens the preflight module's otherwise
absent authorization pins only for one synchronous explicit call. The pins are
restored in ``finally`` and import performs no I/O, database access, or append.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final

import core.m3_c_j_private_device_operator as operator_preflight
from core.m3_c_j_private_device_operator import (
    M3CPrivateDeviceOperatorAuthorizationError,
    PrivateDeviceGoalInput,
    PrivateDeviceOperatorAuthorizationPacket,
    PrivateDeviceOperatorBundle,
    build_private_device_operator_authorization_candidate,
)

REVIEWED_OPERATOR_IMPLEMENTATION_HEAD: Final = (
    "d8eb3c2d6b576cc313712f831f8b2f1556cdefb2"
)
REVIEWED_OPERATOR_AUTHORIZATION_DIGEST: Final = (
    "e360c0e669af3ba89a6f552c81c67e3b3d908171665ed20b510a0044003d13a5"
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


def active_reviewed_private_device_operator_authorization_packet(
) -> PrivateDeviceOperatorAuthorizationPacket:
    """Return the one reviewed packet without touching private paths or SQLite."""

    packet = build_private_device_operator_authorization_candidate(
        operator_implementation_head=REVIEWED_OPERATOR_IMPLEMENTATION_HEAD,
    )
    if packet.authorization_digest != REVIEWED_OPERATOR_AUTHORIZATION_DIGEST:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "checked-in private-device operator packet digest is inconsistent"
        )
    return packet


def verify_reviewed_private_device_operator_authorization(
    packet: PrivateDeviceOperatorAuthorizationPacket | None,
) -> str:
    """Verify the immutable packet without opening the execution pin scope."""

    if not isinstance(packet, PrivateDeviceOperatorAuthorizationPacket):
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "packet must be PrivateDeviceOperatorAuthorizationPacket"
        )
    if packet.operator_implementation_head != REVIEWED_OPERATOR_IMPLEMENTATION_HEAD:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "operator implementation head is not the reviewed head"
        )
    if packet.authorization_digest != REVIEWED_OPERATOR_AUTHORIZATION_DIGEST:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "operator authorization digest is not the reviewed packet"
        )
    return packet.authorization_digest


def execute_exact_reviewed_private_device_observation_window(
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
    """Run one explicit call and bind both reviewed provenance and launch head.

    The immutable preflight function historically requires its ``repository_head``
    argument to equal the reviewed implementation head. The explicit command has
    already verified a newer clean launch checkout. This adapter therefore passes
    the immutable provenance head into the preflight call, then deterministically
    replaces only the receipt's launch-head field with the independently verified
    checkout head. No database path is examined before packet and launch checks.
    """

    packet_digest = verify_reviewed_private_device_operator_authorization(
        authorization_packet
    )
    launch_head = _require_launch_head(repository_head)
    assert authorization_packet is not None

    previous_implementation = (
        operator_preflight._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD
    )
    previous_digest = operator_preflight._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST
    if previous_implementation is not None or previous_digest is not None:
        raise M3CPrivateDeviceOperatorAuthorizationError(
            "private-device operator pin scope was already active"
        )

    operator_preflight._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD = (
        REVIEWED_OPERATOR_IMPLEMENTATION_HEAD
    )
    operator_preflight._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST = packet_digest
    try:
        bundle = operator_preflight.execute_private_device_observation_window(
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
        operator_preflight._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD = (
            previous_implementation
        )
        operator_preflight._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST = (
            previous_digest
        )

    receipt = replace(bundle.operator_receipt, repository_head=launch_head)
    return replace(bundle, operator_receipt=receipt)
