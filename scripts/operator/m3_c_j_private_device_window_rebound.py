#!/usr/bin/env python3
"""Run the single-use M3-C-J operator through the reviewed private-path rebind."""
from __future__ import annotations

from collections.abc import Sequence

import scripts.operator.m3_c_j_private_device_window as base_operator
from core.m3_c_j_private_device_path_rebind import (
    active_rebound_private_device_operator_authorization_packet,
    execute_rebound_private_device_observation_window,
    verify_rebound_private_device_operator_authorization,
)


def main(argv: Sequence[str] | None = None) -> int:
    previous_active = (
        base_operator.active_reviewed_private_device_operator_authorization_packet
    )
    previous_execute = (
        base_operator.execute_exact_reviewed_private_device_observation_window
    )
    previous_verify = base_operator.verify_reviewed_private_device_operator_authorization
    base_operator.active_reviewed_private_device_operator_authorization_packet = (
        active_rebound_private_device_operator_authorization_packet
    )
    base_operator.execute_exact_reviewed_private_device_observation_window = (
        execute_rebound_private_device_observation_window
    )
    base_operator.verify_reviewed_private_device_operator_authorization = (
        verify_rebound_private_device_operator_authorization
    )
    try:
        return base_operator.main(argv)
    finally:
        base_operator.active_reviewed_private_device_operator_authorization_packet = (
            previous_active
        )
        base_operator.execute_exact_reviewed_private_device_observation_window = (
            previous_execute
        )
        base_operator.verify_reviewed_private_device_operator_authorization = (
            previous_verify
        )


if __name__ == "__main__":
    raise SystemExit(main())
