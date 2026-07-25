#!/usr/bin/env python3
"""A11-enabled entrypoint for the bounded guarded M2-E habitat runtime.

The reviewed A1 guarded runtime remains byte-for-byte unchanged. This wrapper
rebinds only its local event/store/replay dependencies to the habitat-scoped A11
implementations, then delegates every CLI/recovery/freeze/evidence behavior to
the merged guarded runtime.
"""
from __future__ import annotations

from core.habitat_event_kernel_a11 import EventEnvelope
from core.habitat_shadow_projection_a11 import (
    ActivationLearnPairShadowState,
    replay_activation_learn_pair,
)
from core.sqlite_shadow_store_habitat_a11 import SQLiteShadowStore
from scripts.habitat import m2_e_window_runtime_guarded as _guarded

_guarded.EventEnvelope = EventEnvelope
_guarded.ActivationLearnPairShadowState = ActivationLearnPairShadowState
_guarded.replay_activation_learn_pair = replay_activation_learn_pair
_guarded.SQLiteShadowStore = SQLiteShadowStore

# Re-export the reviewed control surface so tests/operators can use this module
# exactly like the guarded runtime while keeping the original source immutable.
for _name in (
    "BOUNDED_STREAM",
    "DEFAULT_PRIVATE_ROOT",
    "HabitatError",
    "WindowConfig",
    "WindowState",
    "_enable_io_failure_reason",
    "_ensure_private_root",
    "_event",
    "_load_state",
    "_restore",
    "_save_state",
    "_snapshot_for",
    "resume_reviewed",
    "run_worker",
    "seal_now",
    "status",
):
    globals()[_name] = getattr(_guarded, _name)


def main(argv=None) -> int:
    return _guarded.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
