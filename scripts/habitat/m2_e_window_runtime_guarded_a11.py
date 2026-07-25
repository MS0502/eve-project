#!/usr/bin/env python3
"""A11 persistence wrapper for the bounded guarded M2-E habitat runtime.

The reviewed A1 guarded runtime, EventEnvelope contract, and shadow projection
remain unchanged. This wrapper replaces only the habitat process' SQLite store
binding with the content-addressed A11 persistence implementation, then delegates
all CLI/recovery/freeze/evidence behavior to the merged guarded runtime.
"""
from __future__ import annotations

from core.sqlite_shadow_store_a11 import SQLiteShadowStore
from scripts.habitat import m2_e_window_runtime_guarded as _guarded

# Python resolves this global when the reviewed _store() function is called, so
# only the habitat persistence implementation changes. EventEnvelope creation and
# replay continue to use core.event_kernel/core.shadow_projection unchanged.
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
