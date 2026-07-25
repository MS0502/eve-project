#!/usr/bin/env python3
"""Stable phone/boot entrypoint for the guarded M2-E habitat runtime.

The implementation moved to ``m2_e_window_runtime_guarded`` so the original
operator command and Termux boot hook remain compatible while recovery/freeze
handling is independently testable.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.habitat.m2_e_window_runtime_guarded import (  # noqa: E402,F401
    HabitatError,
    main,
    resume_reviewed,
    run_worker,
    seal_now,
    status,
)


if __name__ == "__main__":
    raise SystemExit(main())
