"""Compatibility import for the retained legacy DefaultModeNetwork class.

Round138 keeps legacy root tests importable without inventing or faking DMN
behavior. The implementation remains the retained v32 module under
``legacy/eve_modules``; this root module only restores the import path expected
by root-level legacy files.
"""

from __future__ import annotations

from legacy.eve_modules.dmn import DefaultModeNetwork

__all__ = ["DefaultModeNetwork"]
