"""Compatibility import for the retained legacy SpreadingActivation class.

Round123 keeps legacy root tests importable without inventing or faking
spreading-activation behavior.  The implementation remains the retained v32
module under ``legacy/eve_modules``; this root module only restores the import
path expected by root-level legacy files.
"""

from __future__ import annotations

from legacy.eve_modules.spreading_activation import SpreadingActivation

__all__ = ["SpreadingActivation"]
