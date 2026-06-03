"""Compatibility import for the retained legacy WorkingMemory class.

Round128 keeps legacy root tests importable without inventing or faking
working-memory behavior. The implementation remains the retained v32 module
under ``legacy/eve_modules``; this root module only restores the import path
expected by root-level legacy files.
"""

from __future__ import annotations

from legacy.eve_modules.working_memory import WMSlot, WorkingMemory

__all__ = ["WMSlot", "WorkingMemory"]
