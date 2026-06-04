"""Compatibility import for the retained legacy DigitalSomatic class.

Round143 keeps legacy root tests importable without inventing or faking
DigitalSomatic behavior. The implementation remains the retained v32 module
under ``legacy/eve_modules``; this root module only restores the import path
expected by root-level legacy files and adapters.
"""

from __future__ import annotations

from legacy.eve_modules.digital_somatic import DigitalSomatic

__all__ = ["DigitalSomatic"]
