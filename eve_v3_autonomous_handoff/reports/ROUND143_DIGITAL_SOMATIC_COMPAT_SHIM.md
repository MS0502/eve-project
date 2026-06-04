# Round143 — DigitalSomatic compatibility shim

Round143 added `digital_somatic.py` as a minimal root import compatibility shim. The shim re-exports `DigitalSomatic` from the retained legacy implementation at `legacy/eve_modules/digital_somatic.py`; it does not implement replacement behavior.

Decision: compatibility shim applied. Production persistence remains NO-GO; `runtime_mapping_enabled` remains default false; enforcement remains disabled.

Validation JSON: `eve_v3_autonomous_handoff/validation/ROUND143_DIGITAL_SOMATIC_COMPAT_SHIM_STATUS.json`.
