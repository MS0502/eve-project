"""Operator report for read-only cross-modal binding preflight schema."""

import json
from adapters.cross_modal_binding_preflight_schema import build_cross_modal_binding_preflight_schema_summary

if __name__ == "__main__":
    summary = build_cross_modal_binding_preflight_schema_summary()
    print(json.dumps(summary, indent=2))
