import json
import inspect
import sys

import adapters.virtual_world_situation_temporal_context_schema as schema
from adapters.virtual_world_situation_temporal_context_schema import (
    build_virtual_world_situation_temporal_context,
    build_virtual_world_situation_temporal_context_schema_summary,
    validate_virtual_world_situation_temporal_context,
)

ANCHOR = {"anchor_id": "operator_anchor", "anchor_kind": "bounded_logical_interval", "label": "operator logical interval", "logical_start": 1, "logical_end": 2, "sequence_index": 0}


def main():
    valid = build_virtual_world_situation_temporal_context(
        temporal_type="situation_interval_candidate",
        situation_id="operator_situation",
        temporal_anchor=ANCHOR,
    )
    invalid = build_virtual_world_situation_temporal_context(
        temporal_type="situation_interval_candidate",
        situation_id="operator_situation",
        temporal_anchor={**ANCHOR, "logical_end": 0},
    )
    stable_a = build_virtual_world_situation_temporal_context(
        temporal_type="symbolic_time_candidate",
        situation_id="operator_situation",
        temporal_anchor={"label": "symbolic", "anchor_kind": "symbolic_marker", "anchor_id": "symbolic_a"},
        metadata={"b": {"z": 1, "a": 2}, "a": "x"},
    )
    stable_b = build_virtual_world_situation_temporal_context(
        temporal_type="symbolic_time_candidate",
        situation_id="operator_situation",
        temporal_anchor={"anchor_id": "symbolic_a", "anchor_kind": "symbolic_marker", "label": "symbolic"},
        metadata={"a": "x", "b": {"a": 2, "z": 1}},
    )
    tampered = dict(valid)
    tampered["temporal_context_id"] = str(tampered["temporal_context_id"]) + "tampered"
    logical_bad = build_virtual_world_situation_temporal_context(
        temporal_type="situation_interval_candidate",
        situation_id="operator_situation",
        temporal_anchor={**ANCHOR, "logical_start": 3, "logical_end": 2},
    )
    malformed_boundary = build_virtual_world_situation_temporal_context(
        temporal_type="situation_interval_candidate",
        situation_id="operator_situation",
        temporal_anchor=ANCHOR,
        metadata={"temporal_boundary_classification": ""},
    )
    malformed_confidence = build_virtual_world_situation_temporal_context(
        temporal_type="situation_interval_candidate",
        situation_id="operator_situation",
        temporal_anchor=ANCHOR,
        metadata={"temporal_confidence_state": []},
    )
    forbidden_nested = build_virtual_world_situation_temporal_context(
        temporal_type="situation_interval_candidate",
        situation_id="operator_situation",
        temporal_anchor=ANCHOR,
        metadata={"nested": [{"memory_write_requested": True}]},
    )
    forbidden_anchor = build_virtual_world_situation_temporal_context(
        temporal_type="situation_interval_candidate",
        situation_id="operator_situation",
        temporal_anchor={**ANCHOR, "preserved": {"timer_requested": True}},
    )
    source = inspect.getsource(schema)
    forbidden_tokens = ["import time", "from time", "datetime", "time.time", "monotonic", "perf_counter", "now()"]
    report = {
        "valid_case_passed": validate_virtual_world_situation_temporal_context(valid),
        "invalid_case_passed": validate_virtual_world_situation_temporal_context(invalid) is False and invalid["blocked_reasons"] == ["logical_end_before_logical_start"],
        "deterministic_id_passed": stable_a["temporal_context_id"] == stable_b["temporal_context_id"],
        "tamper_detection_passed": validate_virtual_world_situation_temporal_context(tampered) is False,
        "logical_interval_validation_passed": logical_bad["blocked_reasons"] == ["logical_end_before_logical_start"],
        "malformed_boundary_validation_passed": malformed_boundary["blocked_reasons"] == ["malformed_temporal_boundary_class"],
        "malformed_confidence_validation_passed": malformed_confidence["blocked_reasons"] == ["malformed_temporal_confidence_state"],
        "recursive_forbidden_metadata_passed": forbidden_nested["blocked_reasons"] == ["memory_write_requested"],
        "recursive_forbidden_anchor_passed": forbidden_anchor["blocked_reasons"] == ["timer_requested"],
        "no_clock_access_passed": all(token not in source for token in forbidden_tokens),
        "schema_summary": build_virtual_world_situation_temporal_context_schema_summary(),
    }
    report["all_checks_passed"] = all(value for key, value in report.items() if key != "schema_summary")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
