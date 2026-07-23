#!/usr/bin/env python3
"""Audit the disconnected immutable legacy 26-axis M3-B capture envelope."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hormone_system import HormoneSystem

from core.m3_b_legacy_affect_capture import (
    CAPTURE_SCHEMA_VERSION,
    LEGACY_AXIS_ORDER,
    capture_legacy_hormone_state,
)

SCHEMA_VERSION = "eve.m3-b.legacy-26-axis-capture-check.v1"
BASELINE_SHA = "97ac96b8bdf54c7fbc74d4b0346ceff49624eaec"
SOURCE_PATH = Path("hormone_system.py")


class LegacyCaptureAuditError(ValueError):
    """Raised when the authoritative legacy catalog cannot be parsed."""


def _string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def parse_authoritative_axis_order(root: Path = ROOT) -> tuple[str, ...]:
    tree = ast.parse(
        (root / SOURCE_PATH).read_text(encoding="utf-8"),
        filename=SOURCE_PATH.as_posix(),
    )
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_init_all_hormones"
    ]
    if len(functions) != 1:
        raise LegacyCaptureAuditError("expected one _init_all_hormones definition")
    axes: list[str] = []
    for node in ast.walk(functions[0]):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "config" for target in targets):
            continue
        value = node.value
        if not isinstance(value, (ast.List, ast.Tuple)):
            continue
        for item in value.elts:
            if not isinstance(item, (ast.List, ast.Tuple)) or len(item.elts) != 6:
                continue
            axis = _string(item.elts[0])
            if axis is not None:
                axes.append(axis)
    if len(axes) != 26 or len(set(axes)) != 26:
        raise LegacyCaptureAuditError("authoritative source must define 26 unique axes")
    return tuple(axes)


def _source_signature(source: HormoneSystem) -> dict[str, Any]:
    return {
        "active_hormones": list(source.active_hormones),
        "axes": [
            {
                "axis": axis,
                "baseline": float(source.hormones[axis].baseline),
                "decay_rate": float(source.hormones[axis].decay_rate),
                "level": float(source.hormones[axis].level),
                "name": source.hormones[axis].name,
                "phase": source.hormones[axis].phase,
                "reactivity": float(source.hormones[axis].reactivity),
                "tier": source.hormones[axis].tier,
            }
            for axis in source.hormones
        ],
        "hormone_order": list(source.hormones),
        "phase": source.phase,
        "sim_hour": float(source.sim_hour),
        "stage": source.stage,
        "time": float(source.time),
    }


def _digest(value: Mapping[str, Any]) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    authoritative_order = parse_authoritative_axis_order(root)
    source = HormoneSystem()
    before = _source_signature(source)
    first = capture_legacy_hormone_state(
        source,
        source_instance_id="m3-b:audit:legacy-source:v1",
        source_snapshot_id="m3-b:audit:legacy-snapshot:v1",
    )
    middle = _source_signature(source)
    second = capture_legacy_hormone_state(
        source,
        source_instance_id="m3-b:audit:legacy-source:v1",
        source_snapshot_id="m3-b:audit:legacy-snapshot:v1",
    )
    after = _source_signature(source)
    observations = first.to_axis_observations()

    errors: list[str] = []
    if authoritative_order != LEGACY_AXIS_ORDER:
        errors.append("capture axis order differs from authoritative source")
    if before != middle or middle != after:
        errors.append("capture changed the source object")
    if first.to_mapping() != second.to_mapping() or first.capture_digest != second.capture_digest:
        errors.append("same source/identity did not produce deterministic capture evidence")
    if len(first.axes) != 26 or tuple(axis.axis for axis in first.axes) != LEGACY_AXIS_ORDER:
        errors.append("capture did not preserve exact 26-axis order")
    if len(observations) != 26 or any(
        observation.source_family != "legacy_mutable_hormone" for observation in observations
    ):
        errors.append("capture did not produce exact legacy AxisObservation output")
    if any(
        (
            first.source_mutated,
            first.persistence_accessed,
            first.event_append_performed,
            first.live_behavior_changed,
            first.observation_window_started,
            first.m3_c_open,
            first.m3_e_authority_open,
            first.cutover_authorized,
        )
    ):
        errors.append("capture gained mutation or authority")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "authority": "shadow_only",
        "axis_count": len(first.axes),
        "axis_order_matches_authoritative_source": authoritative_order == LEGACY_AXIS_ORDER,
        "axis_observation_count": len(observations),
        "before_after_source_equal": before == middle == after,
        "capture_digest": first.capture_digest,
        "deterministic_repeat_equal": first.to_mapping() == second.to_mapping(),
        "exact_source_type_verified": first.exact_source_type_verified,
        "legacy_capture_ready": not errors,
        "legacy_observation_window_started": False,
        "remaining_blockers": ["REGISTRY_OBSERVED_VALUE_OWNER_ABSENT"],
        "m3_b_complete": False,
        "m3_c_open": False,
        "m3_e_authority_open": False,
        "persistence_accessed": False,
        "event_append_performed": False,
        "live_behavior_changed": False,
        "cutover_authorized": False,
        "source_signature_digest": _digest(before),
        "source_integrity_digest": first.source_integrity_digest,
        "errors": errors,
    }
    report["report_digest"] = _digest(report)
    return report


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _write(value: Mapping[str, Any], output: Path | None, pretty: bool) -> None:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    ) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    report = audit_repository()
    output: Mapping[str, Any] = report
    if args.summary_only:
        output = {
            "axis_count": report["axis_count"],
            "axis_observation_count": report["axis_observation_count"],
            "capture_digest": report["capture_digest"],
            "errors": report["errors"],
            "legacy_capture_ready": report["legacy_capture_ready"],
            "m3_b_complete": report["m3_b_complete"],
            "m3_c_open": report["m3_c_open"],
            "remaining_blockers": report["remaining_blockers"],
            "report_digest": report["report_digest"],
        }
    _write(output, args.output, args.pretty)
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
