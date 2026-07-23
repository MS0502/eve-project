#!/usr/bin/env python3
"""Recalculable audit for the registry 37-axis source-manifest preflight."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_b_registry_affect_owner import REGISTRY_AXIS_ORDER  # noqa: E402
from core.m3_b_registry_observation_source_manifest import (  # noqa: E402
    POSITIVE_CONFIDENCE_BLOCKER,
    SOURCE_BINDING_BLOCKER,
    RegistryObservationSourceManifestError,
    registry_observation_source_manifest,
)

SCHEMA_VERSION = "eve.m3-b.registry-observation-source-manifest-audit.v1"
BASELINE_SHA = "03bf2de7294ae21cef0961cd50e516be85330d13"
CORE_PATH = ROOT / "core/m3_b_registry_observation_source_manifest.py"


def _digest(value: Mapping[str, Any]) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _static_surface() -> dict[str, Any]:
    tree = ast.parse(CORE_PATH.read_text(encoding="utf-8"), filename=CORE_PATH.as_posix())
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    forbidden_imports = sorted(
        imports
        & {
            "os",
            "pathlib",
            "persistence",
            "psutil",
            "sqlite3",
            "subprocess",
            "threading",
            "time",
        }
    )
    forbidden_calls = sorted(
        calls
        & {
            "append_event",
            "connect",
            "emit",
            "mkdir",
            "open",
            "poll",
            "save",
            "start",
            "write",
            "write_text",
        }
    )
    return {
        "forbidden_calls": forbidden_calls,
        "forbidden_imports": forbidden_imports,
        "no_io_hardware_polling_scheduler_event_or_runtime_surface": not forbidden_imports
        and not forbidden_calls,
    }


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    del root
    first = registry_observation_source_manifest()
    second = registry_observation_source_manifest()
    static = _static_surface()
    errors: list[str] = []

    if first.axis_count != 37:
        errors.append("source manifest does not contain exactly 37 axes")
    if tuple(item.axis for item in first.entries) != REGISTRY_AXIS_ORDER:
        errors.append("source manifest axis order is not canonical")
    if not first.structurally_complete:
        errors.append("source manifest is not structurally complete")
    if first.to_mapping() != second.to_mapping():
        errors.append("source manifest construction is not deterministic")
    if first.manifest_digest != second.manifest_digest:
        errors.append("source manifest digest is not deterministic")
    if first.real_source_binding_count != 0:
        errors.append("preflight manifest incorrectly claims real source bindings")
    if first.blockers != (SOURCE_BINDING_BLOCKER, POSITIVE_CONFIDENCE_BLOCKER):
        errors.append("source manifest blockers are not exact")
    if not static["no_io_hardware_polling_scheduler_event_or_runtime_surface"]:
        errors.append("source manifest exposes a forbidden live or I/O surface")

    hardware_axes = tuple(
        item.axis for item in first.entries if item.hardware_direct_input_allowed
    )
    expected_hardware_axes = (
        "energy_budget",
        "fatigue_pressure",
        "recovery_need",
        "overload_risk",
    )
    if hardware_axes != expected_hardware_axes:
        errors.append("direct hardware source boundary is not exact")

    rejection_checks: dict[str, bool] = {}
    entry = first.entries[10]
    for name, mutation in (
        ("source_family_mutation", {"source_family": "tampered"}),
        ("raw_fields_mutation", {"required_raw_fields": ("fake_a", "fake_b")}),
        ("minimum_records_mutation", {"minimum_raw_record_count": 99}),
        ("proposal_only", {"proposal_only_allowed": True}),
        ("synthetic_values", {"synthetic_values_allowed": True}),
        ("registry_owner_source", {"registry_owner_as_source_allowed": True}),
        ("real_binding_claim", {"real_source_binding_present": True}),
        ("runtime_capture_claim", {"runtime_capture_installed": True}),
    ):
        try:
            replace(entry, **mutation)
        except RegistryObservationSourceManifestError:
            rejection_checks[name] = True
        else:
            rejection_checks[name] = False
            errors.append(f"forbidden source-manifest mutation accepted: {name}")

    if any(
        (
            first.real_observation_values_present,
            first.real_source_bindings_present,
            first.capture_ready,
            first.runtime_capture_installed,
            first.hardware_polling_installed,
            first.scheduler_installed,
            first.persistence_accessed,
            first.event_append_performed,
            first.observation_window_started,
            first.observation_window_satisfied,
            first.m3_b_complete,
            first.m3_c_open,
            first.m3_e_authority_open,
            first.cutover_authorized,
        )
    ):
        errors.append("source manifest granted forbidden capture, window, or authority")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_registry_observation_source_manifest_preflight",
        "axis_count": first.axis_count,
        "canonical_axis_order": tuple(item.axis for item in first.entries)
        == REGISTRY_AXIS_ORDER,
        "structurally_complete": first.structurally_complete,
        "deterministic_manifest_equal": first.to_mapping() == second.to_mapping(),
        "manifest_digest": first.manifest_digest,
        "entry_digest_count": len({item.entry_digest for item in first.entries}),
        "hardware_direct_axes": list(hardware_axes),
        "real_source_binding_count": first.real_source_binding_count,
        "blockers": list(first.blockers),
        "rejection_checks": rejection_checks,
        "static_surface": static,
        "real_observation_values_present": False,
        "real_source_bindings_present": False,
        "capture_ready": False,
        "runtime_capture_installed": False,
        "hardware_polling_installed": False,
        "scheduler_installed": False,
        "persistence_accessed": False,
        "event_append_performed": False,
        "observation_window_started": False,
        "observation_window_satisfied": False,
        "m3_b_complete": False,
        "m3_c_open": False,
        "m3_e_authority_open": False,
        "cutover_authorized": False,
        "legacy_runtime_authoritative": True,
        "legacy_persistence_authoritative": True,
        "next_required_artifact": "real source bindings and raw-schema contracts for all 37 entries before any capture package",
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
            "blockers": report["blockers"],
            "errors": report["errors"],
            "real_source_binding_count": report["real_source_binding_count"],
            "report_digest": report["report_digest"],
            "structurally_complete": report["structurally_complete"],
        }
    _write(output, args.output, args.pretty)
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
