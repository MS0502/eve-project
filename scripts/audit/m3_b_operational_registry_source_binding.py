#!/usr/bin/env python3
"""Recalculable audit for the four-axis operational source binding."""
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

from core.m3_b_operational_registry_source_binding import (  # noqa: E402
    OPERATIONAL_AXES,
    OperationalRegistryRawRecord,
    OperationalRegistrySourceBindingError,
    derive_operational_axis_evidence,
    operational_raw_observation_digest,
    operational_registry_source_bindings,
)

SCHEMA_VERSION = "eve.m3-b.operational-registry-source-binding-audit.v1"
BASELINE_SHA = "0c0cfdd5a32f25cab52f62e15895475ea64afd63"
CORE_PATH = ROOT / "core/m3_b_operational_registry_source_binding.py"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = (tick - 1) * 0.02
    if axis == "energy_budget":
        value: dict[str, object] = {
            "available_cpu_budget": 0.82 - offset,
            "available_memory_budget": 0.78 - offset,
            "battery_governor_band": 0.74 - offset,
            "foreground_load": 0.24 + offset,
            "sampling_window_ticks": 10,
        }
    elif axis == "fatigue_pressure":
        value = {
            "active_processing_ticks": 4 + tick,
            "queue_pressure": 0.30 + offset,
            "recovery_interval_ticks": 3,
            "sampling_window_ticks": 10,
            "task_switch_count": 2 + tick,
        }
    elif axis == "recovery_need":
        value = {
            "active_processing_ticks": 5 + tick,
            "cooldown_ticks": 2,
            "recent_overload_count": tick,
            "sampling_window_ticks": 10,
            "successful_recovery_count": 1,
        }
    elif axis == "overload_risk":
        value = {
            "concurrent_demand_count": 2 + tick,
            "latency_budget_ratio": 0.32 + offset,
            "memory_pressure_ratio": 0.36 + offset,
            "queue_depth": 1 + tick,
            "thermal_governor_band": 0.28 + offset,
        }
    else:
        raise AssertionError(axis)
    return tuple(sorted(value.items()))


def _record(axis: str, tick: int) -> OperationalRegistryRawRecord:
    observation_id = f"audit:{axis}:observation:{tick}"
    source_instance_id = "audit:operational-source:v1"
    source_snapshot_id = f"audit:{axis}:snapshot:{tick}"
    source_schema_version = "audit.operational-source.v1"
    source_integrity_digest = _sha(f"audit:source:{axis}:{tick}")
    raw_values = _values(axis, tick)
    raw_observation_digest = operational_raw_observation_digest(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        raw_values=raw_values,
    )
    return OperationalRegistryRawRecord(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        raw_observation_digest=raw_observation_digest,
        raw_values=raw_values,
    )


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
        "no_io_polling_scheduler_event_or_runtime_surface": not forbidden_imports
        and not forbidden_calls,
    }


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    del root
    binding_set = operational_registry_source_bindings()
    first_evidence = {
        axis: derive_operational_axis_evidence(
            tuple(_record(axis, tick) for tick in (1, 2, 3))
        )
        for axis in OPERATIONAL_AXES
    }
    second_evidence = {
        axis: derive_operational_axis_evidence(
            tuple(_record(axis, tick) for tick in (1, 2, 3))
        )
        for axis in OPERATIONAL_AXES
    }
    static = _static_surface()
    errors: list[str] = []
    if binding_set.binding_count != 4 or binding_set.remaining_axis_count != 33:
        errors.append("operational binding coverage is not exact 4+33")
    if tuple(first_evidence) != OPERATIONAL_AXES:
        errors.append("derived evidence axis order is not canonical")
    if first_evidence != second_evidence:
        errors.append("operational evidence derivation is not deterministic")
    if any(item.confidence <= 0.0 for item in first_evidence.values()):
        errors.append("operational evidence contains non-positive confidence")
    if any(not 0.0 <= item.value <= 1.0 for item in first_evidence.values()):
        errors.append("operational evidence value is outside registry bounds")
    if not static["no_io_polling_scheduler_event_or_runtime_surface"]:
        errors.append("operational binding exposes a forbidden live or I/O surface")

    sample = _record("energy_budget", 1)
    rejection_checks: dict[str, bool] = {}
    checks = (
        ("raw_digest_mismatch", lambda: replace(sample, raw_observation_digest=_sha("wrong"))),
        ("synthetic", lambda: replace(sample, synthetic=True)),
        ("proposal_only", lambda: replace(sample, proposal_only=True)),
        ("registry_owner_source", lambda: replace(sample, registry_owner_source=True)),
        ("runtime_polled", lambda: replace(sample, runtime_polled=True)),
        (
            "noncanonical_acquisition_method",
            lambda: replace(sample, acquisition_method="unverified_caller_claim"),
        ),
        (
            "noncanonical_verification_method",
            lambda: replace(sample, verification_method="none"),
        ),
        (
            "noncanonical_model_or_rule_version",
            lambda: replace(sample, model_or_rule_version="unversioned"),
        ),
        (
            "noncanonical_source_family",
            lambda: replace(sample, source_family="synthetic_operational_metrics"),
        ),
        (
            "insufficient_record_count",
            lambda: derive_operational_axis_evidence(
                (_record("energy_budget", 1), _record("energy_budget", 2))
            ),
        ),
    )
    for name, action in checks:
        try:
            action()
        except OperationalRegistrySourceBindingError:
            rejection_checks[name] = True
        else:
            rejection_checks[name] = False
            errors.append(f"forbidden operational evidence form accepted: {name}")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_operational_source_binding",
        "audit_fixture_only": True,
        "audit_fixture_is_production_observation": False,
        "binding_count": binding_set.binding_count,
        "remaining_axis_count": binding_set.remaining_axis_count,
        "bound_axes": list(OPERATIONAL_AXES),
        "blockers": list(binding_set.blockers),
        "binding_set_digest": binding_set.binding_set_digest,
        "derived_evidence": {
            axis: {
                "confidence": evidence.confidence,
                "evidence_digest": evidence.evidence_digest,
                "raw_observation_digest": evidence.raw_observation_digest,
                "source_integrity_digest": evidence.source_integrity_digest,
                "value": evidence.value,
            }
            for axis, evidence in first_evidence.items()
        },
        "deterministic_evidence_equal": first_evidence == second_evidence,
        "raw_digest_recalculation_verified": (
            sample.raw_observation_digest
            == sample.recalculated_raw_observation_digest
        ),
        "canonical_provenance_rejection_verified": all(
            rejection_checks[name]
            for name in (
                "noncanonical_acquisition_method",
                "noncanonical_verification_method",
                "noncanonical_model_or_rule_version",
                "noncanonical_source_family",
            )
        ),
        "rejection_checks": rejection_checks,
        "static_surface": static,
        "production_capture_present": False,
        "runtime_capture_installed": False,
        "hardware_polling_installed": False,
        "scheduler_installed": False,
        "persistence_accessed": False,
        "event_append_performed": False,
        "registry_owner_materialized": False,
        "observation_window_started": False,
        "observation_window_satisfied": False,
        "m3_b_complete": False,
        "m3_c_open": False,
        "m3_e_authority_open": False,
        "cutover_authorized": False,
        "legacy_runtime_authoritative": True,
        "legacy_persistence_authoritative": True,
        "next_required_artifact": "appraised source bindings for the remaining 33 axes; real retained operational capture remains separate",
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
            "binding_count": report["binding_count"],
            "blockers": report["blockers"],
            "errors": report["errors"],
            "remaining_axis_count": report["remaining_axis_count"],
            "report_digest": report["report_digest"],
        }
    _write(output, args.output, args.pretty)
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
