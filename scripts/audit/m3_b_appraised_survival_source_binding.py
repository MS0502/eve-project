#!/usr/bin/env python3
"""Recalculable audit for the two-axis appraised survival source binding."""
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

from core.m3_b_appraised_survival_source_binding import (  # noqa: E402
    APPRAISAL_SCHEMA_VERSION,
    APPRAISED_SURVIVAL_AXES,
    AppraisedSurvivalRawRecord,
    AppraisedSurvivalSourceBindingError,
    appraised_survival_raw_observation_digest,
    appraised_survival_source_bindings,
    derive_appraised_survival_axis_evidence,
)

SCHEMA_VERSION = "eve.m3-b.appraised-survival-source-binding-audit.v1"
BASELINE_SHA = "6c89e717c8bd702b86a322d0dbec439ea1ff3944"
CORE_PATH = ROOT / "core/m3_b_appraised_survival_source_binding.py"


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
    if axis == "stress_load":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("controllability_score", 0.74 - offset),
            ("demand_score", 0.40 + offset),
            ("overload_score", 0.26 + offset),
            ("uncertainty_score", 0.30 + offset),
        )
    if axis == "stability_need":
        return (
            ("invariant_failure_count", tick - 1),
            ("pending_migration_count", tick),
            ("replay_divergence_count", tick - 1),
            ("rollback_readiness_score", 0.84 - offset),
            ("sampling_window_ticks", 10),
        )
    raise AssertionError(axis)


def _record(axis: str, tick: int) -> AppraisedSurvivalRawRecord:
    observation_id = f"audit:{axis}:observation:{tick}"
    source_instance_id = "audit:appraised-survival-source:v1"
    source_snapshot_id = f"audit:{axis}:snapshot:{tick}"
    source_schema_version = "audit.appraised-survival-source.v1"
    source_integrity_digest = _sha(f"source:{axis}:{tick}")
    appraisal_trace_id = f"audit:{axis}:appraisal:{tick}"
    appraisal_input_digest = _sha(f"appraisal-input:{axis}:{tick}")
    appraisal_integrity_digest = _sha(f"appraisal-integrity:{axis}:{tick}")
    raw_values = _values(axis, tick)
    raw_observation_digest = appraised_survival_raw_observation_digest(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_values=raw_values,
    )
    return AppraisedSurvivalRawRecord(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
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
        & {"os", "psutil", "sqlite3", "subprocess", "threading", "time"}
    )
    forbidden_calls = sorted(
        calls
        & {
            "append_event",
            "connect",
            "emit",
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
    binding_set = appraised_survival_source_bindings()
    first = {
        axis: derive_appraised_survival_axis_evidence(
            tuple(_record(axis, tick) for tick in (1, 2, 3))
        )
        for axis in APPRAISED_SURVIVAL_AXES
    }
    second = {
        axis: derive_appraised_survival_axis_evidence(
            tuple(_record(axis, tick) for tick in (1, 2, 3))
        )
        for axis in APPRAISED_SURVIVAL_AXES
    }
    static = _static_surface()
    errors: list[str] = []
    if binding_set.appraised_binding_count != 2:
        errors.append("appraised survival binding count is not exactly two")
    if binding_set.total_bound_axis_count != 6 or binding_set.remaining_axis_count != 31:
        errors.append("combined registry binding progress is not exact 6+31")
    if first != second:
        errors.append("appraised survival evidence derivation is not deterministic")
    if any(item.confidence <= 0.0 for item in first.values()):
        errors.append("appraised survival evidence contains non-positive confidence")
    if any(not 0.0 <= item.value <= 1.0 for item in first.values()):
        errors.append("appraised survival evidence value is outside registry bounds")
    if not static["no_io_polling_scheduler_event_or_runtime_surface"]:
        errors.append("appraised survival binding exposes a forbidden live or I/O surface")

    sample = _record("stress_load", 1)
    rejection_checks: dict[str, bool] = {}
    checks = (
        ("raw_digest_mismatch", lambda: replace(sample, raw_observation_digest=_sha("wrong"))),
        ("appraisal_unverified", lambda: replace(sample, appraisal_verified=False)),
        ("raw_social_feedback", lambda: replace(sample, raw_social_feedback_source=True)),
        ("hardware_direct_input", lambda: replace(sample, hardware_direct_input=True)),
        ("synthetic", lambda: replace(sample, synthetic=True)),
        ("proposal_only", lambda: replace(sample, proposal_only=True)),
        ("registry_owner_source", lambda: replace(sample, registry_owner_source=True)),
        ("runtime_polled", lambda: replace(sample, runtime_polled=True)),
        ("noncanonical_acquisition", lambda: replace(sample, acquisition_method="unverified")),
        ("noncanonical_verification", lambda: replace(sample, verification_method="none")),
        ("noncanonical_appraisal_method", lambda: replace(sample, appraisal_method="caller_claim")),
        (
            "insufficient_record_count",
            lambda: derive_appraised_survival_axis_evidence(
                (_record("stress_load", 1), _record("stress_load", 2))
            ),
        ),
    )
    for name, action in checks:
        try:
            action()
        except AppraisedSurvivalSourceBindingError:
            rejection_checks[name] = True
        else:
            rejection_checks[name] = False
            errors.append(f"forbidden appraised survival evidence form accepted: {name}")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_appraised_survival_source_binding",
        "audit_fixture_only": True,
        "audit_fixture_is_production_observation": False,
        "appraised_binding_count": binding_set.appraised_binding_count,
        "total_bound_axis_count": binding_set.total_bound_axis_count,
        "remaining_axis_count": binding_set.remaining_axis_count,
        "bound_axes": list(APPRAISED_SURVIVAL_AXES),
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
            for axis, evidence in first.items()
        },
        "deterministic_evidence_equal": first == second,
        "raw_digest_recalculation_verified": (
            sample.raw_observation_digest == sample.recalculated_raw_observation_digest
        ),
        "appraisal_gate_rejection_verified": all(rejection_checks.values()),
        "rejection_checks": rejection_checks,
        "static_surface": static,
        "production_capture_present": False,
        "runtime_capture_installed": False,
        "hardware_polling_installed": False,
        "raw_social_feedback_ingested": False,
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
        "next_required_artifact": "appraised source bindings for the remaining 31 axes; retained real capture remains separate",
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
            "appraised_binding_count": report["appraised_binding_count"],
            "blockers": report["blockers"],
            "errors": report["errors"],
            "remaining_axis_count": report["remaining_axis_count"],
            "report_digest": report["report_digest"],
            "total_bound_axis_count": report["total_bound_axis_count"],
        }
    _write(output, args.output, args.pretty)
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
