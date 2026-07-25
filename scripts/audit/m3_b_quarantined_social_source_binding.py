#!/usr/bin/env python3
"""Recalculable audit for the seven-axis quarantined social source binding."""
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

from core.m3_b_quarantined_social_source_binding import (  # noqa: E402
    APPRAISAL_SCHEMA_VERSION,
    SOCIAL_RELATIONSHIP_AXES,
    QuarantinedSocialRawRecord,
    QuarantinedSocialSourceBindingError,
    derive_quarantined_social_axis_evidence,
    quarantined_social_raw_observation_digest,
    quarantined_social_source_bindings,
)

SCHEMA_VERSION = "eve.m3-b.quarantined-social-source-binding-audit.v1"
BASELINE_SHA = "d82652d74b9051d227ec415b648ec209b9de8b3d"
CORE_PATH = ROOT / "core/m3_b_quarantined_social_source_binding.py"


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


def _ticks(axis: str) -> tuple[int, ...]:
    if axis == "care_drive":
        return (1, 3)
    if axis == "attachment":
        return (1, 7, 13)
    return (1, 5, 9)


def _offset(tick: int) -> float:
    return min(0.08, ((tick - 1) // 4) * 0.02)


def _values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = _offset(tick)
    if axis == "social_pain":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("injury_evidence_score", 0.26 + offset),
            ("intent_confidence", 0.60 + offset),
            ("recurrence_count", (tick - 1) // 4),
            ("source_trust", 0.80 - offset),
        )
    if axis == "social_trust":
        return (
            ("contradiction_count", (tick - 1) // 8),
            ("fulfilled_commitment_count", 2 + (tick - 1) // 4),
            ("observation_span_ticks", 10 + tick),
            ("repair_count", 1 + (tick - 1) // 8),
            ("source_trust", 0.84 - offset),
        )
    if axis == "attachment":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("interaction_continuity", 0.66 + offset),
            ("mutual_reliability", 0.78 - offset / 2),
            ("relationship_span_ticks", 24 + tick),
            ("separation_tolerance", 0.74 - offset / 2),
        )
    if axis == "care_drive":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("capability_to_help", 0.72 - offset),
            ("consent_status", "granted" if tick == 1 else "limited"),
            ("cost_boundary", 0.22 + offset),
            ("welfare_need_score", 0.54 + offset),
        )
    if axis == "loneliness_pressure":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("available_relationship_context", 0.68 - offset),
            ("chosen_solitude_flag", tick == 1),
            ("meaningful_contact_gap_ticks", 3 + tick),
            ("unmet_connection_signal_count", (tick - 1) // 4),
        )
    if axis == "belonging_need":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("context_span_ticks", 10 + tick),
            ("group_continuity", 0.76 - offset),
            ("reciprocal_inclusion_count", 2 + (tick - 1) // 4),
            ("role_clarity", 0.73 - offset / 2),
        )
    if axis == "rejection_sensitivity":
        return (
            ("ambiguous_signal_count", 1 + (tick - 1) // 4),
            ("false_positive_count", (tick - 1) // 8),
            ("observation_span_ticks", 10 + tick),
            ("source_trust", 0.81 - offset),
            ("verified_rejection_count", (tick - 1) // 8),
        )
    raise AssertionError(axis)


def _record(axis: str, tick: int) -> QuarantinedSocialRawRecord:
    observation_id = f"audit:{axis}:observation:{tick}"
    source_instance_id = "audit:quarantined-social-source:v1"
    source_snapshot_id = f"audit:{axis}:snapshot:{tick}"
    source_schema_version = "audit.quarantined-social-source.v1"
    source_integrity_digest = _sha(f"source:{axis}:{tick}")
    quarantine_trace_id = f"audit:{axis}:quarantine:{tick}"
    quarantine_input_digest = _sha(f"quarantine-input:{axis}:{tick}")
    quarantine_integrity_digest = _sha(f"quarantine-integrity:{axis}:{tick}")
    appraisal_trace_id = f"audit:{axis}:appraisal:{tick}"
    appraisal_input_digest = quarantine_integrity_digest
    appraisal_integrity_digest = _sha(f"appraisal-integrity:{axis}:{tick}")
    raw_values = _values(axis, tick)
    raw_observation_digest = quarantined_social_raw_observation_digest(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        quarantine_trace_id=quarantine_trace_id,
        quarantine_input_digest=quarantine_input_digest,
        quarantine_integrity_digest=quarantine_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_values=raw_values,
    )
    return QuarantinedSocialRawRecord(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        quarantine_trace_id=quarantine_trace_id,
        quarantine_input_digest=quarantine_input_digest,
        quarantine_integrity_digest=quarantine_integrity_digest,
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
        imports & {"os", "psutil", "sqlite3", "subprocess", "threading", "time"}
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
    binding_set = quarantined_social_source_bindings()
    first = {
        axis: derive_quarantined_social_axis_evidence(
            tuple(_record(axis, tick) for tick in _ticks(axis))
        )
        for axis in SOCIAL_RELATIONSHIP_AXES
    }
    second = {
        axis: derive_quarantined_social_axis_evidence(
            tuple(_record(axis, tick) for tick in _ticks(axis))
        )
        for axis in SOCIAL_RELATIONSHIP_AXES
    }
    static = _static_surface()
    errors: list[str] = []
    if binding_set.appraised_binding_count != 7:
        errors.append("quarantined social binding count is not exactly seven")
    if binding_set.total_bound_axis_count != 19 or binding_set.remaining_axis_count != 18:
        errors.append("combined registry binding progress is not exact 19+18")
    if first != second:
        errors.append("quarantined social evidence derivation is not deterministic")
    if any(item.confidence <= 0.0 for item in first.values()):
        errors.append("quarantined social evidence contains non-positive confidence")
    if any(not 0.0 <= item.value <= 1.0 for item in first.values()):
        errors.append("quarantined social evidence value is outside registry bounds")
    if not static["no_io_polling_scheduler_event_or_runtime_surface"]:
        errors.append("quarantined social binding exposes a forbidden live or I/O surface")

    sample = _record("social_pain", 1)
    rejection_checks: dict[str, bool] = {}
    checks = (
        ("raw_digest_mismatch", lambda: replace(sample, raw_observation_digest=_sha("wrong"))),
        ("quarantine_unverified", lambda: replace(sample, quarantine_verified=False)),
        ("appraisal_unverified", lambda: replace(sample, appraisal_verified=False)),
        (
            "quarantine_appraisal_chain_broken",
            lambda: replace(sample, appraisal_input_digest=_sha("wrong-chain")),
        ),
        ("raw_social_feedback", lambda: replace(sample, raw_social_feedback_source=True)),
        ("hardware_direct_input", lambda: replace(sample, hardware_direct_input=True)),
        ("synthetic", lambda: replace(sample, synthetic=True)),
        ("proposal_only", lambda: replace(sample, proposal_only=True)),
        ("registry_owner_source", lambda: replace(sample, registry_owner_source=True)),
        ("runtime_polled", lambda: replace(sample, runtime_polled=True)),
        ("noncanonical_acquisition", lambda: replace(sample, acquisition_method="unverified")),
        ("noncanonical_verification", lambda: replace(sample, verification_method="none")),
        ("noncanonical_quarantine_method", lambda: replace(sample, quarantine_method="bypassed")),
        ("noncanonical_appraisal_method", lambda: replace(sample, appraisal_method="caller_claim")),
        (
            "insufficient_social_record_count",
            lambda: derive_quarantined_social_axis_evidence(
                tuple(_record("social_pain", tick) for tick in (1, 9))
            ),
        ),
    )
    for name, action in checks:
        try:
            action()
        except QuarantinedSocialSourceBindingError:
            rejection_checks[name] = True
        else:
            rejection_checks[name] = False
            errors.append(f"forbidden quarantined social evidence form accepted: {name}")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_quarantined_social_source_binding",
        "audit_fixture_only": True,
        "audit_fixture_is_production_observation": False,
        "appraised_binding_count": binding_set.appraised_binding_count,
        "total_bound_axis_count": binding_set.total_bound_axis_count,
        "remaining_axis_count": binding_set.remaining_axis_count,
        "bound_axes": list(SOCIAL_RELATIONSHIP_AXES),
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
        "quarantine_and_appraisal_gate_rejection_verified": all(rejection_checks.values()),
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
        "next_required_artifact": "appraised source bindings for the remaining 18 axes; retained real capture remains separate",
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
