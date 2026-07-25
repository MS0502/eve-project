#!/usr/bin/env python3
"""Recalculable audit for the six-axis AGP-bounded expression-action binding."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_b_agp_bounded_expression_action_source_binding import (  # noqa: E402
    APPRAISAL_SCHEMA_VERSION,
    EXPRESSION_ACTION_AXES,
    AGPBoundedExpressionActionRawRecord,
    AGPBoundedExpressionActionSourceBindingError,
    agp_bounded_expression_action_raw_observation_digest,
    agp_bounded_expression_action_source_bindings,
    derive_agp_bounded_expression_action_axis_evidence,
)

SCHEMA_VERSION = "eve.m3-b.agp-bounded-expression-action-source-binding-audit.v1"
BASELINE_SHA = "4adbb575ffe517512af74c74fce58a49f7996128"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    if axis == "expression_pressure":
        return (("agp_anchor_coverage", 0.90), ("context_relevance", 0.84), ("pending_expression_count", tick), ("recurrence_count", tick), ("salience_score", 0.80))
    if axis == "expression_inhibition":
        return (("agp_failure_count", tick - 1), ("conflict_risk", 0.24), ("disclosure_risk", 0.20), ("fallback_required", tick > 1), ("uncertainty_score", 0.26))
    if axis == "action_readiness":
        return (("authorization_status", "authorized"), ("capability_available", True), ("feasible_action_count", tick + 1), ("reversibility", 0.86), ("selected_action_confidence", 0.82))
    if axis == "risk_tolerance":
        return (("authorization_scope", "audit-bounded-v1"), ("expected_cost", 0.22), ("reversibility", 0.84), ("safety_margin", 0.78), ("uncertainty_score", 0.24))
    if axis == "patience_level":
        return (("alternative_action_count", tick + 1), ("appraisal_version", APPRAISAL_SCHEMA_VERSION), ("cooldown_remaining", tick), ("deadline_pressure", 0.20), ("uncertainty_resolution_gain", 0.76))
    if axis == "conflict_avoidance":
        return (("appraisal_version", APPRAISAL_SCHEMA_VERSION), ("boundary_cost", 0.64), ("conflict_probability", 0.34), ("deescalation_option_count", tick + 1), ("harm_avoidance_gain", 0.80))
    raise AssertionError(axis)


def _record(axis: str, tick: int) -> AGPBoundedExpressionActionRawRecord:
    agp_integrity_digest = _sha(f"agp-integrity:{axis}:{tick}")
    kwargs = {
        "axis": axis,
        "logical_tick": tick,
        "observation_id": f"audit:{axis}:observation:{tick}",
        "source_instance_id": "audit:expression-action-source:v1",
        "source_snapshot_id": f"audit:{axis}:snapshot:{tick}",
        "source_schema_version": "audit.expression-action-source.v1",
        "source_integrity_digest": _sha(f"source:{axis}:{tick}"),
        "agp_trace_id": f"audit:{axis}:agp:{tick}",
        "agp_input_digest": _sha(f"agp-input:{axis}:{tick}"),
        "agp_integrity_digest": agp_integrity_digest,
        "agp_status": "passed",
        "appraisal_trace_id": f"audit:{axis}:appraisal:{tick}",
        "appraisal_input_digest": agp_integrity_digest,
        "appraisal_integrity_digest": _sha(f"appraisal-integrity:{axis}:{tick}"),
        "raw_values": _values(axis, tick),
    }
    digest = agp_bounded_expression_action_raw_observation_digest(**kwargs)
    return AGPBoundedExpressionActionRawRecord(**kwargs, raw_observation_digest=digest)


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    del root
    binding_set = agp_bounded_expression_action_source_bindings()
    first = {
        axis: derive_agp_bounded_expression_action_axis_evidence(
            (_record(axis, 1), _record(axis, 2))
        )
        for axis in EXPRESSION_ACTION_AXES
    }
    second = {
        axis: derive_agp_bounded_expression_action_axis_evidence(
            (_record(axis, 1), _record(axis, 2))
        )
        for axis in EXPRESSION_ACTION_AXES
    }
    errors: list[str] = []
    if binding_set.total_bound_axis_count != 37 or binding_set.remaining_axis_count != 0:
        errors.append("combined registry binding progress is not exact 37+0")
    if binding_set.retained_real_observation_count != 0:
        errors.append("binding completion fabricated retained real observations")
    if first != second:
        errors.append("expression-action evidence derivation is not deterministic")
    if any(item.confidence <= 0.0 or not 0.0 <= item.value <= 1.0 for item in first.values()):
        errors.append("expression-action evidence is outside bounded positive-confidence contract")

    sample = _record("expression_pressure", 1)
    checks = (
        ("raw_digest_mismatch", lambda: replace(sample, raw_observation_digest=_sha("wrong"))),
        ("agp_unverified", lambda: replace(sample, agp_trace_verified=False)),
        ("appraisal_unverified", lambda: replace(sample, appraisal_verified=False)),
        ("agp_appraisal_chain_broken", lambda: replace(sample, appraisal_input_digest=_sha("wrong-chain"))),
        ("raw_social_feedback", lambda: replace(sample, raw_social_feedback_source=True)),
        ("hardware_direct_input", lambda: replace(sample, hardware_direct_input=True)),
        ("runtime_polled", lambda: replace(sample, runtime_polled=True)),
        ("expression_or_action_executed", lambda: replace(sample, expression_or_action_executed=True)),
        ("memory_write", lambda: replace(sample, memory_write_performed=True)),
        ("cutover_authorized", lambda: replace(sample, cutover_authorized=True)),
        ("insufficient_record_count", lambda: derive_agp_bounded_expression_action_axis_evidence((_record("expression_pressure", 1),))),
    )
    rejection_checks: dict[str, bool] = {}
    for name, action in checks:
        try:
            action()
        except AGPBoundedExpressionActionSourceBindingError:
            rejection_checks[name] = True
        else:
            rejection_checks[name] = False
            errors.append(f"forbidden expression-action evidence form accepted: {name}")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_agp_bounded_expression_action_source_binding",
        "audit_fixture_only": True,
        "audit_fixture_is_production_observation": False,
        "appraised_binding_count": binding_set.appraised_binding_count,
        "total_bound_axis_count": binding_set.total_bound_axis_count,
        "remaining_axis_count": binding_set.remaining_axis_count,
        "retained_real_observation_count": binding_set.retained_real_observation_count,
        "positive_confidence_real_observation_count": binding_set.positive_confidence_real_observation_count,
        "bound_axes": list(EXPRESSION_ACTION_AXES),
        "blockers": list(binding_set.blockers),
        "binding_set_digest": binding_set.binding_set_digest,
        "derived_evidence": {
            axis: {
                "value": item.value,
                "confidence": item.confidence,
                "evidence_digest": item.evidence_digest,
            }
            for axis, item in first.items()
        },
        "deterministic_evidence_equal": first == second,
        "raw_digest_recalculation_verified": sample.raw_observation_digest == sample.recalculated_raw_observation_digest,
        "agp_and_appraisal_gate_rejection_verified": all(rejection_checks.values()),
        "rejection_checks": rejection_checks,
        "production_capture_present": False,
        "expression_or_action_executed": False,
        "memory_write_performed": False,
        "persistence_accessed": False,
        "event_append_performed": False,
        "registry_owner_materialized": False,
        "observation_window_started": False,
        "m3_b_complete": False,
        "m3_c_open": False,
        "m3_e_authority_open": False,
        "cutover_authorized": False,
        "legacy_runtime_authoritative": True,
        "legacy_persistence_authoritative": True,
        "next_required_artifact": "37-axis retained-real-observation capture preflight; production capture remains absent",
        "errors": errors,
    }
    report["report_digest"] = _digest(report)
    return report


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    report = audit_repository()
    text = json.dumps(
        report, ensure_ascii=False, sort_keys=True,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"), allow_nan=False
    ) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
