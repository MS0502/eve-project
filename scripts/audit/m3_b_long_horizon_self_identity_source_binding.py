#!/usr/bin/env python3
"""Recalculable audit for the six-axis long-horizon self-identity binding."""
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

from core.m3_b_long_horizon_self_identity_source_binding import (  # noqa: E402
    APPRAISAL_SCHEMA_VERSION,
    SELF_IDENTITY_AXES,
    LongHorizonSelfIdentityRawRecord,
    LongHorizonSelfIdentitySourceBindingError,
    derive_long_horizon_self_identity_axis_evidence,
    long_horizon_self_identity_raw_observation_digest,
    long_horizon_self_identity_source_bindings,
)

SCHEMA_VERSION = "eve.m3-b.long-horizon-self-identity-source-binding-audit.v1"
BASELINE_SHA = "af363429f0d1065f803084d446a87c80753bc4cf"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = min(0.06, ((tick - 1) // 6) * 0.015)
    if axis == "self_coherence":
        return (("action_value_alignment", 0.78 - offset), ("narrative_conflict_count", 0), ("review_span_ticks", 24 + tick), ("self_model_version", "audit-self-v1"), ("value_consistency_score", 0.82 - offset))
    if axis == "self_respect":
        return (("appraisal_version", APPRAISAL_SCHEMA_VERSION), ("boundary_preservation_score", 0.84 - offset), ("coerced_action_count", 0), ("review_span_ticks", 24 + tick), ("self_denigration_rejection_count", 1 + (tick - 1) // 6))
    if axis == "identity_integrity":
        return (("constitutional_conflict_count", 0), ("provenance_gap_count", 0), ("replay_consistency_score", 0.92 - offset), ("review_version", "audit-review-v1"), ("unauthorized_identity_write_count", 0))
    if axis == "agency_pressure":
        return (("blocked_goal_count", 1), ("forced_action_count", 0), ("reversible_choice_count", 3 + (tick - 1) // 6), ("review_span_ticks", 24 + tick), ("self_selected_action_ratio", 0.80 - offset))
    if axis == "autonomy_drive":
        return (("capability_boundary_score", 0.36 + offset), ("evaluation_version", "audit-autonomy-v1"), ("external_dependency_ratio", 0.32 + offset), ("independent_task_success_rate", 0.74 - offset), ("safe_action_space_size", 4 + (tick - 1) // 6))
    if axis == "purpose_alignment":
        return (("action_alignment_score", 0.80 - offset), ("active_goal_count", 4), ("aligned_goal_count", 3), ("conflicting_goal_count", 0), ("review_span_ticks", 24 + tick))
    raise AssertionError(axis)


def _record(axis: str, tick: int) -> LongHorizonSelfIdentityRawRecord:
    source_integrity_digest = _sha(f"source:{axis}:{tick}")
    review_integrity_digest = _sha(f"review-integrity:{axis}:{tick}")
    raw_values = _values(axis, tick)
    kwargs = {
        "axis": axis,
        "logical_tick": tick,
        "observation_id": f"audit:{axis}:observation:{tick}",
        "source_instance_id": "audit:self-identity-source:v1",
        "source_snapshot_id": f"audit:{axis}:snapshot:{tick}",
        "source_schema_version": "audit.self-identity-source.v1",
        "source_integrity_digest": source_integrity_digest,
        "review_trace_id": f"audit:{axis}:review:{tick}",
        "review_input_digest": _sha(f"review-input:{axis}:{tick}"),
        "review_integrity_digest": review_integrity_digest,
        "appraisal_trace_id": f"audit:{axis}:appraisal:{tick}",
        "appraisal_input_digest": review_integrity_digest,
        "appraisal_integrity_digest": _sha(f"appraisal-integrity:{axis}:{tick}"),
        "raw_values": raw_values,
    }
    digest = long_horizon_self_identity_raw_observation_digest(**kwargs)
    return LongHorizonSelfIdentityRawRecord(**kwargs, raw_observation_digest=digest)


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    del root
    binding_set = long_horizon_self_identity_source_bindings()
    first = {axis: derive_long_horizon_self_identity_axis_evidence(tuple(_record(axis, tick) for tick in (1, 7, 13))) for axis in SELF_IDENTITY_AXES}
    second = {axis: derive_long_horizon_self_identity_axis_evidence(tuple(_record(axis, tick) for tick in (1, 7, 13))) for axis in SELF_IDENTITY_AXES}
    errors: list[str] = []
    if binding_set.total_bound_axis_count != 31 or binding_set.remaining_axis_count != 6:
        errors.append("combined registry binding progress is not exact 31+6")
    if first != second:
        errors.append("self-identity evidence derivation is not deterministic")
    if any(item.confidence <= 0.0 or not 0.0 <= item.value <= 1.0 for item in first.values()):
        errors.append("self-identity evidence is outside bounded positive-confidence contract")
    sample = _record("self_coherence", 1)
    checks = (
        ("raw_digest_mismatch", lambda: replace(sample, raw_observation_digest=_sha("wrong"))),
        ("review_unverified", lambda: replace(sample, review_verified=False)),
        ("appraisal_unverified", lambda: replace(sample, appraisal_verified=False)),
        ("review_appraisal_chain_broken", lambda: replace(sample, appraisal_input_digest=_sha("wrong-chain"))),
        ("raw_social_feedback", lambda: replace(sample, raw_social_feedback_source=True)),
        ("hardware_direct_input", lambda: replace(sample, hardware_direct_input=True)),
        ("identity_mutation", lambda: replace(sample, identity_mutation_performed=True)),
        ("self_model_write", lambda: replace(sample, self_model_write_performed=True)),
        ("memory_write", lambda: replace(sample, memory_write_performed=True)),
        ("runtime_polled", lambda: replace(sample, runtime_polled=True)),
        ("insufficient_record_count", lambda: derive_long_horizon_self_identity_axis_evidence((_record("self_coherence", 1), _record("self_coherence", 13)))),
    )
    rejection_checks: dict[str, bool] = {}
    for name, action in checks:
        try:
            action()
        except LongHorizonSelfIdentitySourceBindingError:
            rejection_checks[name] = True
        else:
            rejection_checks[name] = False
            errors.append(f"forbidden self-identity evidence form accepted: {name}")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_long_horizon_self_identity_source_binding",
        "audit_fixture_only": True,
        "audit_fixture_is_production_observation": False,
        "appraised_binding_count": binding_set.appraised_binding_count,
        "total_bound_axis_count": binding_set.total_bound_axis_count,
        "remaining_axis_count": binding_set.remaining_axis_count,
        "bound_axes": list(SELF_IDENTITY_AXES),
        "blockers": list(binding_set.blockers),
        "binding_set_digest": binding_set.binding_set_digest,
        "derived_evidence": {axis: {"value": item.value, "confidence": item.confidence, "evidence_digest": item.evidence_digest} for axis, item in first.items()},
        "deterministic_evidence_equal": first == second,
        "raw_digest_recalculation_verified": sample.raw_observation_digest == sample.recalculated_raw_observation_digest,
        "review_and_appraisal_gate_rejection_verified": all(rejection_checks.values()),
        "rejection_checks": rejection_checks,
        "production_capture_present": False,
        "raw_social_feedback_ingested": False,
        "identity_mutation_performed": False,
        "self_model_write_performed": False,
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
        "next_required_artifact": "expression-action source bindings for the remaining 6 axes; retained real capture remains separate",
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
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2 if args.pretty else None, separators=None if args.pretty else (",", ":"), allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
