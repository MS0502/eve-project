#!/usr/bin/env python3
"""Recalculable audit for the six-axis validated learning source binding."""
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

from core.m3_b_validated_learning_source_binding import (  # noqa: E402
    APPRAISAL_SCHEMA_VERSION,
    LEARNING_EXPLORATION_AXES,
    ValidatedLearningRawRecord,
    ValidatedLearningSourceBindingError,
    derive_validated_learning_axis_evidence,
    validated_learning_raw_observation_digest,
    validated_learning_source_bindings,
)

SCHEMA_VERSION = "eve.m3-b.validated-learning-source-binding-audit.v1"
BASELINE_SHA = "6c57f41114fbe0a203e559a27b187f6801ad7640"


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
    if axis == "competence_drive":
        return (1, 3, 5)
    if axis == "prediction_error_pressure":
        return (1, 2)
    return (1, 3)


def _values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = min(0.08, (tick - 1) * 0.015)
    if axis == "curiosity_drive":
        return (
            ("exploration_cost", 0.24 + offset),
            ("information_gain_estimate", 0.70 - offset / 2),
            ("relevance_score", 0.74 - offset / 2),
            ("sampling_window_ticks", 6 + tick),
            ("unknown_count", 1 + (tick - 1) // 2),
        )
    if axis == "novelty_seeking":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("expected_information_gain", 0.64 + offset),
            ("novelty_score", 0.60 + offset),
            ("reversibility", 0.84 - offset),
            ("safety_score", 0.86 - offset),
        )
    if axis == "learning_pressure":
        return (
            ("available_training_signal", 0.72 - offset),
            ("competence_gap", 0.42 + offset),
            ("error_recurrence", 1 + (tick - 1) // 2),
            ("task_relevance", 0.78 - offset / 2),
            ("validation_status", "verified"),
        )
    if axis == "memory_consolidation_pressure":
        return (
            ("causal_relevance", 0.64 + offset),
            ("emotional_relevance", 0.46 + offset),
            ("provenance_completeness", 0.90 - offset),
            ("recurrence_count", 1 + (tick - 1) // 2),
            ("salience_score", 0.68 + offset / 2),
        )
    if axis == "prediction_error_pressure":
        return (
            ("model_version", "audit-model-v1"),
            ("normalized_error", 0.22 + offset),
            ("observed_value_digest", _sha(f"observed:{tick}")),
            ("predicted_value_digest", _sha(f"predicted:{tick}")),
            ("verification_status", "verified"),
        )
    if axis == "competence_drive":
        return (
            ("calibrated_error_rate", 0.32 + offset),
            ("evaluation_version", "audit-eval-v1"),
            ("learning_progress", 0.54 + offset / 2),
            ("skill_gap", 0.44 + offset),
            ("success_rate", 0.68 - offset),
        )
    raise AssertionError(axis)


def _record(axis: str, tick: int) -> ValidatedLearningRawRecord:
    observation_id = f"audit:{axis}:observation:{tick}"
    source_instance_id = "audit:validated-learning-source:v1"
    source_snapshot_id = f"audit:{axis}:snapshot:{tick}"
    source_schema_version = "audit.validated-learning-source.v1"
    source_integrity_digest = _sha(f"source:{axis}:{tick}")
    validation_trace_id = f"audit:{axis}:validation:{tick}"
    validation_input_digest = _sha(f"validation-input:{axis}:{tick}")
    validation_integrity_digest = _sha(f"validation-integrity:{axis}:{tick}")
    appraisal_trace_id = f"audit:{axis}:appraisal:{tick}"
    appraisal_input_digest = validation_integrity_digest
    appraisal_integrity_digest = _sha(f"appraisal-integrity:{axis}:{tick}")
    raw_values = _values(axis, tick)
    raw_observation_digest = validated_learning_raw_observation_digest(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        validation_trace_id=validation_trace_id,
        validation_input_digest=validation_input_digest,
        validation_integrity_digest=validation_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_values=raw_values,
    )
    return ValidatedLearningRawRecord(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        validation_trace_id=validation_trace_id,
        validation_input_digest=validation_input_digest,
        validation_integrity_digest=validation_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_observation_digest=raw_observation_digest,
        raw_values=raw_values,
    )


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    del root
    binding_set = validated_learning_source_bindings()
    first = {
        axis: derive_validated_learning_axis_evidence(
            tuple(_record(axis, tick) for tick in _ticks(axis))
        )
        for axis in LEARNING_EXPLORATION_AXES
    }
    second = {
        axis: derive_validated_learning_axis_evidence(
            tuple(_record(axis, tick) for tick in _ticks(axis))
        )
        for axis in LEARNING_EXPLORATION_AXES
    }
    errors: list[str] = []
    if binding_set.appraised_binding_count != 6:
        errors.append("validated learning binding count is not exactly six")
    if binding_set.total_bound_axis_count != 25 or binding_set.remaining_axis_count != 12:
        errors.append("combined registry binding progress is not exact 25+12")
    if first != second:
        errors.append("validated learning evidence derivation is not deterministic")
    if any(item.confidence <= 0.0 for item in first.values()):
        errors.append("validated learning evidence contains non-positive confidence")
    if any(not 0.0 <= item.value <= 1.0 for item in first.values()):
        errors.append("validated learning evidence value is outside registry bounds")

    sample = _record("curiosity_drive", 1)
    rejection_checks: dict[str, bool] = {}
    checks = (
        ("raw_digest_mismatch", lambda: replace(sample, raw_observation_digest=_sha("wrong"))),
        ("validation_unverified", lambda: replace(sample, validation_verified=False)),
        ("appraisal_unverified", lambda: replace(sample, appraisal_verified=False)),
        ("validation_appraisal_chain_broken", lambda: replace(sample, appraisal_input_digest=_sha("wrong-chain"))),
        ("raw_social_feedback", lambda: replace(sample, raw_social_feedback_source=True)),
        ("hardware_direct_input", lambda: replace(sample, hardware_direct_input=True)),
        ("synthetic", lambda: replace(sample, synthetic=True)),
        ("proposal_only", lambda: replace(sample, proposal_only=True)),
        ("registry_owner_source", lambda: replace(sample, registry_owner_source=True)),
        ("runtime_polled", lambda: replace(sample, runtime_polled=True)),
        ("learning_mutation", lambda: replace(sample, learning_mutation_performed=True)),
        ("memory_write", lambda: replace(sample, memory_write_performed=True)),
        ("noncanonical_acquisition", lambda: replace(sample, acquisition_method="unverified")),
        ("noncanonical_verification", lambda: replace(sample, verification_method="none")),
        ("noncanonical_validation", lambda: replace(sample, validation_method="implicit")),
        ("noncanonical_appraisal", lambda: replace(sample, appraisal_method="caller_claim")),
        ("insufficient_record_count", lambda: derive_validated_learning_axis_evidence((sample,))),
    )
    for name, action in checks:
        try:
            action()
        except ValidatedLearningSourceBindingError:
            rejection_checks[name] = True
        else:
            rejection_checks[name] = False
            errors.append(f"forbidden validated learning evidence form accepted: {name}")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_validated_learning_source_binding",
        "audit_fixture_only": True,
        "audit_fixture_is_production_observation": False,
        "appraised_binding_count": binding_set.appraised_binding_count,
        "total_bound_axis_count": binding_set.total_bound_axis_count,
        "remaining_axis_count": binding_set.remaining_axis_count,
        "bound_axes": list(LEARNING_EXPLORATION_AXES),
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
        "raw_digest_recalculation_verified": sample.raw_observation_digest == sample.recalculated_raw_observation_digest,
        "validation_and_appraisal_gate_rejection_verified": all(rejection_checks.values()),
        "rejection_checks": rejection_checks,
        "production_capture_present": False,
        "runtime_capture_installed": False,
        "hardware_polling_installed": False,
        "raw_social_feedback_ingested": False,
        "learning_mutation_performed": False,
        "memory_write_performed": False,
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
        "next_required_artifact": "appraised source bindings for the remaining 12 axes; retained real capture remains separate",
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
