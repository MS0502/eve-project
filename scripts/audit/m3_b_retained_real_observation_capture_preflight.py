#!/usr/bin/env python3
"""Recalculable audit for the 37-axis retained-real-observation capture preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_b_retained_real_observation_capture_preflight import (  # noqa: E402
    PRODUCTION_CAPTURE_FUTURE_PATH,
    RETENTION_SINK_FUTURE_PATH,
    retained_real_observation_capture_preflight,
)

SCHEMA_VERSION = "eve.m3-b.retained-real-observation-capture-preflight-audit.v1"
BASELINE_SHA = "272ae3132547395e03af55731ba87736aa6535e8"


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    first = retained_real_observation_capture_preflight()
    second = retained_real_observation_capture_preflight()
    production_path = root / PRODUCTION_CAPTURE_FUTURE_PATH
    retention_path = root / RETENTION_SINK_FUTURE_PATH
    errors: list[str] = []

    if first != second or first.preflight_digest != second.preflight_digest:
        errors.append("retained-real-observation capture preflight is not deterministic")
    if first.source_binding_count != 37 or first.source_binding_complete is not True:
        errors.append("source-binding coverage is not exact 37/37")
    if tuple(group.cumulative_bound_axis_count for group in first.source_binding_groups) != (
        4,
        6,
        12,
        19,
        25,
        31,
        37,
    ):
        errors.append("source-binding cumulative coverage is not exact")
    if production_path.exists():
        errors.append("future production capture adapter path already exists")
    if retention_path.exists():
        errors.append("future immutable retention sink path already exists")
    if first.production_capture_adapter_present or first.retention_sink_present:
        errors.append("preflight incorrectly claims a production capture component")
    if first.retained_real_observation_count != 0:
        errors.append("preflight fabricated retained real observations")
    if first.positive_confidence_real_observation_count != 0:
        errors.append("preflight fabricated positive-confidence real coverage")
    if first.observation_window_eligible or first.observation_window_started:
        errors.append("preflight opened the observation window boundary")
    if any(
        (
            first.persistence_accessed,
            first.event_append_performed,
            first.registry_owner_mutated,
            first.live_affect_mutated,
            first.live_drive_mutated,
            first.named_state_mutated,
            first.goal_memory_self_expression_mutated,
            first.m3_b_complete,
            first.m3_c_open,
            first.m3_e_authority_open,
            first.cutover_authorized,
        )
    ):
        errors.append("preflight granted mutation, persistence, cutover, or authority")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_retained_real_observation_capture_preflight",
        "source_binding_count": first.source_binding_count,
        "source_binding_complete": first.source_binding_complete,
        "source_binding_cumulative_counts": [
            group.cumulative_bound_axis_count for group in first.source_binding_groups
        ],
        "source_binding_group_digests": {
            group.group_id: group.binding_set_digest for group in first.source_binding_groups
        },
        "preflight_digest": first.preflight_digest,
        "deterministic_preflight_equal": first == second,
        "production_capture_future_path": PRODUCTION_CAPTURE_FUTURE_PATH,
        "production_capture_future_path_present": production_path.exists(),
        "retention_sink_future_path": RETENTION_SINK_FUTURE_PATH,
        "retention_sink_future_path_present": retention_path.exists(),
        "production_capture_adapter_present": first.production_capture_adapter_present,
        "retention_sink_present": first.retention_sink_present,
        "retained_real_observation_count": first.retained_real_observation_count,
        "positive_confidence_real_observation_count": first.positive_confidence_real_observation_count,
        "observation_window_eligible": first.observation_window_eligible,
        "observation_window_started": first.observation_window_started,
        "observation_window_satisfied": first.observation_window_satisfied,
        "blockers": list(first.blockers),
        "runtime_hook_installed": first.runtime_hook_installed,
        "scheduler_installed": first.scheduler_installed,
        "persistence_accessed": first.persistence_accessed,
        "event_append_performed": first.event_append_performed,
        "registry_owner_mutated": first.registry_owner_mutated,
        "live_affect_mutated": first.live_affect_mutated,
        "live_drive_mutated": first.live_drive_mutated,
        "named_state_mutated": first.named_state_mutated,
        "goal_memory_self_expression_mutated": first.goal_memory_self_expression_mutated,
        "m3_b_complete": first.m3_b_complete,
        "m3_c_open": first.m3_c_open,
        "m3_e_authority_open": first.m3_e_authority_open,
        "cutover_authorized": first.cutover_authorized,
        "legacy_runtime_authoritative": True,
        "legacy_persistence_authoritative": True,
        "next_required_artifact": (
            "production capture adapter plus immutable retained-real-observation sink; "
            "real observations must originate from actual production sources and cannot be fabricated"
        ),
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
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
        allow_nan=False,
    ) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
