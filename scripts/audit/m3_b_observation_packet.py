#!/usr/bin/env python3
"""Recalculable audit for the combined 63-axis M3-B packet preflight."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hormone_system import HormoneSystem  # noqa: E402
from core.m3_b_observation_packet import (  # noqa: E402
    WINDOW_BLOCKER_REGISTRY_CONFIDENCE,
    build_m3_b_observation_packet,
)
from core.m3_b_registry_affect_owner import (  # noqa: E402
    apply_validated_registry_proposal,
    create_registry_affect_owner,
)

SCHEMA_VERSION = "eve.m3-b.combined-63-axis-observation-packet-audit.v1"
BASELINE_SHA = "aab991f5c217baf9a6f6e5d2d6115feba9000f5a"
CORE_PATH = ROOT / "core/m3_b_observation_packet.py"


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
    forbidden_imports = sorted(imports & {
        "os",
        "pathlib",
        "persistence",
        "sqlite3",
        "subprocess",
        "threading",
        "time",
    })
    forbidden_calls = sorted(calls & {
        "append_event",
        "connect",
        "emit",
        "mkdir",
        "open",
        "project_shadow_affect",
        "save",
        "start",
        "write",
        "write_text",
    })
    return {
        "forbidden_calls": forbidden_calls,
        "forbidden_imports": forbidden_imports,
        "no_io_persistence_scheduler_event_projection_or_runtime_surface": not forbidden_imports and not forbidden_calls,
    }


def _build(source: HormoneSystem, owner, *, packet_id: str):
    return build_m3_b_observation_packet(
        source,
        owner,
        packet_id=packet_id,
        packet_sequence=1,
        logical_tick=owner.logical_tick,
        legacy_source_instance_id="audit:legacy-owner:v1",
        legacy_source_snapshot_id="audit:legacy-snapshot:v1",
    )


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    del root
    source = HormoneSystem(developmental_stage="adult")
    owner = create_registry_affect_owner(
        owner_instance_id="audit:registry-owner:v1",
        genesis_source_id="audit:registry-genesis:v1",
    )
    source_before = copy.deepcopy(source.__dict__)
    owner_before = owner.to_mapping()
    first = _build(source, owner, packet_id="audit:packet:genesis:v1")
    source_middle = copy.deepcopy(source.__dict__)
    owner_middle = owner.to_mapping()
    second = _build(source, owner, packet_id="audit:packet:genesis:v1")
    source_after = copy.deepcopy(source.__dict__)
    owner_after = owner.to_mapping()

    observed_owner = apply_validated_registry_proposal(
        owner,
        event_category="praise",
        proposed_axis_deltas={"competence_drive": 0.04, "social_trust": 0.03},
        proposal_id="audit:proposal:praise:1",
        proposal_sequence=1,
        proposal_confidence=0.9,
        expected_owner_digest=owner.state_digest,
        operator_authorization_id="audit:operator:praise:1",
    )
    partially_observed = _build(
        HormoneSystem(developmental_stage="adult"),
        observed_owner,
        packet_id="audit:packet:partial:v1",
    )
    static = _static_surface()
    errors: list[str] = []
    if first.axis_count != 63 or first.legacy_axis_count != 26 or first.registry_axis_count != 37:
        errors.append("combined packet does not contain exact 26+37 coverage")
    if first.positive_confidence_count != 26 or first.zero_confidence_count != 37:
        errors.append("genesis packet confidence boundary is incorrect")
    if first.window_blockers != (WINDOW_BLOCKER_REGISTRY_CONFIDENCE,):
        errors.append("genesis packet did not expose the exact registry confidence blocker")
    if first.observation_window_start_eligible:
        errors.append("genesis packet incorrectly opened the observation window")
    if not first.strict_projection_input_ready or not first.structurally_complete:
        errors.append("complete 63-axis shape is not marked structurally ready")
    if first.to_mapping() != second.to_mapping() or first.packet_digest != second.packet_digest:
        errors.append("combined packet is not deterministic")
    if source_before != source_middle or source_middle != source_after:
        errors.append("legacy source changed during repeated packet construction")
    if owner_before != owner_middle or owner_middle != owner_after:
        errors.append("registry owner changed during repeated packet construction")
    if partially_observed.positive_confidence_count != 28 or partially_observed.zero_confidence_count != 35:
        errors.append("partial validated evidence did not promote exactly two registry axes")
    if partially_observed.observation_window_start_eligible:
        errors.append("partial registry evidence incorrectly opened the observation window")
    if not static["no_io_persistence_scheduler_event_projection_or_runtime_surface"]:
        errors.append("packet module exposes a forbidden live or I/O surface")
    if any(
        (
            first.projection_performed,
            first.observation_window_started,
            first.observation_window_satisfied,
            first.persistence_accessed,
            first.event_append_performed,
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
        errors.append("packet granted forbidden mutation, window, or authority")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_explicit_packet_preflight",
        "axis_count": first.axis_count,
        "legacy_axis_count": first.legacy_axis_count,
        "registry_axis_count": first.registry_axis_count,
        "structurally_complete": first.structurally_complete,
        "strict_projection_input_ready": first.strict_projection_input_ready,
        "genesis_positive_confidence_count": first.positive_confidence_count,
        "genesis_zero_confidence_count": first.zero_confidence_count,
        "partial_positive_confidence_count": partially_observed.positive_confidence_count,
        "partial_zero_confidence_count": partially_observed.zero_confidence_count,
        "window_blockers": list(first.window_blockers),
        "observation_window_start_eligible": first.observation_window_start_eligible,
        "deterministic_replay_equal": first.to_mapping() == second.to_mapping(),
        "legacy_source_unchanged": source_before == source_middle == source_after,
        "registry_owner_unchanged": owner_before == owner_middle == owner_after,
        "source_set_digest": first.source_set.digest,
        "packet_digest": first.packet_digest,
        "static_surface": static,
        "next_required_artifact": "positive-confidence observed-value provenance for all remaining registry axes before any M3-B observation window starts",
        "legacy_runtime_authoritative": True,
        "legacy_persistence_authoritative": True,
        "projection_performed": False,
        "observation_window_started": False,
        "observation_window_satisfied": False,
        "persistence_accessed": False,
        "event_append_performed": False,
        "live_affect_mutated": False,
        "live_drive_mutated": False,
        "named_state_mutated": False,
        "goal_memory_self_expression_mutated": False,
        "m3_b_complete": False,
        "m3_c_open": False,
        "m3_e_authority_open": False,
        "cutover_authorized": False,
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
            "errors": report["errors"],
            "genesis_positive_confidence_count": report["genesis_positive_confidence_count"],
            "genesis_zero_confidence_count": report["genesis_zero_confidence_count"],
            "m3_b_complete": report["m3_b_complete"],
            "next_required_artifact": report["next_required_artifact"],
            "observation_window_started": report["observation_window_started"],
            "observation_window_start_eligible": report["observation_window_start_eligible"],
            "report_digest": report["report_digest"],
            "window_blockers": report["window_blockers"],
        }
    _write(output, args.output, args.pretty)
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
