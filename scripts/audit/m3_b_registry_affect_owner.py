#!/usr/bin/env python3
"""Recalculable audit for the detached M3-B registry current-value owner."""
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

from core.m3_b_registry_affect_owner import (  # noqa: E402
    REGISTRY_AXIS_ORDER,
    advance_registry_affect_owner,
    apply_validated_registry_proposal,
    create_registry_affect_owner,
)

SCHEMA_VERSION = "eve.m3-b.registry-37-axis-owner-audit.v1"
BASELINE_SHA = "f20951351b56c7102fdcf7c00f17cbcfac792205"
CORE_PATH = ROOT / "core/m3_b_registry_affect_owner.py"


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
        "save",
        "start",
        "write",
        "write_text",
    })
    return {
        "forbidden_calls": forbidden_calls,
        "forbidden_imports": forbidden_imports,
        "no_io_persistence_scheduler_event_or_runtime_surface": not forbidden_imports and not forbidden_calls,
    }


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    del root
    first = create_registry_affect_owner(
        owner_instance_id="audit:registry-owner:v1",
        genesis_source_id="audit:registry-genesis:v1",
    )
    second = create_registry_affect_owner(
        owner_instance_id="audit:registry-owner:v1",
        genesis_source_id="audit:registry-genesis:v1",
    )
    observations = first.to_axis_observations()
    proposed = apply_validated_registry_proposal(
        first,
        event_category="praise",
        proposed_axis_deltas={"competence_drive": 0.04, "social_trust": 0.03},
        proposal_id="audit:proposal:praise:1",
        proposal_sequence=1,
        proposal_confidence=0.9,
        expected_owner_digest=first.state_digest,
        operator_authorization_id="audit:operator-authorization:1",
    )
    advanced = advance_registry_affect_owner(
        proposed,
        target_tick=20,
        cadence_id="audit:cadence:20",
        expected_owner_digest=proposed.state_digest,
    )
    static = _static_surface()
    errors: list[str] = []
    if len(REGISTRY_AXIS_ORDER) != 37 or len(set(REGISTRY_AXIS_ORDER)) != 37:
        errors.append("registry owner must cover exactly 37 unique axes")
    if first.to_mapping() != second.to_mapping() or first.state_digest != second.state_digest:
        errors.append("registry owner genesis is not deterministic")
    if len(observations) != 37 or tuple(item.axis for item in observations) != REGISTRY_AXIS_ORDER:
        errors.append("registry owner observation snapshot is incomplete")
    if any(item.confidence != 0.0 for item in observations):
        errors.append("registry baseline genesis was incorrectly promoted to positive-confidence observation evidence")
    if first.genesis_is_observation_evidence or first.proposal_metadata_is_current_state:
        errors.append("definitions or proposal metadata masquerade as current observation")
    if proposed.value_for("competence_drive") <= first.value_for("competence_drive"):
        errors.append("validated proposal did not create a detached owner transition")
    if proposed.axes[REGISTRY_AXIS_ORDER.index("competence_drive")].confidence != 0.9:
        errors.append("validated proposal did not establish confidence for its touched axis")
    if proposed.axes[REGISTRY_AXIS_ORDER.index("energy_budget")].confidence != 0.0:
        errors.append("untouched genesis axis gained observation confidence")
    if not first.value_for("competence_drive") < advanced.value_for("competence_drive") < proposed.value_for("competence_drive"):
        errors.append("cadence did not decay the proposal state toward baseline")
    if not static["no_io_persistence_scheduler_event_or_runtime_surface"]:
        errors.append("core owner exposes a forbidden live or I/O surface")
    if any(
        (
            advanced.runtime_hook_installed,
            advanced.scheduler_installed,
            advanced.persistence_accessed,
            advanced.event_append_performed,
            advanced.live_affect_mutated,
            advanced.live_drive_mutated,
            advanced.goal_memory_self_expression_mutated,
            advanced.observation_window_started,
            advanced.m3_b_complete,
            advanced.m3_c_open,
            advanced.m3_e_authority_open,
            advanced.cutover_authorized,
        )
    ):
        errors.append("registry owner granted forbidden authority")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_detached_owner",
        "axis_count": len(REGISTRY_AXIS_ORDER),
        "axis_order": list(REGISTRY_AXIS_ORDER),
        "current_value_owner_contract_found": True,
        "deterministic_initial_state_rule": "registry_baseline_genesis_materialized_as_zero_confidence_current_state_not_observation_evidence",
        "genesis_is_observation_evidence": first.genesis_is_observation_evidence,
        "genesis_unknown_observation_count": sum(item.confidence == 0.0 for item in observations),
        "proposal_metadata_is_current_state": first.proposal_metadata_is_current_state,
        "accepted_proposal_boundary": "existing_read_only_validator_plus_exact_digest_sequence_operator_authorization_identity_and_positive_confidence",
        "proposal_transition_old_state_unchanged": first.value_for("competence_drive") == 0.50,
        "proposal_transition_new_state_digest_changed": proposed.state_digest != first.state_digest,
        "proposal_observed_axis_count": sum(axis.confidence > 0.0 for axis in proposed.axes),
        "proposal_duplicate_rule": "unique_proposal_id_plus_exact_prior_digest_plus_monotonic_sequence",
        "cadence_owner": "explicit_caller_invoked_logical_tick_no_scheduler",
        "cadence_decay_toward_baseline": first.value_for("competence_drive") < advanced.value_for("competence_drive") < proposed.value_for("competence_drive"),
        "absence_unknown_policy": "baseline_genesis_is_owned_state_with_zero_confidence; missing_unknown_or_partial_proposal_input_never_gains_positive_observation_confidence",
        "range_confidence_saturation_policy": "registry_bounds; zero_confidence_genesis; positive_validated_proposal_confidence_then_min; clamp_then_refractory_decay",
        "snapshot_identity_schema_provenance_integrity_complete": len(observations) == 37 and all(item.source_integrity_digest == first.state_digest for item in observations),
        "read_only_observation_count": len(observations),
        "deterministic_genesis_equal": first.to_mapping() == second.to_mapping(),
        "static_surface": static,
        "remaining_source_ownership_blockers": [],
        "next_required_artifact": "real combined 63-axis read-only observation packet and observation-window evidence",
        "legacy_runtime_authoritative": True,
        "legacy_persistence_authoritative": True,
        "runtime_hook_installed": False,
        "scheduler_installed": False,
        "persistence_accessed": False,
        "event_append_performed": False,
        "live_affect_mutated": False,
        "live_drive_mutated": False,
        "goal_memory_self_expression_mutated": False,
        "observation_window_started": False,
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
            "current_value_owner_contract_found": report["current_value_owner_contract_found"],
            "deterministic_genesis_equal": report["deterministic_genesis_equal"],
            "errors": report["errors"],
            "genesis_unknown_observation_count": report["genesis_unknown_observation_count"],
            "m3_b_complete": report["m3_b_complete"],
            "next_required_artifact": report["next_required_artifact"],
            "observation_window_started": report["observation_window_started"],
            "read_only_observation_count": report["read_only_observation_count"],
            "report_digest": report["report_digest"],
        }
    _write(output, args.output, args.pretty)
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
