#!/usr/bin/env python3
"""Recalculable audit for the 37-axis positive-confidence evidence contract."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.affect_hormone_neural_rhythm_registry import (  # noqa: E402
    affect_hormone_axis_registry,
)
from hormone_system import HormoneSystem  # noqa: E402
from core.m3_b_observation_packet import build_m3_b_observation_packet  # noqa: E402
from core.m3_b_registry_affect_owner import (  # noqa: E402
    REGISTRY_AXIS_ORDER,
    create_registry_affect_owner,
)
from core.m3_b_registry_observation_evidence import (  # noqa: E402
    RegistryAxisPositiveConfidenceEvidence,
    RegistryObservationEvidenceError,
    build_registry_positive_confidence_evidence_bundle,
    materialize_registry_observed_owner,
)

SCHEMA_VERSION = "eve.m3-b.registry-positive-confidence-evidence-audit.v1"
BASELINE_SHA = "379d912c4e863b2a692d2c20f9f8113dfa7219cd"
CORE_PATH = ROOT / "core/m3_b_registry_observation_evidence.py"


def _digest_text(text: str) -> str:
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
            "save",
            "start",
            "write",
            "write_text",
        }
    )
    return {
        "forbidden_calls": forbidden_calls,
        "forbidden_imports": forbidden_imports,
        "no_io_persistence_scheduler_event_or_runtime_surface": not forbidden_imports
        and not forbidden_calls,
    }


def _evidence(logical_tick: int = 1) -> tuple[RegistryAxisPositiveConfidenceEvidence, ...]:
    registry = affect_hormone_axis_registry()
    return tuple(
        RegistryAxisPositiveConfidenceEvidence(
            axis=axis,
            value=float(registry[axis]["baseline"]),
            confidence=0.80,
            observed_tick=logical_tick,
            observation_id=f"audit:registry-observation:{index:02d}",
            source_family="audit_verified_internal_observation",
            source_instance_id="audit:observation-source:v1",
            source_snapshot_id=f"audit:source-snapshot:{logical_tick}",
            source_schema_version="audit.registry-observation-source.v1",
            source_integrity_digest=_digest_text(
                f"audit:source:{axis}:{logical_tick}"
            ),
            raw_observation_digest=_digest_text(
                f"audit:raw:{axis}:{logical_tick}"
            ),
            acquisition_method="explicit_recalculable_contract_audit_capture",
            verification_method="deterministic_contract_audit_verification",
            model_or_rule_version="audit:observation-rule:v1",
        )
        for index, axis in enumerate(REGISTRY_AXIS_ORDER)
    )


def _bundle(owner):
    return build_registry_positive_confidence_evidence_bundle(
        owner,
        _evidence(),
        bundle_id="audit:registry-evidence-bundle:v1",
        logical_tick=1,
        source_manifest_schema_version="audit.registry-source-manifest.v1",
        source_manifest_digest=_digest_text("audit:source-manifest:v1"),
        verification_authorization_id="audit:verification-authorization:v1",
        acceptance_policy_version="audit:acceptance-policy:v1",
    )


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    del root
    owner = create_registry_affect_owner(
        owner_instance_id="audit:registry-owner:v1",
        genesis_source_id="audit:registry-genesis:v1",
    )
    owner_before = copy.deepcopy(owner.to_mapping())
    first_bundle = _bundle(owner)
    first_owner = materialize_registry_observed_owner(owner, first_bundle)
    owner_middle = copy.deepcopy(owner.to_mapping())
    second_bundle = _bundle(owner)
    second_owner = materialize_registry_observed_owner(owner, second_bundle)
    owner_after = copy.deepcopy(owner.to_mapping())
    packet = build_m3_b_observation_packet(
        HormoneSystem(developmental_stage="adult"),
        first_owner,
        packet_id="audit:combined-packet:verified-registry:v1",
        packet_sequence=1,
        logical_tick=first_owner.logical_tick,
        legacy_source_instance_id="audit:legacy-owner:v1",
        legacy_source_snapshot_id="audit:legacy-snapshot:v1",
    )
    static = _static_surface()
    errors: list[str] = []

    if first_bundle.positive_confidence_count != 37:
        errors.append("evidence bundle does not contain 37 positive-confidence axes")
    if not first_bundle.exact_positive_confidence_coverage:
        errors.append("evidence bundle does not preserve exact canonical 37-axis coverage")
    if tuple(item.axis for item in first_bundle.observations) != REGISTRY_AXIS_ORDER:
        errors.append("evidence bundle axis order is not canonical")
    if owner_before != owner_middle or owner_middle != owner_after:
        errors.append("predecessor registry owner changed during detached materialization")
    if first_bundle.to_mapping() != second_bundle.to_mapping():
        errors.append("evidence bundle construction is not deterministic")
    if first_bundle.bundle_digest != second_bundle.bundle_digest:
        errors.append("evidence bundle digest is not deterministic")
    if first_owner.to_mapping() != second_owner.to_mapping():
        errors.append("observed owner materialization is not deterministic")
    if first_owner.state_digest != second_owner.state_digest:
        errors.append("observed owner digest is not deterministic")
    if any(axis.confidence <= 0.0 for axis in first_owner.axes):
        errors.append("materialized registry owner contains non-positive confidence")
    if any(axis.last_impulse_tick != 0 for axis in first_owner.axes):
        errors.append("observation materialization incorrectly created an affect impulse")
    if packet.positive_confidence_count != 63 or packet.zero_confidence_count != 0:
        errors.append("verified fixture did not resolve calculated packet confidence coverage")
    if packet.window_blockers:
        errors.append("verified fixture packet retained a confidence blocker")
    if not packet.observation_window_start_eligible:
        errors.append("verified fixture packet did not calculate structural start eligibility")
    if not static["no_io_persistence_scheduler_event_or_runtime_surface"]:
        errors.append("evidence module exposes a forbidden live or I/O surface")

    rejection_checks: dict[str, bool] = {}
    for name, mutation in (
        ("zero_confidence", {"confidence": 0.0}),
        ("genesis_derived", {"genesis_derived": True}),
        ("baseline_derived", {"baseline_derived": True}),
        ("default_derived", {"default_derived": True}),
        ("proposal_only", {"proposal_only": True}),
        ("synthetic", {"synthetic": True}),
        ("missing_raw_reference", {"recalculable_reference_present": False}),
    ):
        try:
            replace(first_bundle.observations[0], **mutation)
        except RegistryObservationEvidenceError:
            rejection_checks[name] = True
        else:
            rejection_checks[name] = False
            errors.append(f"forbidden evidence form was accepted: {name}")

    if any(
        (
            first_bundle.runtime_hook_installed,
            first_bundle.scheduler_installed,
            first_bundle.persistence_accessed,
            first_bundle.event_append_performed,
            first_bundle.live_affect_mutated,
            first_bundle.live_drive_mutated,
            first_bundle.named_state_mutated,
            first_bundle.goal_memory_self_expression_mutated,
            first_bundle.observation_window_started,
            first_bundle.observation_window_satisfied,
            first_bundle.m3_b_complete,
            first_bundle.m3_c_open,
            first_bundle.m3_e_authority_open,
            first_bundle.cutover_authorized,
            packet.observation_window_started,
            packet.observation_window_satisfied,
            packet.m3_b_complete,
            packet.m3_c_open,
            packet.m3_e_authority_open,
            packet.cutover_authorized,
        )
    ):
        errors.append("contract audit granted forbidden mutation, window, or authority")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_positive_confidence_evidence_contract",
        "audit_fixture_only": True,
        "audit_fixture_is_production_observation_evidence": False,
        "axis_count": len(first_bundle.observations),
        "positive_confidence_count": first_bundle.positive_confidence_count,
        "exact_positive_confidence_coverage": first_bundle.exact_positive_confidence_coverage,
        "deterministic_bundle_equal": first_bundle.to_mapping() == second_bundle.to_mapping(),
        "deterministic_owner_equal": first_owner.to_mapping() == second_owner.to_mapping(),
        "predecessor_owner_unchanged": owner_before == owner_middle == owner_after,
        "observation_materialization_created_impulse": any(
            axis.last_impulse_tick != 0 for axis in first_owner.axes
        ),
        "bundle_digest": first_bundle.bundle_digest,
        "materialized_owner_digest": first_owner.state_digest,
        "fixture_packet_positive_confidence_count": packet.positive_confidence_count,
        "fixture_packet_zero_confidence_count": packet.zero_confidence_count,
        "fixture_packet_window_blockers": list(packet.window_blockers),
        "fixture_packet_calculated_start_eligible": packet.observation_window_start_eligible,
        "production_observation_window_started": False,
        "production_observation_window_satisfied": False,
        "rejection_checks": rejection_checks,
        "static_surface": static,
        "next_required_artifact": "real recalculable 37-axis observed-value capture bound to this contract before any observation-window start decision",
        "legacy_runtime_authoritative": True,
        "legacy_persistence_authoritative": True,
        "runtime_hook_installed": False,
        "scheduler_installed": False,
        "persistence_accessed": False,
        "event_append_performed": False,
        "live_affect_mutated": False,
        "live_drive_mutated": False,
        "named_state_mutated": False,
        "goal_memory_self_expression_mutated": False,
        "observation_window_started": False,
        "observation_window_satisfied": False,
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
            "audit_fixture_only": report["audit_fixture_only"],
            "axis_count": report["axis_count"],
            "errors": report["errors"],
            "exact_positive_confidence_coverage": report[
                "exact_positive_confidence_coverage"
            ],
            "next_required_artifact": report["next_required_artifact"],
            "observation_window_started": report["observation_window_started"],
            "positive_confidence_count": report["positive_confidence_count"],
            "report_digest": report["report_digest"],
        }
    _write(output, args.output, args.pretty)
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
