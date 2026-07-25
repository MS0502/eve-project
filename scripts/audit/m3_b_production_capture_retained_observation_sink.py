#!/usr/bin/env python3
"""Recalculable audit for M3-B production-capture and retention-sink machinery."""
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

from core.m3_b_registry_observation_evidence import RegistryAxisPositiveConfidenceEvidence  # noqa: E402
from core.m3_b_registry_production_capture_adapter import (  # noqa: E402
    REGISTERED_PRODUCTION_SOURCE_VERIFIERS,
    ProductionSourceVerification,
    RegistryProductionCaptureAdapter,
    RegistryProductionCaptureError,
    production_capture_capability_status,
)
from core.m3_b_registry_retained_real_observation_sink import (  # noqa: E402
    retention_sink_capability_status,
)

SCHEMA_VERSION = "eve.m3-b.production-capture-retention-sink-audit.v1"
BASELINE_SHA = "b92a16a81e1591e490edc36d0171bc9a2c3bf065"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _candidate_fixture() -> tuple[
    RegistryAxisPositiveConfidenceEvidence,
    ProductionSourceVerification,
]:
    evidence = RegistryAxisPositiveConfidenceEvidence(
        axis="energy_budget",
        value=0.6,
        confidence=0.8,
        observed_tick=20,
        observation_id="audit:energy-budget:observation:20",
        source_family="operational_metrics_or_appraised_load_trace",
        source_instance_id="audit:device-runtime:v1",
        source_snapshot_id="audit:device-runtime:snapshot:20",
        source_schema_version="audit.device-runtime.v1",
        source_integrity_digest=_sha("audit-source:20"),
        raw_observation_digest=_sha("audit-raw:20"),
        acquisition_method="audit_fixture_matching_production_contract",
        verification_method="audit_fixture_exact_digest_verification",
        model_or_rule_version="audit.energy-budget.derivation.v1",
    )
    verification = ProductionSourceVerification(
        axis=evidence.axis,
        source_contract_id="eve:m3-b:registry-source:energy_budget:v1",
        source_family=evidence.source_family,
        source_instance_id=evidence.source_instance_id,
        source_snapshot_id=evidence.source_snapshot_id,
        source_schema_version=evidence.source_schema_version,
        source_integrity_digest=evidence.source_integrity_digest,
        raw_observation_digest=evidence.raw_observation_digest,
        observation_evidence_digest=evidence.evidence_digest,
        verifier_id="audit.unregistered.energy-budget-verifier",
        verifier_version="v1",
        verifier_trace_digest=_sha("audit-verifier-trace:20"),
        verified_logical_tick=20,
    )
    return evidence, verification


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    capture_status = production_capture_capability_status()
    sink_status = retention_sink_capability_status()
    errors: list[str] = []

    evidence, verification = _candidate_fixture()
    unregistered_verifier_rejected = False
    try:
        RegistryProductionCaptureAdapter().capture(
            evidence,
            verification,
            capture_id="audit:capture:must-fail",
            capture_tick=20,
        )
    except RegistryProductionCaptureError as exc:
        unregistered_verifier_rejected = "not registered" in str(exc)
    if not unregistered_verifier_rejected:
        errors.append("unregistered production verifier was not rejected")

    capture_path = root / "core/m3_b_registry_production_capture_adapter.py"
    sink_path = root / "core/m3_b_registry_retained_real_observation_sink.py"
    if not capture_path.exists() or not sink_path.exists():
        errors.append("production capture or retention sink implementation path is absent")
    if REGISTERED_PRODUCTION_SOURCE_VERIFIERS:
        errors.append("capability-only PR unexpectedly registers a production source verifier")
    if capture_status.registered_production_source_verifier_count != 0:
        errors.append("capture status claims a registered production verifier")
    if capture_status.retained_real_observation_count != 0:
        errors.append("capture status fabricates retained real observations")
    if sink_status.retained_real_observation_count != 0:
        errors.append("sink status fabricates retained real observations")
    if capture_status.observation_window_started or sink_status.observation_window_started:
        errors.append("capability machinery starts the observation window")
    if capture_status.m3_e_authority_open or sink_status.m3_e_authority_open:
        errors.append("capability machinery opens M3-E authority")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authority": "shadow_only_production_capture_retention_capability",
        "audit_fixture_only": True,
        "audit_fixture_is_production_observation": False,
        "production_capture_adapter_present": capture_status.production_capture_adapter_present,
        "immutable_retention_sink_present": sink_status.immutable_retention_sink_present,
        "durable_store_type": sink_status.durable_store_type,
        "append_only_chain_required": sink_status.append_only_chain_required,
        "readback_verification_required": sink_status.readback_verification_required,
        "auto_initialize": sink_status.auto_initialize,
        "auto_append": sink_status.auto_append,
        "registered_production_source_verifier_count": capture_status.registered_production_source_verifier_count,
        "registered_production_source_verifiers": dict(REGISTERED_PRODUCTION_SOURCE_VERIFIERS),
        "unregistered_verifier_rejected": unregistered_verifier_rejected,
        "retained_real_observation_count": 0,
        "positive_confidence_real_observation_count": 0,
        "observation_window_eligible": capture_status.observation_window_eligible,
        "observation_window_started": False,
        "m3_b_complete": False,
        "m3_c_open": False,
        "m3_e_authority_open": False,
        "cutover_authorized": False,
        "blockers": list(capture_status.blockers),
        "legacy_runtime_authoritative": True,
        "legacy_persistence_authoritative": True,
        "next_required_artifact": (
            "reviewed source-contract-specific production verifier registration and runtime source bridge; "
            "only then may an actual retained observation be appended"
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
