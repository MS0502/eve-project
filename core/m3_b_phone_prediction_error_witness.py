"""Operator-side phone witness for the first real M3-B prediction-error observation.

This module composes already-merged boundaries without opening production authority:

* C1 operator-private launch attestation;
* the read-only ``prediction_error_pressure`` runtime bridge; and
* detached positive-confidence evidence derivation.

It deliberately does not register a reviewed attestation, install a runtime/source
verifier, append retained-real-observation events, start an observation window, or
open M3-C/M3-E/cutover authority. Raw prediction/error material belongs only in an
operator-private companion outside the repository.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_operator_attestation_trust_root import (
    OperatorLaunchBinding,
    OperatorPublicLaunchAttestation,
    build_operator_public_launch_attestation,
    verify_operator_private_binding,
)
from core.m3_b_prediction_error_runtime_source_bridge import (
    AXIS,
    PredictionErrorRuntimeSourceSnapshot,
    derive_detached_prediction_error_evidence,
)
from core.m3_b_registry_observation_evidence import RegistryAxisPositiveConfidenceEvidence

WITNESS_SCHEMA_VERSION = "eve.m3-b.phone-prediction-error-witness.v1"
PUBLIC_REVIEW_SCHEMA_VERSION = "eve.m3-b.phone-prediction-error-public-review.v1"
ENTRYPOINT_ID = "scripts/operator/m3_b_phone_prediction_error_witness.py:main"
DEFAULT_SOURCE_INSTANCE_ID = "runtime:ai-adapter:primary"
REQUIRED_RAW_RECORD_COUNT = 2
REQUIRED_LOGICAL_SPAN_TICKS = 1


class PhonePredictionErrorWitnessError(ValueError):
    """Raised when phone witness material cannot satisfy the exact C2 preflight contract."""


def _canonical(value: Any, field: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PhonePredictionErrorWitnessError(f"{field} is not canonical JSON material") from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _snapshot_mapping(snapshot: PredictionErrorRuntimeSourceSnapshot) -> dict[str, Any]:
    return {
        "authority": snapshot.authority,
        "bridge_schema_version": snapshot.bridge_schema_version,
        "error": snapshot.error,
        "fixture_only": snapshot.fixture_only,
        "logical_tick": snapshot.logical_tick,
        "prediction": snapshot.prediction,
        "prediction_id": snapshot.prediction_id,
        "production_origin_verified": snapshot.production_origin_verified,
        "production_verifier_registered": snapshot.production_verifier_registered,
        "retained_real_observation": snapshot.retained_real_observation,
        "runtime_source_read_only": snapshot.runtime_source_read_only,
        "schema_version": snapshot.schema_version,
        "source_instance_id": snapshot.source_instance_id,
        "source_integrity_digest": snapshot.source_integrity_digest,
        "source_snapshot_id": snapshot.source_snapshot_id,
    }


def _attestation_review_mapping(
    attestation: OperatorPublicLaunchAttestation,
    local_verification_trace_digest: str,
) -> dict[str, Any]:
    return {
        "attestation_digest": attestation.attestation_digest,
        "fixture_only": attestation.fixture_only,
        "launch_attestation_id": attestation.launch_attestation_id,
        "local_verification_trace_digest": local_verification_trace_digest,
        "private_nonce_commitment_digest": attestation.private_nonce_commitment_digest,
        "repository_head_sha": attestation.repository_head_sha,
        "runtime_instance_id": attestation.runtime_instance_id,
        "source_instance_id": attestation.source_instance_id,
        "trust_domain": attestation.trust_domain,
    }


def _validate_snapshot_sequence(
    snapshots: Sequence[PredictionErrorRuntimeSourceSnapshot],
    *,
    source_instance_id: str,
) -> tuple[PredictionErrorRuntimeSourceSnapshot, ...]:
    items = tuple(snapshots)
    if len(items) != REQUIRED_RAW_RECORD_COUNT:
        raise PhonePredictionErrorWitnessError("phone witness requires exactly two runtime snapshots")
    if any(type(item) is not PredictionErrorRuntimeSourceSnapshot for item in items):
        raise PhonePredictionErrorWitnessError("witness snapshots must use exact immutable bridge type")
    if any(item.fixture_only for item in items):
        raise PhonePredictionErrorWitnessError("phone witness snapshots cannot be fixture_only")
    if any(item.source_instance_id != source_instance_id for item in items):
        raise PhonePredictionErrorWitnessError("attestation and snapshots must bind one source instance")
    ticks = tuple(item.logical_tick for item in items)
    if ticks != tuple(sorted(set(ticks))):
        raise PhonePredictionErrorWitnessError("runtime snapshot ticks must be strictly increasing")
    if ticks[-1] - ticks[0] < REQUIRED_LOGICAL_SPAN_TICKS:
        raise PhonePredictionErrorWitnessError("runtime snapshots do not satisfy minimum logical span")
    return items


@dataclass(frozen=True, slots=True)
class PhonePredictionErrorWitness:
    """Bounded private companion material plus a public digest-only review surface."""

    attestation: OperatorPublicLaunchAttestation
    local_verification_trace_digest: str
    snapshots: tuple[PredictionErrorRuntimeSourceSnapshot, ...]
    evidence: RegistryAxisPositiveConfidenceEvidence
    schema_version: str = WITNESS_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    reviewed_attestation_registered: bool = False
    runtime_provenance_verifier_registered: bool = False
    production_source_verifier_registered: bool = False
    retained_real_observation: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        if type(self.attestation) is not OperatorPublicLaunchAttestation:
            raise PhonePredictionErrorWitnessError("witness requires exact public launch attestation")
        if self.attestation.fixture_only:
            raise PhonePredictionErrorWitnessError("phone witness cannot be fixture_only")
        snapshots = _validate_snapshot_sequence(
            self.snapshots,
            source_instance_id=self.attestation.source_instance_id,
        )
        if type(self.evidence) is not RegistryAxisPositiveConfidenceEvidence:
            raise PhonePredictionErrorWitnessError("witness requires exact positive-confidence evidence")
        if self.evidence.axis != AXIS or self.evidence.source_instance_id != self.attestation.source_instance_id:
            raise PhonePredictionErrorWitnessError("witness evidence does not bind the attested source")
        expected = derive_detached_prediction_error_evidence(snapshots)
        if self.evidence != expected:
            raise PhonePredictionErrorWitnessError("witness evidence is not the exact snapshot derivation")
        if self.schema_version != WITNESS_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise PhonePredictionErrorWitnessError("phone witness must remain exact shadow-only material")
        if any(
            (
                self.reviewed_attestation_registered,
                self.runtime_provenance_verifier_registered,
                self.production_source_verifier_registered,
                self.retained_real_observation,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise PhonePredictionErrorWitnessError(
                "preflight witness cannot claim review registration, verifier, retention, window, or authority"
            )
        object.__setattr__(self, "snapshots", snapshots)

    def private_mapping(self) -> dict[str, Any]:
        """Raw recalculable material for the operator-private companion only."""
        return {
            "authority": self.authority,
            "attestation": self.attestation.to_mapping(),
            "evidence": self.evidence.to_mapping(),
            "local_verification_trace_digest": self.local_verification_trace_digest,
            "schema_version": self.schema_version,
            "snapshots": [_snapshot_mapping(item) for item in self.snapshots],
        }

    @property
    def private_material_digest(self) -> str:
        return _digest(self.private_mapping(), "phone_prediction_error_private_witness")

    def public_review_mapping(self) -> dict[str, Any]:
        """Safe review material: exact launch/evidence digests, never raw trace or nonce."""
        mapping = {
            "authority": self.authority,
            "attestation": self.attestation.to_mapping(),
            "attestation_local_review": _attestation_review_mapping(
                self.attestation,
                self.local_verification_trace_digest,
            ),
            "cutover_authorized": self.cutover_authorized,
            "evidence_digest": self.evidence.evidence_digest,
            "evidence_observed_tick": self.evidence.observed_tick,
            "fixture_only": False,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "observation_window_started": self.observation_window_started,
            "private_material_digest": self.private_material_digest,
            "private_raw_location": "operator_private_companion_only",
            "production_source_verifier_registered": self.production_source_verifier_registered,
            "raw_record_count": len(self.snapshots),
            "retained_real_observation": self.retained_real_observation,
            "reviewed_attestation_registered": self.reviewed_attestation_registered,
            "runtime_provenance_verifier_registered": self.runtime_provenance_verifier_registered,
            "schema_version": PUBLIC_REVIEW_SCHEMA_VERSION,
            "snapshot_integrity_digests": [item.source_integrity_digest for item in self.snapshots],
            "source_instance_id": self.evidence.source_instance_id,
        }
        mapping["public_review_digest"] = _digest(mapping, "phone_prediction_error_public_review")
        return mapping


def build_phone_prediction_error_witness(
    *,
    private_nonce: bytes,
    runtime_instance_id: str,
    source_instance_id: str,
    repository_head_sha: str,
    launch_attestation_id: str,
    snapshots: Sequence[PredictionErrorRuntimeSourceSnapshot],
    launch_logical_tick: int = 0,
    entrypoint_id: str = ENTRYPOINT_ID,
) -> PhonePredictionErrorWitness:
    """Bind one actual phone runtime session without registering or retaining it."""

    items = _validate_snapshot_sequence(
        snapshots,
        source_instance_id=source_instance_id,
    )
    attestation = build_operator_public_launch_attestation(
        OperatorLaunchBinding(
            runtime_instance_id=runtime_instance_id,
            source_instance_id=source_instance_id,
            repository_head_sha=repository_head_sha,
            entrypoint_id=entrypoint_id,
            launch_attestation_id=launch_attestation_id,
            logical_tick=launch_logical_tick,
            fixture_only=False,
        ),
        private_nonce,
    )
    local_trace = verify_operator_private_binding(attestation, private_nonce)
    evidence = derive_detached_prediction_error_evidence(items)
    return PhonePredictionErrorWitness(
        attestation=attestation,
        local_verification_trace_digest=local_trace,
        snapshots=items,
        evidence=evidence,
    )
