"""Phone-side real-runtime appraised witness preflight for M3-B ``stress_load``.

The witness measures bounded process CPU and kernel load-average behavior around
three new full-engine interactions. Those operator-private measurements are
converted by one versioned deterministic appraisal policy into the already
canonical ``stress_load`` appraisal fields. The detached appraisal records are
then evaluated by the merged #180 source-binding contract.

Raw CPU/wall/load observations remain operator-private. Only bounded derived
evidence, versioned method identifiers, and cryptographic digests are exposed
for later review.

This module registers no trust or verifier, appends no retained observation,
starts no M3-B observation window, mutates no registry owner, and grants no
runtime or cutover authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_appraised_survival_source_binding import (
    APPRAISAL_SCHEMA_VERSION,
    AppraisedSurvivalRawRecord,
    appraised_survival_raw_observation_digest,
    derive_appraised_survival_axis_evidence,
)
from core.m3_b_operator_attestation_trust_root import (
    OperatorLaunchBinding,
    OperatorPublicLaunchAttestation,
    build_operator_public_launch_attestation,
    verify_operator_private_binding,
)
from core.m3_b_registry_observation_evidence import (
    RegistryAxisPositiveConfidenceEvidence,
)

AXIS = "stress_load"
SOURCE_SCHEMA_VERSION = "eve.m3-b.phone-stress-load-runtime-snapshot.v1"
BRIDGE_SCHEMA_VERSION = "eve.m3-b.phone-stress-load-appraisal-bridge.v1"
APPRAISAL_POLICY_VERSION = "eve.m3-b.phone-stress-load-appraisal-policy.v1"
WITNESS_SCHEMA_VERSION = "eve.m3-b.phone-stress-load-witness.v1"
PUBLIC_REVIEW_SCHEMA_VERSION = "eve.m3-b.phone-stress-load-public-review.v1"
ENTRYPOINT_ID = "scripts/operator/m3_b_phone_stress_load_witness.py:main"
DEFAULT_SOURCE_INSTANCE_ID = "runtime:phone-appraised-stress:primary"
REQUIRED_RAW_RECORD_COUNT = 3
REQUIRED_LOGICAL_SPAN_TICKS = 2

PROCESS_CPU_METHOD = "os_times_process_cpu_v1"
QUEUE_METHOD = "kernel_loadavg_1m_visible_cpu_ratio_v1"
CONTROLLABILITY_METHOD = "one_minus_mean_overload_and_queue_variability_v1"
DEMAND_METHOD = "mean_process_cpu_and_queue_ratio_v1"
OVERLOAD_METHOD = "max_process_cpu_and_queue_ratio_v1"
UNCERTAINTY_METHOD = "absolute_queue_ratio_delta_v1"


class PhoneStressLoadWitnessError(ValueError):
    """Raised when stress-load witness material violates the preflight contract."""


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
        raise PhoneStressLoadWitnessError(
            f"{field} is not canonical JSON material"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _identifier(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise PhoneStressLoadWitnessError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PhoneStressLoadWitnessError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result == 0:
        raise PhoneStressLoadWitnessError(f"{field} must be positive")
    return result


def _finite(value: Any, field: str, *, lower: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PhoneStressLoadWitnessError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < lower:
        raise PhoneStressLoadWitnessError(
            f"{field} must be finite and >= {lower}"
        )
    return result


def _clip_unit(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True, slots=True)
class PhoneStressLoadRuntimeSnapshot:
    source_instance_id: str
    logical_tick: int
    process_cpu_seconds: float
    wall_seconds: float
    cpu_count: int
    load_average_1m_before: float
    load_average_1m_after: float
    process_cpu_measurement_method: str = PROCESS_CPU_METHOD
    queue_measurement_method: str = QUEUE_METHOD
    controllability_method: str = CONTROLLABILITY_METHOD
    demand_method: str = DEMAND_METHOD
    overload_method: str = OVERLOAD_METHOD
    uncertainty_method: str = UNCERTAINTY_METHOD
    schema_version: str = SOURCE_SCHEMA_VERSION
    bridge_schema_version: str = BRIDGE_SCHEMA_VERSION
    appraisal_policy_version: str = APPRAISAL_POLICY_VERSION
    authority: str = SHADOW_AUTHORITY
    runtime_source_read_only: bool = True
    fixture_only: bool = False
    production_origin_verified: bool = False
    production_verifier_registered: bool = False
    retained_real_observation: bool = False

    def __post_init__(self) -> None:
        _identifier(self.source_instance_id, "source_instance_id")
        _nonnegative_int(self.logical_tick, "logical_tick")
        cpu_seconds = _finite(self.process_cpu_seconds, "process_cpu_seconds")
        wall_seconds = _finite(self.wall_seconds, "wall_seconds")
        cpu_count = _positive_int(self.cpu_count, "cpu_count")
        if wall_seconds <= 0.0:
            raise PhoneStressLoadWitnessError("wall_seconds must be positive")
        if cpu_seconds > wall_seconds * cpu_count + 1e-9:
            raise PhoneStressLoadWitnessError(
                "process CPU exceeds visible CPU capacity"
            )
        _finite(self.load_average_1m_before, "load_average_1m_before")
        _finite(self.load_average_1m_after, "load_average_1m_after")
        expected_methods = {
            "process_cpu_measurement_method": PROCESS_CPU_METHOD,
            "queue_measurement_method": QUEUE_METHOD,
            "controllability_method": CONTROLLABILITY_METHOD,
            "demand_method": DEMAND_METHOD,
            "overload_method": OVERLOAD_METHOD,
            "uncertainty_method": UNCERTAINTY_METHOD,
        }
        for field, expected in expected_methods.items():
            if getattr(self, field) != expected:
                raise PhoneStressLoadWitnessError(
                    f"unsupported {field} for stress-load witness"
                )
        if (
            self.schema_version != SOURCE_SCHEMA_VERSION
            or self.bridge_schema_version != BRIDGE_SCHEMA_VERSION
            or self.appraisal_policy_version != APPRAISAL_POLICY_VERSION
            or self.authority != SHADOW_AUTHORITY
        ):
            raise PhoneStressLoadWitnessError(
                "stress-load snapshot schema/policy drift"
            )
        if self.runtime_source_read_only is not True or self.fixture_only:
            raise PhoneStressLoadWitnessError(
                "phone stress snapshot must be read-only non-fixture material"
            )
        if any(
            (
                self.production_origin_verified,
                self.production_verifier_registered,
                self.retained_real_observation,
            )
        ):
            raise PhoneStressLoadWitnessError(
                "preflight snapshot cannot pre-claim production verification or retention"
            )

    @property
    def process_cpu_ratio(self) -> float:
        return _clip_unit(
            (self.process_cpu_seconds / float(self.cpu_count)) / self.wall_seconds
        )

    @property
    def queue_ratio_before(self) -> float:
        return _clip_unit(self.load_average_1m_before / float(self.cpu_count))

    @property
    def queue_ratio_after(self) -> float:
        return _clip_unit(self.load_average_1m_after / float(self.cpu_count))

    @property
    def uncertainty_score(self) -> float:
        return _clip_unit(abs(self.queue_ratio_after - self.queue_ratio_before))

    @property
    def demand_score(self) -> float:
        return _clip_unit((self.process_cpu_ratio + self.queue_ratio_after) / 2.0)

    @property
    def overload_score(self) -> float:
        return _clip_unit(max(self.process_cpu_ratio, self.queue_ratio_after))

    @property
    def controllability_score(self) -> float:
        return _clip_unit(1.0 - ((self.overload_score + self.uncertainty_score) / 2.0))

    @property
    def appraisal_input_mapping(self) -> dict[str, Any]:
        return {
            "appraisal_policy_version": self.appraisal_policy_version,
            "authority": self.authority,
            "bridge_schema_version": self.bridge_schema_version,
            "cpu_count": self.cpu_count,
            "load_average_1m_after": self.load_average_1m_after,
            "load_average_1m_before": self.load_average_1m_before,
            "logical_tick": self.logical_tick,
            "process_cpu_measurement_method": self.process_cpu_measurement_method,
            "process_cpu_seconds": self.process_cpu_seconds,
            "queue_measurement_method": self.queue_measurement_method,
            "runtime_source_read_only": self.runtime_source_read_only,
            "source_instance_id": self.source_instance_id,
            "source_schema_version": self.schema_version,
            "wall_seconds": self.wall_seconds,
        }

    @property
    def appraisal_input_digest(self) -> str:
        return _digest(self.appraisal_input_mapping, "phone_stress_load_appraisal_input")

    @property
    def appraisal_trace_mapping(self) -> dict[str, Any]:
        return {
            "appraisal_input_digest": self.appraisal_input_digest,
            "appraisal_policy_version": self.appraisal_policy_version,
            "appraisal_version": APPRAISAL_SCHEMA_VERSION,
            "controllability_method": self.controllability_method,
            "controllability_score": self.controllability_score,
            "demand_method": self.demand_method,
            "demand_score": self.demand_score,
            "overload_method": self.overload_method,
            "overload_score": self.overload_score,
            "uncertainty_method": self.uncertainty_method,
            "uncertainty_score": self.uncertainty_score,
        }

    @property
    def appraisal_integrity_digest(self) -> str:
        return _digest(self.appraisal_trace_mapping, "phone_stress_load_appraisal_trace")

    @property
    def appraisal_trace_id(self) -> str:
        return (
            f"phone-stress-load:{self.logical_tick}:"
            f"{self.appraisal_integrity_digest[:20]}"
        )

    @property
    def source_integrity_digest(self) -> str:
        return _digest(
            {
                "appraisal_input_digest": self.appraisal_input_digest,
                "appraisal_integrity_digest": self.appraisal_integrity_digest,
                "appraisal_policy_version": self.appraisal_policy_version,
                "authority": self.authority,
                "bridge_schema_version": self.bridge_schema_version,
                "fixture_only": self.fixture_only,
                "logical_tick": self.logical_tick,
                "production_origin_verified": self.production_origin_verified,
                "production_verifier_registered": self.production_verifier_registered,
                "retained_real_observation": self.retained_real_observation,
                "runtime_source_read_only": self.runtime_source_read_only,
                "source_instance_id": self.source_instance_id,
                "source_schema_version": self.schema_version,
            },
            "phone_stress_load_runtime_snapshot",
        )

    @property
    def source_snapshot_id(self) -> str:
        return f"phone-stress-load:{self.logical_tick}:{self.source_integrity_digest[:20]}"

    @property
    def raw_values(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("controllability_score", self.controllability_score),
            ("demand_score", self.demand_score),
            ("overload_score", self.overload_score),
            ("uncertainty_score", self.uncertainty_score),
        )

    def to_appraised_raw_record(self) -> AppraisedSurvivalRawRecord:
        observation_id = (
            f"phone-stress-load:{self.logical_tick}:"
            f"{self.source_integrity_digest[:24]}"
        )
        raw_digest = appraised_survival_raw_observation_digest(
            axis=AXIS,
            logical_tick=self.logical_tick,
            observation_id=observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.schema_version,
            source_integrity_digest=self.source_integrity_digest,
            appraisal_trace_id=self.appraisal_trace_id,
            appraisal_input_digest=self.appraisal_input_digest,
            appraisal_integrity_digest=self.appraisal_integrity_digest,
            raw_values=self.raw_values,
        )
        return AppraisedSurvivalRawRecord(
            axis=AXIS,
            logical_tick=self.logical_tick,
            observation_id=observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.schema_version,
            source_integrity_digest=self.source_integrity_digest,
            appraisal_trace_id=self.appraisal_trace_id,
            appraisal_input_digest=self.appraisal_input_digest,
            appraisal_integrity_digest=self.appraisal_integrity_digest,
            raw_observation_digest=raw_digest,
            raw_values=self.raw_values,
        )

    def private_mapping(self) -> dict[str, Any]:
        return {
            "appraisal_input": self.appraisal_input_mapping,
            "appraisal_trace": self.appraisal_trace_mapping,
            "appraisal_input_digest": self.appraisal_input_digest,
            "appraisal_integrity_digest": self.appraisal_integrity_digest,
            "appraisal_trace_id": self.appraisal_trace_id,
            "authority": self.authority,
            "bridge_schema_version": self.bridge_schema_version,
            "logical_tick": self.logical_tick,
            "raw_values": [[field, value] for field, value in self.raw_values],
            "schema_version": self.schema_version,
            "source_instance_id": self.source_instance_id,
            "source_integrity_digest": self.source_integrity_digest,
            "source_snapshot_id": self.source_snapshot_id,
        }


def _validate_snapshot_sequence(
    snapshots: Sequence[PhoneStressLoadRuntimeSnapshot],
    *,
    source_instance_id: str,
) -> tuple[PhoneStressLoadRuntimeSnapshot, ...]:
    items = tuple(snapshots)
    if len(items) != REQUIRED_RAW_RECORD_COUNT:
        raise PhoneStressLoadWitnessError(
            "stress-load phone witness requires exactly three snapshots"
        )
    if any(type(item) is not PhoneStressLoadRuntimeSnapshot for item in items):
        raise PhoneStressLoadWitnessError(
            "stress-load witness requires exact immutable snapshot types"
        )
    if any(item.source_instance_id != source_instance_id for item in items):
        raise PhoneStressLoadWitnessError(
            "attestation and snapshots must bind one source instance"
        )
    ticks = tuple(item.logical_tick for item in items)
    if ticks != tuple(sorted(set(ticks))):
        raise PhoneStressLoadWitnessError(
            "stress-load snapshot ticks must be strictly increasing"
        )
    if ticks[-1] - ticks[0] < REQUIRED_LOGICAL_SPAN_TICKS:
        raise PhoneStressLoadWitnessError(
            "stress-load snapshots do not satisfy minimum logical span"
        )
    return items


def derive_detached_stress_load_evidence(
    snapshots: Sequence[PhoneStressLoadRuntimeSnapshot],
) -> RegistryAxisPositiveConfidenceEvidence:
    items = tuple(snapshots)
    if not items:
        raise PhoneStressLoadWitnessError("stress-load evidence requires snapshots")
    validated = _validate_snapshot_sequence(
        items,
        source_instance_id=items[0].source_instance_id,
    )
    return derive_appraised_survival_axis_evidence(
        tuple(item.to_appraised_raw_record() for item in validated)
    )


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


@dataclass(frozen=True, slots=True)
class PhoneStressLoadWitness:
    attestation: OperatorPublicLaunchAttestation
    local_verification_trace_digest: str
    snapshots: tuple[PhoneStressLoadRuntimeSnapshot, ...]
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
        if (
            type(self.attestation) is not OperatorPublicLaunchAttestation
            or self.attestation.fixture_only
        ):
            raise PhoneStressLoadWitnessError(
                "witness requires an exact non-fixture public attestation"
            )
        snapshots = _validate_snapshot_sequence(
            self.snapshots,
            source_instance_id=self.attestation.source_instance_id,
        )
        if type(self.evidence) is not RegistryAxisPositiveConfidenceEvidence:
            raise PhoneStressLoadWitnessError(
                "witness requires exact positive-confidence evidence"
            )
        if (
            self.evidence.axis != AXIS
            or self.evidence.source_instance_id != self.attestation.source_instance_id
        ):
            raise PhoneStressLoadWitnessError(
                "stress-load evidence does not bind the attested source"
            )
        if self.evidence != derive_detached_stress_load_evidence(snapshots):
            raise PhoneStressLoadWitnessError(
                "stress-load evidence is not the exact snapshot appraisal derivation"
            )
        if self.schema_version != WITNESS_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise PhoneStressLoadWitnessError(
                "stress-load witness must remain exact shadow-only material"
            )
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
            raise PhoneStressLoadWitnessError(
                "preflight witness cannot claim review, verifier, retention, window, or authority"
            )
        object.__setattr__(self, "snapshots", snapshots)

    def private_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "attestation": self.attestation.to_mapping(),
            "evidence": self.evidence.to_mapping(),
            "local_verification_trace_digest": self.local_verification_trace_digest,
            "schema_version": self.schema_version,
            "snapshots": [item.private_mapping() for item in self.snapshots],
        }

    @property
    def private_material_digest(self) -> str:
        return _digest(self.private_mapping(), "phone_stress_load_private_witness")

    def public_review_mapping(self) -> dict[str, Any]:
        mapping = {
            "appraisal_input_digests": [
                item.appraisal_input_digest for item in self.snapshots
            ],
            "appraisal_integrity_digests": [
                item.appraisal_integrity_digest for item in self.snapshots
            ],
            "appraisal_policy_version": APPRAISAL_POLICY_VERSION,
            "appraisal_version": APPRAISAL_SCHEMA_VERSION,
            "attestation": self.attestation.to_mapping(),
            "attestation_local_review": _attestation_review_mapping(
                self.attestation,
                self.local_verification_trace_digest,
            ),
            "authority": self.authority,
            "axis": AXIS,
            "controllability_methods": sorted(
                {item.controllability_method for item in self.snapshots}
            ),
            "cutover_authorized": self.cutover_authorized,
            "demand_methods": sorted({item.demand_method for item in self.snapshots}),
            "evidence": self.evidence.to_mapping(),
            "evidence_digest": self.evidence.evidence_digest,
            "evidence_observed_tick": self.evidence.observed_tick,
            "fixture_only": False,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "observation_window_started": self.observation_window_started,
            "overload_methods": sorted(
                {item.overload_method for item in self.snapshots}
            ),
            "private_material_digest": self.private_material_digest,
            "private_raw_location": "operator_private_companion_only",
            "process_cpu_measurement_methods": sorted(
                {item.process_cpu_measurement_method for item in self.snapshots}
            ),
            "production_source_verifier_registered": self.production_source_verifier_registered,
            "queue_measurement_methods": sorted(
                {item.queue_measurement_method for item in self.snapshots}
            ),
            "raw_record_count": len(self.snapshots),
            "retained_real_observation": self.retained_real_observation,
            "reviewed_attestation_registered": self.reviewed_attestation_registered,
            "runtime_provenance_verifier_registered": self.runtime_provenance_verifier_registered,
            "schema_version": PUBLIC_REVIEW_SCHEMA_VERSION,
            "snapshot_integrity_digests": [
                item.source_integrity_digest for item in self.snapshots
            ],
            "source_instance_id": self.evidence.source_instance_id,
            "uncertainty_methods": sorted(
                {item.uncertainty_method for item in self.snapshots}
            ),
        }
        mapping["public_review_digest"] = _digest(
            mapping,
            "phone_stress_load_public_review",
        )
        return mapping


def build_phone_stress_load_witness(
    *,
    private_nonce: bytes,
    runtime_instance_id: str,
    source_instance_id: str,
    repository_head_sha: str,
    launch_attestation_id: str,
    snapshots: Sequence[PhoneStressLoadRuntimeSnapshot],
    launch_logical_tick: int = 0,
    entrypoint_id: str = ENTRYPOINT_ID,
) -> PhoneStressLoadWitness:
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
    evidence = derive_detached_stress_load_evidence(items)
    return PhoneStressLoadWitness(
        attestation=attestation,
        local_verification_trace_digest=local_trace,
        snapshots=items,
        evidence=evidence,
    )
