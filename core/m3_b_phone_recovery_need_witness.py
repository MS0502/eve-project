"""Phone-side real-runtime witness preflight for M3-B ``recovery_need``.

The witness derives the already-governed recovery fields from a real full-engine
interaction followed by a fixed quiet cooldown observation in the same process.
Raw process CPU, wall-clock, and kernel load-average observations remain
operator-private. Only bounded derived evidence, acquisition-method identifiers,
and cryptographic digests are exposed for later review.

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
from core.m3_b_operator_attestation_trust_root import (
    OperatorLaunchBinding,
    OperatorPublicLaunchAttestation,
    build_operator_public_launch_attestation,
    verify_operator_private_binding,
)
from core.m3_b_operational_registry_source_binding import (
    OperationalRegistryRawRecord,
    derive_operational_axis_evidence,
    operational_raw_observation_digest,
)
from core.m3_b_registry_observation_evidence import (
    RegistryAxisPositiveConfidenceEvidence,
)

AXIS = "recovery_need"
SOURCE_SCHEMA_VERSION = "eve.m3-b.phone-recovery-need-runtime-snapshot.v1"
BRIDGE_SCHEMA_VERSION = "eve.m3-b.phone-recovery-need-runtime-source-bridge.v1"
MEASUREMENT_POLICY_VERSION = "eve.m3-b.phone-recovery-need-measurement-policy.v1"
WITNESS_SCHEMA_VERSION = "eve.m3-b.phone-recovery-need-witness.v1"
PUBLIC_REVIEW_SCHEMA_VERSION = "eve.m3-b.phone-recovery-need-public-review.v1"
ENTRYPOINT_ID = "scripts/operator/m3_b_phone_recovery_need_witness.py:main"
DEFAULT_SOURCE_INSTANCE_ID = "runtime:phone-operational-recovery:primary"
REQUIRED_RAW_RECORD_COUNT = 3
REQUIRED_LOGICAL_SPAN_TICKS = 2
TICK_HZ = 1_000_000
FIXED_COOLDOWN_SECONDS = 1.0

PROCESS_CPU_METHOD = "os_times_process_cpu_v1"
QUEUE_METHOD = "kernel_loadavg_1m_capacity_comparison_v1"
COOLDOWN_METHOD = "fixed_post_interaction_quiet_window_1s_v1"
OVERLOAD_COUNT_METHOD = "loadavg_visible_cpu_capacity_breach_count_v1"
RECOVERY_COUNT_METHOD = "cpu_and_queue_nonincrease_indicator_count_v1"


class PhoneRecoveryNeedWitnessError(ValueError):
    """Raised when recovery-need witness material violates the preflight contract."""


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
        raise PhoneRecoveryNeedWitnessError(
            f"{field} is not canonical JSON material"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _identifier(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise PhoneRecoveryNeedWitnessError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PhoneRecoveryNeedWitnessError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result == 0:
        raise PhoneRecoveryNeedWitnessError(f"{field} must be positive")
    return result


def _finite(value: Any, field: str, *, lower: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PhoneRecoveryNeedWitnessError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < lower:
        raise PhoneRecoveryNeedWitnessError(
            f"{field} must be finite and >= {lower}"
        )
    return result


@dataclass(frozen=True, slots=True)
class PhoneRecoveryNeedRuntimeSnapshot:
    source_instance_id: str
    logical_tick: int
    active_process_cpu_seconds: float
    active_wall_seconds: float
    cooldown_process_cpu_seconds: float
    cooldown_wall_seconds: float
    cpu_count: int
    load_average_1m_before: float
    load_average_1m_after_active: float
    load_average_1m_after_cooldown: float
    process_cpu_measurement_method: str = PROCESS_CPU_METHOD
    queue_measurement_method: str = QUEUE_METHOD
    cooldown_measurement_method: str = COOLDOWN_METHOD
    overload_count_method: str = OVERLOAD_COUNT_METHOD
    recovery_count_method: str = RECOVERY_COUNT_METHOD
    schema_version: str = SOURCE_SCHEMA_VERSION
    bridge_schema_version: str = BRIDGE_SCHEMA_VERSION
    measurement_policy_version: str = MEASUREMENT_POLICY_VERSION
    authority: str = SHADOW_AUTHORITY
    runtime_source_read_only: bool = True
    fixture_only: bool = False
    production_origin_verified: bool = False
    production_verifier_registered: bool = False
    retained_real_observation: bool = False

    def __post_init__(self) -> None:
        _identifier(self.source_instance_id, "source_instance_id")
        _nonnegative_int(self.logical_tick, "logical_tick")
        active_cpu = _finite(
            self.active_process_cpu_seconds,
            "active_process_cpu_seconds",
        )
        active_wall = _finite(self.active_wall_seconds, "active_wall_seconds")
        cooldown_cpu = _finite(
            self.cooldown_process_cpu_seconds,
            "cooldown_process_cpu_seconds",
        )
        cooldown_wall = _finite(self.cooldown_wall_seconds, "cooldown_wall_seconds")
        if active_wall <= 0.0 or cooldown_wall <= 0.0:
            raise PhoneRecoveryNeedWitnessError(
                "active and cooldown wall windows must be positive"
            )
        cpu_count = _positive_int(self.cpu_count, "cpu_count")
        for field in (
            "load_average_1m_before",
            "load_average_1m_after_active",
            "load_average_1m_after_cooldown",
        ):
            _finite(getattr(self, field), field)
        if active_cpu > active_wall * cpu_count + 1e-9:
            raise PhoneRecoveryNeedWitnessError(
                "active process CPU exceeds visible CPU capacity"
            )
        if cooldown_cpu > cooldown_wall * cpu_count + 1e-9:
            raise PhoneRecoveryNeedWitnessError(
                "cooldown process CPU exceeds visible CPU capacity"
            )
        expected_methods = {
            "process_cpu_measurement_method": PROCESS_CPU_METHOD,
            "queue_measurement_method": QUEUE_METHOD,
            "cooldown_measurement_method": COOLDOWN_METHOD,
            "overload_count_method": OVERLOAD_COUNT_METHOD,
            "recovery_count_method": RECOVERY_COUNT_METHOD,
        }
        for field, expected in expected_methods.items():
            if getattr(self, field) != expected:
                raise PhoneRecoveryNeedWitnessError(
                    f"unsupported {field} for recovery-need witness"
                )
        if (
            self.schema_version != SOURCE_SCHEMA_VERSION
            or self.bridge_schema_version != BRIDGE_SCHEMA_VERSION
            or self.measurement_policy_version != MEASUREMENT_POLICY_VERSION
            or self.authority != SHADOW_AUTHORITY
        ):
            raise PhoneRecoveryNeedWitnessError(
                "recovery-need snapshot schema/policy drift"
            )
        if self.runtime_source_read_only is not True or self.fixture_only:
            raise PhoneRecoveryNeedWitnessError(
                "phone recovery snapshot must be read-only non-fixture material"
            )
        if any(
            (
                self.production_origin_verified,
                self.production_verifier_registered,
                self.retained_real_observation,
            )
        ):
            raise PhoneRecoveryNeedWitnessError(
                "preflight snapshot cannot pre-claim production verification or retention"
            )

    @property
    def active_window_ticks(self) -> int:
        return max(1, int(round(self.active_wall_seconds * TICK_HZ)))

    @property
    def cooldown_ticks(self) -> int:
        return max(1, int(round(self.cooldown_wall_seconds * TICK_HZ)))

    @property
    def sampling_window_ticks(self) -> int:
        return self.active_window_ticks + self.cooldown_ticks

    @property
    def active_processing_ticks(self) -> int:
        normalized_cpu_seconds = self.active_process_cpu_seconds / float(self.cpu_count)
        return int(normalized_cpu_seconds * TICK_HZ)

    @property
    def active_process_cpu_ratio(self) -> float:
        return float(
            (self.active_process_cpu_seconds / float(self.cpu_count))
            / self.active_wall_seconds
        )

    @property
    def cooldown_process_cpu_ratio(self) -> float:
        return float(
            (self.cooldown_process_cpu_seconds / float(self.cpu_count))
            / self.cooldown_wall_seconds
        )

    @property
    def active_queue_capacity_ratio(self) -> float:
        mean_load = (
            self.load_average_1m_before + self.load_average_1m_after_active
        ) / 2.0
        return float(mean_load / float(self.cpu_count))

    @property
    def cooldown_queue_capacity_ratio(self) -> float:
        mean_load = (
            self.load_average_1m_after_active + self.load_average_1m_after_cooldown
        ) / 2.0
        return float(mean_load / float(self.cpu_count))

    @property
    def recent_overload_count(self) -> int:
        return int(self.load_average_1m_before > float(self.cpu_count)) + int(
            self.load_average_1m_after_active > float(self.cpu_count)
        )

    @property
    def successful_recovery_count(self) -> int:
        return int(
            self.cooldown_process_cpu_ratio <= self.active_process_cpu_ratio
        ) + int(
            self.load_average_1m_after_cooldown
            <= self.load_average_1m_after_active
        )

    @property
    def raw_values(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("active_processing_ticks", self.active_processing_ticks),
            ("cooldown_ticks", self.cooldown_ticks),
            ("recent_overload_count", self.recent_overload_count),
            ("sampling_window_ticks", self.sampling_window_ticks),
            ("successful_recovery_count", self.successful_recovery_count),
        )

    @property
    def source_integrity_digest(self) -> str:
        return _digest(
            {
                "active_process_cpu_seconds": self.active_process_cpu_seconds,
                "active_wall_seconds": self.active_wall_seconds,
                "authority": self.authority,
                "bridge_schema_version": self.bridge_schema_version,
                "cooldown_measurement_method": self.cooldown_measurement_method,
                "cooldown_process_cpu_seconds": self.cooldown_process_cpu_seconds,
                "cooldown_wall_seconds": self.cooldown_wall_seconds,
                "cpu_count": self.cpu_count,
                "fixture_only": self.fixture_only,
                "load_average_1m_after_active": self.load_average_1m_after_active,
                "load_average_1m_after_cooldown": self.load_average_1m_after_cooldown,
                "load_average_1m_before": self.load_average_1m_before,
                "logical_tick": self.logical_tick,
                "measurement_policy_version": self.measurement_policy_version,
                "overload_count_method": self.overload_count_method,
                "process_cpu_measurement_method": self.process_cpu_measurement_method,
                "production_origin_verified": self.production_origin_verified,
                "production_verifier_registered": self.production_verifier_registered,
                "queue_measurement_method": self.queue_measurement_method,
                "recovery_count_method": self.recovery_count_method,
                "retained_real_observation": self.retained_real_observation,
                "runtime_source_read_only": self.runtime_source_read_only,
                "source_instance_id": self.source_instance_id,
                "source_schema_version": self.schema_version,
            },
            "phone_recovery_need_runtime_snapshot",
        )

    @property
    def source_snapshot_id(self) -> str:
        return f"phone-recovery-need:{self.logical_tick}:{self.source_integrity_digest[:20]}"

    def to_operational_raw_record(self) -> OperationalRegistryRawRecord:
        observation_id = (
            f"phone-recovery-need:{self.logical_tick}:"
            f"{self.source_integrity_digest[:24]}"
        )
        raw_digest = operational_raw_observation_digest(
            axis=AXIS,
            logical_tick=self.logical_tick,
            observation_id=observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.schema_version,
            source_integrity_digest=self.source_integrity_digest,
            raw_values=self.raw_values,
        )
        return OperationalRegistryRawRecord(
            axis=AXIS,
            logical_tick=self.logical_tick,
            observation_id=observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.schema_version,
            source_integrity_digest=self.source_integrity_digest,
            raw_observation_digest=raw_digest,
            raw_values=self.raw_values,
        )

    def private_mapping(self) -> dict[str, Any]:
        return {
            "active_process_cpu_seconds": self.active_process_cpu_seconds,
            "active_wall_seconds": self.active_wall_seconds,
            "authority": self.authority,
            "bridge_schema_version": self.bridge_schema_version,
            "cooldown_measurement_method": self.cooldown_measurement_method,
            "cooldown_process_cpu_seconds": self.cooldown_process_cpu_seconds,
            "cooldown_wall_seconds": self.cooldown_wall_seconds,
            "cpu_count": self.cpu_count,
            "fixture_only": self.fixture_only,
            "load_average_1m_after_active": self.load_average_1m_after_active,
            "load_average_1m_after_cooldown": self.load_average_1m_after_cooldown,
            "load_average_1m_before": self.load_average_1m_before,
            "logical_tick": self.logical_tick,
            "measurement_policy_version": self.measurement_policy_version,
            "overload_count_method": self.overload_count_method,
            "process_cpu_measurement_method": self.process_cpu_measurement_method,
            "queue_measurement_method": self.queue_measurement_method,
            "raw_values": [[field, value] for field, value in self.raw_values],
            "recovery_count_method": self.recovery_count_method,
            "runtime_source_read_only": self.runtime_source_read_only,
            "schema_version": self.schema_version,
            "source_instance_id": self.source_instance_id,
            "source_integrity_digest": self.source_integrity_digest,
            "source_snapshot_id": self.source_snapshot_id,
        }


def _validate_snapshot_sequence(
    snapshots: Sequence[PhoneRecoveryNeedRuntimeSnapshot],
    *,
    source_instance_id: str,
) -> tuple[PhoneRecoveryNeedRuntimeSnapshot, ...]:
    items = tuple(snapshots)
    if len(items) != REQUIRED_RAW_RECORD_COUNT:
        raise PhoneRecoveryNeedWitnessError(
            "recovery-need phone witness requires exactly three snapshots"
        )
    if any(type(item) is not PhoneRecoveryNeedRuntimeSnapshot for item in items):
        raise PhoneRecoveryNeedWitnessError(
            "recovery-need witness requires exact immutable snapshot types"
        )
    if any(item.source_instance_id != source_instance_id for item in items):
        raise PhoneRecoveryNeedWitnessError(
            "attestation and snapshots must bind one source instance"
        )
    ticks = tuple(item.logical_tick for item in items)
    if ticks != tuple(sorted(set(ticks))):
        raise PhoneRecoveryNeedWitnessError(
            "recovery-need snapshot ticks must be strictly increasing"
        )
    if ticks[-1] - ticks[0] < REQUIRED_LOGICAL_SPAN_TICKS:
        raise PhoneRecoveryNeedWitnessError(
            "recovery-need snapshots do not satisfy minimum logical span"
        )
    return items


def derive_detached_recovery_need_evidence(
    snapshots: Sequence[PhoneRecoveryNeedRuntimeSnapshot],
) -> RegistryAxisPositiveConfidenceEvidence:
    items = tuple(snapshots)
    if not items:
        raise PhoneRecoveryNeedWitnessError("recovery-need evidence requires snapshots")
    validated = _validate_snapshot_sequence(
        items,
        source_instance_id=items[0].source_instance_id,
    )
    return derive_operational_axis_evidence(
        tuple(item.to_operational_raw_record() for item in validated)
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
class PhoneRecoveryNeedWitness:
    attestation: OperatorPublicLaunchAttestation
    local_verification_trace_digest: str
    snapshots: tuple[PhoneRecoveryNeedRuntimeSnapshot, ...]
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
            raise PhoneRecoveryNeedWitnessError(
                "witness requires an exact non-fixture public attestation"
            )
        snapshots = _validate_snapshot_sequence(
            self.snapshots,
            source_instance_id=self.attestation.source_instance_id,
        )
        if type(self.evidence) is not RegistryAxisPositiveConfidenceEvidence:
            raise PhoneRecoveryNeedWitnessError(
                "witness requires exact positive-confidence evidence"
            )
        if (
            self.evidence.axis != AXIS
            or self.evidence.source_instance_id != self.attestation.source_instance_id
        ):
            raise PhoneRecoveryNeedWitnessError(
                "recovery-need evidence does not bind the attested source"
            )
        if self.evidence != derive_detached_recovery_need_evidence(snapshots):
            raise PhoneRecoveryNeedWitnessError(
                "recovery-need evidence is not the exact snapshot derivation"
            )
        if self.schema_version != WITNESS_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise PhoneRecoveryNeedWitnessError(
                "recovery-need witness must remain exact shadow-only material"
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
            raise PhoneRecoveryNeedWitnessError(
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
        return _digest(self.private_mapping(), "phone_recovery_need_private_witness")

    def public_review_mapping(self) -> dict[str, Any]:
        mapping = {
            "authority": self.authority,
            "attestation": self.attestation.to_mapping(),
            "attestation_local_review": _attestation_review_mapping(
                self.attestation,
                self.local_verification_trace_digest,
            ),
            "axis": AXIS,
            "cooldown_measurement_methods": sorted(
                {item.cooldown_measurement_method for item in self.snapshots}
            ),
            "cutover_authorized": self.cutover_authorized,
            "evidence": self.evidence.to_mapping(),
            "evidence_digest": self.evidence.evidence_digest,
            "evidence_observed_tick": self.evidence.observed_tick,
            "fixture_only": False,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "measurement_policy_version": MEASUREMENT_POLICY_VERSION,
            "observation_window_started": self.observation_window_started,
            "overload_count_methods": sorted(
                {item.overload_count_method for item in self.snapshots}
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
            "recovery_count_methods": sorted(
                {item.recovery_count_method for item in self.snapshots}
            ),
            "retained_real_observation": self.retained_real_observation,
            "reviewed_attestation_registered": self.reviewed_attestation_registered,
            "runtime_provenance_verifier_registered": self.runtime_provenance_verifier_registered,
            "schema_version": PUBLIC_REVIEW_SCHEMA_VERSION,
            "snapshot_integrity_digests": [
                item.source_integrity_digest for item in self.snapshots
            ],
            "source_instance_id": self.evidence.source_instance_id,
            "tick_hz": TICK_HZ,
        }
        mapping["public_review_digest"] = _digest(
            mapping,
            "phone_recovery_need_public_review",
        )
        return mapping


def build_phone_recovery_need_witness(
    *,
    private_nonce: bytes,
    runtime_instance_id: str,
    source_instance_id: str,
    repository_head_sha: str,
    launch_attestation_id: str,
    snapshots: Sequence[PhoneRecoveryNeedRuntimeSnapshot],
    launch_logical_tick: int = 0,
    entrypoint_id: str = ENTRYPOINT_ID,
) -> PhoneRecoveryNeedWitness:
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
    evidence = derive_detached_recovery_need_evidence(items)
    return PhoneRecoveryNeedWitness(
        attestation=attestation,
        local_verification_trace_digest=local_trace,
        snapshots=items,
        evidence=evidence,
    )
