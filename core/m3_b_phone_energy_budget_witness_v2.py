"""Phone-side real-runtime witness preflight for the M3-B ``energy_budget`` axis.

The witness keeps raw CPU/memory/battery/process observations operator-private
while exposing only bounded evidence/digests for later review. Android app
sandboxes may deny ``/proc/stat``, ``/proc/meminfo``, or power-supply sysfs;
the v2 snapshot therefore records the exact observation method used and accepts
only explicit real-kernel/API fallbacks.

This module does not register trust or verifiers, append retained observations,
start the M3-B observation window, mutate registry ownership, or grant authority.
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
from core.m3_b_registry_observation_evidence import RegistryAxisPositiveConfidenceEvidence

AXIS = "energy_budget"
SOURCE_SCHEMA_VERSION = "eve.m3-b.phone-energy-budget-runtime-snapshot.v2"
BRIDGE_SCHEMA_VERSION = "eve.m3-b.phone-energy-budget-runtime-source-bridge.v2"
MEASUREMENT_POLICY_VERSION = "eve.m3-b.phone-energy-budget-measurement-policy.v2"
WITNESS_SCHEMA_VERSION = "eve.m3-b.phone-energy-budget-witness.v2"
PUBLIC_REVIEW_SCHEMA_VERSION = "eve.m3-b.phone-energy-budget-public-review.v2"
ENTRYPOINT_ID = "scripts/operator/m3_b_phone_energy_budget_witness_v2.py:main"
DEFAULT_SOURCE_INSTANCE_ID = "runtime:phone-operational-energy:primary"
REQUIRED_RAW_RECORD_COUNT = 3
REQUIRED_LOGICAL_SPAN_TICKS = 2
SAMPLING_WINDOW_TICKS = 1

CPU_METHOD_PROC_STAT = "proc_stat_idle_delta_v1"
CPU_METHOD_LOADAVG = "kernel_loadavg_1m_headroom_v1"
MEMORY_METHOD_PROC_MEMINFO = "proc_meminfo_available_v1"
MEMORY_METHOD_SYSCONF = "sysconf_avphys_pages_v1"
BATTERY_METHOD_SYSFS = "sysfs_capacity_v1"
BATTERY_METHOD_TERMUX_API = "termux_api_battery_status_v1"


class PhoneEnergyBudgetWitnessError(ValueError):
    """Raised when energy-budget witness material fails the exact preflight contract."""


def _canonical(value: Any, field: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PhoneEnergyBudgetWitnessError(f"{field} is not canonical JSON material") from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _identifier(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise PhoneEnergyBudgetWitnessError(f"{field} must be a bounded non-empty string")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PhoneEnergyBudgetWitnessError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result == 0:
        raise PhoneEnergyBudgetWitnessError(f"{field} must be positive")
    return result


def _finite(value: Any, field: str, *, lower: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PhoneEnergyBudgetWitnessError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < lower:
        raise PhoneEnergyBudgetWitnessError(f"{field} must be finite and >= {lower}")
    return result


@dataclass(frozen=True, slots=True)
class PhoneEnergyBudgetRuntimeSnapshot:
    source_instance_id: str
    logical_tick: int
    cpu_measurement_method: str
    process_cpu_seconds: float
    wall_seconds: float
    cpu_count: int
    mem_total_kib: int
    mem_available_kib: int
    memory_measurement_method: str
    battery_capacity_percent: int
    battery_measurement_method: str
    cpu_total_delta: int | None = None
    cpu_idle_delta: int | None = None
    load_average_1m_before: float | None = None
    load_average_1m_after: float | None = None
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
        _finite(self.process_cpu_seconds, "process_cpu_seconds")
        if _finite(self.wall_seconds, "wall_seconds") <= 0.0:
            raise PhoneEnergyBudgetWitnessError("wall_seconds must be positive")
        _positive_int(self.cpu_count, "cpu_count")
        if self.cpu_measurement_method == CPU_METHOD_PROC_STAT:
            total = _positive_int(self.cpu_total_delta, "cpu_total_delta")
            idle = _nonnegative_int(self.cpu_idle_delta, "cpu_idle_delta")
            if idle > total:
                raise PhoneEnergyBudgetWitnessError("cpu_idle_delta cannot exceed cpu_total_delta")
            if self.load_average_1m_before is not None or self.load_average_1m_after is not None:
                raise PhoneEnergyBudgetWitnessError("proc-stat CPU snapshots cannot also carry load-average observations")
        elif self.cpu_measurement_method == CPU_METHOD_LOADAVG:
            if self.cpu_total_delta is not None or self.cpu_idle_delta is not None:
                raise PhoneEnergyBudgetWitnessError("load-average CPU snapshots cannot carry proc-stat deltas")
            _finite(self.load_average_1m_before, "load_average_1m_before")
            _finite(self.load_average_1m_after, "load_average_1m_after")
        else:
            raise PhoneEnergyBudgetWitnessError("unsupported cpu_measurement_method")
        if self.memory_measurement_method not in {MEMORY_METHOD_PROC_MEMINFO, MEMORY_METHOD_SYSCONF}:
            raise PhoneEnergyBudgetWitnessError("unsupported memory_measurement_method")
        total_mem = _positive_int(self.mem_total_kib, "mem_total_kib")
        available_mem = _nonnegative_int(self.mem_available_kib, "mem_available_kib")
        if available_mem > total_mem:
            raise PhoneEnergyBudgetWitnessError("mem_available_kib cannot exceed mem_total_kib")
        if self.battery_measurement_method not in {BATTERY_METHOD_SYSFS, BATTERY_METHOD_TERMUX_API}:
            raise PhoneEnergyBudgetWitnessError("unsupported battery_measurement_method")
        capacity = _nonnegative_int(self.battery_capacity_percent, "battery_capacity_percent")
        if capacity > 100:
            raise PhoneEnergyBudgetWitnessError("battery_capacity_percent must be inside [0,100]")
        if self.schema_version != SOURCE_SCHEMA_VERSION or self.bridge_schema_version != BRIDGE_SCHEMA_VERSION or self.measurement_policy_version != MEASUREMENT_POLICY_VERSION or self.authority != SHADOW_AUTHORITY:
            raise PhoneEnergyBudgetWitnessError("energy-budget snapshot schema/policy drift")
        if self.runtime_source_read_only is not True or self.fixture_only:
            raise PhoneEnergyBudgetWitnessError("phone energy-budget snapshot must be read-only non-fixture material")
        if any((self.production_origin_verified, self.production_verifier_registered, self.retained_real_observation)):
            raise PhoneEnergyBudgetWitnessError("preflight snapshot cannot pre-claim production verification or retention")

    @property
    def available_cpu_budget(self) -> float:
        if self.cpu_measurement_method == CPU_METHOD_PROC_STAT:
            assert self.cpu_idle_delta is not None and self.cpu_total_delta is not None
            return float(self.cpu_idle_delta / self.cpu_total_delta)
        assert self.load_average_1m_before is not None and self.load_average_1m_after is not None
        mean_load = (self.load_average_1m_before + self.load_average_1m_after) / 2.0
        return float(max(0.0, min(1.0, 1.0 - mean_load / float(self.cpu_count))))

    @property
    def available_memory_budget(self) -> float:
        return float(self.mem_available_kib / self.mem_total_kib)

    @property
    def battery_governor_band(self) -> float:
        return float(self.battery_capacity_percent / 100.0)

    @property
    def foreground_load(self) -> float:
        return float(max(0.0, min(1.0, self.process_cpu_seconds / (self.wall_seconds * self.cpu_count))))

    @property
    def raw_values(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("available_cpu_budget", self.available_cpu_budget),
            ("available_memory_budget", self.available_memory_budget),
            ("battery_governor_band", self.battery_governor_band),
            ("foreground_load", self.foreground_load),
            ("sampling_window_ticks", SAMPLING_WINDOW_TICKS),
        )

    @property
    def source_integrity_digest(self) -> str:
        return _digest({
            "authority": self.authority,
            "battery_capacity_percent": self.battery_capacity_percent,
            "battery_measurement_method": self.battery_measurement_method,
            "bridge_schema_version": self.bridge_schema_version,
            "cpu_count": self.cpu_count,
            "cpu_idle_delta": self.cpu_idle_delta,
            "cpu_measurement_method": self.cpu_measurement_method,
            "cpu_total_delta": self.cpu_total_delta,
            "fixture_only": self.fixture_only,
            "load_average_1m_after": self.load_average_1m_after,
            "load_average_1m_before": self.load_average_1m_before,
            "logical_tick": self.logical_tick,
            "measurement_policy_version": self.measurement_policy_version,
            "mem_available_kib": self.mem_available_kib,
            "mem_total_kib": self.mem_total_kib,
            "memory_measurement_method": self.memory_measurement_method,
            "process_cpu_seconds": self.process_cpu_seconds,
            "production_origin_verified": self.production_origin_verified,
            "production_verifier_registered": self.production_verifier_registered,
            "retained_real_observation": self.retained_real_observation,
            "runtime_source_read_only": self.runtime_source_read_only,
            "source_instance_id": self.source_instance_id,
            "source_schema_version": self.schema_version,
            "wall_seconds": self.wall_seconds,
        }, "phone_energy_budget_runtime_snapshot")

    @property
    def source_snapshot_id(self) -> str:
        return f"phone-energy-budget:{self.logical_tick}:{self.source_integrity_digest[:20]}"

    def to_operational_raw_record(self) -> OperationalRegistryRawRecord:
        observation_id = f"phone-energy-budget:{self.logical_tick}:{self.source_integrity_digest[:24]}"
        raw_digest = operational_raw_observation_digest(axis=AXIS, logical_tick=self.logical_tick, observation_id=observation_id, source_instance_id=self.source_instance_id, source_snapshot_id=self.source_snapshot_id, source_schema_version=self.schema_version, source_integrity_digest=self.source_integrity_digest, raw_values=self.raw_values)
        return OperationalRegistryRawRecord(axis=AXIS, logical_tick=self.logical_tick, observation_id=observation_id, source_instance_id=self.source_instance_id, source_snapshot_id=self.source_snapshot_id, source_schema_version=self.schema_version, source_integrity_digest=self.source_integrity_digest, raw_observation_digest=raw_digest, raw_values=self.raw_values)

    def private_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "battery_capacity_percent": self.battery_capacity_percent,
            "battery_measurement_method": self.battery_measurement_method,
            "bridge_schema_version": self.bridge_schema_version,
            "cpu_count": self.cpu_count,
            "cpu_idle_delta": self.cpu_idle_delta,
            "cpu_measurement_method": self.cpu_measurement_method,
            "cpu_total_delta": self.cpu_total_delta,
            "fixture_only": self.fixture_only,
            "load_average_1m_after": self.load_average_1m_after,
            "load_average_1m_before": self.load_average_1m_before,
            "logical_tick": self.logical_tick,
            "measurement_policy_version": self.measurement_policy_version,
            "mem_available_kib": self.mem_available_kib,
            "mem_total_kib": self.mem_total_kib,
            "memory_measurement_method": self.memory_measurement_method,
            "process_cpu_seconds": self.process_cpu_seconds,
            "raw_values": [[field, value] for field, value in self.raw_values],
            "runtime_source_read_only": self.runtime_source_read_only,
            "schema_version": self.schema_version,
            "source_instance_id": self.source_instance_id,
            "source_integrity_digest": self.source_integrity_digest,
            "source_snapshot_id": self.source_snapshot_id,
            "wall_seconds": self.wall_seconds,
        }


def _validate_snapshot_sequence(snapshots: Sequence[PhoneEnergyBudgetRuntimeSnapshot], *, source_instance_id: str) -> tuple[PhoneEnergyBudgetRuntimeSnapshot, ...]:
    items = tuple(snapshots)
    if len(items) != REQUIRED_RAW_RECORD_COUNT:
        raise PhoneEnergyBudgetWitnessError("energy-budget phone witness requires exactly three snapshots")
    if any(type(item) is not PhoneEnergyBudgetRuntimeSnapshot for item in items):
        raise PhoneEnergyBudgetWitnessError("energy-budget witness requires exact immutable snapshot types")
    if any(item.source_instance_id != source_instance_id for item in items):
        raise PhoneEnergyBudgetWitnessError("attestation and snapshots must bind one source instance")
    ticks = tuple(item.logical_tick for item in items)
    if ticks != tuple(sorted(set(ticks))):
        raise PhoneEnergyBudgetWitnessError("energy-budget snapshot ticks must be strictly increasing")
    if ticks[-1] - ticks[0] < REQUIRED_LOGICAL_SPAN_TICKS:
        raise PhoneEnergyBudgetWitnessError("energy-budget snapshots do not satisfy minimum logical span")
    return items


def derive_detached_energy_budget_evidence(snapshots: Sequence[PhoneEnergyBudgetRuntimeSnapshot]) -> RegistryAxisPositiveConfidenceEvidence:
    items = tuple(snapshots)
    if not items:
        raise PhoneEnergyBudgetWitnessError("energy-budget evidence requires snapshots")
    validated = _validate_snapshot_sequence(items, source_instance_id=items[0].source_instance_id)
    return derive_operational_axis_evidence(tuple(item.to_operational_raw_record() for item in validated))


def _attestation_review_mapping(attestation: OperatorPublicLaunchAttestation, local_verification_trace_digest: str) -> dict[str, Any]:
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
class PhoneEnergyBudgetWitness:
    attestation: OperatorPublicLaunchAttestation
    local_verification_trace_digest: str
    snapshots: tuple[PhoneEnergyBudgetRuntimeSnapshot, ...]
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
        if type(self.attestation) is not OperatorPublicLaunchAttestation or self.attestation.fixture_only:
            raise PhoneEnergyBudgetWitnessError("witness requires an exact non-fixture public attestation")
        snapshots = _validate_snapshot_sequence(self.snapshots, source_instance_id=self.attestation.source_instance_id)
        if type(self.evidence) is not RegistryAxisPositiveConfidenceEvidence:
            raise PhoneEnergyBudgetWitnessError("witness requires exact positive-confidence evidence")
        if self.evidence.axis != AXIS or self.evidence.source_instance_id != self.attestation.source_instance_id:
            raise PhoneEnergyBudgetWitnessError("energy-budget evidence does not bind the attested source")
        if self.evidence != derive_detached_energy_budget_evidence(snapshots):
            raise PhoneEnergyBudgetWitnessError("energy-budget evidence is not the exact snapshot derivation")
        if self.schema_version != WITNESS_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise PhoneEnergyBudgetWitnessError("energy-budget witness must remain exact shadow-only material")
        if any((self.reviewed_attestation_registered, self.runtime_provenance_verifier_registered, self.production_source_verifier_registered, self.retained_real_observation, self.observation_window_started, self.m3_b_complete, self.m3_c_open, self.m3_e_authority_open, self.cutover_authorized)):
            raise PhoneEnergyBudgetWitnessError("preflight witness cannot claim review, verifier, retention, window, or authority")
        object.__setattr__(self, "snapshots", snapshots)

    def private_mapping(self) -> dict[str, Any]:
        return {"authority": self.authority, "attestation": self.attestation.to_mapping(), "evidence": self.evidence.to_mapping(), "local_verification_trace_digest": self.local_verification_trace_digest, "schema_version": self.schema_version, "snapshots": [item.private_mapping() for item in self.snapshots]}

    @property
    def private_material_digest(self) -> str:
        return _digest(self.private_mapping(), "phone_energy_budget_private_witness")

    def public_review_mapping(self) -> dict[str, Any]:
        mapping = {
            "authority": self.authority,
            "attestation": self.attestation.to_mapping(),
            "attestation_local_review": _attestation_review_mapping(self.attestation, self.local_verification_trace_digest),
            "axis": AXIS,
            "battery_measurement_methods": sorted({item.battery_measurement_method for item in self.snapshots}),
            "cpu_measurement_methods": sorted({item.cpu_measurement_method for item in self.snapshots}),
            "cutover_authorized": self.cutover_authorized,
            "evidence": self.evidence.to_mapping(),
            "evidence_digest": self.evidence.evidence_digest,
            "evidence_observed_tick": self.evidence.observed_tick,
            "fixture_only": False,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "measurement_policy_version": MEASUREMENT_POLICY_VERSION,
            "memory_measurement_methods": sorted({item.memory_measurement_method for item in self.snapshots}),
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
        mapping["public_review_digest"] = _digest(mapping, "phone_energy_budget_public_review")
        return mapping


def build_phone_energy_budget_witness(*, private_nonce: bytes, runtime_instance_id: str, source_instance_id: str, repository_head_sha: str, launch_attestation_id: str, snapshots: Sequence[PhoneEnergyBudgetRuntimeSnapshot], launch_logical_tick: int = 0, entrypoint_id: str = ENTRYPOINT_ID) -> PhoneEnergyBudgetWitness:
    items = _validate_snapshot_sequence(snapshots, source_instance_id=source_instance_id)
    attestation = build_operator_public_launch_attestation(OperatorLaunchBinding(runtime_instance_id=runtime_instance_id, source_instance_id=source_instance_id, repository_head_sha=repository_head_sha, entrypoint_id=entrypoint_id, launch_attestation_id=launch_attestation_id, logical_tick=launch_logical_tick, fixture_only=False), private_nonce)
    local_trace = verify_operator_private_binding(attestation, private_nonce)
    evidence = derive_detached_energy_budget_evidence(items)
    return PhoneEnergyBudgetWitness(attestation=attestation, local_verification_trace_digest=local_trace, snapshots=items, evidence=evidence)
