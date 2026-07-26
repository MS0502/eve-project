from __future__ import annotations

import pytest

from core.m3_b_phone_energy_budget_witness import (
    AXIS,
    DEFAULT_SOURCE_INSTANCE_ID,
    MEASUREMENT_POLICY_VERSION,
    PhoneEnergyBudgetRuntimeSnapshot,
    PhoneEnergyBudgetWitnessError,
    build_phone_energy_budget_witness,
    derive_detached_energy_budget_evidence,
)


def _snapshot(
    tick: int,
    *,
    idle: int = 40,
    total: int = 100,
    process_cpu: float = 0.5,
    wall: float = 1.0,
    memory_available: int = 800,
    memory_total: int = 1000,
    battery: int = 80,
) -> PhoneEnergyBudgetRuntimeSnapshot:
    return PhoneEnergyBudgetRuntimeSnapshot(
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        logical_tick=tick,
        cpu_total_delta=total,
        cpu_idle_delta=idle,
        process_cpu_seconds=process_cpu,
        wall_seconds=wall,
        cpu_count=4,
        mem_total_kib=memory_total,
        mem_available_kib=memory_available,
        battery_capacity_percent=battery,
    )


def test_energy_budget_snapshot_maps_exact_operational_contract() -> None:
    snapshot = _snapshot(0)
    assert snapshot.available_cpu_budget == pytest.approx(0.4)
    assert snapshot.available_memory_budget == pytest.approx(0.8)
    assert snapshot.battery_governor_band == pytest.approx(0.8)
    assert snapshot.foreground_load == pytest.approx(0.125)
    assert tuple(field for field, _ in snapshot.raw_values) == (
        "available_cpu_budget",
        "available_memory_budget",
        "battery_governor_band",
        "foreground_load",
        "sampling_window_ticks",
    )

    record = snapshot.to_operational_raw_record()
    assert record.axis == AXIS
    assert record.source_instance_id == DEFAULT_SOURCE_INSTANCE_ID
    assert record.logical_tick == 0
    assert record.runtime_polled is False
    assert record.synthetic is False
    assert record.raw_observation_digest == record.recalculated_raw_observation_digest


def test_three_real_windows_derive_positive_confidence_energy_budget() -> None:
    snapshots = (
        _snapshot(0, idle=35, process_cpu=0.60, memory_available=760, battery=79),
        _snapshot(1, idle=45, process_cpu=0.40, memory_available=780, battery=79),
        _snapshot(2, idle=50, process_cpu=0.30, memory_available=800, battery=78),
    )
    evidence = derive_detached_energy_budget_evidence(snapshots)
    assert evidence.axis == AXIS
    assert evidence.source_instance_id == DEFAULT_SOURCE_INSTANCE_ID
    assert evidence.observed_tick == 2
    assert 0.0 <= evidence.value <= 1.0
    assert 0.5 <= evidence.confidence <= 1.0
    assert evidence.evidence_digest


def test_public_review_is_digest_only_for_raw_device_counters() -> None:
    snapshots = (_snapshot(0), _snapshot(1), _snapshot(2))
    witness = build_phone_energy_budget_witness(
        private_nonce=b"n" * 32,
        runtime_instance_id="runtime:phone:test-energy-budget",
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        repository_head_sha="a" * 40,
        launch_attestation_id="operator-attestation:phone:test-energy-budget",
        snapshots=snapshots,
    )

    private_mapping = witness.private_mapping()
    public_mapping = witness.public_review_mapping()
    assert len(private_mapping["snapshots"]) == 3
    assert "cpu_total_delta" in private_mapping["snapshots"][0]
    assert "snapshots" not in public_mapping
    assert public_mapping["axis"] == AXIS
    assert public_mapping["raw_record_count"] == 3
    assert public_mapping["measurement_policy_version"] == MEASUREMENT_POLICY_VERSION
    assert len(public_mapping["snapshot_integrity_digests"]) == 3
    assert public_mapping["private_raw_location"] == "operator_private_companion_only"
    assert public_mapping["reviewed_attestation_registered"] is False
    assert public_mapping["runtime_provenance_verifier_registered"] is False
    assert public_mapping["production_source_verifier_registered"] is False
    assert public_mapping["retained_real_observation"] is False
    assert public_mapping["observation_window_started"] is False
    assert public_mapping["m3_b_complete"] is False
    assert public_mapping["m3_c_open"] is False
    assert public_mapping["m3_e_authority_open"] is False
    assert public_mapping["cutover_authorized"] is False
    assert public_mapping["public_review_digest"]


def test_witness_rejects_insufficient_record_count_and_span() -> None:
    with pytest.raises(PhoneEnergyBudgetWitnessError, match="exactly three"):
        derive_detached_energy_budget_evidence((_snapshot(0), _snapshot(1)))

    with pytest.raises(PhoneEnergyBudgetWitnessError, match="strictly increasing"):
        derive_detached_energy_budget_evidence((_snapshot(0), _snapshot(1), _snapshot(1)))


def test_snapshot_fails_closed_on_impossible_device_counters() -> None:
    with pytest.raises(PhoneEnergyBudgetWitnessError, match="cpu_idle_delta"):
        _snapshot(0, idle=101, total=100)
    with pytest.raises(PhoneEnergyBudgetWitnessError, match="mem_available_kib"):
        _snapshot(0, memory_available=1001, memory_total=1000)
    with pytest.raises(PhoneEnergyBudgetWitnessError, match="battery_capacity_percent"):
        _snapshot(0, battery=101)
