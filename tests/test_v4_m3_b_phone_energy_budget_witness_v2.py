from __future__ import annotations

from pathlib import Path

import pytest

from core.m3_b_phone_energy_budget_witness_v2 import (
    AXIS,
    BATTERY_METHOD_SYSFS,
    BATTERY_METHOD_TERMUX_API,
    CPU_METHOD_LOADAVG,
    CPU_METHOD_PROC_STAT,
    DEFAULT_SOURCE_INSTANCE_ID,
    MEASUREMENT_POLICY_VERSION,
    MEMORY_METHOD_PROC_MEMINFO,
    MEMORY_METHOD_SYSCONF,
    PhoneEnergyBudgetRuntimeSnapshot,
    PhoneEnergyBudgetWitnessError,
    build_phone_energy_budget_witness,
    derive_detached_energy_budget_evidence,
)
from scripts.operator import m3_b_phone_energy_budget_witness_v2 as cli


def _proc_snapshot(tick: int) -> PhoneEnergyBudgetRuntimeSnapshot:
    return PhoneEnergyBudgetRuntimeSnapshot(
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        logical_tick=tick,
        cpu_measurement_method=CPU_METHOD_PROC_STAT,
        cpu_total_delta=100,
        cpu_idle_delta=40,
        process_cpu_seconds=0.5,
        wall_seconds=1.0,
        cpu_count=4,
        mem_total_kib=1000,
        mem_available_kib=800,
        memory_measurement_method=MEMORY_METHOD_PROC_MEMINFO,
        battery_capacity_percent=80,
        battery_measurement_method=BATTERY_METHOD_SYSFS,
    )


def _load_snapshot(tick: int, before: float = 1.0, after: float = 3.0) -> PhoneEnergyBudgetRuntimeSnapshot:
    return PhoneEnergyBudgetRuntimeSnapshot(
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        logical_tick=tick,
        cpu_measurement_method=CPU_METHOD_LOADAVG,
        load_average_1m_before=before,
        load_average_1m_after=after,
        process_cpu_seconds=0.5,
        wall_seconds=1.0,
        cpu_count=4,
        mem_total_kib=1000,
        mem_available_kib=800,
        memory_measurement_method=MEMORY_METHOD_SYSCONF,
        battery_capacity_percent=80,
        battery_measurement_method=BATTERY_METHOD_TERMUX_API,
    )


def test_v2_loadavg_fallback_maps_to_existing_operational_contract() -> None:
    snapshot = _load_snapshot(0)
    assert snapshot.available_cpu_budget == pytest.approx(0.5)
    assert snapshot.available_memory_budget == pytest.approx(0.8)
    assert snapshot.battery_governor_band == pytest.approx(0.8)
    assert snapshot.foreground_load == pytest.approx(0.125)
    record = snapshot.to_operational_raw_record()
    assert record.axis == AXIS
    assert record.synthetic is False
    assert record.runtime_polled is False
    assert record.raw_observation_digest == record.recalculated_raw_observation_digest


def test_v2_three_mixed_real_methods_derive_positive_confidence() -> None:
    evidence = derive_detached_energy_budget_evidence(
        (_load_snapshot(0), _proc_snapshot(1), _load_snapshot(2, 2.0, 2.0))
    )
    assert evidence.axis == AXIS
    assert evidence.observed_tick == 2
    assert 0.0 <= evidence.value <= 1.0
    assert 0.5 <= evidence.confidence <= 1.0


def test_v2_public_review_names_methods_without_raw_snapshots() -> None:
    witness = build_phone_energy_budget_witness(
        private_nonce=b"n" * 32,
        runtime_instance_id="runtime:phone:test-energy-budget-v2",
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        repository_head_sha="a" * 40,
        launch_attestation_id="operator-attestation:phone:test-energy-budget-v2",
        snapshots=(_load_snapshot(0), _proc_snapshot(1), _load_snapshot(2)),
    )
    public = witness.public_review_mapping()
    assert "snapshots" not in public
    assert public["measurement_policy_version"] == MEASUREMENT_POLICY_VERSION
    assert public["cpu_measurement_methods"] == sorted({CPU_METHOD_LOADAVG, CPU_METHOD_PROC_STAT})
    assert public["memory_measurement_methods"] == sorted({MEMORY_METHOD_PROC_MEMINFO, MEMORY_METHOD_SYSCONF})
    assert public["battery_measurement_methods"] == sorted({BATTERY_METHOD_SYSFS, BATTERY_METHOD_TERMUX_API})
    assert public["retained_real_observation"] is False
    assert public["observation_window_started"] is False
    assert public["cutover_authorized"] is False


def test_v2_cpu_probe_falls_back_when_proc_stat_is_denied(monkeypatch) -> None:
    def denied() -> tuple[int, int]:
        raise PermissionError("blocked")

    monkeypatch.setattr(cli, "_proc_cpu_counters", denied)
    monkeypatch.setattr(cli, "_load_average_1m", lambda: 1.25)
    method, total, idle, load = cli._cpu_probe_start()
    assert (method, total, idle, load) == (CPU_METHOD_LOADAVG, None, None, 1.25)
    assert cli._cpu_probe_finish(method) == (None, None, 1.25)


def test_v2_memory_and_battery_fallbacks_are_explicit(monkeypatch, tmp_path: Path) -> None:
    def denied_memory() -> tuple[int, int]:
        raise PermissionError("blocked")

    monkeypatch.setattr(cli, "_proc_memory_budget", denied_memory)
    monkeypatch.setattr(cli, "_sysconf_memory_budget", lambda: (8000, 3000))
    assert cli._memory_budget() == (MEMORY_METHOD_SYSCONF, 8000, 3000)

    monkeypatch.setattr(cli, "_termux_api_battery_capacity", lambda: 73)
    assert cli._battery_capacity(tmp_path / "blocked-capacity") == (
        BATTERY_METHOD_TERMUX_API,
        73,
    )


def test_v2_snapshot_fails_closed_on_method_mismatch() -> None:
    with pytest.raises(PhoneEnergyBudgetWitnessError, match="cannot carry proc-stat deltas"):
        PhoneEnergyBudgetRuntimeSnapshot(
            source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
            logical_tick=0,
            cpu_measurement_method=CPU_METHOD_LOADAVG,
            cpu_total_delta=100,
            cpu_idle_delta=40,
            load_average_1m_before=1.0,
            load_average_1m_after=1.0,
            process_cpu_seconds=0.1,
            wall_seconds=1.0,
            cpu_count=4,
            mem_total_kib=1000,
            mem_available_kib=800,
            memory_measurement_method=MEMORY_METHOD_SYSCONF,
            battery_capacity_percent=80,
            battery_measurement_method=BATTERY_METHOD_TERMUX_API,
        )
