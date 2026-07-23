from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from core.m3_b_operational_registry_source_binding import (
    BINDING_SCHEMA_VERSION,
    OPERATIONAL_AXES,
    POSITIVE_CONFIDENCE_BLOCKER,
    RAW_SCHEMA_VERSION,
    REMAINING_BINDING_BLOCKER,
    SOURCE_FAMILY,
    OperationalRegistryRawRecord,
    OperationalRegistrySourceBindingError,
    OperationalRegistrySourceBindingSet,
    derive_operational_axis_evidence,
    operational_raw_observation_digest,
    operational_registry_source_bindings,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULE = ROOT / "core/m3_b_operational_registry_source_binding.py"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = (tick - 1) * 0.02
    values: dict[str, object]
    if axis == "energy_budget":
        values = {
            "available_cpu_budget": 0.82 - offset,
            "available_memory_budget": 0.78 - offset,
            "battery_governor_band": 0.74 - offset,
            "foreground_load": 0.24 + offset,
            "sampling_window_ticks": 10,
        }
    elif axis == "fatigue_pressure":
        values = {
            "active_processing_ticks": 4 + tick,
            "queue_pressure": 0.30 + offset,
            "recovery_interval_ticks": 3,
            "sampling_window_ticks": 10,
            "task_switch_count": 2 + tick,
        }
    elif axis == "recovery_need":
        values = {
            "active_processing_ticks": 5 + tick,
            "cooldown_ticks": 2,
            "recent_overload_count": tick,
            "sampling_window_ticks": 10,
            "successful_recovery_count": 1,
        }
    elif axis == "overload_risk":
        values = {
            "concurrent_demand_count": 2 + tick,
            "latency_budget_ratio": 0.32 + offset,
            "memory_pressure_ratio": 0.36 + offset,
            "queue_depth": 1 + tick,
            "thermal_governor_band": 0.28 + offset,
        }
    else:
        raise AssertionError(axis)
    return tuple(sorted(values.items()))


def _record(
    axis: str,
    tick: int,
    *,
    source_instance_id: str = "test:operational-source:v1",
    source_schema_version: str = "test.operational-source.v1",
    observation_id: str | None = None,
    snapshot_id: str | None = None,
    raw_values: tuple[tuple[str, object], ...] | None = None,
) -> OperationalRegistryRawRecord:
    values = _raw_values(axis, tick) if raw_values is None else raw_values
    resolved_observation_id = (
        f"test:{axis}:observation:{tick}"
        if observation_id is None
        else observation_id
    )
    resolved_snapshot_id = (
        f"test:{axis}:snapshot:{tick}" if snapshot_id is None else snapshot_id
    )
    source_integrity_digest = _sha(
        f"source:{axis}:{tick}:{source_instance_id}:{source_schema_version}"
    )
    raw_digest = operational_raw_observation_digest(
        axis=axis,
        logical_tick=tick,
        observation_id=resolved_observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=resolved_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        raw_values=values,
    )
    return OperationalRegistryRawRecord(
        axis=axis,
        logical_tick=tick,
        observation_id=resolved_observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=resolved_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        raw_observation_digest=raw_digest,
        raw_values=values,
    )


def _records(axis: str) -> tuple[OperationalRegistryRawRecord, ...]:
    return tuple(_record(axis, tick) for tick in (1, 2, 3))


def test_binding_set_has_exact_four_operational_axes_and_remaining_blockers():
    result = operational_registry_source_bindings()
    assert result.binding_count == 4
    assert result.remaining_axis_count == 33
    assert tuple(item.axis for item in result.bindings) == OPERATIONAL_AXES
    assert result.blockers == (
        REMAINING_BINDING_BLOCKER,
        POSITIVE_CONFIDENCE_BLOCKER,
    )
    assert len(result.binding_set_digest) == 64
    assert all(len(item.binding_digest) == 64 for item in result.bindings)
    assert all(item.binding_implemented is True for item in result.bindings)
    assert result.production_capture_present is False
    assert result.observation_window_started is False
    assert result.m3_b_complete is False
    assert result.m3_c_open is False
    assert result.m3_e_authority_open is False
    assert result.cutover_authorized is False


@pytest.mark.parametrize("axis", OPERATIONAL_AXES)
def test_exact_raw_records_derive_deterministic_positive_confidence_evidence(axis: str):
    records = _records(axis)
    first = derive_operational_axis_evidence(records)
    second = derive_operational_axis_evidence(records)
    assert first == second
    assert first.axis == axis
    assert 0.0 <= first.value <= 1.0
    assert 0.0 < first.confidence <= 1.0
    assert first.observed_tick == 3
    assert first.source_family == SOURCE_FAMILY
    assert first.source_instance_id == "test:operational-source:v1"
    assert first.source_schema_version == RAW_SCHEMA_VERSION
    assert first.acquisition_method == records[0].acquisition_method
    assert first.verification_method == records[0].verification_method
    assert first.model_or_rule_version == (
        f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1"
    )
    assert len(first.source_integrity_digest) == 64
    assert len(first.raw_observation_digest) == 64
    assert all(
        item.raw_observation_digest == item.recalculated_raw_observation_digest
        for item in records
    )


def test_raw_observation_digest_is_bound_to_identity_time_source_and_values():
    record = _record("energy_budget", 1)
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="digest does not match",
    ):
        replace(record, raw_observation_digest=_sha("tampered"))
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="digest does not match",
    ):
        replace(record, logical_tick=2)
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="digest does not match",
    ):
        replace(record, source_snapshot_id="test:changed-snapshot")
    changed = tuple(
        (field, 0.1 if field == "foreground_load" else value)
        for field, value in record.raw_values
    )
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="digest does not match",
    ):
        replace(record, raw_values=changed)


def test_raw_record_rejects_missing_reordered_duplicate_and_out_of_range_fields():
    valid = _record("energy_budget", 1)
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="canonical axis source plan",
    ):
        _record("energy_budget", 1, raw_values=valid.raw_values[:-1])
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="canonical axis source plan",
    ):
        _record(
            "energy_budget",
            1,
            raw_values=(valid.raw_values[1], valid.raw_values[0], *valid.raw_values[2:]),
        )
    duplicate = valid.raw_values[:-1] + (valid.raw_values[-2],)
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="canonical axis source plan",
    ):
        _record("energy_budget", 1, raw_values=duplicate)
    out_of_range = tuple(
        (field, 1.1 if field == "foreground_load" else value)
        for field, value in valid.raw_values
    )
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="inside \\[0,1\\]",
    ):
        _record("energy_budget", 1, raw_values=out_of_range)


def test_tick_counts_cannot_exceed_sampling_window():
    fatigue = dict(_raw_values("fatigue_pressure", 1))
    fatigue["active_processing_ticks"] = 11
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="cannot exceed sampling window",
    ):
        _record("fatigue_pressure", 1, raw_values=tuple(sorted(fatigue.items())))
    recovery = dict(_raw_values("recovery_need", 1))
    recovery["cooldown_ticks"] = 11
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="cannot exceed sampling window",
    ):
        _record("recovery_need", 1, raw_values=tuple(sorted(recovery.items())))


@pytest.mark.parametrize(
    "field",
    ("synthetic", "proposal_only", "registry_owner_source", "runtime_polled"),
)
def test_forbidden_raw_origins_fail_closed(field: str):
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="synthetic, proposal-only, circular, or runtime-polled",
    ):
        replace(_record("energy_budget", 1), **{field: True})


def test_derivation_rejects_insufficient_count_mixed_axis_and_unsorted_ticks():
    energy = _records("energy_budget")
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="insufficient raw record count",
    ):
        derive_operational_axis_evidence(energy[:2])
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="cannot mix axes",
    ):
        derive_operational_axis_evidence(
            (energy[0], energy[1], _record("fatigue_pressure", 3))
        )
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="ticks must be sorted",
    ):
        derive_operational_axis_evidence((energy[1], energy[0], energy[2]))


def test_derivation_rejects_duplicate_ticks_ids_snapshots_and_mixed_source_contract():
    records = _records("energy_budget")
    duplicate_tick = _record(
        "energy_budget",
        2,
        observation_id="test:energy_budget:observation:duplicate-tick",
        snapshot_id="test:energy_budget:snapshot:duplicate-tick",
    )
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="ticks must be unique",
    ):
        derive_operational_axis_evidence((records[0], records[1], duplicate_tick))
    duplicate_id = _record(
        "energy_budget",
        3,
        observation_id=records[0].observation_id,
    )
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="observation ids must be unique",
    ):
        derive_operational_axis_evidence((records[0], records[1], duplicate_id))
    duplicate_snapshot = _record(
        "energy_budget",
        3,
        snapshot_id=records[0].source_snapshot_id,
    )
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="snapshots must be unique",
    ):
        derive_operational_axis_evidence((records[0], records[1], duplicate_snapshot))
    mixed_schema = _record(
        "energy_budget",
        3,
        source_schema_version="test.operational-source.v2",
    )
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="share one source_schema_version",
    ):
        derive_operational_axis_evidence((records[0], records[1], mixed_schema))
    mixed_instance = _record(
        "energy_budget",
        3,
        source_instance_id="test:other-operational-source:v1",
    )
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="share one source_instance_id",
    ):
        derive_operational_axis_evidence((records[0], records[1], mixed_instance))


def test_binding_and_binding_set_are_frozen_and_cannot_claim_authority():
    binding_set = operational_registry_source_bindings()
    binding = binding_set.bindings[0]
    with pytest.raises(FrozenInstanceError):
        binding.binding_id = "changed"  # type: ignore[misc]
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="canonical plan",
    ):
        replace(binding, minimum_raw_record_count=99)
    for field in (
        "production_capture_present",
        "runtime_capture_installed",
        "hardware_polling_installed",
        "persistence_accessed",
        "event_append_performed",
        "observation_window_started",
        "m3_b_complete",
        "m3_c_open",
        "m3_e_authority_open",
        "cutover_authorized",
    ):
        with pytest.raises(
            OperationalRegistrySourceBindingError,
            match="cannot claim production capture, runtime, window, or authority",
        ):
            replace(binding, **{field: True})
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="exact four-axis order",
    ):
        OperationalRegistrySourceBindingSet(bindings=binding_set.bindings[:-1])


def test_core_module_has_no_io_polling_scheduler_event_or_runtime_surface():
    tree = ast.parse(CORE_MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not imported & {
        "os",
        "pathlib",
        "persistence",
        "psutil",
        "sqlite3",
        "subprocess",
        "threading",
        "time",
    }
    assert not called & {
        "append_event",
        "connect",
        "emit",
        "mkdir",
        "open",
        "poll",
        "save",
        "start",
        "write",
        "write_text",
    }
