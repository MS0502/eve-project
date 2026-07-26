import json

import pytest

from core.m3_b_phone_fatigue_pressure_witness import (
    DEFAULT_SOURCE_INSTANCE_ID,
    TASK_SWITCH_METHOD_PROC_SELF_STATUS,
    TASK_SWITCH_METHOD_RUSAGE,
    PhoneFatiguePressureRuntimeSnapshot,
    PhoneFatiguePressureWitnessError,
    build_phone_fatigue_pressure_witness,
    derive_detached_fatigue_pressure_evidence,
)
from scripts.operator import m3_b_phone_fatigue_pressure_witness as operator_cli


def _snapshot(
    tick: int,
    *,
    process_cpu_seconds: float = 0.8,
    wall_seconds: float = 2.0,
    load_before: float = 0.8,
    load_after: float = 1.2,
    task_switch_count: int = 3,
    method: str = TASK_SWITCH_METHOD_RUSAGE,
) -> PhoneFatiguePressureRuntimeSnapshot:
    return PhoneFatiguePressureRuntimeSnapshot(
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        logical_tick=tick,
        process_cpu_seconds=process_cpu_seconds,
        wall_seconds=wall_seconds,
        cpu_count=2,
        load_average_1m_before=load_before,
        load_average_1m_after=load_after,
        task_switch_count=task_switch_count,
        task_switch_measurement_method=method,
    )


def _snapshots() -> tuple[PhoneFatiguePressureRuntimeSnapshot, ...]:
    return (
        _snapshot(0),
        _snapshot(
            1,
            process_cpu_seconds=1.0,
            load_before=1.2,
            load_after=1.8,
            task_switch_count=4,
        ),
        _snapshot(
            2,
            process_cpu_seconds=0.6,
            load_before=0.4,
            load_after=0.6,
            task_switch_count=2,
            method=TASK_SWITCH_METHOD_PROC_SELF_STATUS,
        ),
    )


def test_snapshot_derives_exact_governed_fatigue_fields() -> None:
    snapshot = _snapshot(0)

    assert snapshot.sampling_window_ticks == 2_000_000
    assert snapshot.active_processing_ticks == 400_000
    assert snapshot.recovery_interval_ticks == 1_600_000
    assert snapshot.queue_pressure == pytest.approx(0.5)
    fields = tuple(field for field, _ in snapshot.raw_values)
    raw = dict(snapshot.raw_values)
    assert fields == (
        "active_processing_ticks",
        "queue_pressure",
        "recovery_interval_ticks",
        "sampling_window_ticks",
        "task_switch_count",
    )
    assert raw["active_processing_ticks"] == 400_000
    assert raw["queue_pressure"] == pytest.approx(0.5)
    assert raw["recovery_interval_ticks"] == 1_600_000
    assert raw["sampling_window_ticks"] == 2_000_000
    assert raw["task_switch_count"] == 3

    record = snapshot.to_operational_raw_record()
    assert record.axis == "fatigue_pressure"
    assert record.source_instance_id == DEFAULT_SOURCE_INSTANCE_ID
    assert record.synthetic is False
    assert record.runtime_polled is False
    assert record.recalculated_raw_observation_digest == record.raw_observation_digest


def test_three_real_shape_snapshots_derive_positive_confidence() -> None:
    evidence = derive_detached_fatigue_pressure_evidence(_snapshots())

    assert evidence.axis == "fatigue_pressure"
    assert evidence.source_instance_id == DEFAULT_SOURCE_INSTANCE_ID
    assert 0.0 <= evidence.value <= 1.0
    assert 0.5 <= evidence.confidence <= 1.0
    assert evidence.observed_tick == 2
    assert evidence.synthetic is False
    assert evidence.proposal_only is False


def test_public_review_is_digest_only_and_claims_no_promotion() -> None:
    witness = build_phone_fatigue_pressure_witness(
        private_nonce=b"fatigue-private-nonce-material-0001",
        runtime_instance_id="phone-runtime-fatigue-001",
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        repository_head_sha="1" * 40,
        launch_attestation_id="fatigue-launch-001",
        snapshots=_snapshots(),
    )

    public = witness.public_review_mapping()
    encoded = json.dumps(public, sort_keys=True)

    assert public["axis"] == "fatigue_pressure"
    assert public["raw_record_count"] == 3
    assert public["private_raw_location"] == "operator_private_companion_only"
    assert public["fixture_only"] is False
    assert public["reviewed_attestation_registered"] is False
    assert public["runtime_provenance_verifier_registered"] is False
    assert public["production_source_verifier_registered"] is False
    assert public["retained_real_observation"] is False
    assert public["observation_window_started"] is False
    assert public["m3_b_complete"] is False
    assert public["m3_c_open"] is False
    assert public["m3_e_authority_open"] is False
    assert public["cutover_authorized"] is False
    assert public["process_cpu_measurement_methods"] == ["os_times_process_cpu_v1"]
    assert public["queue_measurement_methods"] == [
        "kernel_loadavg_1m_normalized_v1"
    ]
    assert public["task_switch_measurement_methods"] == [
        "getrusage_context_switch_delta_v1",
        "proc_self_status_context_switch_delta_v1",
    ]
    assert "process_cpu_seconds" not in encoded
    assert "wall_seconds" not in encoded
    assert "load_average_1m_before" not in encoded
    assert "task_switch_count" not in encoded
    assert len(public["public_review_digest"]) == 64


def test_snapshot_fails_closed_when_process_cpu_exceeds_visible_capacity() -> None:
    with pytest.raises(
        PhoneFatiguePressureWitnessError,
        match="process CPU exceeds visible CPU capacity",
    ):
        _snapshot(0, process_cpu_seconds=4.1, wall_seconds=2.0)


def test_snapshot_sequence_fails_closed_on_duplicate_tick() -> None:
    snapshots = (_snapshot(0), _snapshot(1), _snapshot(1))

    with pytest.raises(
        PhoneFatiguePressureWitnessError,
        match="strictly increasing",
    ):
        derive_detached_fatigue_pressure_evidence(snapshots)


def test_operator_context_switch_probe_uses_proc_self_fallback(monkeypatch) -> None:
    def unavailable() -> int:
        raise OSError("blocked")

    monkeypatch.setattr(operator_cli, "_rusage_task_switches", unavailable)
    monkeypatch.setattr(operator_cli, "_proc_self_task_switches", lambda: 17)

    assert operator_cli._task_switch_probe_start() == (
        TASK_SWITCH_METHOD_PROC_SELF_STATUS,
        17,
    )
